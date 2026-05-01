"""Run the CalibraCT Kaggle Brain MRI benchmark grid."""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from calibract.data.loaders import load_kaggle_brain_mri
from calibract.data.summary import summarize_log
from calibract.experiment_logging import make_logger
from calibract.training.runner import (
    compute_total_experiments,
    get_model_param_combinations,
    train_conv_adapter,
    train_gpaf,
    train_lora,
)


MODEL_PARAM_MAP = {
    "GPAF": {
        "net_lr",
        "num_epochs",
        "batch_size",
        "data_subset",
        "activation_type",
        "act_lr",
        "act_optimizer",
        "modifiers",
    },
    "ConvAdapter": {"net_lr", "num_epochs", "batch_size", "data_subset", "reduction"},
    "LoRA": {"net_lr", "num_epochs", "batch_size", "data_subset", "r", "lora_alpha"},
}

DEFAULT_CONFIG = {
    "experiment": {
        "dataset_name": "Kaggle Brain MRI",
        "dataset_path": None,
        "output_dir": "outputs",
        "seed": 42,
        "reverse_order": False,
        "min_exp": 1,
        "max_exp": None,
        "print_configs_only": False,
    },
    "hyperparameters": {
        "model": ["GPAF", "LoRA", "ConvAdapter"],
        "net_lr": [1e-3],
        "num_epochs": [30],
        "batch_size": [None],
        "activation_type": [
            "full_relu",
            "stage3_4_act2_blockshared_kglap",
            "stage4_act2only_channelwise_kglap",
            "stage3_4_act2_blockshared_prelu",
            "stage4_act2only_channelwise_prelu",
            "stage3_4_act2_blockshared_swishlearn",
            "stage4_act2only_channelwise_swishlearn",
        ],
        "act_lr": [1e-6],
        "act_optimizer": ["adam"],
        "reduction": [16],
        "r": [4, 8],
        "lora_alpha": [8, 16],
        "modifiers": [{"TrainBN": False, "Deferred": None, "Regularization": None}],
        "data_subset": [0.005, 0.01, 0.1, 0.5, 1.0],
    },
}


def load_config(path: Path) -> dict:
    try:
        import yaml
    except ImportError as exc:
        default_path = ROOT / "configs" / "kaggle_brain_mri.yaml"
        if path.resolve() == default_path.resolve():
            print("PyYAML is not installed; using the built-in Kaggle Brain MRI default config.")
            return DEFAULT_CONFIG
        raise SystemExit("PyYAML is required for custom YAML configs. Install with `pip install -e .`.") from exc

    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "kaggle_brain_mri.yaml")
    parser.add_argument("--dataset-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--print-configs-only", action="store_true")
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument("--min-exp", type=int, default=None)
    parser.add_argument("--max-exp", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    experiment = config.get("experiment", {})
    hyperparams = config["hyperparameters"]

    dataset_path = (
        args.dataset_path
        or os.environ.get("CALIBRACT_DATA_DIR")
        or experiment.get("dataset_path")
        or "datasets/Kaggle Brain MRI"
    )
    output_dir = Path(
        args.output_dir
        or os.environ.get("CALIBRACT_OUTPUT_DIR")
        or experiment.get("output_dir", "outputs")
    )
    os.environ["CALIBRACT_OUTPUT_DIR"] = str(output_dir)
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    seed = int(experiment.get("seed", 42))
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model_list = list(hyperparams.get("model", ["GPAF", "LoRA", "ConvAdapter"]))
    reverse_order = args.reverse or bool(experiment.get("reverse_order", False))
    if reverse_order:
        model_list.reverse()

    min_exp = args.min_exp or experiment.get("min_exp", 1)
    max_exp = args.max_exp if args.max_exp is not None else experiment.get("max_exp")
    print_configs_only = args.print_configs_only or bool(experiment.get("print_configs_only", False))

    timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    total_experiments = compute_total_experiments(hyperparams, MODEL_PARAM_MAP)
    exp_i = 0

    for model_name in model_list:
        configs = get_model_param_combinations(model_name, MODEL_PARAM_MAP, hyperparams)
        if reverse_order:
            configs.reverse()

        for model_config in configs:
            exp_i += 1
            if exp_i < min_exp:
                continue
            if max_exp is not None and exp_i > max_exp:
                continue
            if model_config.get("lora_alpha") is not None and model_config.get("r") is not None:
                if model_config["lora_alpha"] / 2 != model_config["r"]:
                    print(f"[SKIP][{model_name}][{exp_i}/{total_experiments}] invalid LoRA alpha/r pair")
                    exp_i -= 1
                    continue

            model_config = dict(model_config)
            model_config["model"] = model_name
            if print_configs_only:
                print(f"[{model_name}] ({exp_i}/{total_experiments}) {model_config}")
                continue

            print(f"\n[{model_name}] ({exp_i}/{total_experiments}) {model_config}")
            train_loader, val_loader, test_loader, num_classes = load_kaggle_brain_mri(
                str(dataset_path),
                batch_size=model_config["batch_size"],
                subset_fraction=model_config["data_subset"],
            )
            if model_config["batch_size"] is None:
                model_config["batch_size"] = train_loader.batch_size

            dataset_summary = summarize_log("Kaggle Brain MRI", train_loader, val_loader, test_loader, num_classes)
            logger = make_logger("UNIFIED", config=model_config, timestamp=timestamp, base_dir=str(log_dir))
            logger.log_dataset_summary(dataset_summary)
            logger.log_hyperparams()

            if model_name == "GPAF":
                train_gpaf(model_config, train_loader, val_loader, test_loader, num_classes, dataset_summary, logger, device)
            elif model_name == "ConvAdapter":
                train_conv_adapter(model_config, train_loader, val_loader, test_loader, num_classes, dataset_summary, logger, device)
            elif model_name == "LoRA":
                train_lora(model_config, train_loader, val_loader, test_loader, num_classes, dataset_summary, logger, device)
            else:
                raise ValueError(f"Unknown model: {model_name}")

            logger.save()

    print(f"\nDone. Logs are under {log_dir}")


if __name__ == "__main__":
    main()
