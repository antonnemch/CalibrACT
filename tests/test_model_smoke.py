import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from calibract.models.activation_configs import activations
from calibract.models.conv_adapter import add_conv_to_resnet, freeze_conv_encoder
from calibract.models.lora_layers import mark_only_lora_as_trainable
from calibract.models.resnet import resnet50_base
from calibract.training.loops import build_activation_map
from calibract.training.runner import split_trainable_parameters


class ModelSmokeTests(unittest.TestCase):
    def test_relu_head_only_model_forward(self):
        model = resnet50_base(pretrained=False, num_classes=4, freeze=True)
        model.set_custom_activation_map(build_activation_map(activations["full_relu"]))
        model.eval()

        with torch.no_grad():
            output = model(torch.randn(1, 3, 64, 64))

        self.assertEqual(tuple(output.shape), (1, 4))

    def test_activation_tuning_exposes_activation_params(self):
        model = resnet50_base(pretrained=False, num_classes=4, freeze=True)
        model.set_custom_activation_map(build_activation_map(activations["stage4_act2only_channelwise_prelu"]))

        network_params, activation_params = split_trainable_parameters(model)

        self.assertGreater(sum(p.numel() for p in network_params), 0)
        self.assertEqual(sum(p.numel() for p in activation_params), 1536)

    def test_conv_adapter_has_trainable_adapters_and_head(self):
        model = resnet50_base(pretrained=False, num_classes=4)
        add_conv_to_resnet(model, reduction=16)
        freeze_conv_encoder(model)

        trainable_names = [name for name, param in model.named_parameters() if param.requires_grad]

        self.assertTrue(any("adapter" in name for name in trainable_names))
        self.assertTrue(any(name.startswith("fc.") for name in trainable_names))

    def test_lora_has_trainable_lora_params_and_head(self):
        model = resnet50_base(
            pretrained=False,
            num_classes=4,
            is_lora=True,
            lora_config={"r": 4, "lora_alpha": 8, "lora_dropout": 0, "merge_weights": True},
        )
        mark_only_lora_as_trainable(model, bias="none")

        trainable_names = [name for name, param in model.named_parameters() if param.requires_grad]

        self.assertTrue(any("lora_" in name for name in trainable_names))
        self.assertTrue(any(name.startswith("fc.") for name in trainable_names))


if __name__ == "__main__":
    unittest.main()
