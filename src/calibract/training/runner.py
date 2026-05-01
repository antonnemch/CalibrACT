import itertools
import os
import time
import torch
from torch import nn, optim

from calibract.models.activations import BaseActivation
from calibract.models.resnet import initialize_basic_model, initialize_lora_model
from calibract.training.loops import EarlyStopping, build_activation_map, evaluate_model, train_one_epoch
from calibract.models.conv_adapter import initialize_conv_model
from calibract.models.activation_configs import activations

def get_model_dir():
    output_dir = os.environ.get("CALIBRACT_OUTPUT_DIR", ".")
    model_dir = os.path.join(output_dir, "saved_models")
    os.makedirs(model_dir, exist_ok=True)
    return model_dir


# === Experiment Grid Generation ===
def get_model_param_combinations(model_name,model_param_map, hyperparams):
    relevant_params = sorted(model_param_map[model_name])
    param_values = [hyperparams[p] for p in relevant_params]
    param_names = relevant_params
    return [dict(zip(param_names, combination)) for combination in itertools.product(*param_values)]

def split_trainable_parameters(model):
    activation_param_ids = set()
    for module in model.modules():
        if isinstance(module, BaseActivation):
            for param in module.parameters(recurse=True):
                activation_param_ids.add(id(param))

    network_params = []
    activation_params = []
    for param in model.parameters():
        if not param.requires_grad:
            continue
        if id(param) in activation_param_ids:
            activation_params.append(param)
        else:
            network_params.append(param)
    return network_params, activation_params

# === Model Training Dispatch ===
def train_gpaf(config, train_loader, val_loader, test_loader, num_classes, dataset_summary, logger, device):
    model_dir = get_model_dir()
    model = initialize_basic_model(num_classes, device, freeze=True)
    activation_map = build_activation_map(activations[config['activation_type']])
    modifiers = config.get('modifiers', {})
    deferred_epochs = modifiers.get('Deferred', None)
    train_bn = modifiers.get('TrainBN', False)
    # Always initialize early_stopper before training loop
    early_stopper = EarlyStopping(patience=5)
    if deferred_epochs is None or deferred_epochs == 0:
        model.set_custom_activation_map(activation_map, train_bn=train_bn)
        activation_map_set = True
        logger.log_param_counts(model)
    else:
        activation_map_set = False
        print(f"Deferring activation map set for {deferred_epochs} epochs.")
    if False:
        from torchsummary import summary
        summary(model, input_size=(3, 224, 224))
    network_params, activation_params = split_trainable_parameters(model)
    net_optimizer = optim.Adam(network_params, lr=config['net_lr']) if network_params else None
    act_optimizer = None
    if config.get('act_lr') and activation_params:
        optimizer_cls = getattr(optim, config['act_optimizer'].capitalize())
        act_optimizer = optimizer_cls(activation_params, lr=config['act_lr'])
    print("\n=== Training with GPAF ===")
    best_val_acc = -float("inf")
    best_model_state = None
    best_epoch = -1
    early_stop_pending = False
    for epoch in range(config['num_epochs']):
        try:
            # Set activation map if deferred and epoch reached
            if (
                not activation_map_set
                and deferred_epochs is not None
                and epoch >= deferred_epochs
            ):
                model.set_custom_activation_map(activation_map, train_bn=train_bn)
                print(
                    f"Custom activation map set at epoch {epoch+1} (deferred {deferred_epochs} epochs)"
                )
                early_stopper = EarlyStopping(patience=5)
                logger.log_param_counts(model)
                activation_map_set = True
                network_params, activation_params = split_trainable_parameters(model)
                net_optimizer = optim.Adam(network_params, lr=config['net_lr']) if network_params else None
                act_optimizer = None
                if config.get('act_lr') and activation_params:
                    optimizer_cls = getattr(optim, config['act_optimizer'].capitalize())
                    act_optimizer = optimizer_cls(activation_params, lr=config['act_lr'])
                # If early stopping was pending, allow at least one more epoch after activation map is set
                if early_stop_pending:
                    early_stop_pending = False
            start = time.time()
            # Training step
            train_loss, acc = train_one_epoch(
                model,
                train_loader,
                net_optimizer,
                device,
                nn.CrossEntropyLoss(),
                epoch,
                logger,
                act_optimizer,
                modifiers,
            )
            # Validation step
            val_loss, val_acc = evaluate_model(
                model, val_loader, device, nn.CrossEntropyLoss(), logger, epoch
            )
            # Logging
            elapsed = time.time() - start
            logger.log_epoch_metrics(
                epoch, train_loss, val_loss, acc, elapsed, torch.cuda.max_memory_allocated()
            )
            print(
                f"GPAF Epoch {epoch+1}/{config['num_epochs']} - Val Acc: {val_acc:.4f} - Val Loss: {val_loss:.4f}"
            )
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = model.state_dict()
                best_epoch = epoch
            # Early stopping check
            if early_stopper.step(val_loss):
                if not activation_map_set:
                    # Defer early stopping until after activation map is set
                    print(f"Early stopping triggered at epoch {epoch+1}, but activation map not set yet. Continuing until after activation map is set.")
                    early_stop_pending = True
                else:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break
        except Exception as e:
            print(f"[ERROR][GPAF][Epoch {epoch+1}] {e}")
            import traceback
            traceback.print_exc()
            break
    # Save best model weights
    best_model_path = os.path.join(
        model_dir, f"{logger.model_name}_{logger.config_id}_best.pt"
    )
    torch.save(best_model_state, best_model_path)
    print(f"Best model (epoch {best_epoch+1}) saved to {best_model_path}")
    # Load best model for test evaluation
    model.load_state_dict(torch.load(best_model_path, weights_only=True))
    # Final test evaluation and logging (handled by evaluate_model)
    try:
        test_acc = evaluate_model(
            model, test_loader, device, nn.CrossEntropyLoss(), logger, phase='test'
        )
        print(f"GPAF Test Acc: {test_acc:.4f}")
    except Exception as e:
        print(f"[ERROR][GPAF][Test] {e}")
        import traceback
        traceback.print_exc()
        test_acc = None
    # Save final model (last epoch)
    final_model_path = os.path.join(
        model_dir, f"{logger.model_name}_{logger.config_id}_final.pt"
    )
    torch.save(model.state_dict(), final_model_path)
    logger.log_model_path(final_model_path)
    # Optionally log final result for unified logger compatibility
    if hasattr(logger, 'log_final_result'):
        logger.log_final_result(model_name="GPAF", config=config, test_acc=test_acc)


def train_conv_adapter(config, train_loader, val_loader, test_loader, num_classes, dataset_summary, logger, device):
    model_dir = get_model_dir()
    model = initialize_conv_model(num_classes, device, reduction=config['reduction'])
    optimizer = optim.Adam(model.parameters(), lr=config['net_lr'])
    early_stopper = EarlyStopping(patience=5)
    logger.log_param_counts(model)
    print("\n=== Training with ConvAdapter ===")
    best_val_acc = -float("inf")
    best_model_state = None
    best_epoch = -1
    for epoch in range(config['num_epochs']):
        try:
            start = time.time()
            train_loss, acc = train_one_epoch(model, train_loader, optimizer, device, nn.CrossEntropyLoss(), epoch, logger)
            val_loss, val_acc = evaluate_model(model, val_loader, device, nn.CrossEntropyLoss(), logger, epoch)
            elapsed = time.time() - start
            logger.log_epoch_metrics(epoch, train_loss, val_loss, acc, elapsed, torch.cuda.max_memory_allocated())
            print(f"ConvAdapter Epoch {epoch+1}/{config['num_epochs']} - Val Acc: {val_acc:.4f} - Val Loss: {val_loss:.4f}")
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = model.state_dict()
                best_epoch = epoch
            if early_stopper.step(val_loss):
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
        except Exception as e:
            print(f"[ERROR][ConvAdapter][Epoch {epoch+1}] {e}")
            import traceback
            traceback.print_exc()
            break
    best_model_path = os.path.join(
        model_dir, f"{logger.model_name}_{logger.config_id}_best.pt"
    )
    torch.save(best_model_state, best_model_path)
    print(f"Best model (epoch {best_epoch+1}) saved to {best_model_path}")
    # Final test evaluation and logging (handled by evaluate_model)
    try:
        model.load_state_dict(torch.load(best_model_path, weights_only=True))
        test_acc = evaluate_model(
            model, test_loader, device, nn.CrossEntropyLoss(), logger, phase='test'
        )
        print(f"ConvAdapter Test Acc: {test_acc:.4f}")
    except Exception as e:
        print(f"[ERROR][ConvAdapter][Test] {e}")
        import traceback
        traceback.print_exc()
        test_acc = None
    final_model_path = os.path.join(
        model_dir, f"{logger.model_name}_{logger.config_id}_final.pt"
    )
    torch.save(model.state_dict(), final_model_path)
    logger.log_model_path(final_model_path)
    if hasattr(logger, 'log_final_result'):
        logger.log_final_result(model_name="ConvAdapter", config=config, test_acc=test_acc)


def train_lora(config, train_loader, val_loader, test_loader, num_classes, dataset_summary, logger, device):
    model_dir = get_model_dir()
    lora_config = {"r": config['r'], "lora_alpha": config['lora_alpha'], "lora_dropout": 0, "merge_weights": True}
    model = initialize_lora_model(num_classes, device, lora_config=lora_config)
    optimizer = optim.Adam(model.parameters(), lr=config['net_lr'])
    early_stopper = EarlyStopping(patience=3)
    logger.log_param_counts(model)
    print("\n=== Training with LoRA ===")
    best_val_acc = -float("inf")
    best_model_state = None
    best_epoch = -1
    for epoch in range(config['num_epochs']):
        try:
            start = time.time()
            train_loss, acc = train_one_epoch(model, train_loader, optimizer, device, nn.CrossEntropyLoss(), epoch, logger)
            val_loss, val_acc = evaluate_model(model, val_loader, device, nn.CrossEntropyLoss(), logger, epoch)
            elapsed = time.time() - start
            logger.log_epoch_metrics(epoch, train_loss, val_loss, acc, elapsed, torch.cuda.max_memory_allocated())
            print(f"LoRA Epoch {epoch+1}/{config['num_epochs']} - Val Acc: {val_acc:.4f} - Val Loss: {val_loss:.4f}")
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = model.state_dict()
                best_epoch = epoch
            if epoch >3:
                if early_stopper.step(val_loss):
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break
        except Exception as e:
            print(f"[ERROR][LoRA][Epoch {epoch+1}] {e}")
            import traceback
            traceback.print_exc()
            break
    best_model_path = os.path.join(
        model_dir, f"{logger.model_name}_{logger.config_id}_best.pt"
    )
    torch.save(best_model_state, best_model_path)
    print(f"Best model (epoch {best_epoch+1}) saved to {best_model_path}")
    # Final test evaluation and logging (handled by evaluate_model)
    try:
        model.load_state_dict(torch.load(best_model_path, weights_only=True))
        test_acc = evaluate_model(
            model, test_loader, device, nn.CrossEntropyLoss(), logger, phase='test'
        )
        print(f"LoRA Test Acc: {test_acc:.4f}")
    except Exception as e:
        print(f"[ERROR][LoRA][Test] {e}")
        import traceback
        traceback.print_exc()
        test_acc = None
    final_model_path = os.path.join(
        model_dir, f"{logger.model_name}_{logger.config_id}_final.pt"
    )
    torch.save(model.state_dict(), final_model_path)
    logger.log_model_path(final_model_path)
    if hasattr(logger, 'log_final_result'):
        logger.log_final_result(model_name="LoRA", config=config, test_acc=test_acc)


# === Compute total number of experiments ===
def compute_total_experiments(hyperparams, model_param_map, run_models=None):
    total = 0
    for model_name in ['GPAF', 'ConvAdapter', 'LoRA']:
        if run_models is not None and not run_models.get(model_name, True):
            continue

        relevant_params = model_param_map[model_name]

        if model_name == "LoRA":
            alpha_vals = hyperparams.get('lora_alpha', [])
            r_vals = hyperparams.get('r', [])
            # Only count pairs where lora_alpha == r
            valid_pairs = [(a, r) for a in alpha_vals for r in r_vals if a/2 == r]
            if not valid_pairs:
                print(f"[SKIPPED][{model_name}] No valid lora_alpha == r pairs")
                continue
            # For other params, multiply their grid sizes
            other_params = [p for p in relevant_params if p not in ('lora_alpha', 'r')]
            other_counts = [len(hyperparams[p]) for p in other_params]
            other_total = 1
            for count in other_counts:
                other_total *= count
            model_total = len(valid_pairs) * other_total
        else:
            param_counts = [len(hyperparams[p]) for p in relevant_params]
            model_total = 1
            for count in param_counts:
                model_total *= count
        total += model_total
    return total
