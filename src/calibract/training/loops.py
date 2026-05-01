import os

import torch
import torch.nn as nn

from calibract.models.activations import (
    BaseActivation,
    ChannelwiseActivation,
    CustomActivationPlaceholder,
    channel_map,
)


def train_one_epoch(
    model,
    dataloader,
    net_optimizer,
    device,
    criterion,
    epoch,
    logger=None,
    act_optimizer=None,
    modifiers=None,
):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    # Separate parameters for activation and network
    if act_optimizer is not None:
        act_params = []
        net_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            # Check if parameter belongs to a BaseActivation module
            module_names = name.split(".")[:-1]
            mod = model
            for mn in module_names:
                if hasattr(mod, mn):
                    mod = getattr(mod, mn)
                else:
                    mod = None
                    break
            if mod is not None and isinstance(mod, BaseActivation):
                act_params.append(param)
            else:
                net_params.append(param)

    for batch_idx, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Add L2 regularization if enabled and strength is set
        reg_strength = None
        if modifiers is not None:
            reg_strength = modifiers.get("Regularization", None)
        if (
            reg_strength is not None
            and isinstance(reg_strength, float)
            and reg_strength > 0
        ):
            reg_loss = 0.0
            for param in model.parameters():
                if param.requires_grad:
                    reg_loss = reg_loss + torch.norm(param, p=2)
            loss = loss + reg_strength * reg_loss  # L2 regularization

        if act_optimizer is not None:
            net_optimizer.zero_grad()
            act_optimizer.zero_grad()
            loss.backward()
            # Only step if params exist (avoid error if group is empty)
            if net_params:
                for p in net_params:
                    if p.grad is not None:
                        pass  # grads already computed
                net_optimizer.step()
            if act_params:
                for p in act_params:
                    if p.grad is not None:
                        pass
                act_optimizer.step()
        else:
            net_optimizer.zero_grad()
            loss.backward()
            net_optimizer.step()

        preds = outputs.argmax(dim=1)
        correct_batch = (preds == labels).sum().item()
        total_batch = labels.size(0)

        total_loss += loss.item() * total_batch
        correct += correct_batch
        total += total_batch

        batch_acc = correct_batch / total_batch
        if logger:
            logger.log_batch_metrics(epoch, batch_idx, loss.item(), batch_acc)

    avg_loss = total_loss / total
    accuracy = correct / total
    print(f"Train Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
    return avg_loss, accuracy


def evaluate_model(
    model, dataloader, device, criterion=None, logger=None, epoch=None, phase="val"
):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    y_true, y_pred = [], []

    # Try to get filenames if possible (for ImageFolder-based datasets)
    has_filenames = hasattr(dataloader.dataset, "dataset") and hasattr(
        dataloader.dataset.dataset, "samples"
    )
    if has_filenames:
        # Subset -> ImageFolder
        all_samples = dataloader.dataset.dataset.samples
        indices = dataloader.dataset.indices
    else:
        all_samples = None
        indices = None

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            confidences = torch.softmax(outputs, dim=1).max(dim=1).values.cpu().tolist()

            # Compute loss if criterion is provided
            if criterion is not None:
                loss = criterion(outputs, labels)
                total_loss += loss.item() * labels.size(0)  # Sum loss over batch

            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            # Per-image logging for test phase
            if logger is not None and phase == "test":
                for i in range(len(labels)):
                    # Filename
                    if has_filenames:
                        img_idx = indices[batch_idx * dataloader.batch_size + i]
                        img_filename = os.path.basename(all_samples[img_idx][0])
                    else:
                        img_filename = "N/A"
                    actual_idx = labels[i].item()
                    pred_idx = preds[i].item()
                    actual_class = (
                        dataloader.dataset.classes[actual_idx]
                        if hasattr(dataloader.dataset, "classes")
                        else str(actual_idx)
                    )
                    predicted_class = (
                        dataloader.dataset.classes[pred_idx]
                        if hasattr(dataloader.dataset, "classes")
                        else str(pred_idx)
                    )
                    prediction_confidence = confidences[i]
                    is_correct = int(actual_idx == pred_idx)
                    # Tumor correct logic
                    tumor_correct = None
                    if hasattr(dataloader.dataset, "tumor_classes") and hasattr(
                        dataloader.dataset, "notumor_class"
                    ):
                        tumor_classes = dataloader.dataset.tumor_classes
                        notumor_class = dataloader.dataset.notumor_class
                        if tumor_classes and notumor_class:
                            is_actual_tumor = actual_class in tumor_classes
                            is_pred_tumor = predicted_class in tumor_classes
                            tumor_correct = int(is_actual_tumor == is_pred_tumor)
                    # Get per-class probabilities for this image
                    out_i = outputs[i].detach().cpu()
                    probs = torch.softmax(out_i, dim=0).tolist()
                    # Only print debug info if something is wrong
                    import numpy as np
                    out_np = out_i.numpy()
                    probs_np = np.array(probs)
                    if (
                        np.any(np.isnan(out_np)) or np.any(np.isinf(out_np)) or
                        np.any(np.isnan(probs_np)) or np.any(np.isinf(probs_np)) or
                        np.allclose(probs_np, 0) or
                        (len(probs_np) > 1 and np.allclose(probs_np, probs_np[0]))
                    ):
                        print(f"[DEBUG] outputs[{i}] for image {img_filename}, config_id: {getattr(logger, 'config_id', 'N/A')}: {out_i}")
                        print(f"[DEBUG] probs for image {img_filename}, config_id: {getattr(logger, 'config_id', 'N/A')}: {probs}")
                    logger.log_test_image(
                        img_filename=img_filename,
                        actual_class=actual_class,
                        predicted_class=predicted_class,
                        prediction_confidence=prediction_confidence,
                        correct=is_correct,
                        tumor_correct=tumor_correct,
                        probs=probs,
                    )

    acc = correct / total
    avg_loss = total_loss / total if criterion is not None else None

    class_names = (
        dataloader.dataset.classes if hasattr(dataloader.dataset, "classes") else None
    )
    if logger is not None and class_names is not None:
        logger.log_confusion_matrix(
            y_true, y_pred, class_names, epoch=epoch if phase == "val" else None
        )
        print(f"Confusion matrix logged for {phase} phase.")

    return (avg_loss, acc) if phase != "test" else acc


# Prints the number of trainable and frozen parameters in a model,
# including breakdown by top-level module


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if p.requires_grad == False)
    print(f"Total parameters:     {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    print(f"Frozen parameters:    {frozen:,}")
    print(f"Trainable ratio:      {trainable / total:.2%}")
    print(f"Frozen ratio:         {frozen / total:.2%}")
    return total, trainable, frozen


def count_parameters_by_module(model):
    print("\n--- Parameter breakdown by module ---")
    for name, module in model.named_children():
        module_total = sum(p.numel() for p in module.parameters())
        module_trainable = sum(
            p.numel() for p in module.parameters() if p.requires_grad
        )
        module_frozen = sum(
            p.numel() for p in module.parameters() if p.requires_grad == False
        )
        print(
            f"{name:20} | total: {module_total:,} | trainable: {module_trainable:,} | frozen: {module_frozen:,}"
        )


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_val = float("inf")
        self.counter = 0

    def step(self, val_loss):
        if self.best_val - val_loss > self.min_delta:
            self.best_val = val_loss
            self.counter = 0
            return False  # No early stop
        else:
            self.counter += 1
            return self.counter >= self.patience


def train_one_epoch_spline(
    model, dataloader, optimizers, device, criterion, epoch, logger=None
):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        # dual optimizer setup
        for opt in optimizers:
            opt.zero_grad()
        loss.backward()
        for opt in optimizers:
            opt.step()

        preds = outputs.argmax(dim=1)
        correct_batch = (preds == labels).sum().item()
        total_batch = labels.size(0)

        total_loss += loss.item() * total_batch
        correct += correct_batch
        total += total_batch

        batch_acc = correct_batch / total_batch
        if logger:
            logger.log_batch_metrics(epoch, batch_idx, loss.item(), batch_acc)

    avg_loss = total_loss / total
    accuracy = correct / total
    print(f"Train Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
    return avg_loss, accuracy


def compute_num_experiments(
    model_name, run_models={"GPAF": True}, hyperparams=None, model_param_map=None
):
    # === Precompute total number of configurations across all models ===
    total_experiments = 0
    for model_name, should_run in run_models.items():
        if not should_run:
            continue
        relevant_params = model_param_map[model_name]
        param_counts = [len(hyperparams[p]) for p in relevant_params]
        model_total = 1
        for count in param_counts:
            model_total *= count
        total_experiments += model_total

    return total_experiments


def print_model_activations(model):
    for module_name, module in model.named_modules():
        for attr_name in dir(module):
            if attr_name.startswith("_"):
                continue  # skip private attributes

            attr_value = getattr(module, attr_name)

            if isinstance(attr_value, nn.Module):
                if isinstance(attr_value, BaseActivation):
                    if isinstance(attr_value, CustomActivationPlaceholder):
                        act_fn = attr_value.act_fn
                        if isinstance(act_fn, nn.Module):
                            print(
                                f"{module_name}.{attr_name}: CustomActivationPlaceholder -> {act_fn}"
                            )
                            for param_name, param in act_fn.named_parameters(
                                recurse=False
                            ):
                                print(
                                    f"  Param {param_name}: {param.detach().cpu().numpy()}"
                                )
                        else:
                            print(
                                f"{module_name}.{attr_name}: CustomActivationPlaceholder -> [UNSET]"
                            )
                    elif isinstance(attr_value, ChannelwiseActivation):
                        print(
                            f"{module_name}.{attr_name}: ChannelwiseActivation with {len(attr_value.activations)} activations"
                        )
                        for i, sub_act in enumerate(attr_value.activations):
                            print(f"  Channel {i}: {sub_act}")
                            for param_name, param in sub_act.named_parameters(
                                recurse=False
                            ):
                                print(
                                    f"    Param {param_name}: {param.detach().cpu().numpy()}"
                                )
                    else:
                        print(f"{module_name}.{attr_name}: {attr_value}")
                        for param_name, param in attr_value.named_parameters(
                            recurse=False
                        ):
                            print(
                                f"  Param {param_name}: {param.detach().cpu().numpy()}"
                            )


def build_activation_map(custom_config):
    """
    Build an activation map for the model.

    Args:
        channel_map (dict): Mapping of activation point names to channel counts.
        custom_config (dict): Mapping of activation point names to dicts:
            {
                'type': activation class (e.g., LaplacianGPAF),
                'mode': 'shared' | 'channelwise',
                'shared_group': optional identifier for shared activations
            }

    Returns:
        dict: Activation map usable by set_custom_activation_map()
    """
    activation_map = {}
    shared_instances = {}

    for name, channels in channel_map.items():
        if name in custom_config:
            config = custom_config[name]
            act_type = config["type"]
            mode = config["mode"]

            if mode == "shared":
                # Determine group key: shared_group if provided, else the type itself
                group_key = config.get("shared_group", act_type)

                if group_key not in shared_instances:
                    shared_instances[group_key] = act_type()
                    #print(
                        #f"Created shared activation for group '{group_key}' at {name}"
                    #)

                activation_map[name] = shared_instances[group_key]

            elif mode == "channelwise":
                activation_map[name] = ChannelwiseActivation(
                    [act_type() for _ in range(channels)]
                )
                # print(f"Created channelwise activation at {name} with {channels} channels")

            else:
                raise ValueError(f"Unknown mode '{mode}' for {name}")

        else:
            # Default to ReLU if not specified in config
            activation_map[name] = nn.ReLU()

    return activation_map


def print_activation_map(activation_map):
    for name, act in activation_map.items():
        if isinstance(act, ChannelwiseActivation):
            print(f"{name}: ChannelwiseActivation with {len(act.activations)} channels")
        else:
            print(f"{name}: {type(act).__name__}")


# L2 regularization (weight decay) penalizes large weights by adding the sum of squared weights to the loss function, scaled by a configurable strength. This helps prevent overfitting by discouraging overly complex models. The regularization strength can be set in the 'modifiers' dictionary as a float value for the 'Regularization' key. If set to None or not provided, no regularization is applied.
