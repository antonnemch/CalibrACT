import hashlib
import json
import os
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import confusion_matrix
import torch
import torch.nn.functional as F

def compute_ece(probs, targets, n_bins=15, norm='l1'):
    """
    Compute Expected Calibration Error (ECE) for multi-class classification.

    Args:
        probs (Tensor): Tensor of shape (N, C), probabilities (or logits) per class
        targets (Tensor): Tensor of shape (N,), ground truth class indices
        n_bins (int): Number of bins to divide [0, 1] confidence range into
        norm (str): One of 'l1' (ECE), 'l2' (RMSE), 'max' (MCE)

    Returns:
        float: Calibration error
    """
    if probs.ndim != 2:
        raise ValueError("probs must be of shape (N, C)")
    if probs.max() > 1 or probs.min() < 0:
        probs = F.softmax(probs, dim=1)  # Convert logits to probs

    confidences, predictions = torch.max(probs, dim=1)
    accuracies = predictions.eq(targets)

    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    ece = torch.tensor(0.0, device=probs.device)
    total = targets.size(0)

    if norm == 'max':
        max_gap = 0.0
        for i in range(n_bins):
            lower = bin_boundaries[i]
            upper = bin_boundaries[i + 1]
            mask = (confidences > lower) & (confidences <= upper)
            if mask.any():
                acc = accuracies[mask].float().mean()
                conf = confidences[mask].mean()
                max_gap = max(max_gap, torch.abs(acc - conf).item())
        return max_gap

    for i in range(n_bins):
        lower = bin_boundaries[i]
        upper = bin_boundaries[i + 1]
        mask = (confidences > lower) & (confidences <= upper)

        if mask.sum() > 0:
            acc = accuracies[mask].float().mean()
            conf = confidences[mask].mean()
            gap = torch.abs(acc - conf)
            weight = mask.float().mean()

            if norm == 'l1':
                ece += gap * weight
            elif norm == 'l2':
                ece += (gap ** 2) * weight
            else:
                raise ValueError(f"Unsupported norm: {norm}")

    return torch.sqrt(ece) if norm == 'l2' else ece

def make_logger(model_name, config, timestamp="", base_dir="logs"):
    return ExcelTrainingLogger(model_name=model_name, config=config, timestamp=timestamp, base_dir=base_dir)

class ExcelTrainingLogger:
    def __init__(self, model_name, config, timestamp, base_dir="logs"):
        self.model_name = model_name
        self.config = config
        self.config_str = json.dumps(config, sort_keys=True)
        self.config_id = hashlib.md5(self.config_str.encode()).hexdigest()[:8]

        os.makedirs(base_dir, exist_ok=True)
        self.path = os.path.join(base_dir, f"{model_name} {timestamp}.xlsx")

        # Tables to append
        self.param_table = []
        self.epoch_metrics = []
        self.batch_metrics = []
        self.meta_lrs = []
        self.confusion_matrices = []
        self.hyperparams = []
        self.dataset_summary = []
        self.model_path = None
        self.test_image_rows = []
        self.metrics = []  # Ensure metrics is always defined

    def log_param_counts(self, model):
        total, trainable, frozen = 0, 0, 0
        module_param_counts = {}
        for name, param in model.named_parameters():
            count = param.numel()
            total += count
            if param.requires_grad:
                trainable += count
            else:
                frozen += count
            # Group by top-level module (e.g., 'layer1.0.conv1.weight' -> 'layer1')
            top_module = name.split('.')[0] if '.' in name else name
            if top_module not in module_param_counts:
                module_param_counts[top_module] = {'total': 0, 'trainable': 0, 'frozen': 0}
            module_param_counts[top_module]['total'] += count
            if param.requires_grad:
                module_param_counts[top_module]['trainable'] += count
            else:
                module_param_counts[top_module]['frozen'] += count
        # Log per-module counts
        for mod_name, counts in module_param_counts.items():
            self.param_table.append({
                "config_id": self.config_id,
                "module": mod_name,
                "total": counts['total'],
                "trainable": counts['trainable'],
                "frozen": counts['frozen']
            })
        self.param_table.append({
            "config_id": self.config_id,
            "module": "TOTAL",
            "total": total,
            "trainable": trainable,
            "frozen": frozen
        })
        #print(f"[LOGGER] Total parameters: {total}, Trainable: {trainable}, Frozen: {frozen}")

    def log_confusion_matrix(self, y_true, y_pred, classes, epoch=None):
        cm = confusion_matrix(y_true, y_pred, labels=range(len(classes)))
        for i, actual_label in enumerate(classes):
            row = {
                "config_id": self.config_id,
                "epoch": epoch if epoch is not None else "final",
                "actual": actual_label
            }
            row.update({pred_label: cm[i, j] for j, pred_label in enumerate(classes)})
            self.confusion_matrices.append(row)

    def log_hyperparams(self):
        row = {"config_id": self.config_id}
        row.update(self.config)
        self.hyperparams.append(row)

    def log_dataset_summary(self, summary: dict):
        row = {"config_id": self.config_id}
        row.update(summary)
        self.dataset_summary.append(row)

    def log_model_path(self, path: str):
        self.model_path = path

    def log_epoch_metrics(self, epoch, train_loss, val_loss, acc, time_sec, gpu_mem_bytes):
        self.epoch_metrics.append({
            "config_id": self.config_id,
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "accuracy": acc,
            "time_sec": round(time_sec, 2),
            "gpu_mem_mb": round(gpu_mem_bytes / 1024 / 1024, 2)
        })

    def log_batch_metrics(self, epoch, batch_idx, loss, acc, meta_loss="N/A"):
        self.batch_metrics.append({
            "config_id": self.config_id,
            "epoch": epoch,
            "batch": batch_idx,
            "loss": loss,
            "accuracy": acc,
            "metaLoss": meta_loss
        })

    def log_metalr_lrs(self, epoch, batch, lrs):
        row = {"config_id": self.config_id, "epoch": epoch, "batch": batch}
        row.update({f"lr_{i}": val for i, val in enumerate(lrs)})
        self.meta_lrs.append(row)

    def log_test_image(self, img_filename, actual_class, predicted_class, prediction_confidence, correct, tumor_correct=None, probs=None):
        row = {
            "config_id": self.config_id,
            "img_filename": img_filename,
            "actual_class": actual_class,
            "predicted_class": predicted_class,
            "prediction_confidence": prediction_confidence,
            "correct": correct
        }
        if tumor_correct is not None:
            row["tumor_correct"] = tumor_correct
        if probs is not None:
            # Store as list of columns: prob_0, prob_1, ...
            for i, p in enumerate(probs):
                row[f"prob_{i}"] = p
        else:
            print(f"[LOGGER WARNING] No probabilities provided for image {img_filename}. Skipping probability logging.")
        self.test_image_rows.append(row)

    def save(self):
        self.compute_and_store_metrics()
        self.compute_and_store_final_results()  # Ensure final_results is up to date

        mode = "a" if os.path.exists(self.path) else "w"
        writer_args = {
            "path": self.path,
            "mode": mode,
            "engine": "openpyxl"
        }
        if mode == "a":
            writer_args["if_sheet_exists"] = "overlay"

        with pd.ExcelWriter(**writer_args) as writer:
            self._append_to_sheet(writer, "Hyperparameters", self.hyperparams)
            self._append_to_sheet(writer, "Parameters", self.param_table)
            self._append_to_sheet(writer, "Dataset Summary", self.dataset_summary)
            self._append_to_sheet(writer, "Epoch Metrics", self.epoch_metrics)
            self._append_to_sheet(writer, "Batch Metrics", self.batch_metrics)
            self._append_to_sheet(writer, "MetaLR LRs", self.meta_lrs)
            self._append_to_sheet(writer, "Metrics", self.metrics)
            if self.test_image_rows:
                self._append_to_sheet(writer, "Test set by Image", self.test_image_rows)
            # Always write Final results sheet, even if empty
            if hasattr(self, "final_results") and self.final_results is not None:
                self._append_to_sheet(writer, "Final Results", self.final_results)
            else:
                # Write empty sheet if not present
                pd.DataFrame().to_excel(writer, sheet_name="Final Results", index=False)
            if self.model_path:
                pd.DataFrame([{
                    "config_id": self.config_id,
                    "model_path": self.model_path
                }]).to_excel(writer, sheet_name="Model File", index=False)
            if self.confusion_matrices:
                df_all_cm = pd.DataFrame(self.confusion_matrices)
                try:
                    existing = pd.read_excel(self.path, sheet_name="Confusion Matrix")
                    df_combined = pd.concat([existing, df_all_cm], ignore_index=True)
                except Exception:
                    df_combined = df_all_cm
                df_combined.to_excel(writer, sheet_name="Confusion Matrix", index=False)

    def _append_to_sheet(self, writer, sheet_name, new_data):
        if not new_data:
            return
        df_new = pd.DataFrame(new_data)
        try:
            existing = pd.read_excel(self.path, sheet_name=sheet_name)
            df_combined = pd.concat([existing, df_new], ignore_index=True)
        except Exception:
            df_combined = df_new
        df_combined.to_excel(writer, sheet_name=sheet_name, index=False)

    def compute_and_store_metrics(self):
        if not self.confusion_matrices:
            return

        df_cm = pd.DataFrame(self.confusion_matrices)
        classes = [col for col in df_cm.columns if col not in ["config_id", "epoch", "actual"]]

        metrics_rows = []

        for (config_id, epoch), group in df_cm.groupby(["config_id", "epoch"]):
            # Rebuild confusion matrix
            cm = []
            for actual_class in classes:
                row = group[group["actual"] == actual_class][classes].values[0]
                cm.append(row)
            cm = np.array(cm)

            for i, cls in enumerate(classes):
                TP = cm[i, i]
                FN = cm[i, :].sum() - TP
                FP = cm[:, i].sum() - TP
                TN = cm.sum() - (TP + FP + FN)
                support = cm[i, :].sum()

                precision = TP / (TP + FP) if (TP + FP) > 0 else 0
                recall = TP / (TP + FN) if (TP + FN) > 0 else 0
                f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                accuracy = (TP + TN) / cm.sum()

                metrics_rows.append({
                    "config_id": config_id,
                    "epoch": epoch,
                    "class": cls,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "accuracy": accuracy,
                    "support": support
                })

        self.metrics = metrics_rows

    def compute_and_store_final_results(self):
        # Debug: print why final_results may be empty
        if not self.test_image_rows:
            print(f"[LOGGER DEBUG] No test_image_rows for config_id {self.config_id}. Final Results will be empty.")
        if not self.epoch_metrics:
            print(f"[LOGGER DEBUG] No epoch_metrics for config_id {self.config_id}. Final Results will be empty.")
        if not self.test_image_rows or not self.epoch_metrics:
            self.final_results = None
            return
        import pandas as pd
        from sklearn.metrics import roc_auc_score
        # --- Aggregate test set info ---
        df_test = pd.DataFrame(self.test_image_rows)
        # Get number of classes
        prob_cols = [c for c in df_test.columns if c.startswith("prob_")]
        n_classes = len(prob_cols)
        # True labels and predicted probabilities for AUC
        y_true = df_test["actual_class"].values
        y_pred = df_test["predicted_class"].values
        y_true_idx = pd.factorize(y_true)[0]
        y_pred_idx = pd.factorize(y_pred)[0]
        # Probabilities as array (n_samples, n_classes)
        if n_classes > 0:
            y_score = df_test[[f"prob_{i}" for i in range(n_classes)]].values
            ece = None
            try:
                # Convert to torch tensors
                probs_tensor = torch.tensor(y_score, dtype=torch.float32)
                labels_tensor = torch.tensor(y_true_idx, dtype=torch.long)

                # Call the ECE function (make sure it's defined or imported)
                ece = compute_ece(probs_tensor, labels_tensor, n_bins=15, norm='l1').item()
            except Exception as e:
                print(f"[ECE ERROR] Could not compute ECE: {e}")
            try:
                auc_micro = roc_auc_score(pd.get_dummies(y_true_idx), y_score, average="micro", multi_class="ovr")
            except Exception:
                auc_micro = None
        else:
            auc_micro = None
        # Test accuracy (simple)
        test_acc = (y_true_idx == y_pred_idx).mean()
        # --- Micro-averaged confusion matrix metrics ---
        # Aggregate all TP, FP, FN over all classes
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true_idx, y_pred_idx, labels=range(n_classes))
        TP = np.trace(cm)
        FP = cm.sum(axis=0) - np.diag(cm)
        FN = cm.sum(axis=1) - np.diag(cm)
        total_FP = FP.sum()
        total_FN = FN.sum()
        total_TP = TP
        precision_micro = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0
        recall_micro = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0
        f1_micro = (2 * precision_micro * recall_micro) / (precision_micro + recall_micro) if (precision_micro + recall_micro) > 0 else 0
        accuracy_micro = total_TP / cm.sum() if cm.sum() > 0 else 0
        # --- Best val accuracy, best val loss, best epoch, total epochs ---
        df_epoch = pd.DataFrame(self.epoch_metrics)
        # Defensive: drop rows with NaN val_loss or accuracy
        if "val_loss" in df_epoch:
            df_epoch = df_epoch.dropna(subset=["val_loss", "accuracy"], how="all")
        else:
            df_epoch = df_epoch.dropna(subset=["accuracy"], how="all")
        if df_epoch.empty:
            best_val_acc = None
            best_val_loss = None
            best_epoch = None
            total_epochs = 0
        else:
            if "val_loss" in df_epoch:
                best_val_idx = df_epoch["val_loss"].idxmin()
            else:
                best_val_idx = df_epoch["accuracy"].idxmax()
            best_val_acc = df_epoch.loc[best_val_idx, "accuracy"] if best_val_idx in df_epoch.index else None
            best_val_loss = df_epoch.loc[best_val_idx, "val_loss"] if "val_loss" in df_epoch and best_val_idx in df_epoch.index else None
            best_epoch = df_epoch.loc[best_val_idx, "epoch"] if best_val_idx in df_epoch.index else None
            total_epochs = df_epoch["epoch"].max() + 1 if not df_epoch["epoch"].empty else 0
        # --- Trainable params ---
        df_params = pd.DataFrame(self.param_table)
        total_trainable = df_params[df_params["module"] == "TOTAL"]["trainable"].values[0] if not df_params.empty else None
        # --- Hyperparams ---
        hp_row = self.hyperparams[-1] if self.hyperparams else {}
        # --- Compose final row ---
        final_row = {
            "config_id": self.config_id,
            "trainable_params": total_trainable,
            "dataset_name": "Kaggle Brain MRI",
            "best_val_accuracy": best_val_acc,
            "test_accuracy": test_acc,
            "AUC_micro": auc_micro,
            "precision_micro": precision_micro,
            "recall_micro": recall_micro,
            "f1_micro": f1_micro,
            "accuracy_micro": accuracy_micro,
            "best_epoch": best_epoch,
            "total_epochs": total_epochs,
            "best_val_loss": best_val_loss,
            "ECE_l1": ece
        }
        # Add all hyperparams
        for k, v in hp_row.items():
            if k not in final_row:
                final_row[k] = v
        self.final_results = [final_row]