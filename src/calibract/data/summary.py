"""Dataset summary helpers used by the experiment logger."""

from calibract.data.loaders import (
    load_kaggle_brain_mri,
    load_isic,
    load_pathmnist_npz
)
import torch

def count_loader_samples(loader):
    return sum(len(batch[0]) for batch in loader)

# Utility to count class occurrences
def count_class_distribution(loader, num_classes):
    class_counts = torch.zeros(num_classes, dtype=torch.int32)
    for images, labels in loader:
        for label in labels:
            class_counts[label] += 1
    return class_counts

# Wrapper to summarize dataset
def summarize(name, train_loader, val_loader, test_loader, num_classes):
    print(f"\n--- {name} Dataset Summary ---")
    print(f"Number of classes: {num_classes}")
    print(f"Train samples: {count_loader_samples(train_loader)}")
    print(f"Val samples:   {count_loader_samples(val_loader)}")
    print(f"Test samples:  {count_loader_samples(test_loader)}")

    print("Class distribution (train):")
    print(count_class_distribution(train_loader, num_classes).tolist())

def summarize_log(name, train_loader, val_loader, test_loader, num_classes):
    summary = {
        "dataset_name": name,
        "num_classes": num_classes,
        "train_samples": count_loader_samples(train_loader),
        "val_samples": count_loader_samples(val_loader),
        "test_samples": count_loader_samples(test_loader),
        "train_class_distribution": count_class_distribution(train_loader, num_classes).tolist()
    }
    return summary

def summarize_all(kaggle_path, isic_image_dir, isic_label_csv, pathmnist_npz):
    # Load and summarize all three datasets
    train, val, test, n_cls = load_kaggle_brain_mri(kaggle_path)
    summarize("Kaggle Brain MRI", train, val, test, n_cls)

    train, val, test, n_cls = load_isic(isic_image_dir, isic_label_csv)
    summarize("ISIC", train, val, test, n_cls)

    train, val, test, n_cls = load_pathmnist_npz(pathmnist_npz)
    summarize("PathMNIST", train, val, test, n_cls)


# Backward-compatible misspelling retained for older notebooks/scripts.
sumarize_all = summarize_all
