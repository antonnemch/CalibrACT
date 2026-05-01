# loaders.py
# Utility functions to load datasets in different formats for training/evaluation

import os
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset, random_split
from PIL import Image
import pandas as pd
import numpy as np

# -------------------------------
# 1. Load Kaggle Brain MRI Dataset (ImageFolder format)
# -------------------------------
def load_kaggle_brain_mri(root_dir, batch_size=32, subset_fraction=1.0, split_ratio=0.8, seed=1):
    import random
    import numpy as np               # <-- you need this
    import torch
    from sklearn.model_selection import StratifiedShuffleSplit
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    dataset = datasets.ImageFolder(root=root_dir, transform=transform)
    targets = [cls for _, cls in dataset.samples]

    dataset.tumor_classes = ["glioma", "meningioma", "pituitary"]
    dataset.notumor_class = "notumor"

    # ---- Step 1: proper unpack (train_idx first, test_idx second)
    full_len = len(dataset)
    test_size = int(0.2 * full_len)
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    indices = np.arange(full_len)
    for train_rem_idx, test_idx in sss1.split(indices, targets):
        pass
    test_set = torch.utils.data.Subset(dataset, test_idx)
    remaining_idx = train_rem_idx
    remaining_targets = [targets[i] for i in remaining_idx]

    # Step 2: optional subset of remaining (keep the "train" side from the split)
    if subset_fraction < 1.0:
        subset_size = int(len(remaining_idx) * subset_fraction)
        sss2 = StratifiedShuffleSplit(n_splits=1,
                                      test_size=len(remaining_idx) - subset_size,
                                      random_state=seed)
        for keep_idx, _ in sss2.split(np.arange(len(remaining_idx)), remaining_targets):
            pass
        remaining_idx = [remaining_idx[i] for i in keep_idx]
        remaining_targets = [remaining_targets[i] for i in keep_idx]

    # Step 3: stratified train/val on remaining
    train_size = int(split_ratio * len(remaining_idx))
    val_size = len(remaining_idx) - train_size
    try:
        sss3 = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
        for tr_local, va_local in sss3.split(np.arange(len(remaining_idx)), remaining_targets):
            pass
        train_idx = [remaining_idx[i] for i in tr_local]
        val_idx   = [remaining_idx[i] for i in va_local]
        train_set = torch.utils.data.Subset(dataset, train_idx)
        val_set   = torch.utils.data.Subset(dataset, val_idx)
    except ValueError as e:
        print(f"[WARNING] Stratified train/val split failed: {e}. Falling back to random split.")
        rng = np.random.default_rng(seed)
        rng.shuffle(remaining_idx)
        train_idx = remaining_idx[:train_size]
        val_idx   = remaining_idx[train_size:]
        train_set = torch.utils.data.Subset(dataset, train_idx)
        val_set   = torch.utils.data.Subset(dataset, val_idx)

    # Attach class metadata
    for subset in [train_set, val_set, test_set]:
        subset.classes = dataset.classes
        subset.tumor_classes = dataset.tumor_classes
        subset.notumor_class = dataset.notumor_class

    # Class balance helper
    from collections import Counter
    def print_class_balance(subset, name):
        labels = [targets[i] for i in subset.indices]
        counts = Counter(labels)
        # Always print in the order of dataset.classes
        class_names = dataset.classes
        ordered_counts = {class_names[i]: counts.get(i, 0) for i in range(len(class_names))}
        print(f"[SPLIT] {name} class balance: {ordered_counts} (total: {len(labels)})")

    print_class_balance(train_set, "Train")
    print_class_balance(val_set,   "Val")
    print_class_balance(test_set,  "Test")

    if batch_size is None:
        dataset_len = len(train_set)
        batch_size = max(4, min(32, dataset_len // 10))
        print(f"[INFO] Auto batch_size set to {batch_size} for train size {dataset_len}")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,  generator=g)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False)

    num_classes = len(dataset.classes)
    return train_loader, val_loader, test_loader, num_classes


# -------------------------------
# 2. Load ISIC dataset from image folder + one-hot label CSV
# -------------------------------
class ISICDataset(Dataset):
    def __init__(self, image_dir, csv_path, transform=None):
        self.data = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.transform = transform
        self.class_names = list(self.data.columns[1:])  # exclude 'image' column
        self.class_map = {name: idx for idx, name in enumerate(self.class_names)}
        self.missing_files = 0

        # Tumor classes info
        self.tumor_classes = []
        self.notumor_class = ""

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        filename = row['image']
        if not filename.lower().endswith(('.jpg', '.png')):
            filename += '.jpg'
        img_path = os.path.join(self.image_dir, filename)

        try:
            image = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            self.missing_files += 1
            return None

        label_name = row[1:].astype(int).idxmax()
        label = self.class_map[label_name]

        if self.transform:
            image = self.transform(image)
        return image, label

def load_isic(image_dir, csv_path, batch_size=32, split_ratio=0.8):
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    raw_dataset = ISICDataset(image_dir, csv_path, transform=transform)

    valid_items = [raw_dataset[i] for i in range(len(raw_dataset)) if raw_dataset[i] is not None]
    if not valid_items:
        raise RuntimeError("No valid ISIC images found after filtering missing files.")
    images, labels = zip(*valid_items)

    tensor_images = torch.stack(images)
    tensor_labels = torch.tensor(labels, dtype=torch.long)
    dataset = torch.utils.data.TensorDataset(tensor_images, tensor_labels)

    num_classes = len(raw_dataset.class_map)
    train_size = int(split_ratio * 0.8 * len(dataset))
    val_size = int(split_ratio * 0.2 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])
    # propagate tumor info to splits
    for subset in [train_set, val_set, test_set]:
        subset.tumor_classes = raw_dataset.tumor_classes
        subset.notumor_class = raw_dataset.notumor_class

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    print(f"[ISIC] Skipped {raw_dataset.missing_files} missing files.")
    return train_loader, val_loader, test_loader, num_classes

# -------------------------------
# 3. Load PathMNIST from .npz file using lazy loading Dataset
# -------------------------------
class PathMNISTDataset(Dataset):
    def __init__(self, npz_path, transform=None):
        self.data = np.load(npz_path)
        self.images = np.concatenate([self.data['train_images'], self.data['val_images'], self.data['test_images']], axis=0)
        self.labels = np.concatenate([self.data['train_labels'], self.data['val_labels'], self.data['test_labels']], axis=0).squeeze()
        self.transform = transform

        # Tumor classes info
        self.tumor_classes = []
        self.notumor_class = ""

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx].astype(np.float32)
        label = int(self.labels[idx])

        image = torch.tensor(image)
        if image.ndim == 3 and image.shape[0] != 3:
            image = image.permute(2, 0, 1)

        if image.max() > 1.0:
            image = image / 255.0

        if self.transform:
            image = self.transform(image)

        return image, label

def load_pathmnist_npz(npz_path, batch_size=32, split_ratio=0.8):
    transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    full_dataset = PathMNISTDataset(npz_path, transform=transform)
    num_classes = len(set(full_dataset.labels))
    train_size = int(split_ratio * 0.8 * len(full_dataset))
    val_size = int(split_ratio * 0.2 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    train_set, val_set, test_set = random_split(full_dataset, [train_size, val_size, test_size])

    # propagate tumor info to splits
    for subset in [train_set, val_set, test_set]:
        subset.tumor_classes = full_dataset.tumor_classes
        subset.notumor_class = full_dataset.notumor_class

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, num_classes
