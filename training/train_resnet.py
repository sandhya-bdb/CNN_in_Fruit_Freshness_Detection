import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms


@dataclass
class TrainingConfig:
    dataset_dir: str
    output_dir: str = "artifacts"
    epochs: int = 10
    batch_size: int = 32
    lr: float = 1e-5
    weight_decay: float = 1e-4
    dropout: float = 0.4
    val_split: float = 0.15
    test_split: float = 0.15
    patience: int = 3
    image_size: int = 224
    num_workers: int = 0
    seed: int = 42
    unfreeze_layer4: bool = True
    label_smoothing: float = 0.0
    augmentation: str = "standard"
    split_strategy: str = "stratified"
    preset: str = "tuned"


class TransformSubset(Dataset):
    def __init__(self, base_dataset: datasets.ImageFolder, indices: List[int], transform):
        self.base_dataset = base_dataset
        self.indices = indices
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        image_idx = self.indices[idx]
        path, label = self.base_dataset.samples[image_idx]
        image = self.base_dataset.loader(path)
        if self.transform is not None:
            image = self.transform(image)
        return image, label


class FruitClassifierResNet(nn.Module):
    def __init__(self, num_classes: int, dropout_rate: float = 0.4, unfreeze_layer4: bool = True):
        super().__init__()
        self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        for param in self.model.parameters():
            param.requires_grad = False

        if unfreeze_layer4:
            for param in self.model.layer4.parameters():
                param.requires_grad = True

        self.model.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.model.fc.in_features, num_classes),
        )

    def forward(self, x):
        return self.model(x)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



def get_transforms(image_size: int, augmentation: str = "standard"):
    if augmentation == "strong":
        train_tf = transforms.Compose(
            [
                transforms.RandomResizedCrop((image_size, image_size), scale=(0.75, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(20),
                transforms.ColorJitter(brightness=0.2, contrast=0.4, saturation=0.2, hue=0.03),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    else:
        train_tf = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.1, contrast=0.3),
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    eval_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    return train_tf, eval_tf



def split_indices_random(n_samples: int, val_split: float, test_split: float, seed: int):
    if val_split + test_split >= 1.0:
        raise ValueError("val_split + test_split must be < 1.0")

    rng = np.random.default_rng(seed)
    indices = np.arange(n_samples)
    rng.shuffle(indices)

    n_test = int(n_samples * test_split)
    n_val = int(n_samples * val_split)

    test_idx = indices[:n_test].tolist()
    val_idx = indices[n_test : n_test + n_val].tolist()
    train_idx = indices[n_test + n_val :].tolist()

    return train_idx, val_idx, test_idx


def split_indices_stratified(targets: List[int], val_split: float, test_split: float, seed: int):
    if val_split + test_split >= 1.0:
        raise ValueError("val_split + test_split must be < 1.0")

    rng = np.random.default_rng(seed)
    targets_np = np.array(targets)
    classes = np.unique(targets_np)

    train_idx: List[int] = []
    val_idx: List[int] = []
    test_idx: List[int] = []

    for cls in classes:
        cls_indices = np.where(targets_np == cls)[0]
        rng.shuffle(cls_indices)

        n_cls = len(cls_indices)
        n_test = int(n_cls * test_split)
        n_val = int(n_cls * val_split)

        test_idx.extend(cls_indices[:n_test].tolist())
        val_idx.extend(cls_indices[n_test : n_test + n_val].tolist())
        train_idx.extend(cls_indices[n_test + n_val :].tolist())

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)
    return train_idx, val_idx, test_idx



def create_dataloaders(config: TrainingConfig):
    base_dataset = datasets.ImageFolder(root=config.dataset_dir)
    num_classes = len(base_dataset.classes)

    train_tf, eval_tf = get_transforms(config.image_size, augmentation=config.augmentation)
    if config.split_strategy == "stratified":
        train_idx, val_idx, test_idx = split_indices_stratified(
            base_dataset.targets, config.val_split, config.test_split, config.seed
        )
    else:
        train_idx, val_idx, test_idx = split_indices_random(
            len(base_dataset), config.val_split, config.test_split, config.seed
        )

    train_ds = TransformSubset(base_dataset, train_idx, train_tf)
    val_ds = TransformSubset(base_dataset, val_idx, eval_tf)
    test_ds = TransformSubset(base_dataset, test_idx, eval_tf)

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return base_dataset.classes, num_classes, train_loader, val_loader, test_loader



def run_epoch(model, loader, criterion, device, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)

    running_loss = 0.0
    running_correct = 0
    n_total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            logits = model(images)
            loss = criterion(logits, labels)
            if is_train:
                loss.backward()
                optimizer.step()

        preds = torch.argmax(logits, dim=1)
        running_loss += loss.item() * labels.size(0)
        running_correct += (preds == labels).sum().item()
        n_total += labels.size(0)

    avg_loss = running_loss / max(n_total, 1)
    avg_acc = running_correct / max(n_total, 1)
    return avg_loss, avg_acc



def evaluate_with_preds(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_correct = 0
    n_total = 0
    y_true: List[int] = []
    y_pred: List[int] = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            preds = torch.argmax(logits, dim=1)

            running_loss += loss.item() * labels.size(0)
            running_correct += (preds == labels).sum().item()
            n_total += labels.size(0)

            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())

    return {
        "loss": running_loss / max(n_total, 1),
        "accuracy": running_correct / max(n_total, 1),
        "y_true": y_true,
        "y_pred": y_pred,
    }



def run_training(config: TrainingConfig) -> Dict:
    config = apply_preset(config)
    set_seed(config.seed)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classes, num_classes, train_loader, val_loader, test_loader = create_dataloaders(config)

    model = FruitClassifierResNet(
        num_classes=num_classes,
        dropout_rate=config.dropout,
        unfreeze_layer4=config.unfreeze_layer4,
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=1
    )

    best_val_acc = 0.0
    best_epoch = -1
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, config.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer=optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device, optimizer=None)

        scheduler.step(val_acc)

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "lr": optimizer.param_groups[0]["lr"],
            }
        )

        print(
            f"Epoch {epoch}/{config.epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(model.state_dict(), output_dir / "best_model.pth")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= config.patience:
            print(f"Early stopping triggered at epoch {epoch}")
            break

    model.load_state_dict(torch.load(output_dir / "best_model.pth", map_location=device))

    val_result = evaluate_with_preds(model, val_loader, criterion, device)
    test_result = evaluate_with_preds(model, test_loader, criterion, device)

    metadata = {
        "device": str(device),
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "val_accuracy": val_result["accuracy"],
        "test_accuracy": test_result["accuracy"],
        "history": history,
        "classes": classes,
        "config": asdict(config),
    }

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    with open(output_dir / "classes.json", "w", encoding="utf-8") as f:
        json.dump(classes, f, indent=2)

    print(
        f"Done. best_val_acc={best_val_acc:.4f}, "
        f"val_acc={val_result['accuracy']:.4f}, test_acc={test_result['accuracy']:.4f}"
    )

    return metadata


def apply_preset(config: TrainingConfig) -> TrainingConfig:
    if config.preset == "realistic_eval":
        config.val_split = 0.2
        config.test_split = 0.2
        config.dropout = max(config.dropout, 0.5)
        config.weight_decay = max(config.weight_decay, 5e-4)
        config.label_smoothing = max(config.label_smoothing, 0.1)
        config.unfreeze_layer4 = False
        config.patience = min(config.patience, 2)
        config.augmentation = "strong"
        config.split_strategy = "stratified"
    return config



def parse_args():
    parser = argparse.ArgumentParser(description="Train tuned ResNet50 for fruit freshness detection")
    parser.add_argument("--dataset-dir", required=True, help="Path to ImageFolder dataset root")
    parser.add_argument("--output-dir", default="artifacts", help="Directory to store checkpoints/metrics")
    parser.add_argument(
        "--preset",
        choices=["tuned", "realistic_eval"],
        default="tuned",
        help="tuned: accuracy-focused; realistic_eval: stricter split and regularization",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--test-split", type=float, default=0.15)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument(
        "--augmentation",
        choices=["standard", "strong"],
        default="standard",
    )
    parser.add_argument(
        "--split-strategy",
        choices=["stratified", "random"],
        default="stratified",
    )
    parser.add_argument(
        "--freeze-layer4",
        action="store_true",
        help="Freeze layer4 as well (default behavior is to unfreeze layer4)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = TrainingConfig(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        preset=args.preset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        val_split=args.val_split,
        test_split=args.test_split,
        patience=args.patience,
        image_size=args.image_size,
        num_workers=args.num_workers,
        seed=args.seed,
        label_smoothing=args.label_smoothing,
        augmentation=args.augmentation,
        split_strategy=args.split_strategy,
        unfreeze_layer4=not args.freeze_layer4,
    )
    run_training(cfg)
