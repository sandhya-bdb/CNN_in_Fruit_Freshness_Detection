import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms


class TransformSubset(Dataset):
    def __init__(self, base_dataset, indices, transform):
        self.base_dataset = base_dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        image_idx = self.indices[idx]
        path, label = self.base_dataset.samples[image_idx]
        image = self.base_dataset.loader(path)
        if self.transform is not None:
            image = self.transform(image)
        return image, label


class FruitClassifierResNet(nn.Module):
    def __init__(self, num_classes=2, dropout=0.4, unfreeze_layer4=True):
        super().__init__()
        self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        for p in self.model.parameters():
            p.requires_grad = False

        if unfreeze_layer4:
            for p in self.model.layer4.parameters():
                p.requires_grad = True

        self.model.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.model.fc.in_features, num_classes),
        )

    def forward(self, x):
        return self.model(x)


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



def get_transforms(image_size=224, strong=False):
    if strong:
        train_tf = transforms.Compose([
            transforms.RandomResizedCrop((image_size, image_size), scale=(0.75, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.4, saturation=0.2, hue=0.03),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        train_tf = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.1, contrast=0.3),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    eval_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_tf, eval_tf



def stratified_split(targets, val_split=0.2, test_split=0.2, seed=42):
    if val_split + test_split >= 1.0:
        raise ValueError("val_split + test_split must be < 1.0")

    rng = random.Random(seed)
    by_class = defaultdict(list)
    for i, y in enumerate(targets):
        by_class[y].append(i)

    train_idx, val_idx, test_idx = [], [], []
    for _, idxs in by_class.items():
        rng.shuffle(idxs)
        n = len(idxs)
        n_test = int(n * test_split)
        n_val = int(n * val_split)
        test_idx.extend(idxs[:n_test])
        val_idx.extend(idxs[n_test:n_test + n_val])
        train_idx.extend(idxs[n_test + n_val:])

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)
    return train_idx, val_idx, test_idx


def grouped_split(dataset, prefix_len=8, val_split=0.2, test_split=0.2, seed=42):
    """
    Splits the dataset ensuring that images with the same prefix (e.g., same fruit) 
    are kept in the same split to avoid data leakage.
    """
    if val_split + test_split >= 1.0:
        raise ValueError("val_split + test_split must be < 1.0")

    rng = random.Random(seed)
    
    # Group indices by prefix
    groups = defaultdict(list)
    for i, (path, _) in enumerate(dataset.samples):
        filename = Path(path).name
        # Use first `prefix_len` characters to group images of the same fruit together
        prefix = filename[:prefix_len]
        groups[prefix].append(i)
        
    group_keys = list(groups.keys())
    rng.shuffle(group_keys)
    
    n_groups = len(group_keys)
    n_test = int(n_groups * test_split)
    n_val = int(n_groups * val_split)
    
    test_groups = group_keys[:n_test]
    val_groups = group_keys[n_test:n_test + n_val]
    train_groups = group_keys[n_test + n_val:]
    
    train_idx, val_idx, test_idx = [], [], []
    for g in test_groups: test_idx.extend(groups[g])
    for g in val_groups: val_idx.extend(groups[g])
    for g in train_groups: train_idx.extend(groups[g])
        
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)
    
    return train_idx, val_idx, test_idx



def run_epoch(model, loader, criterion, device, optimizer=None):
    train_mode = optimizer is not None
    model.train(train_mode)

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        if train_mode:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train_mode):
            logits = model(images)
            loss = criterion(logits, labels)
            if train_mode:
                loss.backward()
                optimizer.step()

        preds = torch.argmax(logits, dim=1)
        total_loss += loss.item() * labels.size(0)
        total_correct += (preds == labels).sum().item()
        total_count += labels.size(0)

    return total_loss / max(1, total_count), total_correct / max(1, total_count)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", default="artifacts/minimal_run")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preset", choices=["tuned", "realistic_eval"], default="realistic_eval")
    parser.add_argument("--group-by-prefix", type=int, default=0, help="Length of filename prefix to group by. 0 disables grouping.")
    args = parser.parse_args()

    val_split = 0.15
    test_split = 0.15
    strong_aug = False
    unfreeze_layer4 = True
    label_smoothing = 0.0
    patience = 3

    if args.preset == "realistic_eval":
        val_split = 0.2
        test_split = 0.2
        args.dropout = max(args.dropout, 0.5)
        args.weight_decay = max(args.weight_decay, 5e-4)
        strong_aug = True
        unfreeze_layer4 = False
        label_smoothing = 0.1
        patience = 2

    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_dataset = datasets.ImageFolder(root=args.dataset_dir)
    train_tf, eval_tf = get_transforms(args.image_size, strong=strong_aug)

    if args.group_by_prefix > 0:
        print(f"Using grouped split by prefix length: {args.group_by_prefix}")
        train_idx, val_idx, test_idx = grouped_split(base_dataset, args.group_by_prefix, val_split, test_split, args.seed)
    else:
        print("Using standard stratified split")
        train_idx, val_idx, test_idx = stratified_split(base_dataset.targets, val_split, test_split, args.seed)
    train_ds = TransformSubset(base_dataset, train_idx, train_tf)
    val_ds = TransformSubset(base_dataset, val_idx, eval_tf)
    test_ds = TransformSubset(base_dataset, test_idx, eval_tf)

    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FruitClassifierResNet(len(base_dataset.classes), args.dropout, unfreeze_layer4).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)

    best_val_acc = 0.0
    best_epoch = -1
    no_improve = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        })

        print(f"Epoch {epoch}/{args.epochs} train_acc={train_acc:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            no_improve = 0
            torch.save(model.state_dict(), output_dir / "best_model.pth")
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    model.load_state_dict(torch.load(output_dir / "best_model.pth", map_location=device))
    val_loss, val_acc = run_epoch(model, val_loader, criterion, device)
    test_loss, test_acc = run_epoch(model, test_loader, criterion, device)

    metrics = {
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "val_acc": val_acc,
        "val_loss": val_loss,
        "test_acc": test_acc,
        "test_loss": test_loss,
        "classes": base_dataset.classes,
        "preset": args.preset,
        "group_by_prefix": args.group_by_prefix,
        "config": vars(args),
        "history": history,
    }

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Done. best_val_acc={best_val_acc:.4f}, test_acc={test_acc:.4f}")


if __name__ == "__main__":
    main()
