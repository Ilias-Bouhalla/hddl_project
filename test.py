"""
Ce script est la base pour faire une inférence sur dataset fournit,

Il renvoie l'accuracy, le f1 score, le recall et la precision du modèle chargé.



"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset

from model import get_model_transforms


class _TransformDataset(Dataset):
    """Apply `transform` to non-tensor samples without mutating the source dataset."""

    def __init__(self, dataset: Dataset, transform) -> None:
        self.dataset = dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        img, label = self.dataset[idx]
        if not isinstance(img, torch.Tensor):
            img = self.transform(img)
        return img, label


def test_inference(
    model: torch.nn.Module,
    dataset,
    batch_size: int = 32,
    num_workers: int = 0,
) -> dict[str, float]:
    """Run inference on `dataset` (applying model transforms if needed) and return metrics."""
    model.eval()
    device = next(model.parameters()).device

    if isinstance(dataset, DataLoader):
        loader = dataset
    else:
        if getattr(dataset, "transform", None) is None:
            dataset = _TransformDataset(dataset, get_model_transforms(model))
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

    all_preds: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []

    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1).cpu()
            all_preds.append(preds)
            all_targets.append(labels.cpu())

    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)

    accuracy = (preds == targets).float().mean().item()

    classes = torch.unique(targets)
    precisions: list[float] = []
    recalls: list[float] = []
    f1s: list[float] = []

    for c in classes:
        tp = ((preds == c) & (targets == c)).sum().item()
        fp = ((preds == c) & (targets != c)).sum().item()
        fn = ((preds != c) & (targets == c)).sum().item()

        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)

    n = len(classes)
    return {
        "accuracy": accuracy,
        "f1": sum(f1s) / n if n > 0 else 0.0,
        "recall": sum(recalls) / n if n > 0 else 0.0,
        "precision": sum(precisions) / n if n > 0 else 0.0,
    }
