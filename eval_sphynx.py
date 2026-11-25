from __future__ import annotations

from collections import Counter
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ConvNeXt_Tiny_Weights

from model import get_model, get_model_transforms

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Indices ImageNet-1K des chats domestiques :
# 281=tabby, 282=tiger cat, 283=Persian cat, 284=Siamese cat, 285=Egyptian cat
CAT_INDICES = {281, 282, 283, 284, 285}
# Indices ImageNet-1K des races de chiens (151..268)
DOG_INDICES = set(range(151, 269))


class SphynxDataset(Dataset):
    def __init__(self, root: str | Path, transform) -> None:
        self.paths = sorted(
            p
            for p in Path(root).rglob("*")
            if p.is_file() and p.suffix.lower() in _IMAGE_EXTENSIONS
        )
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img)


def main() -> None:
    sphynx_root = Path("datasets/sphynx")
    imagenet1k_cats = ConvNeXt_Tiny_Weights.IMAGENET1K_V1.meta["categories"]

    for model_name in ("efficientnet", "convnext", "resnet"):
        model = get_model(model_name, pretrained=True)
        transform = get_model_transforms(model)
        device = next(model.parameters()).device

        ds = SphynxDataset(sphynx_root, transform)
        loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=4)

        preds_1k: list[int] = []
        with torch.inference_mode():
            for images in loader:
                logits = model(images.to(device)).cpu()
                preds_1k.extend(logits.argmax(dim=1).tolist())

        total = len(ds)
        counts = Counter(preds_1k).most_common()

        n_cat = sum(1 for p in preds_1k if p in CAT_INDICES)
        n_dog = sum(1 for p in preds_1k if p in DOG_INDICES)
        n_other = total - n_cat - n_dog

        print(f"\n=== {model_name.upper()} — ImageNet-1000 ({total} images de sphynx) ===")
        print(
            f"Résumé : Chat={n_cat}/{total} ({100*n_cat/total:.1f}%) | "
            f"Chien={n_dog}/{total} ({100*n_dog/total:.1f}%) | "
            f"Autre={n_other}/{total} ({100*n_other/total:.1f}%)"
        )
        print("Détail complet des classes prédites :")
        for cls_idx, count in counts:
            tag = "Chat" if cls_idx in CAT_INDICES else ("Chien" if cls_idx in DOG_INDICES else "Autre")
            print(
                f"  - [{cls_idx:>3}] {imagenet1k_cats[cls_idx]:<30} ({tag:<5}) : "
                f"{count:>3}/{total} ({100 * count / total:5.1f}%)"
            )


if __name__ == "__main__":
    main()
