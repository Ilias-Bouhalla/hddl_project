from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image, ImageDraw
from torchvision import transforms

from model import get_model, get_model_transforms
from test import test_inference
from utils.dataset import ImageNetDataset

# Indices ImageNet-1K correspondant aux 16 classes triées d'ImageNet16 :
# n02009912..n04591713
IMAGENET16_INDICES = [
    132,
    265,
    285,
    294,
    386,
    404,
    444,
    472,
    499,
    508,
    530,
    734,
    765,
    766,
    817,
    907,
]

MODELS = ["efficientnet", "convnext", "resnet"]

CONDITIONS: dict[str, dict[str, bool]] = {
    "classic": {},
    "greyscale": {"greyscale": True},
    "silhouette": {"silhouette": True},
    "edges": {"edges": True},
}


class ImageNet16Adapter(nn.Module):
    """Projette les 1000 logits ImageNet-1K sur les 16 classes d'ImageNet16."""

    def __init__(self, backbone: nn.Module, indices: list[int]) -> None:
        super().__init__()
        self.backbone = backbone
        self.register_buffer("indices", torch.tensor(indices, dtype=torch.long))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)[:, self.indices]


def save_transforms_preview(
    output_path: str | Path = "outputs/transforms_preview.png",
    sample_classes: tuple[str, ...] = ("n02123597", "n02504458", "n02690373", "n02835271"),
    thumb_size: int = 224,
) -> Path:
    """Crée une grille comparant quelques images normales et leurs 3 transformations."""
    base_ds = ImageNetDataset(mode="val")
    sample_indices = [
        next(i for i, p in enumerate(base_ds.images) if wnid in str(p))
        for wnid in sample_classes
    ]

    datasets_by_cond = {
        cond_name: ImageNetDataset(mode="val", **kwargs)
        for cond_name, kwargs in CONDITIONS.items()
    }

    pad = 8
    header_h = 28
    cols = len(CONDITIONS)
    rows = len(sample_indices)
    grid_w = cols * thumb_size + (cols + 1) * pad
    grid_h = header_h + rows * thumb_size + (rows + 1) * pad

    canvas = Image.new("RGB", (grid_w, grid_h), color=(245, 245, 245))
    draw = ImageDraw.Draw(canvas)
    resize = transforms.Resize((thumb_size, thumb_size))

    for col_idx, cond_name in enumerate(CONDITIONS):
        x = pad + col_idx * (thumb_size + pad)
        draw.text((x + 6, 8), cond_name.upper(), fill=(20, 20, 20))

        ds = datasets_by_cond[cond_name]
        for row_idx, sample_idx in enumerate(sample_indices):
            y = header_h + pad + row_idx * (thumb_size + pad)
            img, _ = ds[sample_idx]
            canvas.paste(resize(img), (x, y))

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_file)
    return out_file


def main() -> None:
    preview_path = save_transforms_preview()
    print(f"Grille des transformations sauvegardée : {preview_path}\n")

    print(
        f"{'Modèle':<14} | {'Transformation':<12} | {'Accuracy':>8} | {'F1':>8} | {'Recall':>8} | {'Precision':>9}"
    )
    print("-" * 74)

    for model_name in MODELS:
        backbone = get_model(model_name, pretrained=True)
        transform = get_model_transforms(backbone)
        device = next(backbone.parameters()).device
        model = ImageNet16Adapter(backbone, IMAGENET16_INDICES).to(device)
        model.eval()

        for cond_name, kwargs in CONDITIONS.items():
            ds = ImageNetDataset(mode="val", transform=transform, **kwargs)
            metrics = test_inference(model, ds, batch_size=32, num_workers=4)
            print(
                f"{model_name:<14} | {cond_name:<12} | "
                f"{metrics['accuracy']:>8.4f} | {metrics['f1']:>8.4f} | "
                f"{metrics['recall']:>8.4f} | {metrics['precision']:>9.4f}"
            )


if __name__ == "__main__":
    main()
