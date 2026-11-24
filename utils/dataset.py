"""
Ce fichier introduit la classe basique pour chargé le dataset ImageNet16

Ce fichier charge les transform de simple_style_trnasform.py


"""

from __future__ import annotations

from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

from utils.simple_style_transform import Edges, GreyScale, Silhouette

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class ImageNetDataset(Dataset):
    def __init__(
        self,
        root_path: str | Path = "datasets",
        transform=None,
        mode: str = "train",
        silhouette: bool = False,
        greyscale: bool = False,
        edges: bool = False,
    ) -> None:
        super().__init__()
        self.root_path = Path(root_path)
        self.transform = transform
        self.mode = mode
        self.silhouette = silhouette
        self.greyscale = greyscale
        self.edges = edges

        self._greyscale_tf = GreyScale() if greyscale else None
        self._silhouette_tf = Silhouette() if silhouette else None
        self._edges_tf = Edges() if edges else None

        dataset_root = self._resolve_dataset_root(self.root_path)
        split_dir = dataset_root / mode
        if not split_dir.is_dir():
            raise FileNotFoundError(
                f"Dossier split '{mode}' introuvable dans : {dataset_root}"
            )

        self.classes = sorted(p.name for p in split_dir.iterdir() if p.is_dir())
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.class_names = self._load_class_map(dataset_root / "map.txt")

        self.images: list[Path] = []
        self.labels: list[int] = []
        for cls_name in self.classes:
            cls_dir = split_dir / cls_name
            label = self.class_to_idx[cls_name]
            for img_path in sorted(cls_dir.iterdir()):
                if img_path.is_file() and img_path.suffix.lower() in _IMAGE_EXTENSIONS:
                    self.images.append(img_path)
                    self.labels.append(label)

    @staticmethod
    def _resolve_dataset_root(root_path: Path) -> Path:
        if (root_path / "imagenet16").is_dir():
            return root_path / "imagenet16"
        return root_path

    @staticmethod
    def _load_class_map(map_file: Path) -> dict[str, str]:
        if not map_file.is_file():
            return {}
        mapping: dict[str, str] = {}
        for line in map_file.read_text(encoding="utf-8").splitlines():
            if "=" in line:
                name, wnid = line.split("=", 1)
                mapping[wnid.strip()] = name.strip()
        return mapping

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        img = Image.open(self.images[idx]).convert("RGB")

        if self._greyscale_tf is not None:
            img = self._greyscale_tf(img)
        if self._silhouette_tf is not None:
            img = self._silhouette_tf(img)
        if self._edges_tf is not None:
            img = self._edges_tf(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, self.labels[idx]
