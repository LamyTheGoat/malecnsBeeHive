"""Reading photos and remembering how you swiped on them."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}


def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    """Undo the display gamma so contrast maths happens in linear light."""
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def load_image(path: str | Path, grid: int) -> np.ndarray:
    """Load a photo as a ``(grid, grid, 3)`` linear-light array in [0, 1]."""
    from PIL import Image  # imported lazily so the core model stays dependency-light

    with Image.open(path) as im:
        im = im.convert("RGB")
        # Centre crop to square first: the fly's frontal field is roughly
        # symmetric, and stretching would distort the orientation channels.
        w, h = im.size
        side = min(w, h)
        im = im.crop(
            ((w - side) // 2, (h - side) // 2, (w + side) // 2, (h + side) // 2)
        ).resize((grid, grid), Image.LANCZOS)
        arr = np.asarray(im, dtype=np.float32) / 255.0
    return srgb_to_linear(arr).astype(np.float32)


def list_photos(directory: str | Path) -> list[Path]:
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(f"{directory} bir klasor degil")
    return sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )


class LabelStore:
    """Your swipes, kept next to the photos as plain JSON."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.labels: dict[str, str] = {}
        if self.path.exists():
            self.labels = json.loads(self.path.read_text(encoding="utf-8"))

    def get(self, name: str) -> str | None:
        return self.labels.get(name)

    def set(self, name: str, swipe: str) -> None:
        if swipe not in ("left", "right"):
            raise ValueError("swipe 'left' veya 'right' olmali")
        self.labels[name] = swipe

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self.labels, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def __len__(self) -> int:
        return len(self.labels)
