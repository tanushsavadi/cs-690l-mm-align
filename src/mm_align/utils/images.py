from __future__ import annotations

from pathlib import Path

from PIL import Image


def load_image(path: str | Path) -> Image.Image:
    image = Image.open(path)
    return image.convert("RGB")


def save_image(image: Image.Image, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    image.save(target)
    return target


def make_blank_image(size: tuple[int, int] = (448, 448), color: tuple[int, int, int] = (255, 255, 255)) -> Image.Image:
    return Image.new("RGB", size, color)
