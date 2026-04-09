from __future__ import annotations

import logging
import time
from pathlib import Path

from PIL import Image


def load_image(path: str | Path) -> Image.Image:
    image_path = Path(path)
    last_error: OSError | None = None
    for attempt in range(1, 4):
        try:
            with Image.open(image_path) as image:
                return image.convert("RGB")
        except OSError as error:
            last_error = error
            if attempt == 3:
                break
            logging.warning(
                "Retrying image read for %s after transient I/O failure (%s/%s): %s",
                image_path,
                attempt,
                3,
                error,
            )
            time.sleep(0.5 * attempt)
    assert last_error is not None
    raise last_error


def save_image(image: Image.Image, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    image.save(target)
    return target


def make_blank_image(size: tuple[int, int] = (448, 448), color: tuple[int, int, int] = (255, 255, 255)) -> Image.Image:
    return Image.new("RGB", size, color)
