"""Linear-HDR image output and display transforms used by demo scripts."""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image


DISPLAY_SCALE = 0.12


def save_exr(path: Path, rgb: torch.Tensor) -> None:
    os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
    array = rgb.detach().cpu().numpy().astype(np.float32)
    if not cv2.imwrite(str(path), array[..., ::-1]):
        raise RuntimeError(f"OpenCV failed to write {path}")


def display_transform(rgb: torch.Tensor) -> np.ndarray:
    mapped = torch.log1p(DISPLAY_SCALE * torch.clamp_min(rgb, 0.0)) / np.log(2.0)
    mapped = torch.clamp(mapped, 0.0, 1.0)
    srgb = torch.where(mapped <= 0.0031308, 12.92 * mapped, 1.055 * mapped.pow(1 / 2.4) - 0.055)
    return (srgb.detach().cpu().numpy() * 255.0 + 0.5).astype(np.uint8)


def save_png(path: Path, rgb: torch.Tensor) -> None:
    Image.fromarray(display_transform(rgb), mode="RGB").save(path)
