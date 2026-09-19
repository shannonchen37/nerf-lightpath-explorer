"""Configurable paths for regenerating research artifacts."""

from __future__ import annotations

import os
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(os.environ.get("LIGHTPATH_PROJECT_ROOT", REPOSITORY_ROOT)).expanduser().resolve()
MODEL_ROOT = Path(os.environ.get("LIGHTPATH_MODEL_ROOT", PROJECT_ROOT / "model_runtime")).expanduser().resolve()
ASSET_ROOT = Path(os.environ.get("LIGHTPATH_ASSET_ROOT", MODEL_ROOT / "scene_assets")).expanduser().resolve()
CONFIG_PATH = Path(os.environ.get("LIGHTPATH_CONFIG", MODEL_ROOT / "config.yml")).expanduser().resolve()
CHECKPOINT_PATH = Path(
    os.environ.get("LIGHTPATH_CHECKPOINT", MODEL_ROOT / "nerfstudio_models" / "step-000002319.ckpt")
).expanduser().resolve()
RESULTS_ROOT = Path(os.environ.get("LIGHTPATH_RESULTS_ROOT", PROJECT_ROOT / "results")).expanduser().resolve()


def load_environment(device: str = "cuda:0"):
    """Load the configured neural environment without importing GPU code eagerly."""
    from .neural_environment import EnvironmentRadianceField

    return EnvironmentRadianceField.from_checkpoint(
        CONFIG_PATH,
        CHECKPOINT_PATH,
        device=device,
        model_root=MODEL_ROOT,
    )
