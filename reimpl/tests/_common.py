"""Shared constants and ray generation for R1 diagnostics."""

from __future__ import annotations

from pathlib import Path

import torch

from reimpl.nerf_emitter_query import StandaloneNerfEmitter

PROJECT_ROOT = Path.home() / "nerf_inverse_rendering"
OFFICIAL_ROOT = PROJECT_ROOT / "third_party" / "nerf-emitter"
CONFIG_PATH = OFFICIAL_ROOT / "outputs/synthetic/teapot-unirough-0.2_bedroom_v2/sdf-nerfacto/v1/config.yml"
CHECKPOINT_PATH = CONFIG_PATH.parent / "nerfstudio_models/step-000002319.ckpt"


def load_emitter() -> StandaloneNerfEmitter:
    return StandaloneNerfEmitter.from_official_checkpoint(
        CONFIG_PATH, CHECKPOINT_PATH, device="cuda:0", official_repo_root=OFFICIAL_ROOT
    )


def deterministic_rays(count: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device).manual_seed(20260911 + count)
    origins = 0.15 + 0.70 * torch.rand((count, 3), generator=generator, device=device)
    directions = torch.randn((count, 3), generator=generator, device=device)
    directions = torch.nn.functional.normalize(directions, dim=-1)
    return origins.float(), directions.float()
