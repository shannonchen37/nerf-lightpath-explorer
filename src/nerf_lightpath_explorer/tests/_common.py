"""Shared constants and ray generation for R1 diagnostics."""

from __future__ import annotations

import torch

from nerf_lightpath_explorer.runtime_paths import CHECKPOINT_PATH, CONFIG_PATH, MODEL_ROOT, PROJECT_ROOT, load_environment


def load_emitter():
    return load_environment()


def deterministic_rays(count: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device).manual_seed(20260911 + count)
    origins = 0.15 + 0.70 * torch.rand((count, 3), generator=generator, device=device)
    directions = torch.randn((count, 3), generator=generator, device=device)
    directions = torch.nn.functional.normalize(directions, dim=-1)
    return origins.float(), directions.float()
