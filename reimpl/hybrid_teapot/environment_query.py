"""Separate full-NeRF and PBIR environment-only radiance query APIs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from reimpl.hybrid_teapot.aabb_control import set_aabb_disabled


@dataclass(frozen=True)
class EnvironmentNerfQuery:
    emitter: Any

    @torch.inference_mode()
    def query_full_radiance(
        self, origins: torch.Tensor, directions: torch.Tensor, camera_idx: int,
        near: float | torch.Tensor = 0.0, chunk_size: int = 16384,
    ) -> torch.Tensor:
        """Preserve the R1 full-NeRF semantics, independent of ambient state."""
        with set_aabb_disabled(self.emitter.model, False):
            return self.emitter.query_radiance(origins, directions, camera_idx, near, chunk_size)

    @torch.inference_mode()
    def query_environment_radiance(
        self, origins: torch.Tensor, directions: torch.Tensor, camera_idx: int,
        near: float | torch.Tensor = 0.0, chunk_size: int = 16384,
    ) -> torch.Tensor:
        """Official PBIR emitter semantics: zero density inside object AABB."""
        with set_aabb_disabled(self.emitter.model, True):
            return self.emitter.query_radiance(origins, directions, camera_idx, near, chunk_size)

