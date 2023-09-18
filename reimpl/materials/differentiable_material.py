"""Smooth unconstrained parameterization of the R3 GGX material."""

from __future__ import annotations

import torch

from reimpl.materials.ggx import GGXMaterial


def _logit(value: torch.Tensor) -> torch.Tensor:
    return torch.log(value) - torch.log1p(-value)


class DifferentiableGGXMaterial(torch.nn.Module):
    def __init__(
        self,
        initial_base_color: torch.Tensor,
        initial_roughness: float,
        metallic: float = 0.0,
        roughness_min: float = 0.05,
        roughness_max: float = 0.8,
    ) -> None:
        super().__init__()
        if initial_base_color.shape != (3,) or torch.any(initial_base_color <= 0) or torch.any(initial_base_color >= 1):
            raise ValueError("initial_base_color must be RGB strictly inside (0,1)")
        if not roughness_min < initial_roughness < roughness_max:
            raise ValueError("initial roughness must be strictly inside configured range")
        self.base_color_logits = torch.nn.Parameter(_logit(initial_base_color.detach().clone()))
        normalized_roughness = torch.tensor(
            (initial_roughness - roughness_min) / (roughness_max - roughness_min),
            dtype=initial_base_color.dtype,
            device=initial_base_color.device,
        )
        self.roughness_logit = torch.nn.Parameter(_logit(normalized_roughness))
        self.register_buffer("metallic", torch.tensor(float(metallic), dtype=initial_base_color.dtype, device=initial_base_color.device))
        self.roughness_min = float(roughness_min)
        self.roughness_max = float(roughness_max)

    @property
    def base_color(self) -> torch.Tensor:
        return torch.sigmoid(self.base_color_logits)

    @property
    def roughness(self) -> torch.Tensor:
        span = self.roughness_max - self.roughness_min
        return self.roughness_min + span * torch.sigmoid(self.roughness_logit)

    def as_ggx(self) -> GGXMaterial:
        return GGXMaterial(self.base_color, self.roughness, self.metallic)
