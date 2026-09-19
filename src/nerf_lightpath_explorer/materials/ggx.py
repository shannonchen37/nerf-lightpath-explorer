"""Minimal metallic-roughness GGX BRDF, written explicitly in PyTorch."""

from __future__ import annotations

from dataclasses import dataclass
from math import pi

import torch

DENOMINATOR_EPSILON = 1.0e-8
MIN_ROUGHNESS = 0.02


@dataclass(frozen=True)
class GGXMaterial:
    base_color: torch.Tensor
    roughness: float
    metallic: float

    def validate(self) -> None:
        if self.base_color.shape != (3,) or not torch.isfinite(self.base_color).all():
            raise ValueError("base_color must be a finite RGB tensor")
        if torch.any(self.base_color < 0):
            raise ValueError("base_color must be nonnegative")
        if not (0.0 <= self.metallic <= 1.0):
            raise ValueError("metallic must be in [0,1]")
        if not (self.roughness >= MIN_ROUGHNESS):
            raise ValueError(f"roughness must be >= {MIN_ROUGHNESS}")

    @property
    def alpha(self) -> float:
        return self.roughness * self.roughness


@dataclass(frozen=True)
class GGXEvaluation:
    D: torch.Tensor
    G: torch.Tensor
    F: torch.Tensor
    diffuse: torch.Tensor
    specular: torch.Tensor
    brdf: torch.Tensor
    NoL: torch.Tensor
    NoV: torch.Tensor
    NoH: torch.Tensor
    VoH: torch.Tensor


def fresnel_schlick(voh: torch.Tensor, f0: torch.Tensor) -> torch.Tensor:
    """Schlick Fresnel for surface-to-light/view conventions."""
    voh = torch.clamp(voh, 0.0, 1.0)
    return f0 + (1.0 - f0) * (1.0 - voh[..., None]).pow(5)


def ggx_ndf(noh: torch.Tensor, alpha: float) -> torch.Tensor:
    """Isotropic Trowbridge-Reitz/GGX normal distribution."""
    noh = torch.clamp(noh, 0.0, 1.0)
    alpha2 = alpha * alpha
    denominator = pi * (noh.square() * (alpha2 - 1.0) + 1.0).square()
    return alpha2 / denominator.clamp_min(DENOMINATOR_EPSILON)


def smith_g1_ggx(nox: torch.Tensor, alpha: float) -> torch.Tensor:
    """Exact Smith GGX G1 in a stable cosine form."""
    nox = torch.clamp(nox, 0.0, 1.0)
    alpha2 = alpha * alpha
    root = torch.sqrt(torch.clamp_min(alpha2 + (1.0 - alpha2) * nox.square(), 0.0))
    value = 2.0 * nox / (nox + root).clamp_min(DENOMINATOR_EPSILON)
    return torch.where(nox > 0.0, value, torch.zeros_like(value))


def smith_ggx(nol: torch.Tensor, nov: torch.Tensor, alpha: float) -> torch.Tensor:
    return smith_g1_ggx(nol, alpha) * smith_g1_ggx(nov, alpha)


def evaluate_ggx(
    normal: torch.Tensor,
    wi: torch.Tensor,
    wo: torch.Tensor,
    material: GGXMaterial,
    half_vector: torch.Tensor | None = None,
) -> GGXEvaluation:
    """Evaluate explicit Lambert + Cook-Torrance GGX BRDF."""
    material.validate()
    if normal.shape != wi.shape or normal.shape != wo.shape or normal.shape[-1] != 3:
        raise ValueError("normal, wi and wo must have identical [...,3] shape")
    if half_vector is None:
        half_sum = wi + wo
        half_norm = torch.linalg.vector_norm(half_sum, dim=-1, keepdim=True)
        half_vector = half_sum / half_norm.clamp_min(DENOMINATOR_EPSILON)
    if half_vector.shape != normal.shape:
        raise ValueError("half_vector must match normal shape")

    nol = torch.clamp_min(torch.sum(normal * wi, dim=-1), 0.0)
    nov = torch.clamp_min(torch.sum(normal * wo, dim=-1), 0.0)
    noh = torch.clamp_min(torch.sum(normal * half_vector, dim=-1), 0.0)
    voh = torch.clamp(torch.sum(wo * half_vector, dim=-1), 0.0, 1.0)
    D = ggx_ndf(noh, material.alpha)
    G = smith_ggx(nol, nov, material.alpha)
    f0 = torch.lerp(torch.full_like(material.base_color, 0.04), material.base_color, material.metallic)
    F = fresnel_schlick(voh, f0)
    diffuse = (1.0 - material.metallic) * material.base_color / pi
    diffuse = diffuse.expand(normal.shape)
    denominator = (4.0 * nol * nov).clamp_min(DENOMINATOR_EPSILON)
    specular = D[..., None] * G[..., None] * F / denominator[..., None]
    brdf = diffuse + specular
    for name, value in (("D", D), ("G", G), ("F", F), ("diffuse", diffuse), ("specular", specular), ("brdf", brdf)):
        if not torch.isfinite(value).all():
            raise RuntimeError(f"non-finite GGX {name}")
    return GGXEvaluation(D, G, F, diffuse, specular, brdf, nol, nov, noh, voh)
