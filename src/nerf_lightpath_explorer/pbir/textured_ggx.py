"""R3 GGX equations generalized to reference-benchmark surface textures."""

from __future__ import annotations

from dataclasses import dataclass
from math import pi

import torch

from nerf_lightpath_explorer.materials.ggx import DENOMINATOR_EPSILON, fresnel_schlick


@dataclass(frozen=True)
class TexturedGGXEvaluation:
    brdf: torch.Tensor
    diffuse: torch.Tensor
    specular: torch.Tensor
    D: torch.Tensor
    F: torch.Tensor
    G: torch.Tensor
    NoL: torch.Tensor


def _smith_g1(nox: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    alpha2 = alpha.square()
    root = torch.sqrt(torch.clamp_min(alpha2 + (1.0 - alpha2) * nox.square(), 0.0))
    value = 2.0 * nox / (nox + root).clamp_min(DENOMINATOR_EPSILON)
    return torch.where(nox > 0, value, torch.zeros_like(value))


def evaluate_textured_dielectric_ggx(
    normal: torch.Tensor,
    wi: torch.Tensor,
    wo: torch.Tensor,
    base_color: torch.Tensor,
    roughness: torch.Tensor,
) -> TexturedGGXEvaluation:
    if not (normal.shape == wi.shape == wo.shape == base_color.shape) or normal.shape[-1] != 3:
        raise ValueError("direction/color tensors must have identical [...,3] shape")
    if roughness.shape != normal.shape[:-1]:
        raise ValueError("roughness must match direction batch shape")
    half_vector = torch.nn.functional.normalize(wi + wo, dim=-1)
    nol = torch.clamp_min(torch.sum(normal * wi, dim=-1), 0.0)
    nov = torch.clamp_min(torch.sum(normal * wo, dim=-1), 0.0)
    noh = torch.clamp_min(torch.sum(normal * half_vector, dim=-1), 0.0)
    voh = torch.clamp(torch.sum(wo * half_vector, dim=-1), 0.0, 1.0)
    alpha = torch.clamp_min(roughness, 0.02).square()
    alpha2 = alpha.square()
    D = alpha2 / (pi * (noh.square() * (alpha2 - 1.0) + 1.0).square()).clamp_min(DENOMINATOR_EPSILON)
    G = _smith_g1(nol, alpha) * _smith_g1(nov, alpha)
    f0 = torch.full_like(base_color, 0.04)
    F = fresnel_schlick(voh, f0)
    diffuse = base_color / pi
    specular = D[..., None] * G[..., None] * F / (4.0 * nol * nov).clamp_min(DENOMINATOR_EPSILON)[..., None]
    brdf = diffuse + specular
    if not torch.isfinite(brdf).all():
        raise RuntimeError("non-finite textured GGX")
    return TexturedGGXEvaluation(brdf, diffuse, specular, D, F, G, nol)
