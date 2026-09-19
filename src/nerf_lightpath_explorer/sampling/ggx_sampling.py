"""Isotropic GGX half-vector importance sampling and reflection PDF."""

from __future__ import annotations

from dataclasses import dataclass
from math import pi

import torch

from nerf_lightpath_explorer.materials.ggx import DENOMINATOR_EPSILON, ggx_ndf


@dataclass(frozen=True)
class GGXSamples:
    half_vector: torch.Tensor
    wi: torch.Tensor
    pdf: torch.Tensor
    valid: torch.Tensor
    u1: torch.Tensor
    u2: torch.Tensor


def _basis(normal: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    z = torch.tensor([0.0, 0.0, 1.0], dtype=normal.dtype, device=normal.device).expand_as(normal)
    x = torch.tensor([1.0, 0.0, 0.0], dtype=normal.dtype, device=normal.device).expand_as(normal)
    helper = torch.where((normal[..., 2].abs() < 0.999)[..., None], z, x)
    tangent = torch.nn.functional.normalize(torch.linalg.cross(helper, normal, dim=-1), dim=-1)
    bitangent = torch.linalg.cross(normal, tangent, dim=-1)
    return tangent, bitangent


def ggx_reflection_pdf(
    normal: torch.Tensor, wo: torch.Tensor, half_vector: torch.Tensor, alpha: float
) -> torch.Tensor:
    noh = torch.clamp_min(torch.sum(normal * half_vector, dim=-1), 0.0)
    voh = torch.abs(torch.sum(wo * half_vector, dim=-1))
    p_h = ggx_ndf(noh, alpha) * noh
    pdf = p_h / (4.0 * voh).clamp_min(DENOMINATOR_EPSILON)
    return torch.where((noh > 0) & (voh > 0), pdf, torch.zeros_like(pdf))


def sample_ggx_half_vector(
    normal: torch.Tensor,
    wo: torch.Tensor,
    roughness: float,
    uniforms: torch.Tensor,
) -> GGXSamples:
    """Sample h~D(h)NoH, then map it to wi by specular reflection."""
    if normal.shape != wo.shape or normal.shape[-1] != 3:
        raise ValueError("normal and wo must have identical [...,3] shape")
    if uniforms.shape != normal.shape[:-1] + (2,):
        raise ValueError("uniforms must have shape [...,2]")
    if not torch.isfinite(uniforms).all() or torch.any(uniforms < 0) or torch.any(uniforms >= 1):
        raise ValueError("uniforms must be finite in [0,1)")
    alpha = max(float(roughness), 0.02) ** 2
    u1, u2 = uniforms[..., 0], uniforms[..., 1]
    alpha2 = alpha * alpha
    cos_theta = torch.sqrt((1.0 - u1) / (1.0 + (alpha2 - 1.0) * u1).clamp_min(DENOMINATOR_EPSILON))
    sin_theta = torch.sqrt(torch.clamp_min(1.0 - cos_theta.square(), 0.0))
    phi = 2.0 * pi * u2
    local = torch.stack((sin_theta * torch.cos(phi), sin_theta * torch.sin(phi), cos_theta), dim=-1)
    tangent, bitangent = _basis(normal)
    half_vector = (
        tangent * local[..., 0, None]
        + bitangent * local[..., 1, None]
        + normal * local[..., 2, None]
    )
    half_vector = torch.nn.functional.normalize(half_vector, dim=-1)
    voh = torch.sum(wo * half_vector, dim=-1)
    wi = -wo + 2.0 * voh[..., None] * half_vector
    wi = torch.nn.functional.normalize(wi, dim=-1)
    nol = torch.sum(normal * wi, dim=-1)
    pdf = ggx_reflection_pdf(normal, wo, half_vector, alpha)
    valid = (voh > 0.0) & (nol > 0.0) & (pdf > DENOMINATOR_EPSILON)
    for name, value in (("half_vector", half_vector), ("wi", wi), ("pdf", pdf)):
        if not torch.isfinite(value).all():
            raise RuntimeError(f"non-finite sampled {name}")
    return GGXSamples(half_vector, wi, pdf, valid, u1, u2)
