"""Material-independent cosine-weighted hemisphere sampling."""

from __future__ import annotations

from dataclasses import dataclass
from math import pi

import torch


@dataclass(frozen=True)
class CosineSamples:
    wi: torch.Tensor
    pdf: torch.Tensor
    NoL: torch.Tensor
    tangent: torch.Tensor
    bitangent: torch.Tensor
    u1: torch.Tensor
    u2: torch.Tensor


def stable_onb(normal: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a stable orthonormal basis by choosing a nonparallel helper axis."""
    if normal.shape[-1] != 3:
        raise ValueError("normal must have shape [...,3]")
    if not torch.isfinite(normal).all():
        raise ValueError("normal must be finite")
    normal_length = torch.linalg.vector_norm(normal, dim=-1)
    if not torch.allclose(normal_length, torch.ones_like(normal_length), atol=1e-5, rtol=1e-5):
        raise ValueError("normal must be normalized")
    axis_z = torch.tensor([0., 0., 1.], dtype=normal.dtype, device=normal.device).expand_as(normal)
    axis_x = torch.tensor([1., 0., 0.], dtype=normal.dtype, device=normal.device).expand_as(normal)
    helper = torch.where((normal[..., 2].abs() < 0.999)[..., None], axis_z, axis_x)
    tangent = torch.nn.functional.normalize(torch.linalg.cross(helper, normal, dim=-1), dim=-1)
    bitangent = torch.linalg.cross(normal, tangent, dim=-1)
    return tangent, bitangent


def sample_cosine_hemisphere(normal: torch.Tensor, uniforms: torch.Tensor) -> CosineSamples:
    """Map fixed uniforms to cosine-weighted directions about each normal."""
    if uniforms.shape != normal.shape[:-1] + (2,):
        raise ValueError("uniforms must have shape [...,2]")
    if not torch.isfinite(uniforms).all() or torch.any(uniforms < 0) or torch.any(uniforms >= 1):
        raise ValueError("uniforms must be finite in [0,1)")
    u1, u2 = uniforms[..., 0], uniforms[..., 1]
    radius = torch.sqrt(u1)
    phi = 2.0 * pi * u2
    local_x = radius * torch.cos(phi)
    local_y = radius * torch.sin(phi)
    local_z = torch.sqrt(torch.clamp_min(1.0 - u1, 0.0))
    tangent, bitangent = stable_onb(normal)
    wi = tangent * local_x[..., None] + bitangent * local_y[..., None] + normal * local_z[..., None]
    wi = torch.nn.functional.normalize(wi, dim=-1)
    nol = torch.clamp_min(torch.sum(normal * wi, dim=-1), 0.0)
    pdf = nol / pi
    for name, value in (("wi", wi), ("NoL", nol), ("pdf", pdf)):
        if not torch.isfinite(value).all():
            raise RuntimeError(f"non-finite cosine sample {name}")
    return CosineSamples(wi, pdf, nol, tangent, bitangent, u1, u2)
