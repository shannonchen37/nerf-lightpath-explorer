"""Dependency-free unit tests for explicit R3 GGX math and sampling."""

from __future__ import annotations

import torch

from nerf_lightpath_explorer.materials.ggx import GGXMaterial, evaluate_ggx, fresnel_schlick, ggx_ndf, smith_ggx
from nerf_lightpath_explorer.sampling.ggx_sampling import sample_ggx_half_vector


def test_fresnel() -> None:
    f0 = torch.tensor([0.04, 0.2, 0.8])
    assert torch.allclose(fresnel_schlick(torch.tensor(1.0), f0), f0)
    assert torch.allclose(fresnel_schlick(torch.tensor(0.0), f0), torch.ones(3))


def test_ndf_and_smith() -> None:
    noh = torch.linspace(0, 1, 1001)
    low = ggx_ndf(noh, 0.05 ** 2)
    high = ggx_ndf(noh, 0.5 ** 2)
    assert torch.isfinite(low).all() and torch.isfinite(high).all()
    assert torch.all(low >= 0) and torch.all(high >= 0)
    assert low[-1] > high[-1]
    # Higher roughness is broader away from the normal.
    assert high[800] > low[800]
    G = smith_ggx(torch.linspace(0, 1, 101), torch.ones(101), 0.2 ** 2)
    assert torch.isfinite(G).all() and torch.all((G >= 0) & (G <= 1))


def test_sampling_and_pdf() -> None:
    count = 10000
    n = torch.tensor([[0., 0., 1.]]).expand(count, 3)
    wo = torch.tensor([[0., 0., 1.]]).expand(count, 3)
    u = torch.rand((count, 2), generator=torch.Generator().manual_seed(7))
    sample = sample_ggx_half_vector(n, wo, 0.3, u)
    assert torch.isfinite(sample.pdf).all()
    assert torch.all(sample.pdf[sample.valid] > 0)
    assert torch.all(torch.sum(n[sample.valid] * sample.wi[sample.valid], dim=-1) > 0)
    assert torch.allclose(torch.linalg.vector_norm(sample.wi, dim=-1), torch.ones(count), atol=1e-6)


def test_dielectric_and_metallic_brdf_change() -> None:
    n = torch.tensor([[0., 0., 1.]])
    wi = torch.nn.functional.normalize(torch.tensor([[0.3, 0.1, 1.]]), dim=-1)
    wo = torch.tensor([[0., 0., 1.]])
    color = torch.tensor([0.8, 0.3, 0.1])
    dielectric = evaluate_ggx(n, wi, wo, GGXMaterial(color, 0.2, 0.0))
    metallic = evaluate_ggx(n, wi, wo, GGXMaterial(color, 0.2, 1.0))
    assert torch.isfinite(dielectric.brdf).all() and torch.isfinite(metallic.brdf).all()
    assert not torch.allclose(dielectric.brdf, metallic.brdf)
    assert torch.all(dielectric.diffuse > 0) and torch.all(metallic.diffuse == 0)


if __name__ == "__main__":
    test_fresnel(); test_ndf_and_smith(); test_sampling_and_pdf(); test_dielectric_and_metallic_brdf_change()
    print("R3 GGX UNIT TESTS: PASS")
