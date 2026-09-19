from pathlib import Path

import torch

from nerf_lightpath_explorer.pbir.mitsuba_principled import MitsubaPrincipledConstantBSDF


def test_expected_principled_eval_is_finite_and_pdf_positive():
    device = torch.device("cuda:0")
    bsdf = MitsubaPrincipledConstantBSDF((0.5, 0.2, 0.1), 0.3, specular=1.0)
    n = torch.tensor([[0.0, 0.0, 1.0]] * 4, device=device)
    wi = torch.nn.functional.normalize(torch.tensor([[0.0,0.0,1.0],[0.3,0.0,1.0],[0.8,0.0,0.2],[-0.2,0.4,1.0]], device=device), dim=-1)
    wo = torch.tensor([[0.0, 0.0, 1.0]] * 4, device=device)
    uv = torch.zeros((4, 2), device=device)
    ev = bsdf.evaluate(n, wi, wo, uv)
    assert torch.isfinite(ev.weighted).all()
    assert torch.isfinite(ev.pdf).all()
    assert (ev.weighted > 0).all()
    assert (ev.pdf > 0).all()

