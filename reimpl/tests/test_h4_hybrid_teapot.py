"""Artifact acceptance for H4 hybrid camera/secondary semantics."""

import json
from pathlib import Path


ROOT = Path.home() / "nerf_inverse_rendering"


def main() -> None:
    output = ROOT / "results/h4_hybrid_teapot"
    metrics = json.loads((output / "metrics.json").read_text())
    assert metrics["status"] == "PASS" and all(metrics["conditions"].values())
    render = metrics["render"]
    assert render["camera_formula"].startswith("C_pre")
    assert render["queried_secondary_rays"] > 0
    assert render["hybrid_vs_direct_overlay_mae_on_object"] > 0
    for name in ("full_nerf.png", "disable_aabb_only.png", "hybrid_teapot.png", "hybrid_vs_full_difference.png"):
        assert (output / name).is_file()
    print("H4 HYBRID TEAPOT TESTS: PASS")


if __name__ == "__main__": main()

