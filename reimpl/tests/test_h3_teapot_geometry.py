"""Artifact acceptance for explicit teapot geometry takeover."""

import json
from pathlib import Path


ROOT = Path.home() / "nerf_inverse_rendering"


def main() -> None:
    output = ROOT / "results/h3_teapot_geometry"
    metrics = json.loads((output / "metrics.json").read_text())
    assert metrics["status"] == "PASS" and metrics["hit_pixels"] > 1000
    assert abs(metrics["normal_length_mean"] - 1.0) < 1e-4
    assert metrics["geometry"]["faces"] == 50000
    assert metrics["roughness_min"] >= 0 and metrics["roughness_max"] <= 1
    for name in ("mask.png", "normal.png", "uv.png", "albedo.png", "roughness.png"):
        assert (output / name).is_file()
    print("H3 TEAPOT GEOMETRY TESTS: PASS")


if __name__ == "__main__": main()

