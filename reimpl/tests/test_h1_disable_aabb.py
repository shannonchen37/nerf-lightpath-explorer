"""Artifact acceptance for H1 reference-AABB density removal."""

import json
from pathlib import Path


ROOT = Path.home() / "nerf_inverse_rendering"


def main() -> None:
    output = ROOT / "results/h1_disable_aabb"
    metrics = json.loads((output / "metrics.json").read_text())
    assert metrics["status"] == "PASS"
    audit = metrics["density_audit"]
    assert audit["all_modules_exact_zero_inside"]
    assert len(audit["modules"]) == 3
    for module in audit["modules"].values():
        assert module["full_nonzero_count"] == module["sample_count_inside_bbox"]
        assert module["disabled_nonzero_count"] == 0
        assert module["disabled_max_abs"] == 0
    assert metrics["image_audit"]["max_abs"] > 1e-3
    for name in ("full_nerf.png", "disable_aabb.png", "difference.png"):
        assert (output / name).is_file()
    print("H1 DISABLE AABB TESTS: PASS")


if __name__ == "__main__": main()
