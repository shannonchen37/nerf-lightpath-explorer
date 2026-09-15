"""H2 full/environment API must match reference state control and restore state."""

import torch

from reimpl.hybrid_teapot.aabb_control import aabb_disable_state, set_aabb_disabled
from reimpl.hybrid_teapot.environment_query import EnvironmentNerfQuery
from reimpl.tests._common import deterministic_rays, load_emitter


def main() -> None:
    emitter = load_emitter(); query = EnvironmentNerfQuery(emitter)
    origins, directions = deterministic_rays(128, emitter.device)
    initial = aabb_disable_state(emitter.model)
    direct_full = emitter.query_radiance(origins, directions, 0)
    full = query.query_full_radiance(origins, directions, 0)
    assert torch.equal(full, direct_full)
    assert aabb_disable_state(emitter.model) == initial
    with set_aabb_disabled(emitter.model, True):
        official_environment = emitter.query_radiance(origins, directions, 0)
    environment = query.query_environment_radiance(origins, directions, 0)
    assert aabb_disable_state(emitter.model) == initial
    assert torch.equal(environment, official_environment)
    assert torch.isfinite(full).all() and torch.isfinite(environment).all()
    assert float((full - environment).abs().max()) > 0
    print("H2 ENVIRONMENT QUERY TESTS: PASS")


if __name__ == "__main__":
    main()
