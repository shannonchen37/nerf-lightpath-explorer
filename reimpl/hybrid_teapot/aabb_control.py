"""Official-compatible AABB density-state control without modifying R1."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator


def aabb_disable_state(model) -> tuple[bool, tuple[bool, ...]]:
    return bool(model.field.disable_inside_aabb), tuple(
        bool(network.disable_inside_aabb) for network in model.proposal_networks
    )


@contextmanager
def set_aabb_disabled(model, disabled: bool) -> Iterator[None]:
    """Set the official field+proposal flag and restore the exact prior state."""
    previous_field, previous_proposals = aabb_disable_state(model)
    model.set_disable_aabb(bool(disabled))
    try:
        yield
    finally:
        model.field.disable_inside_aabb = previous_field
        for network, previous in zip(model.proposal_networks, previous_proposals):
            network.disable_inside_aabb = previous

