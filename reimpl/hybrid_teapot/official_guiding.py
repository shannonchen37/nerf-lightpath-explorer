"""Backward-compatible import path for reference-parity vMF guiding.

New code should import :class:`ReferenceVmfMixture` from
``reimpl.hybrid_teapot.reference_guiding``. The historical module and class
alias remain available so existing experiment commands do not break.
"""

from .reference_guiding import OfficialVmfMixture, ReferenceVmfMixture, VmfSamples

__all__ = ["ReferenceVmfMixture", "OfficialVmfMixture", "VmfSamples"]
