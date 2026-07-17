"""Conformance harness against framework loss implementations.

Vendored upstream code lives in ``_vendor`` with pinned provenance; deviations that the
test suite demonstrates are registered in ``deviations.DEVIATIONS`` with neutral wording
and a pytest node id per entry.
"""

from __future__ import annotations

from polgrad.conformance.deviations import DEVIATIONS, Deviation
from polgrad.conformance.harness import (
    VENDORED,
    DeviationReport,
    compare_losses,
    deviation_report,
)

__all__ = [
    "DEVIATIONS",
    "VENDORED",
    "Deviation",
    "DeviationReport",
    "compare_losses",
    "deviation_report",
]
