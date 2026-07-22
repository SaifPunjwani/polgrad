"""Pytest plugin exposing polgrad's conformance-testing helpers as fixtures.

Registered under the ``pytest11`` entry point ``polgrad`` (pyproject.toml), so any
environment with both polgrad and pytest installed gets the ``polgrad_batches`` fixture
without configuration. The pytest import is guarded and nothing in ``polgrad``'s import
graph reaches this module, so importing polgrad never requires pytest.

:func:`polgrad.testing.assert_conforms` is re-exported for convenience, but the
documented import path is ``from polgrad.testing import assert_conforms``.

References:
    tests/test_testing_api.py::test_polgrad_batches_fixture_available_in_fresh_run.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Protocol

import torch
from torch import Tensor

from polgrad.testing import assert_conforms, random_batches

__all__ = [
    "BatchFactory",
    "assert_conforms",
    "polgrad_batches",
]


class BatchFactory(Protocol):
    """Calling signature of :func:`polgrad.testing.random_batches`."""

    def __call__(
        self,
        n_cases: int,
        shapes: Sequence[tuple[int, int]],
        *,
        seed: int,
        max_gap: float = 2.0,
        dtype: torch.dtype = torch.float64,
    ) -> Iterator[dict[str, Tensor]]: ...


try:
    import pytest

    _HAVE_PYTEST = True
except ModuleNotFoundError:  # pragma: no cover - pytest loads this module in tests
    _HAVE_PYTEST = False

if _HAVE_PYTEST:

    @pytest.fixture
    def polgrad_batches() -> BatchFactory:
        """Deterministic batch factory: :func:`polgrad.testing.random_batches`.

        Takes no parameters; call the returned factory as
        ``polgrad_batches(n_cases, shapes, seed=...)`` to iterate keyword-tensor
        batches in polgrad's ``[B, T]`` right-padded convention.

        References:
            tests/test_testing_api.py::test_polgrad_batches_fixture_available_in_fresh_run.
        """
        return random_batches
