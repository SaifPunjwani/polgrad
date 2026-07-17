from __future__ import annotations

import os

import pytest
import torch
from hypothesis import HealthCheck, settings

settings.register_profile(
    "ci",
    max_examples=100,
    deadline=None,
    derandomize=True,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.register_profile(
    "dev",
    max_examples=25,
    deadline=None,
    derandomize=True,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "dev"))


@pytest.fixture
def gen() -> torch.Generator:
    """Seeded generator for the RNG-taking functions (docs/conventions.md, determinism)."""
    return torch.Generator().manual_seed(0)
