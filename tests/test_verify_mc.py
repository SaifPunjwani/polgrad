"""Tests for polgrad.verify.mc (docs/derivations/goldens.md).

Covers the clt_tolerance arithmetic golden, mc_mean recovering known means within
clt_tolerance on seeded generators, exact chunk-combination algebra, determinism,
and input validation.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st
from torch import Tensor

from polgrad.verify.mc import clt_tolerance, mc_mean

Sampler = Callable[[int, torch.Generator], Tensor]


def _stream_from(base: Tensor) -> tuple[list[int], Sampler]:
    """Serve consecutive slices of a fixed tensor, recording each requested chunk size."""
    calls: list[int] = []
    cursor = {"i": 0}

    def fn(k: int, _gen: torch.Generator) -> Tensor:
        calls.append(k)
        i = cursor["i"]
        cursor["i"] = i + k
        return base[i : i + k]

    return calls, fn


def test_clt_tolerance_golden_arithmetic() -> None:
    """clt_tolerance(s, n, z) equals z·s/√n on hand-computed cases.

    4.0·2.0/√100 = 8/10 = 0.8; 2.0·3.0/√9 = 6/3 = 2.0; z·0/√n = 0.
    """
    assert clt_tolerance(2.0, 100) == 0.8
    assert clt_tolerance(3.0, 9, z=2.0) == 2.0
    assert clt_tolerance(0.0, 5) == 0.0


def test_clt_tolerance_rejects_invalid_arguments() -> None:
    """clt_tolerance raises ValueError naming the offending argument."""
    with pytest.raises(ValueError, match="n must be >= 1"):
        clt_tolerance(1.0, 0)
    with pytest.raises(ValueError, match="sample_std"):
        clt_tolerance(-1.0, 10)
    with pytest.raises(ValueError, match="sample_std"):
        clt_tolerance(math.inf, 10)
    with pytest.raises(ValueError, match="z must be"):
        clt_tolerance(1.0, 10, z=0.0)
    with pytest.raises(ValueError, match="z must be"):
        clt_tolerance(1.0, 10, z=math.nan)


def test_mc_mean_matches_direct_mean_and_std_err_on_fixed_samples(gen: torch.Generator) -> None:
    """mc_mean reproduces the hand-derived mean and standard error of [1, 2, 3, 4].

    mean = 10/4 = 2.5; deviations (-1.5, -0.5, 0.5, 1.5) square-sum to 5;
    s² = 5/3 (Bessel); std_err = √(5/3)/√4 = √(5/12). batch=3 forces the
    parallel combine across chunks [1,2,3] and [4]:
    M2 = 2 + 0 + (4-2)²·3·1/4 = 5, matching the single-pass value.
    """
    base = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
    _, fn = _stream_from(base)
    mean, std_err = mc_mean(fn, 4, gen, batch=3)
    assert mean == 2.5
    assert math.isclose(std_err, math.sqrt(5.0 / 12.0), rel_tol=1e-12)
    assert math.isclose(std_err, float(base.std()) / math.sqrt(4), rel_tol=1e-12)


def test_mc_mean_batch_size_invariance_on_fixed_stream(gen: torch.Generator) -> None:
    """mc_mean gives the same (mean, std_err) for any batch size on a fixed sample stream."""
    base = torch.randn(1000, generator=gen, dtype=torch.float64)
    results = []
    for batch in (1000, 7, 1):
        _, fn = _stream_from(base)
        results.append(mc_mean(fn, 1000, gen, batch=batch))
    for mean, std_err in results:
        assert math.isclose(mean, float(base.mean()), rel_tol=0, abs_tol=1e-12)
        assert math.isclose(std_err, float(base.std()) / math.sqrt(1000), rel_tol=0, abs_tol=1e-12)


def test_mc_mean_requests_chunks_of_at_most_batch(gen: torch.Generator) -> None:
    """mc_mean asks fn for consecutive chunks of size <= batch summing to n."""
    calls, fn = _stream_from(torch.zeros(8000, dtype=torch.float64))
    mc_mean(fn, 8000, gen, batch=3000)
    assert calls == [3000, 3000, 2000]


def test_mc_mean_recovers_standard_normal_mean_within_clt_tolerance(
    gen: torch.Generator,
) -> None:
    """mc_mean recovers E[Z] = 0 for Z ~ Normal(0, 1) within clt_tolerance(1, n), seeded."""
    n = 65536

    def fn(k: int, g: torch.Generator) -> Tensor:
        return torch.randn(k, generator=g, dtype=torch.float64)

    mean, std_err = mc_mean(fn, n, gen)
    assert abs(mean) <= clt_tolerance(1.0, n)
    # The sample std of n draws deviates from 1 by ~1/sqrt(2n) ≈ 0.003; 5% is loose.
    assert abs(std_err * math.sqrt(n) - 1.0) < 0.05


def test_mc_mean_recovers_shifted_scaled_uniform_mean(gen: torch.Generator) -> None:
    """mc_mean recovers E[2 + 3U] = 3.5 for U ~ Uniform(0, 1) within clt_tolerance, seeded."""
    n = 40960
    true_std = 3.0 / math.sqrt(12.0)

    def fn(k: int, g: torch.Generator) -> Tensor:
        return 2.0 + 3.0 * torch.rand(k, generator=g, dtype=torch.float64)

    mean, _ = mc_mean(fn, n, gen)
    assert abs(mean - 3.5) <= clt_tolerance(true_std, n)


def test_mc_mean_recovers_bernoulli_mean(gen: torch.Generator) -> None:
    """mc_mean recovers E[X] = 0.25 for X ~ Bernoulli(0.25) within clt_tolerance, seeded."""
    n = 40960
    true_std = math.sqrt(0.25 * 0.75)

    def fn(k: int, g: torch.Generator) -> Tensor:
        return (torch.rand(k, generator=g, dtype=torch.float64) < 0.25).to(torch.float64)

    mean, _ = mc_mean(fn, n, gen)
    assert abs(mean - 0.25) <= clt_tolerance(true_std, n)


@given(
    mu=st.floats(-5.0, 5.0, allow_nan=False, allow_infinity=False, width=32),
    sigma=st.floats(0.125, 3.0, allow_nan=False, allow_infinity=False, width=32),
    seed=st.integers(0, 2**31 - 1),
)
def test_mc_mean_recovers_gaussian_mean_within_clt_tolerance(
    mu: float, sigma: float, seed: int
) -> None:
    """mc_mean recovers μ of μ + sigma·Z within clt_tolerance(sigma, n) on seeded generators."""
    n = 16384
    g = torch.Generator().manual_seed(seed)

    def fn(k: int, gen_: torch.Generator) -> Tensor:
        return mu + sigma * torch.randn(k, generator=gen_, dtype=torch.float64)

    mean, std_err = mc_mean(fn, n, g)
    assert abs(mean - mu) <= clt_tolerance(sigma, n)
    assert abs(std_err - sigma / math.sqrt(n)) <= 0.1 * sigma / math.sqrt(n)


def test_mc_mean_is_deterministic_given_seed() -> None:
    """mc_mean returns bitwise-identical results for identically seeded generators."""

    def fn(k: int, g: torch.Generator) -> Tensor:
        return torch.randn(k, generator=g, dtype=torch.float64)

    first = mc_mean(fn, 8192, torch.Generator().manual_seed(123), batch=1024)
    second = mc_mean(fn, 8192, torch.Generator().manual_seed(123), batch=1024)
    assert first == second


def test_mc_mean_rejects_invalid_n_and_batch(gen: torch.Generator) -> None:
    """mc_mean raises ValueError for n < 2 (standard error undefined) and batch < 1."""

    def fn(k: int, g: torch.Generator) -> Tensor:
        return torch.zeros(k, dtype=torch.float64)

    with pytest.raises(ValueError, match="n must be >= 2"):
        mc_mean(fn, 1, gen)
    with pytest.raises(ValueError, match="n must be >= 2"):
        mc_mean(fn, 0, gen)
    with pytest.raises(ValueError, match="batch must be >= 1"):
        mc_mean(fn, 4, gen, batch=0)


def test_mc_mean_rejects_bad_fn_output(gen: torch.Generator) -> None:
    """mc_mean raises ValueError when fn output is not 1-D [k] or is non-finite."""

    def two_d(k: int, g: torch.Generator) -> Tensor:
        return torch.zeros((k, 2), dtype=torch.float64)

    def wrong_len(k: int, g: torch.Generator) -> Tensor:
        return torch.zeros(k + 1, dtype=torch.float64)

    def non_finite(k: int, g: torch.Generator) -> Tensor:
        return torch.full((k,), math.inf, dtype=torch.float64)

    with pytest.raises(ValueError, match="must be 1-D"):
        mc_mean(two_d, 4, gen)
    with pytest.raises(ValueError, match=r"must return shape \[k\]"):
        mc_mean(wrong_len, 4, gen)
    with pytest.raises(ValueError, match="non-finite"):
        mc_mean(non_finite, 4, gen)
