"""Monte Carlo estimation helpers for polgrad's verification suites.

polgrad's MC certification tests (KL estimator bias, ESS calibration, bandit
gradients) compare sample means against closed-form targets. This module owns
the two pieces those tests share: a CLT-derived tolerance for such comparisons
and a batched, generator-threaded sample-mean estimator. Per
``docs/conventions.md``, randomness enters only through an explicit
``torch.Generator``.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
from torch import Tensor

from polgrad._validation import check_1d, check_finite

__all__ = ["clt_tolerance", "mc_mean"]


def clt_tolerance(sample_std: float, n: int, *, z: float = 4.0) -> float:
    """Return a CLT-based tolerance for comparing an ``n``-sample mean to its target.

    ``tol = z · sample_std / √n``. By the central limit theorem the sample mean of
    ``n`` iid draws with standard deviation ``s`` is approximately
    ``Normal(μ, s²/n)``, so ``|mean - μ| > tol`` occurs with probability
    ``≈ 2·Φ(-z)``; for the default ``z = 4.0`` that is ``≈ 6.3e-5``.

    Args:
        sample_std: Standard deviation of one sample (exact or estimated); must be
            finite and non-negative.
        n: Number of iid samples averaged; must be ``>= 1``.
        z: Number of standard errors of slack; must be finite and positive.

    Returns:
        The tolerance ``z · sample_std / √n`` as a Python float.

    Raises:
        ValueError: If ``sample_std`` is negative or non-finite, ``n < 1``, or
            ``z`` is non-positive or non-finite.

    References:
        docs/conventions.md (error rules);
        tests/test_verify_mc.py::test_clt_tolerance_golden_arithmetic.
    """
    if not math.isfinite(sample_std) or sample_std < 0.0:
        raise ValueError(f"sample_std must be finite and non-negative; got {sample_std}")
    if n < 1:
        raise ValueError(f"n must be >= 1; got {n}")
    if not math.isfinite(z) or z <= 0.0:
        raise ValueError(f"z must be finite and positive; got {z}")
    return z * sample_std / math.sqrt(n)


def mc_mean(
    fn: Callable[[int, torch.Generator], Tensor],
    n: int,
    generator: torch.Generator,
    *,
    batch: int = 4096,
) -> tuple[float, float]:
    """Estimate ``E[X]`` from ``n`` draws of ``fn``, returning ``(mean, std_err)``.

    ``mean = (1/n) Σᵢ xᵢ`` and ``std_err = s / √n`` with the Bessel-corrected
    sample variance ``s² = (1/(n-1)) Σᵢ (xᵢ - mean)²``. Samples are requested in
    chunks of at most ``batch`` via ``fn(k, generator)`` (the same generator is
    threaded through every call, so a seeded generator makes the result
    deterministic). Chunk statistics are combined exactly with the parallel
    update ``M2 = M2_a + M2_b + δ²·n_a·n_b/(n_a+n_b)`` where ``δ`` is the
    difference of chunk means, accumulated in float64.

    Args:
        fn: Sampler; ``fn(k, generator)`` must return a 1-D tensor of shape
            ``[k]`` of finite values.
        n: Total number of samples; must be ``>= 2`` so the standard error is
            defined.
        generator: Explicit RNG threaded through every ``fn`` call.
        batch: Maximum chunk size per ``fn`` call; must be ``>= 1``.

    Returns:
        ``(mean, std_err)`` as Python floats.

    Raises:
        ValueError: If ``n < 2`` or ``batch < 1``, or if any ``fn(k, generator)``
            output is not 1-D of shape ``[k]`` or contains non-finite values.

    References:
        docs/conventions.md (determinism rules);
        tests/test_verify_mc.py::test_mc_mean_matches_direct_mean_and_std_err_on_fixed_samples.
    """
    if n < 2:
        raise ValueError(f"n must be >= 2 so the standard error is defined; got {n}")
    if batch < 1:
        raise ValueError(f"batch must be >= 1; got {batch}")

    count = 0
    mean = 0.0
    m2 = 0.0
    while count < n:
        k = min(batch, n - count)
        chunk = fn(k, generator)
        check_1d("fn(k, generator)", chunk)
        if chunk.shape[0] != k:
            raise ValueError(
                f"fn(k, generator) must return shape [k] = [{k}]; got {tuple(chunk.shape)}"
            )
        check_finite("fn(k, generator)", chunk)
        chunk64 = chunk.to(torch.float64)
        chunk_mean = float(chunk64.mean())
        chunk_m2 = float(((chunk64 - chunk_mean) ** 2).sum())
        new_count = count + k
        delta = chunk_mean - mean
        m2 += chunk_m2 + delta * delta * count * k / new_count
        mean += delta * k / new_count
        count = new_count

    # Rounding in the parallel update can leave M2 a hair below zero for
    # constant samples; the true value is never negative.
    variance = max(m2, 0.0) / (n - 1)
    std_err = math.sqrt(variance / n)
    return mean, std_err
