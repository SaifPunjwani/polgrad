"""Policy-entropy diagnostics: sampled-token entropy estimate and collapse detection.

``token_entropy_estimate`` turns the sampled-token log-probabilities into the Monte
Carlo cross-entropy estimate of the policy entropy (unbiased on-policy only).
``entropy_trend`` watches a per-step entropy series for the collapse pathology: a
Theil-Sen slope for drift plus a CUSUM changepoint test whose rejection threshold is
calibrated by permutation, so the false-positive rate is at most ``alpha`` under the
exchangeable null (verified by Monte Carlo in the tests).

This module is one of the RNG exceptions of ``docs/conventions.md``:
``entropy_trend`` consumes an explicit ``torch.Generator`` for its permutations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from polgrad._validation import check_1d, check_2d, check_finite, check_mask

__all__ = ["EntropyReport", "TrendReport", "entropy_trend", "token_entropy_estimate"]


@dataclass(frozen=True)
class EntropyReport:
    """Sampled-token entropy estimate for one batch.

    Attributes:
        n_tokens: Total number of response tokens pooled.
        entropy_estimate: ``-masked-mean(logprobs)`` over all response tokens.
        per_seq_entropy: ``[B]`` per-sequence ``-masked-mean`` of the row's logprobs.

    References:
        docs/diagnostics/entropy.md; enforced by
        ``tests/test_diagnostics_entropy.py::test_token_entropy_golden_case``.
    """

    n_tokens: int
    entropy_estimate: float
    per_seq_entropy: Tensor

    def summary(self) -> str:
        """Return a compact human-readable multi-line description of the report."""
        per_seq_min = float(self.per_seq_entropy.min())
        per_seq_max = float(self.per_seq_entropy.max())
        return (
            f"token entropy estimate: {self.entropy_estimate:.4g} nats"
            f" over n_tokens={self.n_tokens}\n"
            f"per-sequence entropy: min={per_seq_min:.4g} max={per_seq_max:.4g}"
            f" (B={self.per_seq_entropy.numel()})"
        )


@dataclass(frozen=True)
class TrendReport:
    """Entropy-trend diagnostic over the trailing window of a per-step series.

    Attributes:
        slope: Theil-Sen slope (median of pairwise slopes) over the window, in entropy
            units per step.
        changepoint_index: Index into ``entropy_per_step`` of the estimated last
            pre-change step (CUSUM argmax), or ``None`` if the permutation test does
            not reject the no-change null at level ``alpha``.
        cusum_stat: Observed ``max_k |S_k|`` of the centered cumulative sums.
        threshold: Permutation-calibrated rejection threshold: the
            ``floor(alpha·(n_perm+1))``-th largest permuted statistic.
        alpha: Nominal false-positive rate of the changepoint test.
        n_perm: Number of permutations used for calibration.

    References:
        docs/diagnostics/entropy.md; enforced by
        ``tests/test_diagnostics_entropy.py::test_trend_false_positive_rate_calibrated``.
    """

    slope: float
    changepoint_index: int | None
    cusum_stat: float
    threshold: float
    alpha: float
    n_perm: int

    def summary(self) -> str:
        """Return a compact human-readable multi-line description of the report."""
        change = (
            "none detected"
            if self.changepoint_index is None
            else f"detected at step {self.changepoint_index}"
        )
        return (
            f"entropy trend: Theil-Sen slope={self.slope:.4g} per step\n"
            f"CUSUM stat={self.cusum_stat:.4g} vs threshold={self.threshold:.4g}"
            f" (alpha={self.alpha:g}, n_perm={self.n_perm})\n"
            f"changepoint: {change}"
        )


def token_entropy_estimate(logprobs: Tensor, response_mask: Tensor) -> EntropyReport:
    """Monte Carlo cross-entropy estimate of the policy entropy from sampled tokens.

    ``entropy_estimate = -(Σ_{b,t} m·logprobs) / Σ_{b,t} m`` and
    ``per_seq_entropy_b = -(Σ_t m·logprobs) / Σ_t m``. Because
    ``H(π) = E_{y~π}[-log π(y)]``, the sampled-token ``-logprobs`` is an unbiased
    entropy estimate **only when the tokens were sampled from the same policy that is
    scored** (on-policy); off-policy it estimates the cross-entropy
    ``H(sampling policy, π)`` instead.

    Args:
        logprobs: ``[B, T]`` sampled-token log-probabilities ``log π(y_t | y_<t, x)``.
        response_mask: ``[B, T]`` bool mask of response tokens.

    Returns:
        An :class:`EntropyReport`; ``per_seq_entropy`` has the input dtype and is
        detached.

    Raises:
        ValueError: If ``logprobs`` is not 2-D, the mask is invalid (dtype, shape, or a
            row with zero response tokens), or a response position holds a non-finite
            value.

    References:
        docs/diagnostics/entropy.md; enforced by
        ``tests/test_diagnostics_entropy.py::test_token_entropy_golden_case`` and
        ``tests/test_diagnostics_entropy.py::test_pooled_entropy_is_length_weighted_mean_of_per_seq``.
    """
    check_2d("logprobs", logprobs)
    check_mask(response_mask, like=logprobs)
    pooled = logprobs[response_mask]
    check_finite("logprobs (response positions)", pooled)
    lengths = response_mask.sum(dim=1).to(logprobs.dtype)
    row_sums = logprobs.masked_fill(~response_mask, 0.0).sum(dim=1)
    # Diagnostic output only; the entropy trace must never feed a gradient path.
    per_seq_entropy = (-row_sums / lengths).detach()
    return EntropyReport(
        n_tokens=int(pooled.numel()),
        entropy_estimate=float(-pooled.mean()),
        per_seq_entropy=per_seq_entropy,
    )


def _cusum_stat(centered: Tensor) -> Tensor:
    """``max_k |S_k|`` of cumulative sums along the last dim, for mean-centered input."""
    return centered.cumsum(dim=-1).abs().amax(dim=-1)


def entropy_trend(
    entropy_per_step: Tensor,
    *,
    window: int,
    n_perm: int = 999,
    alpha: float = 0.05,
    generator: torch.Generator,
) -> TrendReport:
    """Theil-Sen slope and permutation-calibrated CUSUM changepoint over a trailing window.

    Over the last ``window`` values ``x_0, …, x_{w-1}`` of ``entropy_per_step``:

    - slope = median{ (x_j - x_i) / (j - i) : i < j } (Theil-Sen);
    - CUSUM statistic ``max_k |S_k|`` with ``S_k = Σ_{i≤k} (x_i - x̄)``;
    - threshold = the ``m``-th largest of ``n_perm`` permuted statistics, where
      ``m = floor(alpha·(n_perm + 1))``. A changepoint is reported iff the observed
      statistic exceeds the threshold, which bounds the false-positive rate by
      ``m / (n_perm + 1) ≤ alpha`` under the exchangeable null
      (docs/diagnostics/entropy.md derives the bound).

    Args:
        entropy_per_step: ``[S]`` per-training-step entropy estimates.
        window: Trailing window length analyzed; ``2 ≤ window ≤ S``.
        n_perm: Number of permutations for threshold calibration; ``≥ 1``.
        alpha: Nominal false-positive rate, in ``(0, 1)``; must satisfy
            ``floor(alpha·(n_perm + 1)) ≥ 1`` so the test is able to reject.
        generator: Explicit RNG for the permutations (RNG exception of
            ``docs/conventions.md``).

    Returns:
        A :class:`TrendReport`; ``changepoint_index`` is a global index into
        ``entropy_per_step`` (the estimated last pre-change step) or ``None``.
        Statistics are computed internally in float64; the report carries Python
        floats only.

    Raises:
        ValueError: If ``entropy_per_step`` is not 1-D or contains non-finite values,
            ``window`` is out of range, ``n_perm < 1``, ``alpha`` is outside ``(0, 1)``,
            or ``alpha`` is too small for ``n_perm`` to allow any rejection.

    References:
        docs/diagnostics/entropy.md; enforced by
        ``tests/test_diagnostics_entropy.py::test_trend_false_positive_rate_calibrated``
        and
        ``tests/test_diagnostics_entropy.py::test_cusum_detects_mean_shift_and_localizes_changepoint``.
    """
    check_1d("entropy_per_step", entropy_per_step)
    check_finite("entropy_per_step", entropy_per_step)
    n_steps = int(entropy_per_step.numel())
    if window < 2:
        raise ValueError(f"window must be >= 2; got window={window}")
    if window > n_steps:
        raise ValueError(f"window must be <= len(entropy_per_step)={n_steps}; got window={window}")
    if n_perm < 1:
        raise ValueError(f"n_perm must be >= 1; got n_perm={n_perm}")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1); got alpha={alpha}")
    m = math.floor(alpha * (n_perm + 1))
    if m < 1:
        raise ValueError(
            f"alpha={alpha} is too small for n_perm={n_perm}: the permutation test needs"
            f" floor(alpha·(n_perm + 1)) >= 1 to be able to reject"
        )
    segment = entropy_per_step[-window:].to(torch.float64)

    pairs = torch.triu_indices(window, window, offset=1)
    steps = torch.arange(window, dtype=torch.float64)
    pair_slopes = (segment[pairs[1]] - segment[pairs[0]]) / (steps[pairs[1]] - steps[pairs[0]])
    slope = float(torch.quantile(pair_slopes, 0.5))

    cusum = (segment - segment.mean()).cumsum(dim=0)
    stat = float(cusum.abs().max())

    uniforms = torch.rand(n_perm, window, generator=generator, dtype=torch.float64)
    permuted = segment[uniforms.argsort(dim=1)]
    perm_stats = _cusum_stat(permuted - permuted.mean(dim=1, keepdim=True))
    threshold = float(perm_stats.sort(descending=True).values[m - 1])

    changepoint_index: int | None = None
    if stat > threshold:
        changepoint_index = n_steps - window + int(cusum.abs().argmax())
    return TrendReport(
        slope=slope,
        changepoint_index=changepoint_index,
        cusum_stat=stat,
        threshold=threshold,
        alpha=alpha,
        n_perm=n_perm,
    )
