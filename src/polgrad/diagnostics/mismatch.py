"""Rollout-vs-trainer logprob mismatch diagnostics.

``old_logprobs`` recomputed by the trainer and ``rollout_logprobs`` reported by the
inference engine describe the same policy in exact arithmetic, but kernels, precision,
and batching make them differ in practice (``docs/conventions.md``). That gap
``Δ_t = trainer - rollout`` silently turns on-policy training off-policy;
``logprob_mismatch`` measures it: moment and quantile statistics of ``Δ``, KL estimates
between the two streams, per-sequence accumulated log-ratios, catastrophic single-token
outliers, and the perplexity ratio implied by the mean gap.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from polgrad._validation import check_2d, check_finite, check_mask, check_same_shape
from polgrad.kl import KLEstimator, kl_estimate

__all__ = ["MismatchReport", "logprob_mismatch"]

_QUANTILE_LEVELS = (0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99)


@dataclass(frozen=True)
class MismatchReport:
    """Statistics of the trainer-vs-rollout logprob gap ``Δ = trainer - rollout``.

    Attributes:
        n_tokens: Total number of response tokens pooled.
        gap_mean: Mean of ``Δ`` over response tokens.
        gap_std: Bessel-corrected std of ``Δ`` (``0.0`` when ``n_tokens == 1``, where
            no spread estimate exists).
        gap_abs_max: ``max |Δ|`` over response tokens.
        gap_quantiles: Quantiles of ``Δ`` at levels (q01, q05, q25, q50, q75, q95, q99).
        kl_k1: k1 estimate of KL(trainer‖rollout): mean of ``Δ``.
        kl_k2: k2 estimate of KL(trainer‖rollout): mean of ``Δ²/2``.
        kl_k3: k3 estimate of KL(trainer‖rollout): mean of ``exp(-Δ) - 1 + Δ``.
        kl_k3_reversed: k3 estimate of KL(rollout‖trainer): mean of ``exp(Δ) - 1 - Δ``.
        seq_log_ratio_mean: Mean over sequences of ``Σ_t m·Δ``.
        seq_log_ratio_std: Bessel-corrected std of ``Σ_t m·Δ`` (``0.0`` when ``B == 1``).
        seq_log_ratio_abs_max: ``max_b |Σ_t m·Δ|``.
        catastrophic_count: Number of response tokens with ``|Δ| > catastrophic_gap``.
        catastrophic_indices: ``[N, 2]`` long tensor of ``(b, t)`` positions of those
            tokens, in row-major order.
        ppl_ratio: ``exp(-masked-mean Δ)`` — the ratio of trainer perplexity to rollout
            perplexity on the sampled tokens.

    References:
        docs/diagnostics/mismatch.md; enforced by
        ``tests/test_diagnostics_mismatch.py::test_golden_case_hand_derived``.
    """

    n_tokens: int
    gap_mean: float
    gap_std: float
    gap_abs_max: float
    gap_quantiles: tuple[float, ...]
    kl_k1: float
    kl_k2: float
    kl_k3: float
    kl_k3_reversed: float
    seq_log_ratio_mean: float
    seq_log_ratio_std: float
    seq_log_ratio_abs_max: float
    catastrophic_count: int
    catastrophic_indices: Tensor
    ppl_ratio: float

    def summary(self) -> str:
        """Return a compact human-readable multi-line description of the report."""
        quantiles = " ".join(
            f"q{int(100 * level):02d}={value:.4g}"
            for level, value in zip(_QUANTILE_LEVELS, self.gap_quantiles, strict=True)
        )
        return (
            f"logprob mismatch (Δ = trainer - rollout) over n_tokens={self.n_tokens}\n"
            f"gap: mean={self.gap_mean:.4g} std={self.gap_std:.4g}"
            f" |max|={self.gap_abs_max:.4g} ppl_ratio={self.ppl_ratio:.4g}\n"
            f"gap quantiles: {quantiles}\n"
            f"KL(trainer‖rollout): k1={self.kl_k1:.4g} k2={self.kl_k2:.4g}"
            f" k3={self.kl_k3:.4g}; KL(rollout‖trainer) k3={self.kl_k3_reversed:.4g}\n"
            f"sequence log-ratio: mean={self.seq_log_ratio_mean:.4g}"
            f" std={self.seq_log_ratio_std:.4g} |max|={self.seq_log_ratio_abs_max:.4g}\n"
            f"catastrophic tokens: {self.catastrophic_count}"
        )


def _std_or_zero(values: Tensor) -> float:
    """Bessel-corrected std; a single observation has no spread estimate, reported as 0.0."""
    if values.numel() < 2:
        return 0.0
    return float(values.std())


def logprob_mismatch(
    trainer_logprobs: Tensor,
    rollout_logprobs: Tensor,
    response_mask: Tensor,
    *,
    catastrophic_gap: float = 5.0,
) -> MismatchReport:
    """Measure the gap between trainer-recomputed and rollout-reported logprobs.

    With ``Δ_t = trainer_logprobs_t - rollout_logprobs_t`` on response tokens, the
    report pools moment and quantile statistics of ``Δ``, the k1/k2/k3 estimates of
    KL(trainer‖rollout) from ``polgrad.kl.kl_estimate`` (and the k3 estimate of the
    reversed direction), per-sequence sums ``Σ_t m·Δ``, tokens with
    ``|Δ| > catastrophic_gap`` (strict), and

        ppl_ratio = exp(-masked-mean Δ) = PPL_trainer / PPL_rollout

    since ``PPL = exp(-masked-mean logprobs)`` on the sampled tokens.

    Args:
        trainer_logprobs: ``[B, T]`` logprobs recomputed by the trainer
            (``old_logprobs`` stream).
        rollout_logprobs: ``[B, T]`` logprobs reported by the inference engine.
        response_mask: ``[B, T]`` bool mask of response tokens.
        catastrophic_gap: Strict threshold on ``|Δ|`` for flagging single-token
            outliers; must be positive.

    Returns:
        A :class:`MismatchReport`; ``catastrophic_indices`` is a ``[N, 2]`` long tensor
        of ``(b, t)`` positions.

    Raises:
        ValueError: If the logprob tensors are not 2-D with identical shapes, the mask
            is invalid (dtype, shape, or a row with zero response tokens), a response
            position holds a non-finite value, or ``catastrophic_gap <= 0``.

    References:
        docs/diagnostics/mismatch.md; enforced by
        ``tests/test_diagnostics_mismatch.py::test_golden_case_hand_derived`` and
        ``tests/test_diagnostics_mismatch.py::test_catastrophic_indices_consistency``.
    """
    check_2d("trainer_logprobs", trainer_logprobs)
    check_2d("rollout_logprobs", rollout_logprobs)
    check_same_shape("trainer_logprobs", trainer_logprobs, "rollout_logprobs", rollout_logprobs)
    check_mask(response_mask, like=trainer_logprobs)
    check_finite("trainer_logprobs (response positions)", trainer_logprobs[response_mask])
    check_finite("rollout_logprobs (response positions)", rollout_logprobs[response_mask])
    if not catastrophic_gap > 0:
        raise ValueError(f"catastrophic_gap must be > 0; got {catastrophic_gap}")

    delta = trainer_logprobs - rollout_logprobs
    gaps = delta[response_mask]
    n_tokens = int(gaps.numel())
    levels = torch.tensor(_QUANTILE_LEVELS, dtype=torch.float64)
    # torch.quantile requires float32/float64 input; the report carries Python floats,
    # so upcasting does not alter any returned tensor dtype.
    quantiles = torch.quantile(gaps.to(torch.float64), levels)

    def _kl(logprobs: Tensor, ref_logprobs: Tensor, kind: KLEstimator) -> float:
        per_token = kl_estimate(logprobs, ref_logprobs, kind, response_mask=response_mask)
        return float(per_token[response_mask].mean())

    seq_log_ratio = delta.masked_fill(~response_mask, 0.0).sum(dim=1)
    catastrophic = response_mask & (delta.abs() > catastrophic_gap)
    catastrophic_indices = torch.nonzero(catastrophic)
    return MismatchReport(
        n_tokens=n_tokens,
        gap_mean=float(gaps.mean()),
        gap_std=_std_or_zero(gaps),
        gap_abs_max=float(gaps.abs().max()),
        gap_quantiles=tuple(float(q) for q in quantiles),
        kl_k1=_kl(trainer_logprobs, rollout_logprobs, KLEstimator.K1),
        kl_k2=_kl(trainer_logprobs, rollout_logprobs, KLEstimator.K2),
        kl_k3=_kl(trainer_logprobs, rollout_logprobs, KLEstimator.K3),
        kl_k3_reversed=_kl(rollout_logprobs, trainer_logprobs, KLEstimator.K3),
        seq_log_ratio_mean=float(seq_log_ratio.mean()),
        seq_log_ratio_std=_std_or_zero(seq_log_ratio),
        seq_log_ratio_abs_max=float(seq_log_ratio.abs().max()),
        catastrophic_count=int(catastrophic_indices.shape[0]),
        catastrophic_indices=catastrophic_indices,
        ppl_ratio=float(torch.exp(-gaps.mean())),
    )
