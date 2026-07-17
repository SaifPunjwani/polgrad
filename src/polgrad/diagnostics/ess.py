"""Effective sample size (ESS) diagnostics for importance-weighted policy updates.

When the trainer policy drifts away from the policy that generated the rollouts, the
importance weights ``w = exp(logprobs_new - logprobs_old)`` become uneven and the update
behaves as if it had fewer samples than the batch contains. The standard
importance-sampling effective sample size

    ESS = (Σᵢ wᵢ)² / Σᵢ wᵢ²

quantifies that loss: ``ESS/n == 1`` iff all weights are equal, and ``ESS/n → 0`` as one
weight dominates. Null calibration (derived in ``docs/diagnostics/ess.md``): identical
policies give ``ess_ratio == 1`` exactly, and iid Normal(0, σ²) log-weight perturbations
give ``E[ESS/n] → exp(-σ²)`` as ``n`` grows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

from polgrad._validation import check_logprob_streams, std_or_zero

__all__ = ["ESSReport", "importance_ess", "sliding_ess"]


@dataclass(frozen=True)
class ESSReport:
    """Effective-sample-size diagnostic for one batch of importance weights.

    Attributes:
        level: ``"sequence"`` or ``"token"`` — which weights were pooled.
        n: Number of weights (``B`` at sequence level, total response tokens at token
            level).
        ess: ``(Σw)² / Σw²``.
        ess_ratio: ``ess / n``, in ``(0, 1]``; ``1`` iff all weights are equal.
        log_weight_mean: Mean of the log-weights.
        log_weight_std: Bessel-corrected standard deviation of the log-weights
            (``0.0`` when ``n == 1``, where no spread estimate exists).
        log_weight_min: Smallest log-weight.
        log_weight_max: Largest log-weight.

    References:
        docs/diagnostics/ess.md; enforced by
        ``tests/test_diagnostics_ess.py::test_report_log_weight_stats_match_recomputation``.
    """

    level: str
    n: int
    ess: float
    ess_ratio: float
    log_weight_mean: float
    log_weight_std: float
    log_weight_min: float
    log_weight_max: float

    def summary(self) -> str:
        """Return a compact human-readable multi-line description of the report."""
        return (
            f"importance ESS ({self.level} level): ESS={self.ess:.4g} of n={self.n}"
            f" (ESS/n={self.ess_ratio:.4g})\n"
            f"log-weights: mean={self.log_weight_mean:.4g} std={self.log_weight_std:.4g}"
            f" min={self.log_weight_min:.4g} max={self.log_weight_max:.4g}"
        )


def _sequence_log_weights(
    logprobs_new: Tensor, logprobs_old: Tensor, response_mask: Tensor
) -> Tensor:
    """Per-sequence log importance weights ``Σ_t m·(logprobs_new - logprobs_old)``, shape [B]."""
    gaps = (logprobs_new - logprobs_old).masked_fill(~response_mask, 0.0)
    return gaps.sum(dim=1)


def _ess_from_log_weights(log_weights: Tensor) -> Tensor:
    """ESS = (Σw)²/Σw² for ``w = exp(log_weights)`` along the last dim.

    ESS is invariant under positive rescaling of the weights (both numerator and
    denominator scale by c²), so the exponent is shifted by its maximum to avoid
    overflow; see docs/diagnostics/ess.md for the one-line algebra.
    """
    shifted = torch.exp(log_weights - log_weights.max(dim=-1, keepdim=True).values)
    totals = shifted.sum(dim=-1)
    return totals * totals / (shifted * shifted).sum(dim=-1)


def _validate_streams(logprobs_new: Tensor, logprobs_old: Tensor, response_mask: Tensor) -> None:
    check_logprob_streams(
        "logprobs_new",
        logprobs_new,
        "logprobs_old",
        logprobs_old,
        response_mask,
        check_b_2d=True,
        finite_suffix=" (response positions)",
    )


def importance_ess(
    logprobs_new: Tensor,
    logprobs_old: Tensor,
    response_mask: Tensor,
    *,
    level: Literal["token", "sequence"] = "sequence",
) -> ESSReport:
    """Effective sample size of the importance weights between two policies.

    ``ESS = (Σᵢ wᵢ)² / Σᵢ wᵢ²`` and ``ess_ratio = ESS / n``, where at sequence level

        wᵢ = exp(Σ_t m_{i,t}·(logprobs_new_{i,t} - logprobs_old_{i,t}))

    is the exact (UNnormalized) sequence importance weight — NOT the length-normalized
    ``RatioKind.SEQUENCE`` exponent used by GSPO, which divides the summed gap by the
    sequence length before exponentiating. At token level the per-token ratios
    ``exp(logprobs_new - logprobs_old)`` are pooled over all response tokens of the
    batch, so ``n = Σ_{b,t} m``.

    Args:
        logprobs_new: ``[B, T]`` log-probabilities under the numerator policy.
        logprobs_old: ``[B, T]`` log-probabilities under the denominator (sampling)
            policy.
        response_mask: ``[B, T]`` bool mask of response tokens.
        level: ``"sequence"`` (default) pools per-sequence weights (``n = B``);
            ``"token"`` pools per-token ratios.

    Returns:
        An :class:`ESSReport`; identical policies give ``ess_ratio == 1.0`` exactly.

    Raises:
        ValueError: If the logprob tensors are not 2-D with identical shapes, the mask
            is invalid (dtype, shape, or a row with zero response tokens), a response
            position holds a non-finite value, or ``level`` is not ``"token"`` /
            ``"sequence"``.

    References:
        docs/diagnostics/ess.md; enforced by
        ``tests/test_diagnostics_ess.py::test_identical_policies_ess_ratio_is_exactly_one``,
        ``tests/test_diagnostics_ess.py::test_mc_calibration_mean_ess_ratio_approaches_exp_neg_var``.
    """
    _validate_streams(logprobs_new, logprobs_old, response_mask)
    if level == "sequence":
        log_weights = _sequence_log_weights(logprobs_new, logprobs_old, response_mask)
    elif level == "token":
        log_weights = (logprobs_new - logprobs_old)[response_mask]
    else:
        raise ValueError(f"level must be 'token' or 'sequence'; got {level!r}")
    n = int(log_weights.numel())
    ess = float(_ess_from_log_weights(log_weights))
    return ESSReport(
        level=level,
        n=n,
        ess=ess,
        ess_ratio=ess / n,
        log_weight_mean=float(log_weights.mean()),
        log_weight_std=std_or_zero(log_weights),
        log_weight_min=float(log_weights.min()),
        log_weight_max=float(log_weights.max()),
    )


def sliding_ess(
    logprobs_new: Tensor,
    logprobs_old: Tensor,
    response_mask: Tensor,
    *,
    window: int,
    step: int = 1,
) -> Tensor:
    """Sequence-level ``ESS/window`` over sliding windows in batch order.

    Batch order is rollout-chronological, so this traces how the effective sample size
    decays across a rollout buffer. Window ``k`` covers rows
    ``[k·step, k·step + window)`` and reports

        ESS_k / window,  ESS_k = (Σ_{i∈window} wᵢ)² / Σ_{i∈window} wᵢ²

    with the unnormalized sequence weights ``wᵢ`` of :func:`importance_ess`.

    Args:
        logprobs_new: ``[B, T]`` log-probabilities under the numerator policy.
        logprobs_old: ``[B, T]`` log-probabilities under the denominator policy.
        response_mask: ``[B, T]`` bool mask of response tokens.
        window: Number of consecutive sequences per window; ``2 ≤ window ≤ B``.
        step: Stride between window starts; ``≥ 1``.

    Returns:
        Tensor of shape ``[(B - window) // step + 1]`` (input dtype) holding
        ``ESS/window`` per window, each in ``(0, 1]``.

    Raises:
        ValueError: If ``window < 2``, ``window > B``, ``step < 1``, or the tensor /
            mask validation of :func:`importance_ess` fails.

    References:
        docs/diagnostics/ess.md; enforced by
        ``tests/test_diagnostics_ess.py::test_sliding_ess_golden_case`` and
        ``tests/test_diagnostics_ess.py::test_sliding_ess_window_and_step_shapes``.
    """
    _validate_streams(logprobs_new, logprobs_old, response_mask)
    batch = int(logprobs_new.shape[0])
    if window < 2:
        raise ValueError(f"window must be >= 2; got window={window}")
    if window > batch:
        raise ValueError(f"window must be <= batch size B={batch}; got window={window}")
    if step < 1:
        raise ValueError(f"step must be >= 1; got step={step}")
    log_weights = _sequence_log_weights(logprobs_new, logprobs_old, response_mask)
    windows = log_weights.unfold(0, window, step)
    # Diagnostic output only; the ESS trace must never feed a gradient path.
    return (_ess_from_log_weights(windows) / window).detach()
