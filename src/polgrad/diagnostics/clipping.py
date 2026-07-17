"""PPO clip-band diagnostics: quadrant clip fractions and the killed-gradient census.

The PG_CLIP surrogate ``-min(r_t A_t, clip(r_t, 1-ε_lo, 1+ε_hi) A_t)`` silences the
per-token gradient in some (advantage sign, ratio) regions and lets it flow in others.
The two are not the same partition: crossing the band above with ``A_t < 0`` clips
nothing — the gradient flows with unbounded magnitude ``r_t·|A_t|`` unless dual-clip
caps it at ``r_t > c`` (the known PPO pathology). :func:`clip_report` measures both:
how often each side of the band is crossed within each advantage sign, and exactly
where the PG_CLIP gradient is zero. The zero-gradient condition is derived branch by
branch in ``docs/diagnostics/clipping.md`` and verified against autograd of
:func:`polgrad.losses.policy_loss`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from torch import Tensor

from polgrad._validation import check_2d, check_finite, check_mask, check_same_shape
from polgrad.losses import ClipConfig

__all__ = ["ClipReport", "clip_report"]


@dataclass(frozen=True)
class ClipReport:
    """Clip-band census of a PG_CLIP batch.

    Every fraction has the same denominator ``N = Σ_{b,t} m`` (total response tokens),
    so the four quadrant fractions are directly comparable and never divide by zero
    when one advantage sign is absent. ``clipped high`` means ``r_t > 1 + ε_hi`` and
    ``clipped low`` means ``r_t < 1 - ε_lo``; the fractions count band crossings, not
    killed gradients — for ``A_t < 0`` a high crossing clips nothing unless dual-clip
    caps it (docs/diagnostics/clipping.md).

    Attributes:
        eps_low: Lower clip width; the band floor is ``1 - eps_low``.
        eps_high: Upper clip width; the band ceiling is ``1 + eps_high``.
        frac_pos_adv_clipped_high: Fraction of response tokens with ``A_t > 0`` and
            ``r_t > 1 + ε_hi`` (gradient killed).
        frac_pos_adv_clipped_low: Fraction with ``A_t > 0`` and ``r_t < 1 - ε_lo``
            (gradient flows; PPO's ``min`` is one-sided pessimism).
        frac_neg_adv_clipped_high: Fraction with ``A_t < 0`` and ``r_t > 1 + ε_hi``
            (gradient flows unbounded unless dual-clip caps it at ``r_t > c``).
        frac_neg_adv_clipped_low: Fraction with ``A_t < 0`` and ``r_t < 1 - ε_lo``
            (gradient killed).
        gradient_killed_mask: ``[B, T]`` bool, ``True`` exactly where the PG_CLIP
            per-token gradient w.r.t. ``logprobs`` is ``0``: ``(A_t > 0 and r_t >
            1+ε_hi) or (A_t < 0 and r_t < 1-ε_lo) or (dual-clip: A_t < 0 and r_t > c)
            or A_t == 0``; ``False`` at masked positions.
        gradient_killed_frac: ``Σ gradient_killed_mask / N``.

    References:
        docs/diagnostics/clipping.md; enforced by
        ``tests/test_diagnostics_clipping.py::test_quadrant_fractions_constructed_case_dual_clip``
        and
        ``tests/test_diagnostics_clipping.py::test_gradient_killed_matches_policy_loss_autograd``.
    """

    eps_low: float
    eps_high: float
    frac_pos_adv_clipped_high: float
    frac_pos_adv_clipped_low: float
    frac_neg_adv_clipped_high: float
    frac_neg_adv_clipped_low: float
    gradient_killed_mask: Tensor
    gradient_killed_frac: float

    def summary(self) -> str:
        """Return a compact human-readable multi-line description of the report."""
        return (
            f"PG_CLIP band [1 - eps_low, 1 + eps_high] ="
            f" [{1.0 - self.eps_low:.4g}, {1.0 + self.eps_high:.4g}];"
            f" fractions are of all response tokens\n"
            f"A > 0: clipped high {self.frac_pos_adv_clipped_high:.4g} (gradient killed),"
            f" clipped low {self.frac_pos_adv_clipped_low:.4g} (gradient flows)\n"
            f"A < 0: clipped low {self.frac_neg_adv_clipped_low:.4g} (gradient killed),"
            f" clipped high {self.frac_neg_adv_clipped_high:.4g}"
            f" (gradient flows unless dual-clip caps it)\n"
            f"gradient killed on {self.gradient_killed_frac:.4g} of response tokens"
        )


def _validate_clip(clip: ClipConfig) -> tuple[float, float]:
    """Enforce the clip_report requirements on ``ClipConfig`` (contract section 4.6)."""
    if clip.eps_low is None or clip.eps_high is None:
        raise ValueError(
            f"clip_report requires a ClipConfig with eps_low and eps_high non-None; "
            f"got clip={clip!r}"
        )
    for name, eps in (("eps_low", clip.eps_low), ("eps_high", clip.eps_high)):
        if not (math.isfinite(eps) and eps > 0.0):
            raise ValueError(f"ClipConfig.{name} must be a positive finite float; got {eps}")
    if clip.ratio_cap is not None and not (math.isfinite(clip.ratio_cap) and clip.ratio_cap > 1.0):
        raise ValueError(
            f"ClipConfig.ratio_cap must be a finite float > 1 (dual-clip); got {clip.ratio_cap}"
        )
    return clip.eps_low, clip.eps_high


def _broadcast_advantages(advantages: Tensor, like: Tensor, response_mask: Tensor) -> Tensor:
    """Validate ``[B]`` or ``[B, T]`` advantages and return a ``[B, T]`` view."""
    if advantages.dim() == 1:
        if advantages.shape[0] != like.shape[0]:
            raise ValueError(
                f"advantages [B] must have B = {like.shape[0]} rows; "
                f"got shape {tuple(advantages.shape)}"
            )
        check_finite("advantages", advantages)
        return advantages.unsqueeze(1).expand_as(like)
    if advantages.dim() == 2:
        check_same_shape("advantages", advantages, "ratio", like)
        check_finite("advantages (response positions)", advantages[response_mask])
        return advantages
    raise ValueError(f"advantages must be [B] or [B, T]; got shape {tuple(advantages.shape)}")


def clip_report(
    ratio: Tensor, advantages: Tensor, response_mask: Tensor, clip: ClipConfig
) -> ClipReport:
    """Classify every response token by clip-band crossing and killed PG_CLIP gradient.

    With band ``(1 - ε_lo, 1 + ε_hi)``, each quadrant fraction counts the response
    tokens whose ratio crosses one side of the band within one advantage sign, divided
    by ``N = Σ m`` (all response tokens). The killed-gradient mask marks where the
    PG_CLIP per-token gradient w.r.t. ``logprobs`` is exactly zero:

        killed_t ⇔ (A_t > 0 and r_t > 1+ε_hi) or (A_t < 0 and r_t < 1-ε_lo)
                   or (dual-clip: A_t < 0 and r_t > c) or A_t == 0

    derived branch by branch in docs/diagnostics/clipping.md for ``RatioKind.TOKEN``
    (where ``∇r_t = r_t·∇logprobs_t`` and ``r_t > 0``) and verified against autograd of
    :func:`polgrad.losses.policy_loss`. Ratios exactly on a boundary sit at a
    non-differentiable tie point and are classified by the strict inequalities above.

    Args:
        ratio: ``[B, T]`` importance ratios actually used by the surrogate (for
            example ``PolicyLossResult.ratio``); must be strictly positive at response
            positions.
        advantages: ``[B]`` (broadcast across the row's tokens) or ``[B, T]``.
        response_mask: ``[B, T]`` bool mask of real response tokens.
        clip: Clip bounds; ``eps_low`` and ``eps_high`` must be non-None, and
            ``ratio_cap`` (if set) enables the dual-clip branch of the condition.

    Returns:
        A :class:`ClipReport`; ``gradient_killed_mask`` is ``False`` at masked
        positions.

    Raises:
        ValueError: If ``eps_low`` or ``eps_high`` is ``None`` or non-positive,
            ``ratio_cap`` is set but not a finite float > 1, shapes or the mask are
            invalid, a response position holds a non-finite value, or ``ratio`` is not
            strictly positive at a response position.

    References:
        Schulman et al., "Proximal Policy Optimization Algorithms", arXiv 1707.06347;
        Ye et al., "Mastering Complex Control in MOBA Games with Deep Reinforcement
        Learning", arXiv 1912.09729 (dual-clip PPO);
        docs/diagnostics/clipping.md;
        tests/test_diagnostics_clipping.py::test_gradient_killed_matches_policy_loss_autograd,
        tests/test_diagnostics_clipping.py::test_quadrant_fractions_match_python_oracle.
    """
    eps_low, eps_high = _validate_clip(clip)
    check_2d("ratio", ratio)
    check_mask(response_mask, like=ratio)
    check_finite("ratio (response positions)", ratio[response_mask])
    if not bool((ratio[response_mask] > 0).all()):
        raise ValueError(
            "ratio must be strictly positive at response positions "
            "(an importance ratio is exp(logprobs - old_logprobs) > 0)"
        )
    adv = _broadcast_advantages(advantages, ratio, response_mask)

    # Every region indicator is intersected with the mask, so junk values at masked
    # positions (including non-finite ones) never reach any output (mask invariance).
    pos = response_mask & (adv > 0)
    neg = response_mask & (adv < 0)
    above = ratio > 1.0 + eps_high
    below = ratio < 1.0 - eps_low
    n_tokens = float(response_mask.sum())

    def frac(region: Tensor) -> float:
        return float(region.sum()) / n_tokens

    killed = (pos & above) | (neg & below) | (response_mask & (adv == 0))
    if clip.ratio_cap is not None:
        killed = killed | (neg & (ratio > clip.ratio_cap))
    return ClipReport(
        eps_low=eps_low,
        eps_high=eps_high,
        frac_pos_adv_clipped_high=frac(pos & above),
        frac_pos_adv_clipped_low=frac(pos & below),
        frac_neg_adv_clipped_high=frac(neg & above),
        frac_neg_adv_clipped_low=frac(neg & below),
        gradient_killed_mask=killed,
        gradient_killed_frac=frac(killed),
    )
