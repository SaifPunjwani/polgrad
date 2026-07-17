"""Policy-gradient surrogate and value losses for LLM RL post-training.

Defines the exact per-token semantics of the clipped policy-gradient family: PG_CLIP
(PPO, with optional dual-clip), unclipped PG, REINFORCE, and CISPO surrogates; the three
importance-ratio kinds (per-token, GSPO sequence-level, GSPO token-level); truncated
importance-sampling (TIS) correction for the rollout/trainer logprob mismatch; optional
as-loss KL composition; and the clipped value loss. All losses are quantities to
minimize (the surrogate is already negated). The stop-gradient placements
(``sg[·]`` = ``.detach()``) are load-bearing; every branch and gradient is derived in
``docs/derivations/losses.md``.
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

from polgrad._validation import check_2d, check_finite, check_mask, check_same_shape
from polgrad.aggregate import Aggregation, aggregate
from polgrad.kl import KLLossConfig, kl_loss

__all__ = [
    "ClipConfig",
    "ISCorrectionConfig",
    "PolicyLossConfig",
    "PolicyLossResult",
    "RatioKind",
    "SurrogateKind",
    "ValueLossResult",
    "policy_loss",
    "value_loss",
]


class RatioKind(enum.Enum):
    """Importance-ratio construction used by the ratio-based surrogates.

    With per-token log-ratio ``z_t = logprobs_t - old_logprobs_t``, mask ``m``, and row
    lengths ``L_i = Σ_t m_{i,t}``:

    Attributes:
        TOKEN: ``r_t = exp(z_t)`` — the PPO per-token ratio.
        SEQUENCE: ``s_i = exp(Σ_t m·z_t / L_i)`` — the GSPO length-normalized sequence
            ratio, broadcast to every token of row ``i``; the gradient flows through
            every response token inside the masked mean.
        SEQUENCE_TOKEN: ``s_{i,t} = sg[s_i] · r_t / sg[r_t]`` — GSPO-token: numerically
            equal to ``sg[s_i]``, with token-local gradient ``sg[s_i] · ∇logprobs_t``.

    References:
        Zheng et al., "Group Sequence Policy Optimization", arXiv 2507.18071;
        docs/derivations/losses.md;
        tests/test_losses.py::test_gspo_sequence_token_value_equals_sequence_ratio_value,
        tests/test_losses.py::test_gspo_sequence_token_gradient_is_token_local.
    """

    TOKEN = "token"
    SEQUENCE = "sequence"
    SEQUENCE_TOKEN = "sequence_token"


@dataclass(frozen=True)
class ClipConfig:
    """Clipping bounds for the PG_CLIP and CISPO surrogates.

    PPO clips the ratio below at ``1 - eps_low`` and above at ``1 + eps_high``
    (DAPO's clip-higher sets ``eps_high > eps_low``). ``ratio_cap`` enables dual-clip
    PPO: for ``A < 0`` the objective is floored via the constant ``c = ratio_cap > 1``
    (exact branch in :func:`policy_loss`). The config is inert frozen data; all
    validation fires at :func:`policy_loss` entry.

    Attributes:
        eps_low: Lower clip width, ``1 - eps_low``; ``None`` only where a surrogate
            permits one-sided clipping (CISPO).
        eps_high: Upper clip width, ``1 + eps_high``.
        ratio_cap: Dual-clip constant ``c > 1``; ``None`` disables dual-clip.

    References:
        Schulman et al., "Proximal Policy Optimization Algorithms", arXiv 1707.06347;
        Ye et al., "Mastering Complex Control in MOBA Games with Deep Reinforcement
        Learning", arXiv 1912.09729 (dual-clip PPO);
        docs/derivations/losses.md;
        tests/test_losses.py::test_policy_loss_config_validation_errors.
    """

    eps_low: float | None
    eps_high: float | None
    ratio_cap: float | None = None


class SurrogateKind(enum.Enum):
    """Per-token policy-gradient surrogate (all negated: quantities to minimize).

    With ratio ``r_t`` (see :class:`RatioKind`) and advantage ``A_t``:

    Attributes:
        PG_CLIP: ``-min(r_t A_t, clip(r_t, 1-ε_lo, 1+ε_hi) A_t)``, plus the dual-clip
            floor for ``A_t < 0`` when ``ClipConfig.ratio_cap`` is set.
        PG: ``-r_t A_t`` — unclipped importance-sampled policy gradient.
        REINFORCE: ``-A_t · logprobs_t`` — no ratio; ``old_logprobs`` is ignored and
            ``RatioKind.TOKEN`` is required (the ratio field is unused).
        CISPO: ``-sg[ŵ_t] · A_t · logprobs_t`` with ``ŵ_t = min(r_t, 1+ε_hi)`` (or the
            two-sided ``clamp``); the gradient flows only through the REINFORCE factor.

    References:
        arXiv 1707.06347 (PPO); MiniMax-M1, arXiv 2506.13585 eq. 4-5 (CISPO);
        docs/derivations/losses.md;
        tests/test_losses.py::test_fp64_gradcheck_policy_loss_valid_combinations.
    """

    PG_CLIP = "pg_clip"
    PG = "pg"
    REINFORCE = "reinforce"
    CISPO = "cispo"


@dataclass(frozen=True)
class ISCorrectionConfig:
    """Truncated importance-sampling (TIS) correction for rollout/trainer mismatch.

    The rollout engine's logprobs (``rollout_logprobs``) differ from the trainer's
    recomputed ``old_logprobs``; TIS reweights the surrogate by the truncated ratio of
    the two, applied as a detached factor ``sg[w] · surrogate_t``:

    - ``level="token"``: ``w_t = min(exp(old_logprobs_t - rollout_logprobs_t), cap)``.
    - ``level="sequence"``: ``w_i = min(exp(Σ_t m·(old_logprobs - rollout_logprobs)),
      cap)``, broadcast to the row (the exponent is the unnormalized masked sum).

    Attributes:
        cap: Truncation cap; must be positive and finite.
        level: ``"token"`` or ``"sequence"``.

    References:
        TIS, verl PR #2953; docs/derivations/losses.md;
        tests/test_losses.py::test_is_correction_weight_one_is_noop,
        tests/test_losses.py::test_is_correction_cap_binds.
    """

    cap: float
    level: Literal["token", "sequence"] = "token"


@dataclass(frozen=True)
class PolicyLossConfig:
    """Full specification of a policy loss: ratio, surrogate, clip, aggregation, KL.

    The config is inert frozen data; every validation rule (surrogate/clip
    compatibility, ``ratio_cap > 1``, required call-time tensors) fires at
    :func:`policy_loss` entry, so partially-specified configs (for example
    ``norm_len=None`` for Dr.GRPO) may be constructed and stored freely.

    Attributes:
        ratio: Importance-ratio construction.
        surrogate: Per-token surrogate.
        clip: Clip bounds; required for PG_CLIP and CISPO, must be ``None`` for PG and
            REINFORCE.
        aggregation: Reduction of the per-token surrogate to a scalar.
        norm_len: Fixed generation budget; required iff ``aggregation`` is
            ``Aggregation.TOKEN_SUM_NORM``.
        is_correction: Optional TIS correction; requires ``rollout_logprobs`` at call
            time.
        kl: Optional as-loss KL term ``loss += kl.coef · kl_loss(...)``; requires
            ``ref_logprobs`` at call time.

    References:
        docs/derivations/losses.md;
        tests/test_losses.py::test_policy_loss_config_validation_errors.
    """

    ratio: RatioKind
    surrogate: SurrogateKind
    clip: ClipConfig | None
    aggregation: Aggregation
    norm_len: int | None = None
    is_correction: ISCorrectionConfig | None = None
    kl: KLLossConfig | None = None


@dataclass(frozen=True)
class PolicyLossResult:
    """Outputs of :func:`policy_loss`.

    Attributes:
        loss: Scalar, differentiable w.r.t. ``logprobs``; includes the KL term if
            configured.
        per_token_objective: ``[B, T]`` per-token surrogate, post-mask and
            post-IS-correction, pre-KL; exactly ``0`` at masked positions and
            differentiable, so ``aggregate(per_token_objective, ...)`` reproduces the
            surrogate part of ``loss``.
        ratio: ``[B, T]`` detached ratio actually used by the surrogate; ``1.0`` at
            masked positions; all-ones for REINFORCE.
        clipped_low: ``[B, T]`` bool, ``True`` where the lower clip bound was the branch
            autograd took; all-``False`` for PG/REINFORCE and at masked positions.
        clipped_high: ``[B, T]`` bool, ``True`` where the upper clip bound (or the
            dual-clip cap, for ``A < 0``) was the branch autograd took; all-``False``
            for PG/REINFORCE and at masked positions.
        kl_loss: The unscaled ``kl_loss(...)`` scalar (the ``coef`` multiplies it inside
            ``loss``), differentiable; ``None`` if ``config.kl`` is ``None``.

    References:
        docs/derivations/losses.md;
        tests/test_losses.py::test_pg_clip_masks_match_autograd_branch,
        tests/test_losses.py::test_policy_loss_kl_term_composition.
    """

    loss: Tensor
    per_token_objective: Tensor
    ratio: Tensor
    clipped_low: Tensor
    clipped_high: Tensor
    kl_loss: Tensor | None


@dataclass(frozen=True)
class ValueLossResult:
    """Outputs of :func:`value_loss`.

    Attributes:
        loss: Scalar, differentiable w.r.t. ``values``.
        clipped_frac: Fraction of response tokens where the clipped squared error
            strictly exceeded the unclipped one (the branch the pessimistic ``max``
            took); ``0.0`` when ``clip_eps`` is ``None``.

    References:
        docs/derivations/losses.md;
        tests/test_losses.py::test_value_loss_golden_clip_branches.
    """

    loss: Tensor
    clipped_frac: float


def _validate_policy_config(config: PolicyLossConfig) -> None:
    """Enforce the surrogate/clip compatibility rules of contract section 4.3."""
    clip = config.clip
    surrogate = config.surrogate
    if surrogate is SurrogateKind.PG_CLIP:
        if clip is None or clip.eps_low is None or clip.eps_high is None:
            raise ValueError(
                f"SurrogateKind.PG_CLIP requires a ClipConfig with eps_low and eps_high "
                f"non-None; got clip={clip!r}"
            )
        if clip.ratio_cap is not None and not (
            math.isfinite(clip.ratio_cap) and clip.ratio_cap > 1.0
        ):
            raise ValueError(
                f"ClipConfig.ratio_cap must be a finite float > 1 (dual-clip); got {clip.ratio_cap}"
            )
    elif surrogate is SurrogateKind.CISPO:
        if clip is None or clip.eps_high is None:
            raise ValueError(
                f"SurrogateKind.CISPO requires a ClipConfig with eps_high non-None; "
                f"got clip={clip!r}"
            )
        if clip.ratio_cap is not None:
            raise ValueError(
                f"SurrogateKind.CISPO does not support dual-clip; got ratio_cap={clip.ratio_cap}"
            )
    elif clip is not None:
        raise ValueError(f"SurrogateKind.{surrogate.name} requires clip=None; got clip={clip!r}")
    if surrogate is SurrogateKind.REINFORCE and config.ratio is not RatioKind.TOKEN:
        raise ValueError(
            f"SurrogateKind.REINFORCE requires ratio=RatioKind.TOKEN (the ratio is unused); "
            f"got {config.ratio}"
        )
    if clip is not None:
        for name, eps in (("eps_low", clip.eps_low), ("eps_high", clip.eps_high)):
            if eps is not None and not (math.isfinite(eps) and eps > 0.0):
                raise ValueError(f"ClipConfig.{name} must be a positive finite float; got {eps}")
    if config.is_correction is not None:
        correction = config.is_correction
        if not (math.isfinite(correction.cap) and correction.cap > 0.0):
            raise ValueError(
                f"ISCorrectionConfig.cap must be a positive finite float; got {correction.cap}"
            )
        if correction.level not in ("token", "sequence"):
            raise ValueError(
                f"ISCorrectionConfig.level must be 'token' or 'sequence'; got {correction.level!r}"
            )


def _broadcast_advantages(advantages: Tensor, like: Tensor, response_mask: Tensor) -> Tensor:
    """Validate ``[B]`` or ``[B, T]`` advantages and return them ``[B, T]``, 0 at masked."""
    if advantages.dim() == 1:
        if advantages.shape[0] != like.shape[0]:
            raise ValueError(
                f"advantages [B] must have B = {like.shape[0]} rows; "
                f"got shape {tuple(advantages.shape)}"
            )
        check_finite("advantages", advantages)
        expanded = advantages.unsqueeze(1).expand_as(like)
    elif advantages.dim() == 2:
        check_same_shape("advantages", advantages, "logprobs", like)
        check_finite("advantages", advantages[response_mask])
        expanded = advantages
    else:
        raise ValueError(f"advantages must be [B] or [B, T]; got shape {tuple(advantages.shape)}")
    zero = torch.zeros((), dtype=expanded.dtype, device=expanded.device)
    # Masked positions are zeroed so padded advantage junk reaches neither the surrogate
    # values nor the backward formulas (mask invariance).
    return torch.where(response_mask, expanded, zero)


def _ratio(
    kind: RatioKind, logprobs: Tensor, old_logprobs: Tensor, response_mask: Tensor
) -> Tensor:
    """Differentiable ``[B, T]`` ratio; inputs are pre-zeroed at masked positions."""
    log_ratio = logprobs - old_logprobs
    if kind is RatioKind.TOKEN:
        return torch.exp(log_ratio)
    lengths = response_mask.sum(dim=1, keepdim=True).to(logprobs.dtype)
    sequence = torch.exp(log_ratio.sum(dim=1, keepdim=True) / lengths)
    if kind is RatioKind.SEQUENCE:
        return sequence.expand_as(logprobs)
    if kind is RatioKind.SEQUENCE_TOKEN:
        token = torch.exp(log_ratio)
        # detach: GSPO-token freezes the sequence weight (sg[s_i]); no gradient flows
        # through the length-normalized mean.
        frozen_sequence = sequence.detach()
        # detach: sg[r_t] in the denominator makes r_t / sg[r_t] exactly 1 in value while
        # keeping gradient ∇r_t / r_t = ∇logprobs_t, so ∇s_{i,t} = sg[s_i] · ∇logprobs_t.
        return frozen_sequence * (token / token.detach())
    raise ValueError(f"unknown RatioKind: {kind!r}")


def _pg_clip(
    ratio: Tensor,
    advantages: Tensor,
    eps_low: float,
    eps_high: float,
    ratio_cap: float | None,
    response_mask: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """PG_CLIP per-token loss and the clip masks matching the branch autograd took."""
    low, high = 1.0 - eps_low, 1.0 + eps_high
    unclipped = ratio * advantages
    clipped = ratio.clamp(low, high) * advantages
    objective = torch.minimum(unclipped, clipped)
    clipped_high = (advantages > 0) & (ratio > high)
    clipped_low = (advantages < 0) & (ratio < low)
    if ratio_cap is not None:
        floor = ratio_cap * advantages
        objective = torch.where(advantages < 0, torch.maximum(objective, floor), objective)
        clipped_high = clipped_high | ((advantages < 0) & (ratio > ratio_cap))
    return -objective, clipped_low & response_mask, clipped_high & response_mask


def _cispo(
    ratio: Tensor,
    advantages: Tensor,
    logprobs: Tensor,
    eps_low: float | None,
    eps_high: float,
    response_mask: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """CISPO per-token loss ``-sg[ŵ]·A·logprobs`` and the weight-clip masks."""
    high = 1.0 + eps_high
    if eps_low is None:
        weight = ratio.clamp(max=high)
        clipped_low = torch.zeros_like(response_mask)
    else:
        weight = ratio.clamp(1.0 - eps_low, high)
        clipped_low = (ratio < 1.0 - eps_low) & response_mask
    clipped_high = (ratio > high) & response_mask
    # detach: CISPO's ŵ = clip(r) is an importance weight held constant (sg[ŵ]); the
    # gradient flows only through the REINFORCE factor logprobs (arXiv 2506.13585 eq. 5).
    frozen_weight = weight.detach()
    return -(frozen_weight * advantages * logprobs), clipped_low, clipped_high


def _is_weights(
    old_logprobs: Tensor,
    rollout_logprobs: Tensor,
    response_mask: Tensor,
    correction: ISCorrectionConfig,
) -> Tensor:
    """Detached TIS weights ``[B, T]``; inputs are pre-zeroed at masked positions."""
    log_weight = old_logprobs - rollout_logprobs
    if correction.level == "token":
        weight = torch.exp(log_weight).clamp(max=correction.cap)
    else:
        row = torch.exp(log_weight.sum(dim=1, keepdim=True)).clamp(max=correction.cap)
        weight = row.expand_as(old_logprobs)
    # detach: the truncated IS weight corrects the sampling distribution as data
    # (sg[w] · surrogate); it must not open a gradient path even if callers pass
    # differentiable logprob streams (TIS, verl PR #2953).
    return weight.detach()


def policy_loss(
    config: PolicyLossConfig,
    *,
    logprobs: Tensor,
    old_logprobs: Tensor,
    advantages: Tensor,
    response_mask: Tensor,
    ref_logprobs: Tensor | None = None,
    rollout_logprobs: Tensor | None = None,
) -> PolicyLossResult:
    """Compute the configured policy-gradient surrogate loss.

    ``loss = aggregate(sg[w_TIS] · surrogate_t(r_t, A_t), aggregation)
    [+ kl.coef · kl_loss(logprobs, ref_logprobs, ...)]`` with the surrogate per-token
    forms of :class:`SurrogateKind`, the ratio forms of :class:`RatioKind`, and the
    dual-clip branch ``-max(min(rA, clip(r)A), cA)`` for ``A < 0`` when
    ``ClipConfig.ratio_cap = c`` is set. Every branch and stop-gradient is derived in
    docs/derivations/losses.md.

    Args:
        config: Loss specification; validated here (contract section 4.3).
        logprobs: ``[B, T]`` current-policy sampled-token logprobs (differentiable).
        old_logprobs: ``[B, T]`` behavior-policy logprobs (constant); ignored by
            REINFORCE.
        advantages: ``[B]`` (broadcast across the row's tokens) or ``[B, T]``.
        response_mask: ``[B, T]`` bool mask of real response tokens.
        ref_logprobs: ``[B, T]`` frozen reference logprobs; required iff ``config.kl``
            is set.
        rollout_logprobs: ``[B, T]`` inference-engine logprobs; required iff
            ``config.is_correction`` is set.

    Returns:
        :class:`PolicyLossResult`; masked positions are ``0`` in
        ``per_token_objective``, ``1.0`` in ``ratio``, ``False`` in ``clipped_*``.

    Raises:
        ValueError: On any contract-section-4.3 config violation (PG_CLIP/CISPO clip
            requirements, PG/REINFORCE ``clip=None``, REINFORCE ``RatioKind.TOKEN``,
            ``ratio_cap > 1``), on missing ``rollout_logprobs``/``ref_logprobs``, on
            shape/mask/dtype violations, or on non-finite response-position values.

    References:
        arXiv 1707.06347 (PPO); arXiv 1912.09729 (dual-clip); arXiv 2507.18071 (GSPO);
        arXiv 2506.13585 eq. 4-5 (CISPO); verl PR #2953 (TIS);
        docs/derivations/losses.md;
        tests/test_losses.py::test_fp64_gradcheck_policy_loss_valid_combinations,
        tests/test_losses.py::test_pg_clip_golden_two_token_ragged_mixed_branches.
    """
    _validate_policy_config(config)
    check_2d("logprobs", logprobs)
    check_same_shape("logprobs", logprobs, "old_logprobs", old_logprobs)
    check_mask(response_mask, like=logprobs)
    check_finite("logprobs", logprobs[response_mask])
    check_finite("old_logprobs", old_logprobs[response_mask])
    if config.is_correction is not None and rollout_logprobs is None:
        raise ValueError("config.is_correction is set but rollout_logprobs is None")
    if config.kl is not None and ref_logprobs is None:
        raise ValueError("config.kl is set but ref_logprobs is None")
    if rollout_logprobs is not None:
        check_same_shape("logprobs", logprobs, "rollout_logprobs", rollout_logprobs)
        check_finite("rollout_logprobs", rollout_logprobs[response_mask])
    if ref_logprobs is not None:
        check_same_shape("logprobs", logprobs, "ref_logprobs", ref_logprobs)

    zero = torch.zeros((), dtype=logprobs.dtype, device=logprobs.device)
    # Masked positions are zeroed before exp() so padding junk reaches neither the
    # forward values nor the backward formulas, and the masked ratio is exactly 1.
    lp = torch.where(response_mask, logprobs, zero)
    olp = torch.where(response_mask, old_logprobs, zero)
    adv = _broadcast_advantages(advantages, logprobs, response_mask)

    all_false = torch.zeros_like(response_mask)
    if config.surrogate is SurrogateKind.REINFORCE:
        ratio = torch.ones_like(lp)
        per_token = -(adv * lp)
        clipped_low, clipped_high = all_false, all_false
    else:
        ratio = _ratio(config.ratio, lp, olp, response_mask)
        if config.surrogate is SurrogateKind.PG:
            per_token = -(ratio * adv)
            clipped_low, clipped_high = all_false, all_false
        elif config.surrogate is SurrogateKind.PG_CLIP:
            clip = config.clip
            assert clip is not None and clip.eps_low is not None and clip.eps_high is not None
            per_token, clipped_low, clipped_high = _pg_clip(
                ratio, adv, clip.eps_low, clip.eps_high, clip.ratio_cap, response_mask
            )
        else:
            clip = config.clip
            assert clip is not None and clip.eps_high is not None
            per_token, clipped_low, clipped_high = _cispo(
                ratio, adv, lp, clip.eps_low, clip.eps_high, response_mask
            )

    if config.is_correction is not None:
        assert rollout_logprobs is not None
        rlp = torch.where(response_mask, rollout_logprobs, zero)
        per_token = per_token * _is_weights(olp, rlp, response_mask, config.is_correction)

    per_token_objective = torch.where(response_mask, per_token, zero)
    surrogate_loss = aggregate(
        per_token_objective, response_mask, config.aggregation, norm_len=config.norm_len
    )

    kl_term: Tensor | None = None
    loss = surrogate_loss
    if config.kl is not None:
        assert ref_logprobs is not None
        kl_aggregation = (
            config.kl.aggregation if config.kl.aggregation is not None else config.aggregation
        )
        kl_norm_len = config.kl.norm_len if config.kl.norm_len is not None else config.norm_len
        kl_term = kl_loss(
            logprobs,
            ref_logprobs,
            config.kl.kind,
            kl_aggregation,
            response_mask=response_mask,
            norm_len=kl_norm_len,
        )
        loss = surrogate_loss + config.kl.coef * kl_term

    one = torch.ones((), dtype=logprobs.dtype, device=logprobs.device)
    # detach: the reported ratio is a diagnostic output, not a gradient path.
    ratio_report = torch.where(response_mask, ratio.detach(), one)
    return PolicyLossResult(
        loss=loss,
        per_token_objective=per_token_objective,
        ratio=ratio_report,
        clipped_low=clipped_low,
        clipped_high=clipped_high,
        kl_loss=kl_term,
    )


def value_loss(
    values: Tensor,
    old_values: Tensor,
    returns: Tensor,
    response_mask: Tensor,
    *,
    clip_eps: float | None,
    aggregation: Aggregation,
    norm_len: int | None = None,
) -> ValueLossResult:
    """Clipped (PPO) or plain squared-error value loss.

    Per token: ``½·max((v - R)², (clip(v, v_old - ε, v_old + ε) - R)²)`` when
    ``clip_eps = ε`` is set, else ``½·(v - R)²``; reduced with ``aggregation``. The
    pessimistic ``max`` takes the clipped branch — killing the gradient — exactly where
    the clipped squared error exceeds the unclipped one (docs/derivations/losses.md).

    Args:
        values: ``[B, T]`` current value predictions (differentiable).
        old_values: ``[B, T]`` value predictions at rollout time (constant).
        returns: ``[B, T]`` return targets.
        response_mask: ``[B, T]`` bool mask of real response tokens.
        clip_eps: Clip width ``ε > 0``, or ``None`` for the unclipped loss.
        aggregation: Reduction of the per-token loss to a scalar.
        norm_len: Required iff ``aggregation`` is ``Aggregation.TOKEN_SUM_NORM``.

    Returns:
        :class:`ValueLossResult` with the scalar loss (input dtype, differentiable
        w.r.t. ``values``) and the clipped-branch fraction over response tokens.

    Raises:
        ValueError: On shape/mask violations, non-finite response-position values, a
            non-positive ``clip_eps``, or a missing ``norm_len`` under
            ``TOKEN_SUM_NORM``.

    References:
        arXiv 1707.06347 (PPO); docs/derivations/losses.md;
        tests/test_losses.py::test_value_loss_golden_clip_branches,
        tests/test_losses.py::test_fp64_gradcheck_value_loss.
    """
    check_2d("values", values)
    check_same_shape("values", values, "old_values", old_values)
    check_same_shape("values", values, "returns", returns)
    check_mask(response_mask, like=values)
    check_finite("values", values[response_mask])
    check_finite("old_values", old_values[response_mask])
    check_finite("returns", returns[response_mask])
    if clip_eps is not None and not (math.isfinite(clip_eps) and clip_eps > 0.0):
        raise ValueError(f"clip_eps must be a positive finite float or None; got {clip_eps}")

    zero = torch.zeros((), dtype=values.dtype, device=values.device)
    # Masked positions are zeroed so padding junk reaches neither the squared errors
    # nor their backward formulas (mask invariance).
    v = torch.where(response_mask, values, zero)
    v_old = torch.where(response_mask, old_values, zero)
    target = torch.where(response_mask, returns, zero)
    error_sq = (v - target).pow(2)
    if clip_eps is None:
        per_token = 0.5 * error_sq
        clipped = torch.zeros_like(response_mask)
    else:
        # clamp (not v_old + clamp(v - v_old)) so v_clip == v bitwise inside the band and
        # the strict clipped-branch comparison cannot pick up 1-ulp rounding artifacts.
        v_clip = torch.clamp(v, v_old - clip_eps, v_old + clip_eps)
        clip_error_sq = (v_clip - target).pow(2)
        per_token = 0.5 * torch.maximum(error_sq, clip_error_sq)
        clipped = (clip_error_sq > error_sq) & response_mask
    loss = aggregate(per_token, response_mask, aggregation, norm_len=norm_len)
    clipped_frac = float(clipped.sum().item()) / float(response_mask.sum().item())
    return ValueLossResult(loss=loss, clipped_frac=clipped_frac)
