"""Enforces the surrogate-loss semantics of docs/derivations/losses.md: fp64 gradcheck
over every valid surrogate x ratio x aggregation combination, hand-derived PG_CLIP and
dual-clip goldens, the on-policy PG_CLIP == PG == REINFORCE gradient equivalence,
CISPO's stop-gradient semantics, the GSPO sequence/sequence-token decompositions, TIS
correction, KL composition, mask invariance, validation errors, and the clipped value
loss."""

from __future__ import annotations

import dataclasses
import math
from typing import NamedTuple

import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st
from strategies import MASKED_JUNK, padded_masks
from torch.testing import assert_close

from polgrad.aggregate import Aggregation, aggregate, effective_token_weights
from polgrad.kl import KLEstimator, KLLossConfig, kl_loss
from polgrad.losses import (
    ClipConfig,
    ISCorrectionConfig,
    PolicyLossConfig,
    PolicyLossResult,
    RatioKind,
    SurrogateKind,
    ValueLossResult,
    policy_loss,
    value_loss,
)

MODES = tuple(Aggregation)
NORM_LEN = 4

# Every valid (surrogate, ratio) pair: REINFORCE only with TOKEN (docs/derivations/losses.md).
VALID_COMBOS = [
    (surrogate, ratio)
    for surrogate in SurrogateKind
    for ratio in RatioKind
    if not (surrogate is SurrogateKind.REINFORCE and ratio is not RatioKind.TOKEN)
]


def _norm_len(mode: Aggregation) -> int | None:
    return NORM_LEN if mode is Aggregation.TOKEN_SUM_NORM else None


def make_config(
    surrogate: SurrogateKind,
    ratio: RatioKind,
    aggregation: Aggregation = Aggregation.TOKEN_MEAN,
    *,
    ratio_cap: float | None = None,
    cispo_eps_low: float | None = None,
    is_correction: ISCorrectionConfig | None = None,
    kl: KLLossConfig | None = None,
) -> PolicyLossConfig:
    """Valid config for the pair, with the clip the surrogate requires."""
    clip: ClipConfig | None
    if surrogate is SurrogateKind.PG_CLIP:
        clip = ClipConfig(eps_low=0.2, eps_high=0.3, ratio_cap=ratio_cap)
    elif surrogate is SurrogateKind.CISPO:
        clip = ClipConfig(eps_low=cispo_eps_low, eps_high=0.3)
    else:
        clip = None
    return PolicyLossConfig(
        ratio=ratio,
        surrogate=surrogate,
        clip=clip,
        aggregation=aggregation,
        norm_len=_norm_len(aggregation),
        is_correction=is_correction,
        kl=kl,
    )


class LossBatch(NamedTuple):
    """Local [B, T] float64 batch with all four logprob streams and token advantages."""

    logprobs: torch.Tensor
    old_logprobs: torch.Tensor
    ref_logprobs: torch.Tensor
    rollout_logprobs: torch.Tensor
    advantages: torch.Tensor
    response_mask: torch.Tensor


@st.composite
def loss_batches(draw: st.DrawFn, *, max_b: int = 6, max_t: int = 8) -> LossBatch:
    """Like strategies.logprob_batches but with tighter gaps (|old/ref gaps| <= 1.5,
    |rollout gap| <= 1.0) so PG_CLIP ratios exercise both clip branches without
    overflow. Masked positions hold MASKED_JUNK."""
    mask = draw(padded_masks(max_b=max_b, max_t=max_t))
    b, t = mask.shape

    def fill(low: float, high: float, junk: float) -> torch.Tensor:
        vals = [
            draw(st.floats(low, high, allow_nan=False, allow_infinity=False, width=32))
            for _ in range(b * t)
        ]
        x = torch.tensor(vals, dtype=torch.float64).reshape(b, t)
        return torch.where(mask, x, torch.full_like(x, junk))

    logprobs = fill(-8.0, -0.0625, MASKED_JUNK)
    junk = torch.full_like(logprobs, MASKED_JUNK)
    old = torch.where(mask, logprobs + fill(-1.5, 1.5, 0.0), junk)
    ref = torch.where(mask, logprobs + fill(-1.5, 1.5, 0.0), junk)
    rollout = torch.where(mask, old + fill(-1.0, 1.0, 0.0), junk)
    advantages = fill(-3.0, 3.0, MASKED_JUNK)
    return LossBatch(logprobs, old, ref, rollout, advantages, mask)


def _grad_of_loss(
    config: PolicyLossConfig, batch: LossBatch, old_logprobs: torch.Tensor | None = None
) -> tuple[PolicyLossResult, torch.Tensor]:
    leaf = batch.logprobs.clone().requires_grad_(True)
    result = policy_loss(
        config,
        logprobs=leaf,
        old_logprobs=batch.old_logprobs if old_logprobs is None else old_logprobs,
        advantages=batch.advantages,
        response_mask=batch.response_mask,
    )
    (grad,) = torch.autograd.grad(result.loss, leaf)
    return result, grad


# --- fp64 gradcheck over every valid combination (docs/derivations/losses.md) -------
#
# Ragged 2x3 batch. Token ratios exp(lp - olp) = [e^0.1, e^-0.2, e^0.2, e^-0.1, e^0.2]
# = [1.105, 0.819, 1.221, 0.905, 1.221] and sequence ratios [e^{0.1/3}, e^{0.05}]
# = [1.034, 1.051] all sit > 0.018 away from the clip bounds 1 - 0.2 = 0.8 and
# 1 + 0.3 = 1.3, so no branch flips inside gradcheck's finite-difference perturbation.
GC_MASK = torch.tensor([[True, True, True], [True, True, False]])
GC_LP = torch.tensor([[-0.9, -1.6, -0.3], [-0.6, -1.1, MASKED_JUNK]], dtype=torch.float64)
GC_OLP = torch.tensor([[-1.0, -1.4, -0.5], [-0.5, -1.3, MASKED_JUNK]], dtype=torch.float64)
GC_ADV = torch.tensor([[1.5, -2.0, 0.8], [-1.0, 2.2, MASKED_JUNK]], dtype=torch.float64)


def _sg_frozen_fn(
    config: PolicyLossConfig,
    lp0: torch.Tensor,
    olp: torch.Tensor,
    adv: torch.Tensor,
    mask: torch.Tensor,
) -> object:
    """gradcheck target with the stop-gradient factors frozen at ``lp0``.

    Finite differences see through ``.detach()``, so a loss with internal sg[.] cannot
    be gradchecked directly; the semantically-checked function holds the detached
    factors constant. CISPO (sg[w]) freezes the weight into REINFORCE advantages;
    SEQUENCE_TOKEN (sg[s_i], sg[r_t]) becomes a TOKEN-ratio call whose old_logprobs
    are shifted so exp(lp0 - old') = s_i(lp0). Combos without internal detach are
    returned as the plain policy_loss call.
    """
    zero = torch.zeros((), dtype=torch.float64)
    z0 = torch.where(mask, lp0 - olp, zero)
    lengths = mask.sum(dim=1, keepdim=True).to(torch.float64)
    s0 = torch.exp(z0.sum(dim=1, keepdim=True) / lengths)
    if config.surrogate is SurrogateKind.CISPO:
        ratio0 = torch.exp(z0) if config.ratio is RatioKind.TOKEN else s0.expand_as(lp0)
        clip = config.clip
        assert clip is not None and clip.eps_high is not None
        high = 1.0 + clip.eps_high
        if clip.eps_low is None:
            w0 = ratio0.clamp(max=high)
        else:
            w0 = ratio0.clamp(1.0 - clip.eps_low, high)
        frozen = dataclasses.replace(
            config, surrogate=SurrogateKind.REINFORCE, ratio=RatioKind.TOKEN, clip=None
        )
        frozen_adv = w0 * torch.where(mask, adv, zero)

        def fn_cispo(x: torch.Tensor) -> torch.Tensor:
            return policy_loss(
                frozen, logprobs=x, old_logprobs=olp, advantages=frozen_adv, response_mask=mask
            ).loss

        return fn_cispo
    if config.ratio is RatioKind.SEQUENCE_TOKEN:
        frozen = dataclasses.replace(config, ratio=RatioKind.TOKEN)
        shifted_old = lp0 - torch.log(s0)

        def fn_gspo_token(x: torch.Tensor) -> torch.Tensor:
            return policy_loss(
                frozen, logprobs=x, old_logprobs=shifted_old, advantages=adv, response_mask=mask
            ).loss

        return fn_gspo_token

    def fn_plain(x: torch.Tensor) -> torch.Tensor:
        return policy_loss(
            config, logprobs=x, old_logprobs=olp, advantages=adv, response_mask=mask
        ).loss

    return fn_plain


def _gradcheck_semantic(config: PolicyLossConfig, olp: torch.Tensor) -> None:
    """gradcheck the sg-frozen target, then assert the real config's autograd gradient
    equals the frozen target's at the evaluation point."""
    fn = _sg_frozen_fn(config, GC_LP, olp, GC_ADV, GC_MASK)
    leaf = GC_LP.clone().requires_grad_(True)
    assert torch.autograd.gradcheck(fn, (leaf,))  # type: ignore[arg-type]
    frozen_leaf = GC_LP.clone().requires_grad_(True)
    (frozen_grad,) = torch.autograd.grad(fn(frozen_leaf), frozen_leaf)  # type: ignore[operator]
    real_leaf = GC_LP.clone().requires_grad_(True)
    real = policy_loss(
        config, logprobs=real_leaf, old_logprobs=olp, advantages=GC_ADV, response_mask=GC_MASK
    )
    (real_grad,) = torch.autograd.grad(real.loss, real_leaf)
    assert_close(real_grad, frozen_grad, rtol=1e-12, atol=1e-12)


def _gradcheck_params() -> list[object]:
    params: list[object] = []
    for aggregation in MODES:
        for surrogate, ratio in VALID_COMBOS:
            config = make_config(surrogate, ratio, aggregation)
            params.append(
                pytest.param(config, id=f"{surrogate.value}-{ratio.value}-{aggregation.value}")
            )
    return params


@pytest.mark.parametrize("config", _gradcheck_params())
def test_fp64_gradcheck_policy_loss_valid_combinations(config: PolicyLossConfig) -> None:
    """torch.autograd.gradcheck of policy_loss on a small ragged fp64 batch for every
    valid SurrogateKind x RatioKind x Aggregation combination; stop-gradient factors
    are frozen at the evaluation point and the real config's gradient is asserted
    equal to the frozen equivalent's (docs/derivations/losses.md, gradient tables)."""
    _gradcheck_semantic(config, GC_OLP)


@pytest.mark.parametrize("ratio", tuple(RatioKind))
def test_fp64_gradcheck_policy_loss_dual_clip(ratio: RatioKind) -> None:
    """gradcheck of the dual-clip branch for every RatioKind: token (0,1) has
    A = -2 < 0; its token ratio is e^2.4 = 11.02 > ratio_cap = 2 and row 0's sequence
    ratio is e^{(0.1+2.4+0.2)/3} = e^0.9 = 2.4596 > 2, so the cap floor is the active
    branch there under TOKEN, SEQUENCE, and SEQUENCE_TOKEN ratios
    (docs/derivations/losses.md, dual-clip)."""
    olp = GC_OLP.clone()
    olp[0, 1] = -4.0
    _gradcheck_semantic(make_config(SurrogateKind.PG_CLIP, ratio, ratio_cap=2.0), olp)


def test_fp64_gradcheck_policy_loss_with_is_correction_and_kl() -> None:
    """gradcheck of the full composition: PG_CLIP surrogate x detached TIS weight +
    coef * kl_loss(K3) (docs/derivations/losses.md, composition)."""
    config = make_config(
        SurrogateKind.PG_CLIP,
        RatioKind.TOKEN,
        is_correction=ISCorrectionConfig(cap=1.5, level="token"),
        kl=KLLossConfig(kind=KLEstimator.K3, coef=0.07),
    )
    ref = GC_LP + 0.3
    rollout = GC_OLP - 0.2
    leaf = GC_LP.clone().requires_grad_(True)

    def fn(x: torch.Tensor) -> torch.Tensor:
        return policy_loss(
            config,
            logprobs=x,
            old_logprobs=GC_OLP,
            advantages=GC_ADV,
            response_mask=GC_MASK,
            ref_logprobs=ref,
            rollout_logprobs=rollout,
        ).loss

    assert torch.autograd.gradcheck(fn, (leaf,))


# --- hand-derived PG_CLIP goldens (arithmetic shown; docs/derivations/losses.md) -----


def _one_token(
    lp: float, olp: float, adv: float, clip: ClipConfig
) -> tuple[PolicyLossResult, torch.Tensor]:
    config = PolicyLossConfig(
        ratio=RatioKind.TOKEN,
        surrogate=SurrogateKind.PG_CLIP,
        clip=clip,
        aggregation=Aggregation.TOKEN_MEAN,
    )
    leaf = torch.tensor([[lp]], dtype=torch.float64, requires_grad=True)
    result = policy_loss(
        config,
        logprobs=leaf,
        old_logprobs=torch.tensor([[olp]], dtype=torch.float64),
        advantages=torch.tensor([adv], dtype=torch.float64),
        response_mask=torch.ones((1, 1), dtype=torch.bool),
    )
    (grad,) = torch.autograd.grad(result.loss, leaf)
    return result, grad


def test_pg_clip_golden_one_token_inside_clip() -> None:
    """1-token PG_CLIP, ratio inside the clip band (docs/derivations/losses.md).
    r = e^{-0.4-(-0.5)} = e^0.1 = 1.10517... in [0.8, 1.2]; A = 2:
    loss = -min(rA, rA) = -2e^0.1 = -2.21034...; dloss/dlp = -A*r = -2e^0.1."""
    result, grad = _one_token(-0.4, -0.5, 2.0, ClipConfig(eps_low=0.2, eps_high=0.2))
    expected = -2.0 * math.exp(0.1)
    assert_close(result.loss, torch.tensor(expected, dtype=torch.float64), rtol=1e-12, atol=0.0)
    assert_close(grad, torch.tensor([[expected]], dtype=torch.float64), rtol=1e-12, atol=0.0)
    assert_close(
        result.ratio,
        torch.tensor([[math.exp(0.1)]], dtype=torch.float64),
        rtol=1e-12,
        atol=0.0,
    )
    assert not bool(result.clipped_low.any()) and not bool(result.clipped_high.any())


def test_pg_clip_golden_one_token_clipped_high_positive_advantage() -> None:
    """1-token PG_CLIP above 1+eps_high with A > 0 (docs/derivations/losses.md).
    r = e^{-0.1-(-0.9)} = e^0.8 = 2.22554 > 1.2; A = 1.5:
    unclipped = 1.5*e^0.8 = 3.33831 > clipped = 1.5*1.2 = 1.8, so the min takes the
    constant clipped branch: loss = -1.8, gradient exactly 0, clipped_high True."""
    result, grad = _one_token(-0.1, -0.9, 1.5, ClipConfig(eps_low=0.2, eps_high=0.2))
    assert torch.equal(result.loss, torch.tensor(-(1.2 * 1.5), dtype=torch.float64))
    assert torch.equal(grad, torch.zeros((1, 1), dtype=torch.float64))
    assert bool(result.clipped_high.all()) and not bool(result.clipped_low.any())


def test_pg_clip_golden_one_token_clipped_low_negative_advantage() -> None:
    """1-token PG_CLIP below 1-eps_low with A < 0 (docs/derivations/losses.md).
    r = e^{-1.2-(-0.4)} = e^{-0.8} = 0.44933 < 0.8; A = -1:
    unclipped = -e^{-0.8} = -0.44933; clipped = -0.8; min = -0.8, so
    loss = 0.8, gradient exactly 0, clipped_low True."""
    result, grad = _one_token(-1.2, -0.4, -1.0, ClipConfig(eps_low=0.2, eps_high=0.2))
    assert torch.equal(result.loss, torch.tensor(0.8, dtype=torch.float64))
    assert torch.equal(grad, torch.zeros((1, 1), dtype=torch.float64))
    assert bool(result.clipped_low.all()) and not bool(result.clipped_high.any())


def test_pg_clip_golden_dual_clip_negative_advantage_above_cap() -> None:
    """1-token dual-clip branch, A < 0 and r > c (docs/derivations/losses.md).
    r = e^{-0.1-(-1.6)} = e^1.5 = 4.48169 > c = 3; A = -2:
    min branch = min(-2e^1.5, 1.2*(-2)) = min(-8.96338, -2.4) = -8.96338; the dual-clip
    floor cA = -6 wins the max: objective = -6, loss = 6, gradient exactly 0; the cap
    binding is reported in clipped_high."""
    result, grad = _one_token(
        -0.1, -1.6, -2.0, ClipConfig(eps_low=0.2, eps_high=0.2, ratio_cap=3.0)
    )
    assert torch.equal(result.loss, torch.tensor(6.0, dtype=torch.float64))
    assert torch.equal(grad, torch.zeros((1, 1), dtype=torch.float64))
    assert bool(result.clipped_high.all()) and not bool(result.clipped_low.any())


def test_pg_clip_golden_dual_clip_inactive_between_high_and_cap() -> None:
    """1-token, A < 0 with 1+eps_high < r < c: the unclipped branch flows
    (docs/derivations/losses.md, dual-clip branch table).
    r = e^{-0.3-(-0.8)} = e^0.5 = 1.64872, c = 3; A = -1:
    min(-e^0.5, -1.2) = -e^0.5 (unclipped); max(-e^0.5, -3) = -e^0.5, so
    loss = e^0.5 and dloss/dlp = -A*r = +e^0.5; no clip mask is set even though
    r > 1.2, because the upper clip was not the branch autograd took."""
    result, grad = _one_token(
        -0.3, -0.8, -1.0, ClipConfig(eps_low=0.2, eps_high=0.2, ratio_cap=3.0)
    )
    expected = math.exp(0.5)
    assert_close(result.loss, torch.tensor(expected, dtype=torch.float64), rtol=1e-12, atol=0.0)
    assert_close(grad, torch.tensor([[expected]], dtype=torch.float64), rtol=1e-12, atol=0.0)
    assert not bool(result.clipped_low.any()) and not bool(result.clipped_high.any())


def test_pg_clip_golden_two_token_ragged_mixed_branches() -> None:
    """2-token ragged PG_CLIP golden mixing all three branches
    (docs/derivations/losses.md). eps = 0.2, TOKEN_MEAN (N = 3, weight 1/3):
    (0,0): r = e^{-0.2-(-0.5)} = e^0.3 = 1.34986 > 1.2, A = 1 -> obj -1.2, grad 0;
    (0,1): r = e^{-1.0-(-0.7)} = e^{-0.3} = 0.74082 < 0.8, A = -1 -> obj 0.8, grad 0;
    (1,0): r = e^{-0.5-(-0.6)} = e^0.1 = 1.10517 inside, A = 2 -> obj -2e^0.1,
           dloss/dlp = -(1/3)*2*e^0.1;
    loss = (-1.2 + 0.8 - 2e^0.1)/3 = -0.87011...; masked (1,1) contributes nothing."""
    config = PolicyLossConfig(
        ratio=RatioKind.TOKEN,
        surrogate=SurrogateKind.PG_CLIP,
        clip=ClipConfig(eps_low=0.2, eps_high=0.2),
        aggregation=Aggregation.TOKEN_MEAN,
    )
    mask = torch.tensor([[True, True], [True, False]])
    leaf = torch.tensor(
        [[-0.2, -1.0], [-0.5, MASKED_JUNK]], dtype=torch.float64, requires_grad=True
    )
    result = policy_loss(
        config,
        logprobs=leaf,
        old_logprobs=torch.tensor([[-0.5, -0.7], [-0.6, MASKED_JUNK]], dtype=torch.float64),
        advantages=torch.tensor([[1.0, -1.0], [2.0, MASKED_JUNK]], dtype=torch.float64),
        response_mask=mask,
    )
    (grad,) = torch.autograd.grad(result.loss, leaf)
    e01 = math.exp(0.1)
    assert_close(
        result.loss,
        torch.tensor((-1.2 + 0.8 - 2.0 * e01) / 3.0, dtype=torch.float64),
        rtol=1e-12,
        atol=0.0,
    )
    assert_close(
        result.per_token_objective,
        torch.tensor([[-1.2, 0.8], [-2.0 * e01, 0.0]], dtype=torch.float64),
        rtol=1e-12,
        atol=0.0,
    )
    assert_close(
        grad,
        torch.tensor([[0.0, 0.0], [-2.0 * e01 / 3.0, 0.0]], dtype=torch.float64),
        rtol=1e-12,
        atol=0.0,
    )
    assert_close(
        result.ratio,
        torch.tensor([[math.exp(0.3), math.exp(-0.3)], [e01, 1.0]], dtype=torch.float64),
        rtol=1e-12,
        atol=0.0,
    )
    assert torch.equal(result.clipped_high, torch.tensor([[True, False], [False, False]]))
    assert torch.equal(result.clipped_low, torch.tensor([[False, True], [False, False]]))
    assert result.kl_loss is None


# --- equivalence and stop-gradient properties ----------------------------------------


@pytest.mark.parametrize("mode", MODES)
@given(batch=loss_batches())
def test_on_policy_pg_clip_pg_reinforce_gradients_coincide(
    mode: Aggregation, batch: LossBatch
) -> None:
    """At old_logprobs == logprobs.detach() (r = 1 everywhere), PG_CLIP, PG, dual-clip
    PG_CLIP, and REINFORCE have bitwise-identical gradients for identical advantages
    (docs/derivations/losses.md, on-policy collapse)."""
    on_policy_old = batch.logprobs.clone()
    grads = [
        _grad_of_loss(make_config(surrogate, RatioKind.TOKEN, mode), batch, on_policy_old)[1]
        for surrogate in (SurrogateKind.PG_CLIP, SurrogateKind.PG, SurrogateKind.REINFORCE)
    ]
    dual = make_config(SurrogateKind.PG_CLIP, RatioKind.TOKEN, mode, ratio_cap=2.0)
    grads.append(_grad_of_loss(dual, batch, on_policy_old)[1])
    for other in grads[1:]:
        assert torch.equal(grads[0], other)


@pytest.mark.parametrize(("surrogate", "ratio"), VALID_COMBOS)
@given(batch=loss_batches())
def test_zero_advantages_give_zero_surrogate_gradient(
    surrogate: SurrogateKind, ratio: RatioKind, batch: LossBatch
) -> None:
    """A == 0 makes the surrogate gradient exactly zero for every valid combination,
    both [B, T] and [B] advantage shapes (docs/derivations/losses.md)."""
    config = make_config(surrogate, ratio)
    for advantages in (
        torch.zeros_like(batch.advantages),
        torch.zeros(batch.response_mask.shape[0], dtype=torch.float64),
    ):
        leaf = batch.logprobs.clone().requires_grad_(True)
        result = policy_loss(
            config,
            logprobs=leaf,
            old_logprobs=batch.old_logprobs,
            advantages=advantages,
            response_mask=batch.response_mask,
        )
        (grad,) = torch.autograd.grad(result.loss, leaf)
        assert torch.equal(grad, torch.zeros_like(grad))


@pytest.mark.parametrize("cispo_eps_low", [None, 0.15])
@pytest.mark.parametrize("ratio", [RatioKind.TOKEN, RatioKind.SEQUENCE])
@given(batch=loss_batches())
def test_cispo_gradient_equals_detached_weight_scaled_reinforce(
    cispo_eps_low: float | None, ratio: RatioKind, batch: LossBatch
) -> None:
    """CISPO == REINFORCE run on advantages pre-scaled by the detached clipped weight
    sg[w] (one-sided min or two-sided clamp), bitwise in loss, per-token objective, and
    gradient (docs/derivations/losses.md, CISPO stop-gradient; arXiv 2506.13585
    eq. 4-5)."""
    mask = batch.response_mask
    config = make_config(SurrogateKind.CISPO, ratio, cispo_eps_low=cispo_eps_low)
    result, grad = _grad_of_loss(config, batch)
    zero = torch.zeros((), dtype=torch.float64)
    log_ratio = torch.where(mask, batch.logprobs - batch.old_logprobs, zero)
    if ratio is RatioKind.TOKEN:
        raw = torch.exp(log_ratio)
    else:
        lengths = mask.sum(dim=1, keepdim=True).to(torch.float64)
        raw = torch.exp(log_ratio.sum(dim=1, keepdim=True) / lengths).expand_as(log_ratio)
    weight = raw.clamp(max=1.3) if cispo_eps_low is None else raw.clamp(1.0 - cispo_eps_low, 1.3)
    scaled_advantages = weight * torch.where(mask, batch.advantages, zero)
    reinforce = make_config(SurrogateKind.REINFORCE, RatioKind.TOKEN)
    scaled_batch = batch._replace(advantages=scaled_advantages)
    expected, expected_grad = _grad_of_loss(reinforce, scaled_batch)
    assert torch.equal(result.loss, expected.loss)
    assert torch.equal(result.per_token_objective, expected.per_token_objective)
    assert torch.equal(grad, expected_grad)


def test_cispo_clipped_masks_report_weight_clipping() -> None:
    """CISPO clip masks report where the weight was clipped, not a branch of a min
    with the advantage (docs/derivations/losses.md, CISPO). r = [e^0.5, e^-0.5, e^0.1]
    = [1.64872, 0.60653, 1.10517]; eps_high = 0.2 -> 1.2. One-sided: w =
    [1.2, e^-0.5, e^0.1], clipped_high only at t0. Two-sided eps_low = 0.1 -> 0.9:
    w = [1.2, 0.9, e^0.1], clipped_low at t1. Objective -w*A*lp checked per token:
    t0: -1.2*1.0*(-0.3) = 0.36; t1 one-sided: -e^-0.5*(-2)*(-1) = -2e^-0.5;
    t1 two-sided: -0.9*(-2)*(-1) = -1.8; t2: -e^0.1*0.5*(-0.9) = 0.45e^0.1."""
    lp = torch.tensor([[-0.3, -1.0, -0.9]], dtype=torch.float64)
    olp = torch.tensor([[-0.8, -0.5, -1.0]], dtype=torch.float64)
    adv = torch.tensor([[1.0, -2.0, 0.5]], dtype=torch.float64)
    mask = torch.ones((1, 3), dtype=torch.bool)
    one_sided = PolicyLossConfig(
        ratio=RatioKind.TOKEN,
        surrogate=SurrogateKind.CISPO,
        clip=ClipConfig(eps_low=None, eps_high=0.2),
        aggregation=Aggregation.TOKEN_MEAN,
    )
    result = policy_loss(
        one_sided, logprobs=lp, old_logprobs=olp, advantages=adv, response_mask=mask
    )
    assert torch.equal(result.clipped_high, torch.tensor([[True, False, False]]))
    assert torch.equal(result.clipped_low, torch.zeros((1, 3), dtype=torch.bool))
    expected = torch.tensor(
        [[0.36, -2.0 * math.exp(-0.5), 0.45 * math.exp(0.1)]], dtype=torch.float64
    )
    assert_close(result.per_token_objective, expected, rtol=1e-12, atol=0.0)
    two_sided = dataclasses.replace(one_sided, clip=ClipConfig(eps_low=0.1, eps_high=0.2))
    result2 = policy_loss(
        two_sided, logprobs=lp, old_logprobs=olp, advantages=adv, response_mask=mask
    )
    assert torch.equal(result2.clipped_low, torch.tensor([[False, True, False]]))
    assert torch.equal(result2.clipped_high, torch.tensor([[True, False, False]]))
    expected2 = torch.tensor([[0.36, -1.8, 0.45 * math.exp(0.1)]], dtype=torch.float64)
    assert_close(result2.per_token_objective, expected2, rtol=1e-12, atol=0.0)


@pytest.mark.parametrize("ratio_cap", [None, 2.0])
@given(batch=loss_batches())
def test_pg_clip_masks_match_autograd_branch(ratio_cap: float | None, batch: LossBatch) -> None:
    """clipped_low | clipped_high is exactly the set of response tokens with A != 0
    where the per-token PG_CLIP gradient is 0 — the constant branch autograd took —
    including the dual-clip cap branch (docs/derivations/losses.md, branch tables)."""
    mask = batch.response_mask
    config = make_config(SurrogateKind.PG_CLIP, RatioKind.TOKEN, ratio_cap=ratio_cap)
    leaf = batch.logprobs.clone().requires_grad_(True)
    result = policy_loss(
        config,
        logprobs=leaf,
        old_logprobs=batch.old_logprobs,
        advantages=batch.advantages,
        response_mask=mask,
    )
    (grad,) = torch.autograd.grad(result.per_token_objective.sum(), leaf)
    active = mask & (torch.where(mask, batch.advantages, torch.zeros(())) != 0)
    killed = grad == 0
    clipped = result.clipped_low | result.clipped_high
    assert torch.equal(clipped[active], killed[active])
    assert not bool(clipped[~mask].any())


# --- GSPO decompositions (sg[] placements; arXiv 2507.18071) --------------------------


@given(batch=loss_batches())
def test_gspo_sequence_ratio_value_matches_masked_mean_exponent(batch: LossBatch) -> None:
    """RatioKind.SEQUENCE uses s_i = exp(sum_t m*(lp-olp) / L_i) broadcast to the row;
    the reported ratio is 1.0 at masked positions and the per-token objective is
    -s_i * A_t (docs/derivations/losses.md, GSPO-seq)."""
    mask = batch.response_mask
    config = make_config(SurrogateKind.PG, RatioKind.SEQUENCE, Aggregation.SEQ_MEAN_TOKEN_MEAN)
    result = policy_loss(
        config,
        logprobs=batch.logprobs,
        old_logprobs=batch.old_logprobs,
        advantages=batch.advantages,
        response_mask=mask,
    )
    zero = torch.zeros((), dtype=torch.float64)
    log_ratio = torch.where(mask, batch.logprobs - batch.old_logprobs, zero)
    lengths = mask.sum(dim=1, keepdim=True).to(torch.float64)
    s = torch.exp(log_ratio.sum(dim=1, keepdim=True) / lengths).expand_as(log_ratio)
    assert torch.equal(result.ratio, torch.where(mask, s, torch.ones_like(s)))
    expected = torch.where(mask, -(s * torch.where(mask, batch.advantages, zero)), zero)
    assert torch.equal(result.per_token_objective, expected)
    assert torch.equal(result.loss, aggregate(expected, mask, Aggregation.SEQ_MEAN_TOKEN_MEAN))


@pytest.mark.parametrize("mode", [Aggregation.SEQ_MEAN_TOKEN_MEAN, Aggregation.TOKEN_MEAN])
@given(batch=loss_batches())
def test_gspo_sequence_gradient_matches_coupled_analytic_formula(
    mode: Aggregation, batch: LossBatch
) -> None:
    """The GSPO-seq gradient couples the whole row through the masked mean:
    dL/dlp_{it} = -(sum_tau w_{i,tau} A_{i,tau}) * s_i * m_{it} / L_i
    (docs/derivations/losses.md, GSPO-seq gradient)."""
    mask = batch.response_mask
    config = make_config(SurrogateKind.PG, RatioKind.SEQUENCE, mode)
    _, grad = _grad_of_loss(config, batch)
    zero = torch.zeros((), dtype=torch.float64)
    log_ratio = torch.where(mask, batch.logprobs - batch.old_logprobs, zero)
    lengths = mask.sum(dim=1, keepdim=True).to(torch.float64)
    s = torch.exp(log_ratio.sum(dim=1, keepdim=True) / lengths)
    weights = effective_token_weights(mask, mode)
    adv = torch.where(mask, batch.advantages, zero)
    row_sum = (weights * adv).sum(dim=1, keepdim=True)
    expected = torch.where(mask, -row_sum * s / lengths, zero)
    assert_close(grad, expected, rtol=1e-12, atol=1e-12)


@given(batch=loss_batches())
def test_gspo_sequence_token_value_equals_sequence_ratio_value(batch: LossBatch) -> None:
    """GSPO-token s_{i,t} = sg[s_i] * r_t / sg[r_t] equals sg[s_i] numerically:
    loss, per-token objective, and reported ratio are bitwise identical to the
    SEQUENCE-ratio config (docs/derivations/losses.md, GSPO-token value)."""
    sequence = make_config(SurrogateKind.PG, RatioKind.SEQUENCE, Aggregation.SEQ_MEAN_TOKEN_MEAN)
    token = make_config(SurrogateKind.PG, RatioKind.SEQUENCE_TOKEN, Aggregation.SEQ_MEAN_TOKEN_MEAN)
    result_seq, _ = _grad_of_loss(sequence, batch)
    result_tok, _ = _grad_of_loss(token, batch)
    assert torch.equal(result_seq.loss, result_tok.loss)
    assert torch.equal(result_seq.per_token_objective, result_tok.per_token_objective)
    assert torch.equal(result_seq.ratio, result_tok.ratio)


@given(batch=loss_batches())
def test_gspo_sequence_token_gradient_is_token_local(batch: LossBatch) -> None:
    """The GSPO-token gradient is token-local: dL/dlp_{it} = -w_{it} * A_{it} * sg[s_i]
    — no coupling through the masked mean (docs/derivations/losses.md, GSPO-token
    gradient)."""
    mask = batch.response_mask
    config = make_config(SurrogateKind.PG, RatioKind.SEQUENCE_TOKEN)
    _, grad = _grad_of_loss(config, batch)
    zero = torch.zeros((), dtype=torch.float64)
    log_ratio = torch.where(mask, batch.logprobs - batch.old_logprobs, zero)
    lengths = mask.sum(dim=1, keepdim=True).to(torch.float64)
    s = torch.exp(log_ratio.sum(dim=1, keepdim=True) / lengths)
    weights = effective_token_weights(mask, Aggregation.TOKEN_MEAN)
    adv = torch.where(mask, batch.advantages, zero)
    expected = torch.where(mask, -(weights * adv * s), zero)
    assert_close(grad, expected, rtol=1e-12, atol=1e-12)


def test_gspo_sequence_and_sequence_token_gradients_differ() -> None:
    """Fixed demonstration that the sg[] placement changes the gradient while values
    agree: gaps [0.2, -0.2] give s = e^0 = 1; A = [1, -2], TOKEN_MEAN (w = 1/2).
    GSPO-seq:   dL/dlp_t = -(1/2*1 + 1/2*(-2))*1/2 = +0.25 at both tokens;
    GSPO-token: dL/dlp_t = -w_t*A_t*s = [-0.5, +1.0]
    (docs/derivations/losses.md, GSPO gradients)."""
    mask = torch.ones((1, 2), dtype=torch.bool)
    lp = torch.tensor([[-0.5, -1.0]], dtype=torch.float64)
    olp = torch.tensor([[-0.7, -0.8]], dtype=torch.float64)
    adv = torch.tensor([[1.0, -2.0]], dtype=torch.float64)

    def grad_for(ratio: RatioKind) -> torch.Tensor:
        leaf = lp.clone().requires_grad_(True)
        result = policy_loss(
            make_config(SurrogateKind.PG, ratio),
            logprobs=leaf,
            old_logprobs=olp,
            advantages=adv,
            response_mask=mask,
        )
        (grad,) = torch.autograd.grad(result.loss, leaf)
        return grad

    assert_close(
        grad_for(RatioKind.SEQUENCE),
        torch.tensor([[0.25, 0.25]], dtype=torch.float64),
        rtol=1e-12,
        atol=1e-12,
    )
    assert_close(
        grad_for(RatioKind.SEQUENCE_TOKEN),
        torch.tensor([[-0.5, 1.0]], dtype=torch.float64),
        rtol=1e-12,
        atol=1e-12,
    )


# --- TIS correction (verl PR #2953) ---------------------------------------------------


@pytest.mark.parametrize("level", ["token", "sequence"])
@given(batch=loss_batches())
def test_is_correction_weight_one_is_noop(level: str, batch: LossBatch) -> None:
    """rollout_logprobs == old_logprobs makes every TIS weight min(e^0, cap) = 1
    (cap >= 1), so loss, per-token objective, and gradient are bitwise identical to the
    uncorrected config (docs/derivations/losses.md, TIS)."""
    correction = ISCorrectionConfig(cap=1.5, level=level)  # type: ignore[arg-type]
    config = make_config(SurrogateKind.PG_CLIP, RatioKind.TOKEN, is_correction=correction)
    plain = make_config(SurrogateKind.PG_CLIP, RatioKind.TOKEN)
    leaf = batch.logprobs.clone().requires_grad_(True)
    corrected = policy_loss(
        config,
        logprobs=leaf,
        old_logprobs=batch.old_logprobs,
        advantages=batch.advantages,
        response_mask=batch.response_mask,
        rollout_logprobs=batch.old_logprobs.clone(),
    )
    (grad,) = torch.autograd.grad(corrected.loss, leaf)
    expected, expected_grad = _grad_of_loss(plain, batch)
    assert torch.equal(corrected.loss, expected.loss)
    assert torch.equal(corrected.per_token_objective, expected.per_token_objective)
    assert torch.equal(grad, expected_grad)


@pytest.mark.parametrize("level", ["token", "sequence"])
@given(batch=loss_batches())
def test_is_correction_cap_binds(level: str, batch: LossBatch) -> None:
    """With old - rollout = 2 at every response token, both levels saturate the cap
    (token: e^2 = 7.389 > 3; sequence: e^{2L} >= e^2 > 3), so every per-token objective
    is exactly cap * the uncorrected objective (docs/derivations/losses.md, TIS)."""
    mask = batch.response_mask
    rollout = torch.where(mask, batch.old_logprobs - 2.0, batch.old_logprobs)
    correction = ISCorrectionConfig(cap=3.0, level=level)  # type: ignore[arg-type]
    config = make_config(SurrogateKind.PG_CLIP, RatioKind.TOKEN, is_correction=correction)
    plain = make_config(SurrogateKind.PG_CLIP, RatioKind.TOKEN)
    corrected = policy_loss(
        config,
        logprobs=batch.logprobs,
        old_logprobs=batch.old_logprobs,
        advantages=batch.advantages,
        response_mask=mask,
        rollout_logprobs=rollout,
    )
    expected, _ = _grad_of_loss(plain, batch)
    assert torch.equal(corrected.per_token_objective, expected.per_token_objective * 3.0)


def test_is_correction_sequence_level_uses_unnormalized_sum() -> None:
    """The sequence-level TIS exponent is the raw masked sum, not a length-normalized
    mean: with old - rollout = 0.5 per token and L = 2, the token weight is
    min(e^0.5, 1.9) = e^0.5 = 1.64872 (uncapped) while the sequence weight is
    min(e^{0.5*2}, 1.9) = min(2.71828, 1.9) = 1.9 (capped). On r = 1, A = 1 the
    uncorrected objective is -1 per token (docs/derivations/losses.md, TIS)."""
    mask = torch.ones((1, 2), dtype=torch.bool)
    lp = torch.tensor([[-0.5, -1.0]], dtype=torch.float64)
    rollout = lp - 0.5
    adv = torch.tensor([[1.0, 1.0]], dtype=torch.float64)

    def objective(level: str) -> torch.Tensor:
        config = make_config(
            SurrogateKind.PG,
            RatioKind.TOKEN,
            is_correction=ISCorrectionConfig(cap=1.9, level=level),  # type: ignore[arg-type]
        )
        return policy_loss(
            config,
            logprobs=lp,
            old_logprobs=lp.clone(),
            advantages=adv,
            response_mask=mask,
            rollout_logprobs=rollout,
        ).per_token_objective

    token_expected = torch.full((1, 2), -math.exp(0.5), dtype=torch.float64)
    assert_close(objective("token"), token_expected, rtol=1e-12, atol=0.0)
    assert torch.equal(objective("sequence"), torch.full((1, 2), -1.9, dtype=torch.float64))


# --- KL composition -------------------------------------------------------------------


@given(batch=loss_batches())
def test_policy_loss_kl_term_composition(batch: LossBatch) -> None:
    """loss = aggregate(per_token_objective) + coef * kl_loss(...), with the result's
    kl_loss field carrying the unscaled KL scalar, and kl_loss None without config.kl
    (docs/derivations/losses.md, KL composition)."""
    mask = batch.response_mask
    config = make_config(
        SurrogateKind.PG_CLIP,
        RatioKind.TOKEN,
        Aggregation.SEQ_MEAN_TOKEN_MEAN,
        kl=KLLossConfig(kind=KLEstimator.K3, coef=0.13),
    )
    result = policy_loss(
        config,
        logprobs=batch.logprobs,
        old_logprobs=batch.old_logprobs,
        advantages=batch.advantages,
        response_mask=mask,
        ref_logprobs=batch.ref_logprobs,
    )
    assert result.kl_loss is not None
    expected_kl = kl_loss(
        batch.logprobs,
        batch.ref_logprobs,
        KLEstimator.K3,
        Aggregation.SEQ_MEAN_TOKEN_MEAN,
        response_mask=mask,
    )
    assert torch.equal(result.kl_loss, expected_kl)
    surrogate = aggregate(result.per_token_objective, mask, Aggregation.SEQ_MEAN_TOKEN_MEAN)
    assert torch.equal(result.loss, surrogate + 0.13 * result.kl_loss)
    plain = policy_loss(
        make_config(SurrogateKind.PG_CLIP, RatioKind.TOKEN, Aggregation.SEQ_MEAN_TOKEN_MEAN),
        logprobs=batch.logprobs,
        old_logprobs=batch.old_logprobs,
        advantages=batch.advantages,
        response_mask=mask,
    )
    assert plain.kl_loss is None
    assert torch.equal(plain.loss, surrogate)


def test_policy_loss_kl_inherits_aggregation_and_norm_len() -> None:
    """KLLossConfig.aggregation/norm_len of None inherit the policy config's values;
    non-None values override them (docs/derivations/losses.md, KL composition)."""
    mask = torch.tensor([[True, True], [True, False]])
    lp = torch.tensor([[-0.5, -1.0], [-0.8, MASKED_JUNK]], dtype=torch.float64)
    olp = lp - 0.1
    ref = lp + 0.2
    adv = torch.tensor([1.0, -0.5], dtype=torch.float64)

    def kl_field(kl: KLLossConfig) -> torch.Tensor:
        config = make_config(
            SurrogateKind.PG_CLIP, RatioKind.TOKEN, Aggregation.TOKEN_SUM_NORM, kl=kl
        )
        result = policy_loss(
            config,
            logprobs=lp,
            old_logprobs=olp,
            advantages=adv,
            response_mask=mask,
            ref_logprobs=ref,
        )
        assert result.kl_loss is not None
        return result.kl_loss

    inherited = kl_field(KLLossConfig(kind=KLEstimator.K2, coef=1.0))
    assert torch.equal(
        inherited,
        kl_loss(
            lp,
            ref,
            KLEstimator.K2,
            Aggregation.TOKEN_SUM_NORM,
            response_mask=mask,
            norm_len=NORM_LEN,
        ),
    )
    overridden = kl_field(
        KLLossConfig(kind=KLEstimator.K2, coef=1.0, aggregation=Aggregation.TOKEN_MEAN)
    )
    assert torch.equal(
        overridden, kl_loss(lp, ref, KLEstimator.K2, Aggregation.TOKEN_MEAN, response_mask=mask)
    )
    own_norm = kl_field(
        KLLossConfig(
            kind=KLEstimator.K2,
            coef=1.0,
            aggregation=Aggregation.TOKEN_SUM_NORM,
            norm_len=3,
        )
    )
    assert torch.equal(
        own_norm,
        kl_loss(
            lp,
            ref,
            KLEstimator.K2,
            Aggregation.TOKEN_SUM_NORM,
            response_mask=mask,
            norm_len=3,
        ),
    )


# --- masking, broadcast, dtype, and inertness properties ------------------------------


@pytest.mark.parametrize(("surrogate", "ratio"), VALID_COMBOS)
@given(batch=loss_batches())
def test_policy_loss_mask_invariance(
    surrogate: SurrogateKind, ratio: RatioKind, batch: LossBatch
) -> None:
    """Perturbing every masked input leaves all outputs bitwise unchanged; masked
    positions are 0 in per_token_objective, 1.0 in ratio, False in clipped_*
    (docs/conventions.md, masked positions)."""
    mask = batch.response_mask
    config = make_config(surrogate, ratio)

    def run(lp: torch.Tensor, olp: torch.Tensor, adv: torch.Tensor) -> PolicyLossResult:
        return policy_loss(
            config, logprobs=lp, old_logprobs=olp, advantages=adv, response_mask=mask
        )

    result = run(batch.logprobs, batch.old_logprobs, batch.advantages)
    perturbed = run(
        torch.where(mask, batch.logprobs, batch.logprobs - 11.5),
        torch.where(mask, batch.old_logprobs, batch.old_logprobs + 4.25),
        torch.where(mask, batch.advantages, batch.advantages - 7.0),
    )
    assert torch.equal(result.loss, perturbed.loss)
    assert torch.equal(result.per_token_objective, perturbed.per_token_objective)
    assert torch.equal(result.ratio, perturbed.ratio)
    assert torch.equal(result.clipped_low, perturbed.clipped_low)
    assert torch.equal(result.clipped_high, perturbed.clipped_high)
    off = ~mask
    assert torch.equal(
        result.per_token_objective[off], torch.zeros_like(result.per_token_objective[off])
    )
    assert bool((result.ratio[off] == 1.0).all())
    assert not bool(result.clipped_low[off].any())
    assert not bool(result.clipped_high[off].any())
    assert torch.equal(result.loss, aggregate(result.per_token_objective, mask, config.aggregation))


@given(batch=loss_batches())
def test_policy_loss_mask_invariance_with_is_correction_and_kl(batch: LossBatch) -> None:
    """Mask invariance extends to the rollout and reference streams of the full
    TIS + KL composition (docs/conventions.md, masked positions)."""
    mask = batch.response_mask
    config = make_config(
        SurrogateKind.PG_CLIP,
        RatioKind.SEQUENCE,
        is_correction=ISCorrectionConfig(cap=2.0, level="sequence"),
        kl=KLLossConfig(kind=KLEstimator.K3, coef=0.1),
    )

    def run(shift: float) -> PolicyLossResult:
        def bump(x: torch.Tensor) -> torch.Tensor:
            return torch.where(mask, x, x + shift)

        return policy_loss(
            config,
            logprobs=bump(batch.logprobs),
            old_logprobs=bump(batch.old_logprobs),
            advantages=bump(batch.advantages),
            response_mask=mask,
            ref_logprobs=bump(batch.ref_logprobs),
            rollout_logprobs=bump(batch.rollout_logprobs),
        )

    result, perturbed = run(0.0), run(-9.5)
    assert result.kl_loss is not None and perturbed.kl_loss is not None
    assert torch.equal(result.loss, perturbed.loss)
    assert torch.equal(result.per_token_objective, perturbed.per_token_objective)
    assert torch.equal(result.ratio, perturbed.ratio)
    assert torch.equal(result.kl_loss, perturbed.kl_loss)


@pytest.mark.parametrize(("surrogate", "ratio"), VALID_COMBOS)
@given(batch=loss_batches())
def test_policy_loss_gradient_is_zero_at_masked_positions(
    surrogate: SurrogateKind, ratio: RatioKind, batch: LossBatch
) -> None:
    """d loss / d logprobs is exactly 0 at masked positions for every valid combination
    (docs/conventions.md, masked positions)."""
    _, grad = _grad_of_loss(make_config(surrogate, ratio), batch)
    off = ~batch.response_mask
    assert torch.equal(grad[off], torch.zeros_like(grad[off]))


@given(batch=loss_batches())
def test_policy_loss_sequence_advantages_broadcast(batch: LossBatch) -> None:
    """[B] advantages are exactly the [B, T] broadcast of themselves
    (docs/derivations/losses.md)."""
    b, t = batch.response_mask.shape
    seq_adv = batch.advantages[:, 0].clone()
    config = make_config(SurrogateKind.PG_CLIP, RatioKind.TOKEN)
    result_seq = policy_loss(
        config,
        logprobs=batch.logprobs,
        old_logprobs=batch.old_logprobs,
        advantages=seq_adv,
        response_mask=batch.response_mask,
    )
    result_full = policy_loss(
        config,
        logprobs=batch.logprobs,
        old_logprobs=batch.old_logprobs,
        advantages=seq_adv.unsqueeze(1).expand(b, t),
        response_mask=batch.response_mask,
    )
    assert torch.equal(result_seq.loss, result_full.loss)
    assert torch.equal(result_seq.per_token_objective, result_full.per_token_objective)


@given(batch=loss_batches())
def test_reinforce_ignores_old_logprobs(batch: LossBatch) -> None:
    """REINFORCE has no ratio: changing old_logprobs leaves every output bitwise
    unchanged and the reported ratio is all-ones (docs/derivations/losses.md)."""
    config = make_config(SurrogateKind.REINFORCE, RatioKind.TOKEN)
    result_a, grad_a = _grad_of_loss(config, batch)
    result_b, grad_b = _grad_of_loss(config, batch, batch.old_logprobs - 1.25)
    assert torch.equal(result_a.loss, result_b.loss)
    assert torch.equal(grad_a, grad_b)
    assert torch.equal(result_a.ratio, torch.ones_like(batch.logprobs))
    assert not bool(result_a.clipped_low.any()) and not bool(result_a.clipped_high.any())


def test_policy_loss_preserves_input_dtype() -> None:
    """policy_loss returns the input dtype with no silent casts
    (docs/conventions.md, dtypes)."""
    for dtype in (torch.float32, torch.float64):
        lp = torch.tensor([[-0.5, -1.0]], dtype=dtype)
        olp = torch.tensor([[-0.6, -0.8]], dtype=dtype)
        adv = torch.tensor([[1.0, -1.0]], dtype=dtype)
        mask = torch.ones((1, 2), dtype=torch.bool)
        result = policy_loss(
            make_config(SurrogateKind.PG_CLIP, RatioKind.TOKEN),
            logprobs=lp,
            old_logprobs=olp,
            advantages=adv,
            response_mask=mask,
        )
        assert result.loss.dtype == dtype
        assert result.per_token_objective.dtype == dtype
        assert result.ratio.dtype == dtype


def test_configs_are_inert_frozen_data() -> None:
    """The config and result dataclasses are frozen; constructing a PG_CLIP config
    with clip=None is legal — validation fires at policy_loss entry
    (docs/derivations/losses.md)."""
    config = PolicyLossConfig(
        ratio=RatioKind.TOKEN,
        surrogate=SurrogateKind.PG_CLIP,
        clip=None,
        aggregation=Aggregation.TOKEN_MEAN,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.norm_len = 3  # type: ignore[misc]
    clip = ClipConfig(eps_low=0.2, eps_high=0.2)
    with pytest.raises(dataclasses.FrozenInstanceError):
        clip.eps_low = 0.3  # type: ignore[misc]
    correction = ISCorrectionConfig(cap=2.0)
    assert correction.level == "token"
    with pytest.raises(dataclasses.FrozenInstanceError):
        correction.cap = 1.0  # type: ignore[misc]


# --- validation errors ----------------------------------------------------------------


def _call(config: PolicyLossConfig, **overrides: torch.Tensor | None) -> PolicyLossResult:
    mask = torch.tensor([[True, True], [True, False]])
    inputs: dict[str, torch.Tensor | None] = {
        "logprobs": torch.tensor([[-0.5, -1.0], [-0.8, MASKED_JUNK]], dtype=torch.float64),
        "old_logprobs": torch.tensor([[-0.6, -0.8], [-0.7, MASKED_JUNK]], dtype=torch.float64),
        "advantages": torch.tensor([[1.0, -1.0], [0.5, MASKED_JUNK]], dtype=torch.float64),
        "response_mask": mask,
    }
    inputs.update(overrides)
    return policy_loss(config, **inputs)  # type: ignore[arg-type]


def test_policy_loss_config_validation_errors() -> None:
    """Every surrogate/clip compatibility rule (docs/derivations/losses.md) raises
    ValueError at policy_loss entry."""
    base = dict(ratio=RatioKind.TOKEN, aggregation=Aggregation.TOKEN_MEAN)
    pg_clip_bad = [
        None,
        ClipConfig(eps_low=None, eps_high=0.2),
        ClipConfig(eps_low=0.2, eps_high=None),
    ]
    for clip in pg_clip_bad:
        with pytest.raises(ValueError, match="PG_CLIP requires"):
            _call(PolicyLossConfig(surrogate=SurrogateKind.PG_CLIP, clip=clip, **base))
    for cap in (1.0, 0.5, float("inf")):
        with pytest.raises(ValueError, match="ratio_cap must be"):
            _call(
                PolicyLossConfig(
                    surrogate=SurrogateKind.PG_CLIP,
                    clip=ClipConfig(eps_low=0.2, eps_high=0.2, ratio_cap=cap),
                    **base,
                )
            )
    for clip in (None, ClipConfig(eps_low=0.2, eps_high=None)):
        with pytest.raises(ValueError, match="CISPO requires"):
            _call(PolicyLossConfig(surrogate=SurrogateKind.CISPO, clip=clip, **base))
    with pytest.raises(ValueError, match="CISPO does not support dual-clip"):
        _call(
            PolicyLossConfig(
                surrogate=SurrogateKind.CISPO,
                clip=ClipConfig(eps_low=None, eps_high=0.2, ratio_cap=2.0),
                **base,
            )
        )
    for surrogate in (SurrogateKind.PG, SurrogateKind.REINFORCE):
        with pytest.raises(ValueError, match="requires clip=None"):
            _call(
                PolicyLossConfig(
                    surrogate=surrogate, clip=ClipConfig(eps_low=0.2, eps_high=0.2), **base
                )
            )
    for ratio in (RatioKind.SEQUENCE, RatioKind.SEQUENCE_TOKEN):
        with pytest.raises(ValueError, match=r"REINFORCE requires ratio=RatioKind\.TOKEN"):
            _call(
                PolicyLossConfig(
                    ratio=ratio,
                    surrogate=SurrogateKind.REINFORCE,
                    clip=None,
                    aggregation=Aggregation.TOKEN_MEAN,
                )
            )
    with pytest.raises(ValueError, match="eps_high must be a positive finite float"):
        _call(
            PolicyLossConfig(
                surrogate=SurrogateKind.PG_CLIP,
                clip=ClipConfig(eps_low=0.2, eps_high=0.0),
                **base,
            )
        )
    with pytest.raises(ValueError, match="eps_low must be a positive finite float"):
        _call(
            PolicyLossConfig(
                surrogate=SurrogateKind.PG_CLIP,
                clip=ClipConfig(eps_low=-0.1, eps_high=0.2),
                **base,
            )
        )
    for cap in (0.0, -1.0, float("inf")):
        with pytest.raises(ValueError, match="cap must be a positive finite float"):
            _call(
                make_config(
                    SurrogateKind.PG,
                    RatioKind.TOKEN,
                    is_correction=ISCorrectionConfig(cap=cap),
                )
            )
    with pytest.raises(ValueError, match="level must be 'token' or 'sequence'"):
        _call(
            make_config(
                SurrogateKind.PG,
                RatioKind.TOKEN,
                is_correction=ISCorrectionConfig(cap=2.0, level="row"),  # type: ignore[arg-type]
            )
        )


def test_policy_loss_call_time_validation_errors() -> None:
    """Missing call-time tensors and shape/mask/finiteness violations raise ValueError
    naming the argument (docs/conventions.md, errors)."""
    config = make_config(SurrogateKind.PG_CLIP, RatioKind.TOKEN)
    with pytest.raises(ValueError, match="is_correction is set but rollout_logprobs is None"):
        _call(
            make_config(
                SurrogateKind.PG_CLIP,
                RatioKind.TOKEN,
                is_correction=ISCorrectionConfig(cap=2.0),
            )
        )
    with pytest.raises(ValueError, match="kl is set but ref_logprobs is None"):
        _call(
            make_config(
                SurrogateKind.PG_CLIP,
                RatioKind.TOKEN,
                kl=KLLossConfig(kind=KLEstimator.K1, coef=0.1),
            )
        )
    with pytest.raises(ValueError, match=r"logprobs must be 2-D"):
        _call(config, logprobs=torch.tensor([-0.5], dtype=torch.float64))
    with pytest.raises(ValueError, match=r"logprobs and old_logprobs"):
        _call(config, old_logprobs=torch.zeros((2, 3), dtype=torch.float64))
    with pytest.raises(ValueError, match=r"dtype torch\.bool"):
        _call(config, response_mask=torch.ones((2, 2)))
    with pytest.raises(ValueError, match=r"zero response tokens"):
        _call(config, response_mask=torch.tensor([[True, True], [False, False]]))
    with pytest.raises(ValueError, match=r"advantages \[B\] must have B = 2"):
        _call(config, advantages=torch.zeros(3, dtype=torch.float64))
    with pytest.raises(ValueError, match=r"advantages and logprobs"):
        _call(config, advantages=torch.zeros((2, 3), dtype=torch.float64))
    with pytest.raises(ValueError, match=r"advantages must be \[B\] or \[B, T\]"):
        _call(config, advantages=torch.zeros((2, 2, 1), dtype=torch.float64))
    with pytest.raises(ValueError, match="logprobs contains non-finite"):
        _call(
            config,
            logprobs=torch.tensor([[float("nan"), -1.0], [-0.8, 0.0]], dtype=torch.float64),
        )
    with pytest.raises(ValueError, match="advantages contains non-finite"):
        _call(
            config,
            advantages=torch.tensor([[1.0, float("inf")], [0.5, 0.0]], dtype=torch.float64),
        )
    with pytest.raises(ValueError, match=r"logprobs and rollout_logprobs"):
        _call(
            make_config(
                SurrogateKind.PG_CLIP,
                RatioKind.TOKEN,
                is_correction=ISCorrectionConfig(cap=2.0),
            ),
            rollout_logprobs=torch.zeros((2, 3), dtype=torch.float64),
        )
    with pytest.raises(ValueError, match=r"logprobs and ref_logprobs"):
        _call(
            make_config(
                SurrogateKind.PG_CLIP,
                RatioKind.TOKEN,
                kl=KLLossConfig(kind=KLEstimator.K1, coef=0.1),
            ),
            ref_logprobs=torch.zeros((2, 3), dtype=torch.float64),
        )
    with pytest.raises(ValueError, match="norm_len is required"):
        _call(
            PolicyLossConfig(
                ratio=RatioKind.TOKEN,
                surrogate=SurrogateKind.PG,
                clip=None,
                aggregation=Aggregation.TOKEN_SUM_NORM,
            )
        )


def test_policy_loss_tolerates_non_finite_junk_at_masked_positions() -> None:
    """Non-finite values confined to masked positions neither raise nor perturb any
    output (mask invariance extends to validation; docs/conventions.md)."""
    config = make_config(SurrogateKind.PG_CLIP, RatioKind.SEQUENCE)
    mask = torch.tensor([[True, False]])
    lp = torch.tensor([[-0.5, float("nan")]], dtype=torch.float64)
    clean = torch.tensor([[-0.5, -1.0]], dtype=torch.float64)
    olp = torch.tensor([[-0.6, float("inf")]], dtype=torch.float64)
    clean_olp = torch.tensor([[-0.6, -0.9]], dtype=torch.float64)
    adv = torch.tensor([1.0], dtype=torch.float64)
    junky = policy_loss(config, logprobs=lp, old_logprobs=olp, advantages=adv, response_mask=mask)
    tidy = policy_loss(
        config, logprobs=clean, old_logprobs=clean_olp, advantages=adv, response_mask=mask
    )
    assert torch.equal(junky.loss, tidy.loss)
    assert torch.equal(junky.ratio, tidy.ratio)


# --- value loss -----------------------------------------------------------------------


def test_value_loss_golden_clip_branches() -> None:
    """Hand-derived clipped value loss (docs/derivations/losses.md, value loss).
    v = [1.5, 0.5], v_old = 1, R = 0, eps = 0.2, band [0.8, 1.2]:
    t0: clip(1.5) = 1.2; max(0.5*1.5^2, 0.5*1.2^2) = max(1.125, 0.72) = 1.125
        (unclipped branch; gradient w*(v-R) = 0.5*1.5 = 0.75);
    t1: clip(0.5) = 0.8; max(0.5*0.25, 0.5*0.64) = 0.32 (clipped branch; gradient 0);
    TOKEN_MEAN loss = (1.125 + 0.32)/2 = 0.7225; clipped_frac = 1/2."""
    values = torch.tensor([[1.5, 0.5]], dtype=torch.float64, requires_grad=True)
    old_values = torch.tensor([[1.0, 1.0]], dtype=torch.float64)
    returns = torch.zeros((1, 2), dtype=torch.float64)
    mask = torch.ones((1, 2), dtype=torch.bool)
    result = value_loss(
        values, old_values, returns, mask, clip_eps=0.2, aggregation=Aggregation.TOKEN_MEAN
    )
    assert torch.equal(result.loss, torch.tensor(0.7225, dtype=torch.float64))
    assert result.clipped_frac == 0.5
    (grad,) = torch.autograd.grad(result.loss, values)
    assert torch.equal(grad, torch.tensor([[0.75, 0.0]], dtype=torch.float64))


def test_value_loss_golden_unclipped() -> None:
    """Hand-derived unclipped value loss: 0.5*[(1.5)^2 + (0.5)^2]/2 = (1.125 + 0.125)/2
    = 0.625; gradients w*(v-R) = [0.75, 0.25]; clipped_frac = 0
    (docs/derivations/losses.md, value loss)."""
    values = torch.tensor([[1.5, 0.5]], dtype=torch.float64, requires_grad=True)
    old_values = torch.tensor([[1.0, 1.0]], dtype=torch.float64)
    returns = torch.zeros((1, 2), dtype=torch.float64)
    mask = torch.ones((1, 2), dtype=torch.bool)
    result = value_loss(
        values, old_values, returns, mask, clip_eps=None, aggregation=Aggregation.TOKEN_MEAN
    )
    assert torch.equal(result.loss, torch.tensor(0.625, dtype=torch.float64))
    assert result.clipped_frac == 0.0
    (grad,) = torch.autograd.grad(result.loss, values)
    assert torch.equal(grad, torch.tensor([[0.75, 0.25]], dtype=torch.float64))


V_MASK = torch.tensor([[True, True, True], [True, True, False]])
V_VALUES = torch.tensor([[0.4, -0.3, 1.2], [0.9, -1.1, MASKED_JUNK]], dtype=torch.float64)
V_OLD = torch.tensor([[0.2, 0.1, 1.0], [0.5, -0.9, MASKED_JUNK]], dtype=torch.float64)
V_RETURNS = torch.tensor([[0.0, 0.5, 0.8], [1.5, -0.5, MASKED_JUNK]], dtype=torch.float64)


@pytest.mark.parametrize("clip_eps", [None, 0.3])
@pytest.mark.parametrize("mode", MODES)
def test_fp64_gradcheck_value_loss(clip_eps: float | None, mode: Aggregation) -> None:
    """torch.autograd.gradcheck of value_loss on a ragged fp64 batch, clipped and
    unclipped, every aggregation. With eps = 0.3 the only |v - v_old| > eps tokens are
    (0,1) (unclipped branch wins: 0.64 > 0.49) and (1,0) (clipped branch wins:
    0.49 > 0.36), both far from the branch tie (docs/derivations/losses.md)."""
    leaf = V_VALUES.clone().requires_grad_(True)

    def fn(v: torch.Tensor) -> torch.Tensor:
        return value_loss(
            v,
            V_OLD,
            V_RETURNS,
            V_MASK,
            clip_eps=clip_eps,
            aggregation=mode,
            norm_len=_norm_len(mode),
        ).loss

    assert torch.autograd.gradcheck(fn, (leaf,))


def test_value_loss_clipped_frac_counts_clipped_branch_tokens() -> None:
    """clipped_frac counts response tokens where the clipped squared error strictly
    exceeds the unclipped one: exactly token (1,0) out of 5 here, and the gradient is
    0 exactly there (docs/derivations/losses.md, value loss)."""
    leaf = V_VALUES.clone().requires_grad_(True)
    result = value_loss(
        leaf, V_OLD, V_RETURNS, V_MASK, clip_eps=0.3, aggregation=Aggregation.TOKEN_MEAN
    )
    assert result.clipped_frac == pytest.approx(0.2)
    (grad,) = torch.autograd.grad(result.loss, leaf)
    assert grad[1, 0].item() == 0.0
    assert grad[0, 0].item() != 0.0


def test_value_loss_wide_clip_equals_unclipped_bitwise() -> None:
    """A clip band wider than any |v - v_old| reproduces the unclipped loss bitwise:
    clamp returns v exactly inside the band (docs/derivations/losses.md)."""
    wide = value_loss(
        V_VALUES, V_OLD, V_RETURNS, V_MASK, clip_eps=100.0, aggregation=Aggregation.TOKEN_MEAN
    )
    plain = value_loss(
        V_VALUES, V_OLD, V_RETURNS, V_MASK, clip_eps=None, aggregation=Aggregation.TOKEN_MEAN
    )
    assert torch.equal(wide.loss, plain.loss)
    assert wide.clipped_frac == 0.0


@st.composite
def value_batches(draw: st.DrawFn) -> tuple[torch.Tensor, ...]:
    """[B, T] float64 (values, old_values, returns, mask) with junk at masked slots."""
    mask = draw(padded_masks(max_b=6, max_t=8))
    b, t = mask.shape

    def fill() -> torch.Tensor:
        vals = [
            draw(st.floats(-3.0, 3.0, allow_nan=False, allow_infinity=False, width=32))
            for _ in range(b * t)
        ]
        x = torch.tensor(vals, dtype=torch.float64).reshape(b, t)
        return torch.where(mask, x, torch.full_like(x, MASKED_JUNK))

    return fill(), fill(), fill(), mask


@pytest.mark.parametrize("clip_eps", [None, 0.3])
@given(data=value_batches())
def test_value_loss_mask_invariance(clip_eps: float | None, data: tuple[torch.Tensor, ...]) -> None:
    """Perturbing masked values/old_values/returns leaves loss and clipped_frac
    bitwise unchanged (docs/conventions.md, masked positions)."""
    values, old_values, returns, mask = data
    result = value_loss(
        values, old_values, returns, mask, clip_eps=clip_eps, aggregation=Aggregation.TOKEN_MEAN
    )
    perturbed = value_loss(
        torch.where(mask, values, values + 3.5),
        torch.where(mask, old_values, old_values - 2.0),
        torch.where(mask, returns, returns + 8.25),
        mask,
        clip_eps=clip_eps,
        aggregation=Aggregation.TOKEN_MEAN,
    )
    assert torch.equal(result.loss, perturbed.loss)
    assert result.clipped_frac == perturbed.clipped_frac


def test_value_loss_validation_errors() -> None:
    """Shape/mask/finiteness/clip_eps violations raise ValueError naming the argument;
    TOKEN_SUM_NORM without norm_len raises at call time
    (docs/conventions.md, errors)."""
    good = torch.zeros((1, 2), dtype=torch.float64)
    mask = torch.ones((1, 2), dtype=torch.bool)
    with pytest.raises(ValueError, match=r"values must be 2-D"):
        value_loss(
            torch.zeros(2, dtype=torch.float64),
            good,
            good,
            mask,
            clip_eps=None,
            aggregation=Aggregation.TOKEN_MEAN,
        )
    with pytest.raises(ValueError, match=r"values and old_values"):
        value_loss(
            good,
            torch.zeros((1, 3), dtype=torch.float64),
            good,
            mask,
            clip_eps=None,
            aggregation=Aggregation.TOKEN_MEAN,
        )
    with pytest.raises(ValueError, match=r"values and returns"):
        value_loss(
            good,
            good,
            torch.zeros((1, 3), dtype=torch.float64),
            mask,
            clip_eps=None,
            aggregation=Aggregation.TOKEN_MEAN,
        )
    with pytest.raises(ValueError, match=r"dtype torch\.bool"):
        value_loss(
            good,
            good,
            good,
            torch.ones((1, 2)),
            clip_eps=None,
            aggregation=Aggregation.TOKEN_MEAN,
        )
    with pytest.raises(ValueError, match="returns contains non-finite"):
        value_loss(
            good,
            good,
            torch.tensor([[1.0, float("nan")]], dtype=torch.float64),
            mask,
            clip_eps=None,
            aggregation=Aggregation.TOKEN_MEAN,
        )
    for eps in (0.0, -0.5, float("inf")):
        with pytest.raises(ValueError, match="clip_eps must be a positive finite float"):
            value_loss(good, good, good, mask, clip_eps=eps, aggregation=Aggregation.TOKEN_MEAN)
    with pytest.raises(ValueError, match="norm_len is required"):
        value_loss(good, good, good, mask, clip_eps=None, aggregation=Aggregation.TOKEN_SUM_NORM)


def test_value_loss_token_sum_norm_uses_norm_len() -> None:
    """TOKEN_SUM_NORM divides the per-token sum by B * norm_len: golden batch sum
    1.125 + 0.32 = 1.445 over B = 1, norm_len = 4 gives 0.361250
    (docs/derivations/losses.md, value loss)."""
    values = torch.tensor([[1.5, 0.5]], dtype=torch.float64)
    old_values = torch.tensor([[1.0, 1.0]], dtype=torch.float64)
    returns = torch.zeros((1, 2), dtype=torch.float64)
    mask = torch.ones((1, 2), dtype=torch.bool)
    result = value_loss(
        values,
        old_values,
        returns,
        mask,
        clip_eps=0.2,
        aggregation=Aggregation.TOKEN_SUM_NORM,
        norm_len=4,
    )
    assert torch.equal(result.loss, torch.tensor(1.445 / 4.0, dtype=torch.float64))


def test_value_loss_preserves_input_dtype_and_result_is_frozen() -> None:
    """value_loss preserves the input dtype and returns frozen data
    (docs/conventions.md, dtypes)."""
    mask = torch.ones((1, 2), dtype=torch.bool)
    for dtype in (torch.float32, torch.float64):
        v = torch.tensor([[1.5, 0.5]], dtype=dtype)
        result = value_loss(
            v,
            torch.ones((1, 2), dtype=dtype),
            torch.zeros((1, 2), dtype=dtype),
            mask,
            clip_eps=0.2,
            aggregation=Aggregation.TOKEN_MEAN,
        )
        assert result.loss.dtype == dtype
        assert isinstance(result, ValueLossResult)
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.clipped_frac = 1.0  # type: ignore[misc]
