"""Cross-module test obligations.

Each obligation ties at least two polgrad modules together end to end:

1. GRPO-style ``SEQ_MEAN_TOKEN_MEAN`` vs Dr.GRPO-style ``TOKEN_SUM_NORM`` aggregation
   collapse on equal-length batches, and the ``effective_token_weights`` prediction of
   their divergence on ragged batches (docs/derivations/variants.md).
2. k2-as-loss gradient == ``reverse_kl_grad_surrogate`` gradient, exactly, under every
   aggregation (docs/derivations/kl.md).
3. ``clip_report.gradient_killed_mask`` == the exact-zero set of the ``policy_loss``
   autograd gradient (docs/diagnostics/clipping.md).
4. ``microbatch_token_weights`` == autograd weights of an explicit micro-batched
   ``policy_loss`` loop (docs/derivations/aggregation.md).
5. The ``rloo`` registry entry == a manually composed leave-one-out REINFORCE
   surrogate (docs/derivations/variants.md).
6. End-to-end 8-arm softmax-bandit training with every ``ALGORITHMS`` entry
   (docs/derivations/goldens.md).
"""

from __future__ import annotations

import dataclasses
from typing import NamedTuple

import pytest
import torch
from hypothesis import assume, given
from hypothesis import strategies as st
from strategies import MASKED_JUNK, padded_masks
from torch.testing import assert_close

from polgrad.advantages import (
    GAEConfig,
    GroupNormConfig,
    ReinforcePPConfig,
    gae,
    grpo_advantages,
    reinforce_pp_advantages,
    rloo_advantages,
)
from polgrad.aggregate import Aggregation, effective_token_weights, microbatch_token_weights
from polgrad.diagnostics.clipping import clip_report
from polgrad.kl import KLEstimator, kl_in_reward, kl_loss, reverse_kl_grad_surrogate
from polgrad.losses import (
    ClipConfig,
    PolicyLossConfig,
    PolicyLossResult,
    RatioKind,
    SurrogateKind,
    policy_loss,
)
from polgrad.registry import ALGORITHMS, AlgorithmSpec, get
from polgrad.verify.goldens import BanditBatch, SoftmaxBandit

MODES = tuple(Aggregation)
NORM_LEN = 7


def _norm_len(mode: Aggregation) -> int | None:
    return NORM_LEN if mode is Aggregation.TOKEN_SUM_NORM else None


def _masked_fill(
    draw: st.DrawFn, mask: torch.Tensor, low: float, high: float, junk: float = MASKED_JUNK
) -> torch.Tensor:
    """[B, T] float64 draws in [low, high] with ``junk`` at masked positions."""
    b, t = mask.shape
    vals = [
        draw(st.floats(low, high, allow_nan=False, allow_infinity=False, width=32))
        for _ in range(b * t)
    ]
    out = torch.tensor(vals, dtype=torch.float64).reshape(b, t)
    return torch.where(mask, out, torch.full_like(out, junk))


def _seq_advantages(draw: st.DrawFn, b: int) -> torch.Tensor:
    vals = [
        draw(st.floats(-3.0, 3.0, allow_nan=False, allow_infinity=False, width=32))
        for _ in range(b)
    ]
    return torch.tensor(vals, dtype=torch.float64)


class CrossBatch(NamedTuple):
    """Local [B, T] float64 batch with the streams the cross obligations need."""

    logprobs: torch.Tensor
    old_logprobs: torch.Tensor
    ref_logprobs: torch.Tensor
    advantages: torch.Tensor
    response_mask: torch.Tensor


@st.composite
def cross_batches(draw: st.DrawFn) -> CrossBatch:
    """Like strategies.logprob_batches but a 5-field CrossBatch without a rollout
    stream, which the cross obligations do not use. Masked positions hold
    MASKED_JUNK."""
    mask = draw(padded_masks())
    logprobs = _masked_fill(draw, mask, -8.0, -0.0625)
    junk = torch.full_like(logprobs, MASKED_JUNK)
    old = torch.where(mask, logprobs + _masked_fill(draw, mask, -2.0, 2.0, junk=0.0), junk)
    ref = torch.where(mask, logprobs + _masked_fill(draw, mask, -2.0, 2.0, junk=0.0), junk)
    advantages = _masked_fill(draw, mask, -3.0, 3.0)
    return CrossBatch(logprobs, old, ref, advantages, mask)


# ---------------------------------------------------------------------------
# Obligation 1: aggregation collapse, GRPO-style vs Dr.GRPO-style
# ---------------------------------------------------------------------------


def _grpo_style_config() -> PolicyLossConfig:
    """The grpo registry loss with the KL term stripped (surrogate aggregation only)."""
    return dataclasses.replace(get("grpo").loss, kl=None)


def _dr_grpo_style_config(norm_len: int) -> PolicyLossConfig:
    return dataclasses.replace(get("dr_grpo").loss, norm_len=norm_len)


class EqualLengthBatch(NamedTuple):
    logprobs: torch.Tensor
    old_logprobs: torch.Tensor
    advantages: torch.Tensor
    response_mask: torch.Tensor
    length: int


@st.composite
def equal_length_policy_batches(draw: st.DrawFn) -> EqualLengthBatch:
    """Batches where every row has exactly ``length`` response tokens, right-padded."""
    b = draw(st.integers(1, 6))
    length = draw(st.integers(1, 6))
    pad = draw(st.integers(0, 3))
    mask = torch.zeros((b, length + pad), dtype=torch.bool)
    mask[:, :length] = True
    logprobs = _masked_fill(draw, mask, -8.0, -0.0625)
    gap = _masked_fill(draw, mask, -2.0, 2.0, junk=0.0)
    old_logprobs = torch.where(mask, logprobs + gap, torch.full_like(logprobs, MASKED_JUNK))
    return EqualLengthBatch(logprobs, old_logprobs, _seq_advantages(draw, b), mask, length)


def test_equal_length_grpo_loss_is_dr_grpo_loss_times_norm_len_over_length_exactly() -> None:
    """Cross-module obligation 1: with identical advantages fed to both configs
    (advantage normalization bypassed), the grpo-style SEQ_MEAN_TOKEN_MEAN loss on an
    equal-length batch equals the dr_grpo-style TOKEN_SUM_NORM loss times norm_len/L.
    Here L = 4 and norm_len = 8, so the factor 2 and both weight sets (1/8 and 1/16)
    are powers of two and the identity holds bitwise (docs/derivations/variants.md)."""
    f64 = torch.float64
    junk = MASKED_JUNK
    mask = torch.tensor([[True] * 4 + [False], [True] * 4 + [False]])
    logprobs = torch.tensor(
        [[-0.5, -1.0, -0.25, -2.0, junk], [-1.5, -0.75, -3.0, -0.125, junk]], dtype=f64
    )
    old_logprobs = torch.tensor(
        [[-0.9, -0.8, -0.5, -1.5, junk], [-1.0, -1.25, -2.5, -0.25, junk]], dtype=f64
    )
    advantages = torch.tensor([1.5, -2.0], dtype=f64)
    kwargs = {
        "logprobs": logprobs,
        "old_logprobs": old_logprobs,
        "advantages": advantages,
        "response_mask": mask,
    }
    loss_grpo = policy_loss(_grpo_style_config(), **kwargs).loss
    loss_dr = policy_loss(_dr_grpo_style_config(norm_len=8), **kwargs).loss
    assert torch.equal(loss_grpo, 2.0 * loss_dr)


@given(batch=equal_length_policy_batches(), norm_len=st.integers(1, 9))
def test_equal_length_grpo_dr_grpo_collapse_property(
    batch: EqualLengthBatch, norm_len: int
) -> None:
    """Cross-module obligation 1: on every generated equal-length batch, with
    identical advantages fed to both registry-derived configs, grpo-style loss ==
    dr_grpo-style loss * (norm_len / L) (docs/derivations/variants.md)."""
    kwargs = {
        "logprobs": batch.logprobs,
        "old_logprobs": batch.old_logprobs,
        "advantages": batch.advantages,
        "response_mask": batch.response_mask,
    }
    loss_grpo = policy_loss(_grpo_style_config(), **kwargs).loss
    loss_dr = policy_loss(_dr_grpo_style_config(norm_len=norm_len), **kwargs).loss
    assert_close(loss_grpo, (norm_len / batch.length) * loss_dr, rtol=1e-12, atol=1e-12)


def test_ragged_grpo_dr_grpo_diverge_and_weight_ratio_matches_effective_token_weights() -> None:
    """Cross-module obligation 1: on a ragged batch the two losses differ (no
    single equal-length factor relates them), and the per-sequence weight ratio matches
    the effective_token_weights prediction w_grpo/w_dr = norm_len/L_i (per-token weight
    1/(B*L_i) vs 1/(B*norm_len)); each loss reconstructs bitwise from its weights and
    the shared per_token_objective. Hand arithmetic (docs/derivations/variants.md):
    row 0 (L=1): r = e^0.4 > 1.2, A = 1 -> surrogate -1.2; row 1 (L=3): r = 1,
    A = [1, -1, 2] -> surrogate [-1, 1, -2]; grpo loss = ((-1.2)/1 + (-2)/3)/2
    = -14/15; dr_grpo loss (norm_len=2) = (-1.2 - 2)/(2*2) = -0.8."""
    f64 = torch.float64
    junk = MASKED_JUNK
    norm_len = 2
    mask = torch.tensor([[True, False, False], [True, True, True]])
    logprobs = torch.tensor([[-0.5, junk, junk], [-1.0, -0.5, -2.0]], dtype=f64)
    old_logprobs = torch.tensor([[-0.9, junk, junk], [-1.0, -0.5, -2.0]], dtype=f64)
    advantages = torch.tensor([[1.0, junk, junk], [1.0, -1.0, 2.0]], dtype=f64)
    kwargs = {
        "logprobs": logprobs,
        "old_logprobs": old_logprobs,
        "advantages": advantages,
        "response_mask": mask,
    }
    result_grpo = policy_loss(_grpo_style_config(), **kwargs)
    result_dr = policy_loss(_dr_grpo_style_config(norm_len=norm_len), **kwargs)
    assert_close(result_grpo.loss, torch.tensor(-14.0 / 15.0, dtype=f64), rtol=1e-12, atol=0.0)
    assert_close(result_dr.loss, torch.tensor(-0.8, dtype=f64), rtol=1e-12, atol=0.0)

    lengths = mask.sum(dim=1).to(f64)
    for length in lengths.tolist():
        factor = norm_len / length
        assert not torch.isclose(result_grpo.loss, factor * result_dr.loss)

    w_grpo = effective_token_weights(mask, Aggregation.SEQ_MEAN_TOKEN_MEAN)
    w_dr = effective_token_weights(mask, Aggregation.TOKEN_SUM_NORM, norm_len=norm_len)
    predicted_ratio = (norm_len / lengths).unsqueeze(1).expand_as(w_grpo)
    assert_close(w_grpo[mask] / w_dr[mask], predicted_ratio[mask], rtol=1e-12, atol=0.0)

    assert torch.equal(result_grpo.loss, (w_grpo * result_grpo.per_token_objective).sum())
    assert torch.equal(result_dr.loss, (w_dr * result_dr.per_token_objective).sum())


# ---------------------------------------------------------------------------
# Obligation 2: k2-as-loss gradient == reverse-KL surrogate gradient
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("aggregation", MODES)
@given(batch=cross_batches())
def test_k2_as_loss_gradient_equals_reverse_kl_grad_surrogate_gradient(
    aggregation: Aggregation, batch: CrossBatch
) -> None:
    """Cross-module obligation 2: the autograd gradient of kl_loss with k2 is
    bitwise equal to the autograd gradient of reverse_kl_grad_surrogate under every
    aggregation mode: both are w * (logprobs - ref_logprobs) per token
    (docs/derivations/kl.md)."""
    norm_len = _norm_len(aggregation)
    leaf_k2 = batch.logprobs.clone().requires_grad_(True)
    loss_k2 = kl_loss(
        leaf_k2,
        batch.ref_logprobs,
        KLEstimator.K2,
        aggregation,
        response_mask=batch.response_mask,
        norm_len=norm_len,
    )
    (grad_k2,) = torch.autograd.grad(loss_k2, leaf_k2)
    leaf_surrogate = batch.logprobs.clone().requires_grad_(True)
    surrogate = reverse_kl_grad_surrogate(
        leaf_surrogate,
        batch.ref_logprobs,
        aggregation,
        response_mask=batch.response_mask,
        norm_len=norm_len,
    )
    (grad_surrogate,) = torch.autograd.grad(surrogate, leaf_surrogate)
    assert torch.equal(grad_k2, grad_surrogate)


# ---------------------------------------------------------------------------
# Obligation 3: gradient_killed_mask == autograd zero set of policy_loss
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ratio_cap", [None, 1.5])
@given(batch=cross_batches())
def test_gradient_killed_mask_equals_policy_loss_autograd_zero_set(
    ratio_cap: float | None, batch: CrossBatch
) -> None:
    """Cross-module obligation 3: clip_report.gradient_killed_mask ==
    (per-token autograd gradient of the PG_CLIP policy_loss == 0) & response_mask on
    generated inputs, with and without dual-clip. Under TOKEN_MEAN every response
    token carries weight 1/N > 0, so the aggregate gradient at a token is 0 iff its
    per-token PG_CLIP gradient is 0. Tie points (ratio exactly on a clip bound or the
    dual-clip cap) are excluded by the generator, and advantages below 1e-6 in
    magnitude are snapped to 0 so no gradient underflows to an accidental exact zero
    (docs/diagnostics/clipping.md)."""
    clip = ClipConfig(eps_low=0.2, eps_high=0.3, ratio_cap=ratio_cap)
    advantages = torch.where(batch.advantages.abs() < 1e-6, 0.0, batch.advantages)
    leaf = batch.logprobs.clone().requires_grad_(True)
    config = PolicyLossConfig(
        ratio=RatioKind.TOKEN,
        surrogate=SurrogateKind.PG_CLIP,
        clip=clip,
        aggregation=Aggregation.TOKEN_MEAN,
    )
    result = policy_loss(
        config,
        logprobs=leaf,
        old_logprobs=batch.old_logprobs,
        advantages=advantages,
        response_mask=batch.response_mask,
    )
    bounds = [1.0 - 0.2, 1.0 + 0.3] + ([ratio_cap] if ratio_cap is not None else [])
    for bound in bounds:
        assume(not bool(((result.ratio == bound) & batch.response_mask).any()))
    (grad,) = torch.autograd.grad(result.loss, leaf)
    report = clip_report(result.ratio, advantages, batch.response_mask, clip)
    assert torch.equal(report.gradient_killed_mask, (grad == 0) & batch.response_mask)


# ---------------------------------------------------------------------------
# Obligation 4: microbatch weights == explicit micro-batched policy_loss loop
# ---------------------------------------------------------------------------


class PartitionedBatch(NamedTuple):
    logprobs: torch.Tensor
    advantages: torch.Tensor
    response_mask: torch.Tensor
    sizes: tuple[int, ...]


@st.composite
def partitioned_policy_batches(draw: st.DrawFn) -> PartitionedBatch:
    """A batch plus a drawn partition of its rows into micro-batch sizes."""
    mask = draw(padded_masks(min_b=2))
    logprobs = _masked_fill(draw, mask, -8.0, -0.0625)
    advantages = _masked_fill(draw, mask, -3.0, 3.0)
    sizes: list[int] = []
    remaining = mask.shape[0]
    while remaining > 0:
        size = draw(st.integers(1, remaining))
        sizes.append(size)
        remaining -= size
    return PartitionedBatch(logprobs, advantages, mask, tuple(sizes))


def _microbatched_reinforce_loss(
    leaf: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    mode: Aggregation,
    sizes: tuple[int, ...],
    norm_len: int | None,
    loss_scale: str,
) -> torch.Tensor:
    """Explicit micro-batch loop: policy_loss per chunk, then mean or sum combine."""
    config = PolicyLossConfig(
        ratio=RatioKind.TOKEN,
        surrogate=SurrogateKind.REINFORCE,
        clip=None,
        aggregation=mode,
        norm_len=norm_len,
    )
    losses = []
    start = 0
    for size in sizes:
        stop = start + size
        result = policy_loss(
            config,
            logprobs=leaf[start:stop],
            old_logprobs=leaf[start:stop].detach(),
            advantages=advantages[start:stop],
            response_mask=mask[start:stop],
        )
        losses.append(result.loss)
        start = stop
    stacked = torch.stack(losses)
    return stacked.mean() if loss_scale == "mean" else stacked.sum()


@pytest.mark.parametrize("loss_scale", ["mean", "sum"])
@pytest.mark.parametrize("mode", MODES)
@given(batch=partitioned_policy_batches())
def test_microbatch_weights_match_explicit_microbatched_policy_loss_loop(
    loss_scale: str, mode: Aggregation, batch: PartitionedBatch
) -> None:
    """Cross-module obligation 4: microbatch_token_weights equals the autograd
    weights of an explicit micro-batched policy_loss loop. With the REINFORCE
    surrogate d loss/d logprobs = -A_t * w_t per token, so the loop gradient must be
    -(A * microbatch_token_weights): bitwise for loss_scale="sum", and up to one
    rounding of the 1/K combine factor for "mean"
    (docs/derivations/aggregation.md, micro-batch algebra)."""
    norm_len = _norm_len(mode)
    leaf = batch.logprobs.clone().requires_grad_(True)
    total = _microbatched_reinforce_loss(
        leaf, batch.advantages, batch.response_mask, mode, batch.sizes, norm_len, loss_scale
    )
    (grad,) = torch.autograd.grad(total, leaf)
    weights = microbatch_token_weights(
        batch.response_mask, mode, batch.sizes, norm_len=norm_len, loss_scale=loss_scale
    )
    masked_adv = torch.where(batch.response_mask, batch.advantages, 0.0)
    predicted = -(masked_adv * weights)
    if loss_scale == "sum":
        assert torch.equal(grad, predicted)
    else:
        assert_close(grad, predicted, rtol=1e-15, atol=0.0)


# ---------------------------------------------------------------------------
# Obligation 5: rloo registry entry == manual leave-one-out REINFORCE
# ---------------------------------------------------------------------------


def _manual_loo_advantages(rewards: torch.Tensor, group_ids: torch.Tensor) -> torch.Tensor:
    """Leave-one-out baseline composed with explicit Python loops."""
    b = int(rewards.shape[0])
    rows = []
    for i in range(b):
        peers = [rewards[j] for j in range(b) if int(group_ids[j]) == int(group_ids[i]) and j != i]
        rows.append(rewards[i] - torch.stack(peers).mean())
    return torch.stack(rows)


def _manual_loo_reinforce_loss(
    leaf: torch.Tensor, advantages: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """mean_b( -A_i * sum_{t in mask_i} logprobs_{i,t} ), composed row by row."""
    rows = [-(advantages[i] * leaf[i][mask[i]].sum()) for i in range(int(leaf.shape[0]))]
    return torch.stack(rows).sum() / leaf.shape[0]


def test_rloo_registry_equals_manual_loo_reinforce_golden() -> None:
    """Cross-module obligation 5, hand-derived case: policy_loss under
    ALGORITHMS["rloo"].loss (REINFORCE, SEQ_MEAN_TOKEN_SUM) fed rloo_advantages equals
    the manually composed per-group LOO REINFORCE surrogate, bitwise on dyadic inputs.
    Arithmetic: rewards [1, 0, 2, 4], groups [0, 0, 1, 1] -> LOO baselines
    [0, 1, 4, 2] -> A = [1, -1, -2, 2]; masked logprob row sums
    [-1.5, -2.0, -1.0, -3.0]; loss = (1/4) * (-(1)(-1.5) - (-1)(-2.0) - (-2)(-1.0)
    - (2)(-3.0)) = (1.5 - 2.0 - 2.0 + 6.0)/4 = 0.875; grad at response tokens is
    -A_i/4 = [-0.25, +0.25, +0.5, -0.5] per row (docs/derivations/variants.md)."""
    f64 = torch.float64
    junk = MASKED_JUNK
    spec = get("rloo")
    assert spec.loss.surrogate is SurrogateKind.REINFORCE
    assert spec.loss.aggregation is Aggregation.SEQ_MEAN_TOKEN_SUM
    rewards = torch.tensor([1.0, 0.0, 2.0, 4.0], dtype=f64)
    group_ids = torch.tensor([0, 0, 1, 1])
    advantages = rloo_advantages(rewards, group_ids)
    assert torch.equal(advantages, torch.tensor([1.0, -1.0, -2.0, 2.0], dtype=f64))

    mask = torch.tensor([[True, True], [True, False], [True, True], [True, False]])
    logprobs = torch.tensor([[-0.5, -1.0], [-2.0, junk], [-0.25, -0.75], [-3.0, junk]], dtype=f64)
    leaf = logprobs.clone().requires_grad_(True)
    result = policy_loss(
        spec.loss,
        logprobs=leaf,
        old_logprobs=logprobs.detach(),
        advantages=advantages,
        response_mask=mask,
    )
    (grad,) = torch.autograd.grad(result.loss, leaf)
    assert torch.equal(result.loss, torch.tensor(0.875, dtype=f64))
    expected_grad = torch.tensor([[-0.25, -0.25], [0.25, 0.0], [0.5, 0.5], [-0.5, 0.0]], dtype=f64)
    assert torch.equal(grad, expected_grad)


class RlooBatch(NamedTuple):
    logprobs: torch.Tensor
    rewards: torch.Tensor
    group_ids: torch.Tensor
    response_mask: torch.Tensor


@st.composite
def rloo_batches(draw: st.DrawFn) -> RlooBatch:
    """Masked logprobs plus per-row rewards and a group assignment with sizes >= 2."""
    mask = draw(padded_masks(min_b=2, max_b=8, max_t=8))
    b = mask.shape[0]
    logprobs = _masked_fill(draw, mask, -8.0, -0.0625)
    rewards = _seq_advantages(draw, b)
    order = draw(st.permutations(list(range(b))))
    group_ids = torch.zeros(b, dtype=torch.long)
    assigned, gid = 0, 0
    while assigned < b:
        remaining = b - assigned
        legal = [s for s in range(2, remaining + 1) if remaining - s != 1]
        size = draw(st.sampled_from(legal))
        for i in range(assigned, assigned + size):
            group_ids[order[i]] = gid
        assigned += size
        gid += 1
    return RlooBatch(logprobs, rewards, group_ids, mask)


@given(batch=rloo_batches())
def test_rloo_registry_equals_manual_loo_reinforce_property(batch: RlooBatch) -> None:
    """Cross-module obligation 5: on generated batches, rloo_advantages fed to
    policy_loss under ALGORITHMS["rloo"].loss matches the manually composed per-group
    LOO baseline and REINFORCE surrogate in advantages, loss, and gradient. The
    entry's in-reward k1 shaping acts upstream on the rewards identically for both
    compositions and is exercised in obligation 6 (docs/derivations/variants.md)."""
    spec = get("rloo")
    advantages = rloo_advantages(batch.rewards, batch.group_ids)
    manual_advantages = _manual_loo_advantages(batch.rewards, batch.group_ids)
    assert_close(advantages, manual_advantages, rtol=1e-12, atol=1e-12)

    leaf_registry = batch.logprobs.clone().requires_grad_(True)
    result = policy_loss(
        spec.loss,
        logprobs=leaf_registry,
        old_logprobs=batch.logprobs.detach(),
        advantages=advantages,
        response_mask=batch.response_mask,
    )
    (grad_registry,) = torch.autograd.grad(result.loss, leaf_registry)

    leaf_manual = batch.logprobs.clone().requires_grad_(True)
    manual_loss = _manual_loo_reinforce_loss(leaf_manual, manual_advantages, batch.response_mask)
    (grad_manual,) = torch.autograd.grad(manual_loss, leaf_manual)

    assert_close(result.loss, manual_loss, rtol=1e-12, atol=1e-12)
    assert_close(grad_registry, grad_manual, rtol=1e-12, atol=1e-12)


# ---------------------------------------------------------------------------
# Obligation 6: bandit end-to-end smoke for every ALGORITHMS entry
# ---------------------------------------------------------------------------

BANDIT_ARMS = 8
BANDIT_BATCH = 64
BANDIT_MAX_STEPS = 300
BANDIT_THRESHOLD = 0.6
OPTIMAL_ARM = BANDIT_ARMS - 1


def _bandit_advantages(spec: AlgorithmSpec, batch: BanditBatch) -> torch.Tensor:
    """Advantages for one bandit batch under the spec's estimator and KL placement."""
    token_rewards = batch.rewards.unsqueeze(1)
    if spec.kl_placement == "reward":
        assert spec.kl_reward is not None
        token_rewards = kl_in_reward(
            token_rewards,
            batch.old_logprobs,
            batch.ref_logprobs,
            spec.kl_reward.kind,
            spec.kl_reward.coef,
            response_mask=batch.response_mask,
        )
    if spec.advantage == "gae":
        advantages, _ = gae(
            token_rewards,
            torch.zeros_like(token_rewards),
            config=GAEConfig(gamma=1.0, lam=1.0),
            response_mask=batch.response_mask,
        )
        return advantages
    seq_rewards = token_rewards.sum(dim=1)
    group_ids = torch.zeros(seq_rewards.shape[0], dtype=torch.long)
    if spec.advantage == "grpo":
        group_config = spec.advantage_config
        assert isinstance(group_config, GroupNormConfig)
        return grpo_advantages(seq_rewards, group_ids, group_config)
    if spec.advantage == "rloo":
        return rloo_advantages(seq_rewards, group_ids)
    assert spec.advantage == "reinforce_pp"
    pp_config = spec.advantage_config
    assert isinstance(pp_config, ReinforcePPConfig)
    return reinforce_pp_advantages(seq_rewards, None, config=pp_config)


def _bandit_loss(
    spec: AlgorithmSpec, loss_config: PolicyLossConfig, batch: BanditBatch
) -> PolicyLossResult:
    return policy_loss(
        loss_config,
        logprobs=batch.logprobs,
        old_logprobs=batch.old_logprobs,
        advantages=_bandit_advantages(spec, batch),
        response_mask=batch.response_mask,
        ref_logprobs=batch.ref_logprobs if loss_config.kl is not None else None,
        rollout_logprobs=(
            batch.rollout_logprobs if loss_config.is_correction is not None else None
        ),
    )


@pytest.mark.parametrize("name", sorted(ALGORITHMS))
def test_bandit_end_to_end_reaches_near_greedy_optimal_arm(name: str) -> None:
    """Cross-module obligation 6: an 8-arm SoftmaxBandit trained on-policy from
    a uniform start with each ALGORITHMS entry (Adam on theta, <= 300 steps, one
    seeded generator per entry) drives the optimal arm's probability above 0.6.
    ppo uses values = zeros with GAEConfig(1, 1), so GAE reduces to the raw
    reward-to-go; dr_grpo runs with norm_len = 1 via dataclasses.replace; the
    in-reward KL entries (rloo, reinforce_pp) shape rewards through kl_in_reward.
    Each entry converges in well under a second of compute; wall-clock time is
    deliberately not asserted (it depends on machine load, not correctness)."""
    spec = get(name)
    loss_config = spec.loss
    if name == "dr_grpo":
        loss_config = dataclasses.replace(loss_config, norm_len=1)
    theta = torch.zeros(BANDIT_ARMS, dtype=torch.float64, requires_grad=True)
    arm_rewards = torch.linspace(0.0, 1.0, BANDIT_ARMS, dtype=torch.float64)
    bandit = SoftmaxBandit(theta, arm_rewards)
    assert torch.equal(
        bandit.probs(), torch.full((BANDIT_ARMS,), 1.0 / BANDIT_ARMS, dtype=torch.float64)
    )
    generator = torch.Generator().manual_seed(1234)
    optimizer = torch.optim.Adam([theta], lr=0.05)
    for _ in range(BANDIT_MAX_STEPS):
        batch = bandit.sample(BANDIT_BATCH, generator)
        result = _bandit_loss(spec, loss_config, batch)
        optimizer.zero_grad()
        result.loss.backward()
        optimizer.step()
        if float(bandit.probs()[OPTIMAL_ARM]) > 0.7:
            break
    assert float(bandit.probs()[OPTIMAL_ARM]) > BANDIT_THRESHOLD


# ---------------------------------------------------------------------------
# Cross-pipeline mask invariance
# ---------------------------------------------------------------------------


def test_cross_pipeline_mask_invariance_policy_loss_and_clip_report() -> None:
    """The masked-position output rule of docs/conventions.md across modules, supporting
    cross-module obligations 1 and 3: perturbing every masked position of every input
    stream of the grpo_tis pipeline (PG_CLIP + as-loss k3 KL + token TIS) leaves the
    policy_loss outputs, the logprobs gradient, and the downstream clip_report
    bitwise unchanged."""
    f64 = torch.float64
    spec = get("grpo_tis")
    assert spec.loss.clip is not None
    mask = torch.tensor([[True] * 4, [True, True, False, False], [True, True, True, False]])

    def batch(junk: float) -> dict[str, torch.Tensor]:
        base = {
            "logprobs": torch.tensor(
                [[-0.5, -1.0, -0.25, -2.0], [-1.5, -0.75, 0.0, 0.0], [-3.0, -0.125, -1.0, 0.0]],
                dtype=f64,
            ),
            "old_logprobs": torch.tensor(
                [[-0.9, -0.8, -0.5, -1.5], [-1.0, -1.25, 0.0, 0.0], [-2.5, -0.25, -1.5, 0.0]],
                dtype=f64,
            ),
            "ref_logprobs": torch.tensor(
                [[-0.7, -1.1, -0.3, -1.8], [-1.2, -0.9, 0.0, 0.0], [-2.8, -0.2, -1.2, 0.0]],
                dtype=f64,
            ),
            "rollout_logprobs": torch.tensor(
                [[-0.8, -0.9, -0.6, -1.4], [-1.1, -1.3, 0.0, 0.0], [-2.6, -0.3, -1.4, 0.0]],
                dtype=f64,
            ),
            "advantages": torch.tensor(
                [[1.5, -2.0, 0.5, 3.0], [-1.0, 2.0, 0.0, 0.0], [0.25, -0.75, 1.0, 0.0]],
                dtype=f64,
            ),
        }
        return {
            name: torch.where(mask, tensor, torch.full_like(tensor, junk))
            for name, tensor in base.items()
        }

    outputs = []
    for junk in (MASKED_JUNK, -55.5):
        streams = batch(junk)
        leaf = streams["logprobs"].requires_grad_(True)
        result = policy_loss(
            spec.loss,
            logprobs=leaf,
            old_logprobs=streams["old_logprobs"],
            advantages=streams["advantages"],
            response_mask=mask,
            ref_logprobs=streams["ref_logprobs"],
            rollout_logprobs=streams["rollout_logprobs"],
        )
        (grad,) = torch.autograd.grad(result.loss, leaf)
        report = clip_report(result.ratio, streams["advantages"], mask, spec.loss.clip)
        assert result.kl_loss is not None
        outputs.append(
            (
                result.loss,
                result.per_token_objective,
                result.ratio,
                result.clipped_low,
                result.clipped_high,
                result.kl_loss,
                grad,
                report.gradient_killed_mask,
                torch.tensor(
                    [
                        report.frac_pos_adv_clipped_high,
                        report.frac_pos_adv_clipped_low,
                        report.frac_neg_adv_clipped_high,
                        report.frac_neg_adv_clipped_low,
                        report.gradient_killed_frac,
                    ],
                    dtype=f64,
                ),
            )
        )
    for reference, perturbed in zip(outputs[0], outputs[1], strict=True):
        assert torch.equal(reference, perturbed)
