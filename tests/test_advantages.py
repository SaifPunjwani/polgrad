"""Enforces the docs/derivations/advantages.md semantics for polgrad.advantages.

Closed-form cases carry their hand arithmetic in comments; the matching derivations
are in docs/derivations/advantages.md. Property tests use Hypothesis with the shared
strategies from tests/strategies.py plus module-local grouped-reward strategies.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import pytest
import torch
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st
from strategies import MASKED_JUNK, padded_masks

from polgrad.advantages import (
    GAEConfig,
    GroupNormConfig,
    ReinforcePPConfig,
    broadcast_to_tokens,
    gae,
    grpo_advantages,
    reinforce_pp_advantages,
    rloo_advantages,
    whiten,
)


class GroupedRewards(NamedTuple):
    rewards: torch.Tensor
    group_ids: torch.Tensor


class MaskedTensor(NamedTuple):
    x: torch.Tensor
    response_mask: torch.Tensor


class GAEBatch(NamedTuple):
    token_rewards: torch.Tensor
    values: torch.Tensor
    response_mask: torch.Tensor
    gamma: float
    lam: float


def _grouped_ids(draw: st.DrawFn, sizes: list[int]) -> torch.Tensor:
    """Non-contiguous, unique labels per group, interleaved by a random permutation."""
    labels = draw(
        st.lists(st.integers(0, 50), unique=True, min_size=len(sizes), max_size=len(sizes))
    )
    ids = [label for label, size in zip(labels, sizes, strict=True) for _ in range(size)]
    perm = draw(st.permutations(range(len(ids))))
    return torch.tensor(ids, dtype=torch.long)[torch.tensor(perm, dtype=torch.long)]


@st.composite
def grouped_rewards(
    draw: st.DrawFn,
    *,
    min_group: int = 2,
    max_group: int = 4,
    max_groups: int = 4,
) -> GroupedRewards:
    """Reward batches with every group of size >= min_group and arbitrary labels."""
    n_groups = draw(st.integers(1, max_groups))
    sizes = [draw(st.integers(min_group, max_group)) for _ in range(n_groups)]
    group_ids = _grouped_ids(draw, sizes)
    vals = [
        draw(st.floats(-5.0, 5.0, allow_nan=False, allow_infinity=False, width=32))
        for _ in range(int(group_ids.numel()))
    ]
    return GroupedRewards(torch.tensor(vals, dtype=torch.float64), group_ids)


@st.composite
def dyadic_grouped_rewards(draw: st.DrawFn) -> GroupedRewards:
    """Rewards k/8 with group sizes in {2, 3, 5} (G - 1 a power of two), so both RLOO
    forms are exactly representable in float64 and bitwise comparison is valid."""
    n_groups = draw(st.integers(1, 3))
    sizes = [draw(st.sampled_from([2, 3, 5])) for _ in range(n_groups)]
    group_ids = _grouped_ids(draw, sizes)
    vals = [draw(st.integers(-64, 64)) / 8.0 for _ in range(int(group_ids.numel()))]
    return GroupedRewards(torch.tensor(vals, dtype=torch.float64), group_ids)


def _fill_masked(
    draw: st.DrawFn, mask: torch.Tensor, low: float, high: float, junk: float = MASKED_JUNK
) -> torch.Tensor:
    b, t = mask.shape
    vals = [
        draw(st.floats(low, high, allow_nan=False, allow_infinity=False, width=32))
        for _ in range(b * t)
    ]
    out = torch.tensor(vals, dtype=torch.float64).reshape(b, t)
    return torch.where(mask, out, torch.full_like(out, junk))


@st.composite
def masked_tensors(draw: st.DrawFn, *, max_b: int = 6, max_t: int = 10) -> MaskedTensor:
    mask = draw(padded_masks(max_b=max_b, max_t=max_t))
    return MaskedTensor(_fill_masked(draw, mask, -3.0, 3.0), mask)


@st.composite
def gae_batches(draw: st.DrawFn, *, max_b: int = 5, max_t: int = 10) -> GAEBatch:
    mask = draw(padded_masks(max_b=max_b, max_t=max_t))
    rewards = _fill_masked(draw, mask, -3.0, 3.0)
    values = _fill_masked(draw, mask, -3.0, 3.0)
    gamma = draw(st.floats(0.0, 1.0, allow_nan=False, allow_infinity=False, width=32))
    lam = draw(st.floats(0.0, 1.0, allow_nan=False, allow_infinity=False, width=32))
    return GAEBatch(rewards, values, mask, gamma, lam)


def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.to(x.dtype)
    return (x * m).sum() / m.sum()


def _masked_var(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.to(x.dtype)
    n = m.sum()
    centered = (x - (x * m).sum() / n) * m
    return (centered * centered).sum() / (n - 1)


def _gae_slow(
    token_rewards: torch.Tensor,
    values: torch.Tensor,
    gamma: float,
    lam: float,
    response_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """O(T^2) oracle: A_t = sum_{s>=t} (gamma*lam)^(s-t) * delta_s per row, evaluated
    directly from the definition over the L real tokens of each right-padded row."""
    b, _ = token_rewards.shape
    adv = torch.zeros_like(token_rewards)
    ret = torch.zeros_like(token_rewards)
    for i in range(b):
        length = int(response_mask[i].sum())
        for t in range(length):
            acc = 0.0
            for s in range(t, length):
                bootstrap = gamma * float(values[i, s + 1]) if s + 1 < length else 0.0
                delta = float(token_rewards[i, s]) + bootstrap - float(values[i, s])
                acc += (gamma * lam) ** (s - t) * delta
            adv[i, t] = acc
            ret[i, t] = acc + float(values[i, t])
    return adv, ret


# ---------------------------------------------------------------------------
# grpo_advantages
# ---------------------------------------------------------------------------


def test_grpo_closed_form_single_group() -> None:
    """GRPO defaults on one group match the hand-derived normalized rewards
    (docs/derivations/advantages.md, "Group normalization")."""
    # rewards [1, 2, 3]: mean = 2, centered = [-1, 0, 1], squared-deviation sum = 2,
    # Bessel denominator = 3 - 1 = 2, var = 1, std = 1, A = centered / (1 + 1e-4).
    rewards = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    group_ids = torch.zeros(3, dtype=torch.long)
    out = grpo_advantages(rewards, group_ids, GroupNormConfig())
    expected = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float64) / (1.0 + 1e-4)
    assert torch.equal(out, expected)


def test_grpo_unbiased_false_uses_population_std() -> None:
    """unbiased=False divides by the population std sqrt(sum((r - mean)^2) / G)."""
    # Same batch as above: population var = 2 / 3, std = sqrt(2/3).
    rewards = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    group_ids = torch.zeros(3, dtype=torch.long)
    out = grpo_advantages(rewards, group_ids, GroupNormConfig(unbiased=False))
    expected = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float64) / (math.sqrt(2.0 / 3.0) + 1e-4)
    assert torch.equal(out, expected)


def test_grpo_center_false_divides_raw_rewards() -> None:
    """center=False keeps the raw rewards in the numerator; only sigma_g is applied."""
    # std of [1, 2, 3] is 1 (Bessel), so A = rewards / (1 + 1e-4).
    rewards = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    group_ids = torch.zeros(3, dtype=torch.long)
    out = grpo_advantages(rewards, group_ids, GroupNormConfig(center=False))
    assert torch.equal(out, rewards / (1.0 + 1e-4))


@given(batch=grouped_rewards())
def test_grpo_scale_none_equals_per_group_centering(batch: GroupedRewards) -> None:
    """GRPO(scale="none", center=True) equals per-group centered rewards (Dr.GRPO;
    docs/derivations/advantages.md, "Group normalization")."""
    out = grpo_advantages(batch.rewards, batch.group_ids, GroupNormConfig(scale="none"))
    expected = torch.empty_like(batch.rewards)
    for label in batch.group_ids.unique().tolist():
        idx = batch.group_ids == label
        expected[idx] = batch.rewards[idx] - batch.rewards[idx].mean()
    torch.testing.assert_close(out, expected, rtol=0.0, atol=1e-12)


@pytest.mark.parametrize("scale", ["std", "none"])
@given(batch=grouped_rewards(), shift=st.floats(-4.0, 4.0, allow_nan=False, width=32))
def test_grpo_shift_invariance(batch: GroupedRewards, shift: float, scale: str) -> None:
    """Adding a constant to every reward leaves group-normalized advantages unchanged
    (mean shifts with the rewards, std is shift-invariant)."""
    config = GroupNormConfig(scale=scale)  # type: ignore[arg-type]
    base = grpo_advantages(batch.rewards, batch.group_ids, config)
    shifted = grpo_advantages(batch.rewards + shift, batch.group_ids, config)
    torch.testing.assert_close(shifted, base, rtol=1e-9, atol=1e-9)


@given(batch=grouped_rewards(), a=st.floats(0.25, 4.0, allow_nan=False, width=32))
def test_grpo_scale_none_is_scale_equivariant(batch: GroupedRewards, a: float) -> None:
    """Dr.GRPO advantages scale linearly with the rewards: A(a*r) = a*A(r)."""
    config = GroupNormConfig(scale="none")
    base = grpo_advantages(batch.rewards, batch.group_ids, config)
    scaled = grpo_advantages(a * batch.rewards, batch.group_ids, config)
    torch.testing.assert_close(scaled, a * base, rtol=1e-9, atol=1e-9)


@settings(suppress_health_check=[HealthCheck.filter_too_much], deadline=None)
@given(batch=grouped_rewards(), a=st.floats(0.25, 4.0, allow_nan=False, width=32))
def test_grpo_std_scale_is_scale_invariant_at_eps_zero(batch: GroupedRewards, a: float) -> None:
    """With eps=0, std scaling cancels a positive reward rescaling: A(a*r) = A(r)."""
    for label in batch.group_ids.unique().tolist():
        idx = batch.group_ids == label
        assume(float(batch.rewards[idx].std()) > 1e-2)
    config = GroupNormConfig(eps=0.0)
    base = grpo_advantages(batch.rewards, batch.group_ids, config)
    scaled = grpo_advantages(a * batch.rewards, batch.group_ids, config)
    torch.testing.assert_close(scaled, base, rtol=1e-9, atol=1e-9)


def test_grpo_group_labels_are_arbitrary_nonnegative_ints() -> None:
    """Group ids need not be contiguous or sorted; only row partitioning matters."""
    rewards = torch.tensor([1.0, 2.0, 3.0, 6.0], dtype=torch.float64)
    interleaved = torch.tensor([7, 42, 7, 42])
    config = GroupNormConfig(scale="none")
    out = grpo_advantages(rewards, interleaved, config)
    # groups: {1, 3} (mean 2) and {2, 6} (mean 4) -> [-1, -2, 1, 2]
    assert torch.equal(out, torch.tensor([-1.0, -2.0, 1.0, 2.0], dtype=torch.float64))
    relabeled = grpo_advantages(rewards, torch.tensor([0, 1, 0, 1]), config)
    assert torch.equal(out, relabeled)


def test_dr_grpo_difficulty_bias_exact_std_factor() -> None:
    """Two groups with identical centered rewards but different stds: GRPO reweights
    each group by exactly 1/(sigma_g + eps), the difficulty bias Dr.GRPO removes
    (docs/derivations/advantages.md, "Group normalization and the Dr.GRPO difficulty
    bias")."""
    # Group 0 = [1, -1]           : mean 0, sq-dev sum 2, Bessel 1, sigma_0 = sqrt(2)
    # Group 1 = [1, -1, 1, -1]    : mean 0, sq-dev sum 4, Bessel 3, sigma_1 = sqrt(4/3)
    # Centered rewards are elementwise identical (+-1) in both groups.
    rewards = torch.tensor([1.0, -1.0, 1.0, -1.0, 1.0, -1.0], dtype=torch.float64)
    group_ids = torch.tensor([0, 0, 1, 1, 1, 1])
    a_none = grpo_advantages(rewards, group_ids, GroupNormConfig(scale="none"))
    a_std = grpo_advantages(rewards, group_ids, GroupNormConfig(scale="std"))
    assert torch.equal(a_none, rewards)
    sigma_0, sigma_1 = math.sqrt(2.0), math.sqrt(4.0 / 3.0)
    sigma = torch.tensor(
        [sigma_0, sigma_0, sigma_1, sigma_1, sigma_1, sigma_1], dtype=torch.float64
    )
    assert torch.equal(a_std, a_none / (sigma + 1e-4))
    # The bias: Dr.GRPO treats both groups identically, GRPO does not.
    assert torch.equal(a_none[:2], a_none[2:4])
    assert not torch.equal(a_std[:2], a_std[2:4])


def test_grpo_scale_std_group_of_one_raises() -> None:
    """scale="std" with a size-1 group raises: its std is undefined. Frameworks emit
    0/eps here; polgrad raises (recorded in the conformance deviation docs)."""
    rewards = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    group_ids = torch.tensor([0, 0, 1])
    with pytest.raises(ValueError, match=r"scale='std'.*group id\(s\) \[1\]"):
        grpo_advantages(rewards, group_ids, GroupNormConfig())


def test_grpo_scale_none_allows_group_of_one() -> None:
    """Without std scaling a singleton group is well defined: its advantage is 0."""
    rewards = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    group_ids = torch.tensor([0, 0, 1])
    out = grpo_advantages(rewards, group_ids, GroupNormConfig(scale="none"))
    assert torch.equal(out, torch.tensor([-0.5, 0.5, 0.0], dtype=torch.float64))


def test_grpo_rejects_unknown_scale() -> None:
    """A config carrying an unknown scale value raises at call time."""
    rewards = torch.tensor([1.0, 2.0], dtype=torch.float64)
    group_ids = torch.tensor([0, 0])
    config = GroupNormConfig(scale="bogus")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=r"config\.scale"):
        grpo_advantages(rewards, group_ids, config)


# ---------------------------------------------------------------------------
# rloo_advantages
# ---------------------------------------------------------------------------


def test_rloo_closed_form_two_groups() -> None:
    """RLOO leave-one-out baselines match the hand-derived values
    (docs/derivations/advantages.md, "The RLOO identity")."""
    # Group 0 = [1, 3]: A = [1 - 3, 3 - 1] = [-2, 2].
    # Group 1 = [2, 4, 6]: A = [2 - (4+6)/2, 4 - (2+6)/2, 6 - (2+4)/2] = [-3, 0, 3].
    rewards = torch.tensor([1.0, 3.0, 2.0, 4.0, 6.0], dtype=torch.float64)
    group_ids = torch.tensor([0, 0, 1, 1, 1])
    out = rloo_advantages(rewards, group_ids)
    assert torch.equal(out, torch.tensor([-2.0, 2.0, -3.0, 0.0, 3.0], dtype=torch.float64))


@given(batch=dyadic_grouped_rewards())
def test_rloo_two_form_identity_exact_on_dyadic_inputs(batch: GroupedRewards) -> None:
    """r_i - (S - r_i)/(G - 1) and (G*r_i - S)/(G - 1) are bitwise equal on dyadic
    rewards with G - 1 a power of two, and the implementation matches both; the
    (G/(G-1))*(r_i - S/G) arrangement agrees to fp rounding
    (docs/derivations/advantages.md, "The RLOO identity")."""
    out = rloo_advantages(batch.rewards, batch.group_ids)
    form_loo = torch.empty_like(batch.rewards)
    form_scaled = torch.empty_like(batch.rewards)
    form_mean = torch.empty_like(batch.rewards)
    for label in batch.group_ids.unique().tolist():
        idx = batch.group_ids == label
        g = int(idx.sum())
        s = float(batch.rewards[idx].sum())
        r = batch.rewards[idx]
        form_loo[idx] = r - (s - r) / (g - 1)
        form_scaled[idx] = (g * r - s) / (g - 1)
        form_mean[idx] = (g / (g - 1)) * (r - s / g)
    assert torch.equal(out, form_loo)
    assert torch.equal(out, form_scaled)
    torch.testing.assert_close(out, form_mean, rtol=0.0, atol=1e-12)


@given(batch=grouped_rewards())
def test_rloo_two_form_identity_general_floats(batch: GroupedRewards) -> None:
    """On generic float rewards the implementation agrees with both RLOO forms up to
    fp rounding."""
    out = rloo_advantages(batch.rewards, batch.group_ids)
    form_loo = torch.empty_like(batch.rewards)
    form_mean = torch.empty_like(batch.rewards)
    for label in batch.group_ids.unique().tolist():
        idx = batch.group_ids == label
        g = int(idx.sum())
        r = batch.rewards[idx]
        form_loo[idx] = r - (float(r.sum()) - r) / (g - 1)
        form_mean[idx] = (g / (g - 1)) * (r - r.mean())
    torch.testing.assert_close(out, form_loo, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(out, form_mean, rtol=1e-10, atol=1e-10)


@given(batch=grouped_rewards())
def test_rloo_group_advantages_sum_to_zero(batch: GroupedRewards) -> None:
    """RLOO advantages sum to zero within each group: sum_i A_i = (G/(G-1))(S - S)."""
    out = rloo_advantages(batch.rewards, batch.group_ids)
    for label in batch.group_ids.unique().tolist():
        assert abs(float(out[batch.group_ids == label].sum())) <= 1e-10


def test_rloo_group_of_one_raises() -> None:
    """A size-1 group has no leave-one-out baseline; rloo_advantages raises."""
    rewards = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    group_ids = torch.tensor([5, 5, 9])
    with pytest.raises(ValueError, match=r"leave-one-out.*group id\(s\) \[9\]"):
        rloo_advantages(rewards, group_ids)


# ---------------------------------------------------------------------------
# reinforce_pp_advantages
# ---------------------------------------------------------------------------


def test_reinforce_pp_global_baseline_closed_form() -> None:
    """group_ids=None subtracts the global batch mean (REINFORCE++)."""
    # rewards [1, 2, 3, 6]: mean = 3, A = [-2, -1, 0, 3].
    rewards = torch.tensor([1.0, 2.0, 3.0, 6.0], dtype=torch.float64)
    out = reinforce_pp_advantages(rewards, config=ReinforcePPConfig(batch_norm=False))
    assert torch.equal(out, torch.tensor([-2.0, -1.0, 0.0, 3.0], dtype=torch.float64))


def test_reinforce_pp_group_baseline_closed_form() -> None:
    """group_ids selects the per-group mean baseline (REINFORCE++-baseline)."""
    # Group 0 = [1, 3] (mean 2), group 1 = [2, 4, 6] (mean 4): A = [-1, 1, -2, 0, 2].
    rewards = torch.tensor([1.0, 3.0, 2.0, 4.0, 6.0], dtype=torch.float64)
    group_ids = torch.tensor([0, 0, 1, 1, 1])
    out = reinforce_pp_advantages(rewards, group_ids, config=ReinforcePPConfig(batch_norm=False))
    assert torch.equal(out, torch.tensor([-1.0, 1.0, -2.0, 0.0, 2.0], dtype=torch.float64))


def test_reinforce_pp_batch_norm_closed_form() -> None:
    """batch_norm divides by the Bessel-corrected global std of the centered
    advantages plus eps (docs/derivations/advantages.md, "REINFORCE++ baseline
    variants")."""
    # A = [-2, -1, 0, 3] (mean 0); squared sum = 4 + 1 + 0 + 9 = 14, Bessel
    # denominator = 3, std = sqrt(14/3); out = A / (sqrt(14/3) + 1e-8).
    rewards = torch.tensor([1.0, 2.0, 3.0, 6.0], dtype=torch.float64)
    out = reinforce_pp_advantages(rewards, config=ReinforcePPConfig(batch_norm=True))
    centered = torch.tensor([-2.0, -1.0, 0.0, 3.0], dtype=torch.float64)
    expected = centered / (math.sqrt(14.0 / 3.0) + 1e-8)
    assert torch.equal(out, expected)


@given(batch=grouped_rewards())
def test_reinforce_pp_group_baseline_matches_grpo_scale_none(batch: GroupedRewards) -> None:
    """REINFORCE++-baseline without batch_norm is exactly per-group centering, i.e.
    GRPO with scale="none" (docs/derivations/advantages.md)."""
    a_rpp = reinforce_pp_advantages(
        batch.rewards, batch.group_ids, config=ReinforcePPConfig(batch_norm=False)
    )
    a_grpo = grpo_advantages(batch.rewards, batch.group_ids, GroupNormConfig(scale="none"))
    assert torch.equal(a_rpp, a_grpo)


def test_reinforce_pp_singleton_group_allowed() -> None:
    """A singleton group is well defined for a mean baseline: its advantage is 0."""
    rewards = torch.tensor([5.0, 1.0, 3.0], dtype=torch.float64)
    group_ids = torch.tensor([0, 1, 1])
    out = reinforce_pp_advantages(rewards, group_ids, config=ReinforcePPConfig(batch_norm=False))
    assert torch.equal(out, torch.tensor([0.0, -1.0, 1.0], dtype=torch.float64))


def test_reinforce_pp_batch_norm_single_reward_raises() -> None:
    """batch_norm=True with fewer than 2 rewards raises: Bessel std is undefined."""
    with pytest.raises(ValueError, match="at least 2"):
        reinforce_pp_advantages(
            torch.tensor([2.0], dtype=torch.float64), config=ReinforcePPConfig(batch_norm=True)
        )


# ---------------------------------------------------------------------------
# shared grouped-input validation
# ---------------------------------------------------------------------------

GROUPED_FNS = [
    pytest.param(lambda r, g: grpo_advantages(r, g, GroupNormConfig()), id="grpo"),
    pytest.param(lambda r, g: rloo_advantages(r, g), id="rloo"),
    pytest.param(
        lambda r, g: reinforce_pp_advantages(r, g, config=ReinforcePPConfig(batch_norm=False)),
        id="reinforce_pp",
    ),
]

BAD_GROUPED_INPUTS = [
    pytest.param(torch.zeros(2, 2), torch.tensor([0, 0, 1, 1]), r"must be 1-D", id="rewards-2d"),
    pytest.param(
        torch.zeros(3), torch.tensor([0, 0, 1, 1]), "identical shapes", id="length-mismatch"
    ),
    pytest.param(torch.zeros(4), torch.zeros(4), "integer dtype", id="float-ids"),
    pytest.param(torch.zeros(4), torch.tensor([-1, -1, 0, 0]), "non-negative", id="negative-ids"),
    pytest.param(
        torch.zeros(0), torch.zeros(0, dtype=torch.long), "at least one sequence", id="empty"
    ),
    pytest.param(
        torch.tensor([float("nan"), 0.0, 1.0, 1.0]),
        torch.tensor([0, 0, 1, 1]),
        "non-finite",
        id="nan-rewards",
    ),
]


@pytest.mark.parametrize(("rewards", "group_ids", "message"), BAD_GROUPED_INPUTS)
@pytest.mark.parametrize("fn", GROUPED_FNS)
def test_grouped_input_validation(fn, rewards, group_ids, message) -> None:  # type: ignore[no-untyped-def]
    """All grouped estimators reject malformed rewards/group_ids with a ValueError
    naming the argument (docs/conventions.md, "Errors and determinism")."""
    with pytest.raises(ValueError, match=message):
        fn(rewards, group_ids)


# ---------------------------------------------------------------------------
# gae
# ---------------------------------------------------------------------------


def test_gae_closed_form_ragged_batch() -> None:
    """GAE on a hand-derived ragged batch, junk in the padding
    (docs/derivations/advantages.md, "GAE over right-padded responses")."""
    # gamma = lam = 1/2, so gamma*lam = 1/4.
    # Row 0 (L=3), r = [1, 2, 3], V = [4, 2, 1]:
    #   delta_2 = 3 - 1 = 2;  delta_1 = 2 + 0.5*1 - 2 = 0.5;  delta_0 = 1 + 0.5*2 - 4 = -2
    #   A_2 = 2;  A_1 = 0.5 + 0.25*2 = 1;  A_0 = -2 + 0.25*1 = -1.75
    #   R = A + V = [2.25, 3, 3]
    # Row 1 (L=2), r = [1, 1], V = [1, 1]:
    #   delta_1 = 1 - 1 = 0;  delta_0 = 1 + 0.5*1 - 1 = 0.5
    #   A_1 = 0;  A_0 = 0.5 + 0.25*0 = 0.5;  R = [1.5, 1], padding -> 0
    junk = MASKED_JUNK
    token_rewards = torch.tensor([[1.0, 2.0, 3.0], [1.0, 1.0, junk]], dtype=torch.float64)
    values = torch.tensor([[4.0, 2.0, 1.0], [1.0, 1.0, junk]], dtype=torch.float64)
    mask = torch.tensor([[True, True, True], [True, True, False]])
    adv, ret = gae(token_rewards, values, config=GAEConfig(gamma=0.5, lam=0.5), response_mask=mask)
    expected_adv = torch.tensor([[-1.75, 1.0, 2.0], [0.5, 0.0, 0.0]], dtype=torch.float64)
    expected_ret = torch.tensor([[2.25, 3.0, 3.0], [1.5, 1.0, 0.0]], dtype=torch.float64)
    assert torch.equal(adv, expected_adv)
    assert torch.equal(ret, expected_ret)


@given(batch=gae_batches())
def test_gae_matches_slow_oracle(batch: GAEBatch) -> None:
    """The O(T) reverse scan agrees with the O(T^2) definitional oracle _gae_slow on
    ragged batches, and padded positions are exactly 0."""
    config = GAEConfig(gamma=batch.gamma, lam=batch.lam)
    adv, ret = gae(
        batch.token_rewards, batch.values, config=config, response_mask=batch.response_mask
    )
    slow_adv, slow_ret = _gae_slow(
        batch.token_rewards, batch.values, batch.gamma, batch.lam, batch.response_mask
    )
    torch.testing.assert_close(adv, slow_adv, rtol=1e-9, atol=1e-9)
    torch.testing.assert_close(ret, slow_ret, rtol=1e-9, atol=1e-9)
    assert bool((adv[~batch.response_mask] == 0).all())
    assert bool((ret[~batch.response_mask] == 0).all())


@given(batch=gae_batches())
def test_gae_gamma_lambda_one_reduces_to_reward_to_go(batch: GAEBatch) -> None:
    """At gamma = lam = 1 the deltas telescope: advantages = reward-to-go - values and
    returns = reward-to-go (docs/derivations/advantages.md, "GAE over right-padded
    responses")."""
    adv, ret = gae(
        batch.token_rewards,
        batch.values,
        config=GAEConfig(gamma=1.0, lam=1.0),
        response_mask=batch.response_mask,
    )
    masked_rewards = torch.where(
        batch.response_mask, batch.token_rewards, torch.zeros_like(batch.token_rewards)
    )
    rtg = torch.flip(torch.cumsum(torch.flip(masked_rewards, [1]), 1), [1])
    zero = torch.zeros_like(rtg)
    expected_adv = torch.where(batch.response_mask, rtg - batch.values, zero)
    expected_ret = torch.where(batch.response_mask, rtg, zero)
    torch.testing.assert_close(adv, expected_adv, rtol=1e-9, atol=1e-9)
    torch.testing.assert_close(ret, expected_ret, rtol=1e-9, atol=1e-9)


@given(batch=gae_batches())
def test_gae_mask_invariance_bitwise(batch: GAEBatch) -> None:
    """Perturbing rewards/values at masked positions leaves both GAE outputs
    bitwise unchanged (docs/conventions.md, "Masked positions")."""
    config = GAEConfig(gamma=batch.gamma, lam=batch.lam)
    adv, ret = gae(
        batch.token_rewards, batch.values, config=config, response_mask=batch.response_mask
    )
    rewards_perturbed = torch.where(
        batch.response_mask, batch.token_rewards, torch.full_like(batch.token_rewards, -55.5)
    )
    values_perturbed = torch.where(
        batch.response_mask, batch.values, torch.full_like(batch.values, 7.25)
    )
    adv_p, ret_p = gae(
        rewards_perturbed, values_perturbed, config=config, response_mask=batch.response_mask
    )
    assert torch.equal(adv, adv_p)
    assert torch.equal(ret, ret_p)


def test_gae_rejects_non_right_padded_mask() -> None:
    """A real token after a padded position violates the right-padding convention and
    raises (docs/derivations/advantages.md, "GAE over right-padded responses")."""
    token_rewards = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float64)
    values = torch.zeros_like(token_rewards)
    mask = torch.tensor([[True, False, True]])
    with pytest.raises(ValueError, match=r"right-padded.*rows \[0\]"):
        gae(token_rewards, values, config=GAEConfig(gamma=1.0, lam=1.0), response_mask=mask)


def test_gae_rejects_shape_and_mask_violations() -> None:
    """gae raises ValueError on rank, shape, mask-dtype, and finiteness violations."""
    config = GAEConfig(gamma=1.0, lam=1.0)
    ok = torch.zeros((2, 3), dtype=torch.float64)
    mask = torch.ones((2, 3), dtype=torch.bool)
    with pytest.raises(ValueError, match=r"must be 2-D"):
        gae(torch.zeros(3), torch.zeros(3), config=config, response_mask=mask)
    with pytest.raises(ValueError, match="identical shapes"):
        gae(ok, torch.zeros((2, 4), dtype=torch.float64), config=config, response_mask=mask)
    with pytest.raises(ValueError, match=r"torch\.bool"):
        gae(ok, ok, config=config, response_mask=torch.ones((2, 3)))
    bad = ok.clone()
    bad[0, 0] = float("inf")
    with pytest.raises(ValueError, match="non-finite"):
        gae(ok, bad, config=config, response_mask=mask)


# ---------------------------------------------------------------------------
# broadcast_to_tokens
# ---------------------------------------------------------------------------


def test_broadcast_to_tokens_closed_form() -> None:
    """Each sequence value is repeated over its real tokens and 0 elsewhere
    (docs/derivations/advantages.md, "Broadcasting and whitening")."""
    per_seq = torch.tensor([2.0, -3.0], dtype=torch.float64)
    mask = torch.tensor([[True, True, False], [True, False, False]])
    out = broadcast_to_tokens(per_seq, mask)
    expected = torch.tensor([[2.0, 2.0, 0.0], [-3.0, 0.0, 0.0]], dtype=torch.float64)
    assert torch.equal(out, expected)
    assert bool((out[~mask] == 0).all())


def test_broadcast_to_tokens_validation() -> None:
    """broadcast_to_tokens raises on rank, emptiness, and batch-size violations."""
    mask = torch.tensor([[True, True], [True, False]])
    with pytest.raises(ValueError, match=r"must be 1-D"):
        broadcast_to_tokens(torch.zeros((2, 2)), mask)
    with pytest.raises(ValueError, match="at least one sequence"):
        broadcast_to_tokens(torch.zeros(0), mask)
    with pytest.raises(ValueError, match="batch size"):
        broadcast_to_tokens(torch.zeros(3), mask)
    with pytest.raises(ValueError, match=r"torch\.bool"):
        broadcast_to_tokens(torch.zeros(2), torch.ones((2, 2)))


# ---------------------------------------------------------------------------
# whiten
# ---------------------------------------------------------------------------


def test_whiten_closed_form() -> None:
    """Whitening matches the hand-derived masked moments
    (docs/derivations/advantages.md, "Broadcasting and whitening")."""
    # x = [1, 2, 3, 4]: mean = 2.5, centered = [-1.5, -0.5, 0.5, 1.5],
    # squared sum = 2.25 + 0.25 + 0.25 + 2.25 = 5, Bessel denominator 3, var = 5/3.
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float64)
    mask = torch.ones((1, 4), dtype=torch.bool)
    out = whiten(x, mask)
    scale = torch.rsqrt(torch.tensor(5.0 / 3.0 + 1e-8, dtype=torch.float64))
    assert torch.equal(out, (x - 2.5) * scale)


def test_whiten_shift_mean_false_restores_mean() -> None:
    """shift_mean=False adds the masked mean back after scaling; masked positions stay
    exactly 0."""
    junk = MASKED_JUNK
    x = torch.tensor([[1.0, 2.0, junk], [3.0, 4.0, junk]], dtype=torch.float64)
    mask = torch.tensor([[True, True, False], [True, True, False]])
    out_true = whiten(x, mask)
    out_false = whiten(x, mask, shift_mean=False)
    expected = torch.where(mask, out_true + 2.5, torch.zeros_like(out_true))
    assert torch.equal(out_false, expected)
    torch.testing.assert_close(
        _masked_mean(out_false, mask), _masked_mean(x, mask), rtol=1e-12, atol=1e-12
    )


@settings(suppress_health_check=[HealthCheck.filter_too_much], deadline=None)
@given(batch=masked_tensors())
def test_whiten_masked_moments(batch: MaskedTensor) -> None:
    """Whitened output has masked mean ~0 and masked Bessel variance ~1 (up to the
    eps regularizer)."""
    assume(int(batch.response_mask.sum()) >= 2)
    assume(float(_masked_var(batch.x, batch.response_mask)) > 1e-3)
    out = whiten(batch.x, batch.response_mask)
    assert abs(float(_masked_mean(out, batch.response_mask))) <= 1e-9
    assert abs(float(_masked_var(out, batch.response_mask)) - 1.0) <= 1e-3


@given(batch=masked_tensors())
def test_whiten_mask_invariance_bitwise(batch: MaskedTensor) -> None:
    """Perturbing x at masked positions leaves the whitened output bitwise unchanged
    (docs/conventions.md, "Masked positions")."""
    assume(int(batch.response_mask.sum()) >= 2)
    out = whiten(batch.x, batch.response_mask)
    perturbed = torch.where(batch.response_mask, batch.x, torch.full_like(batch.x, -55.5))
    out_p = whiten(perturbed, batch.response_mask)
    assert torch.equal(out, out_p)
    assert bool((out[~batch.response_mask] == 0).all())


def test_whiten_fewer_than_two_tokens_raises() -> None:
    """The Bessel-corrected masked variance needs >= 2 response tokens."""
    with pytest.raises(ValueError, match="at least 2 response tokens"):
        whiten(torch.tensor([[3.0]], dtype=torch.float64), torch.tensor([[True]]))


# ---------------------------------------------------------------------------
# dtype preservation
# ---------------------------------------------------------------------------


def test_dtype_preserved_float32() -> None:
    """Every public function returns tensors in the input dtype; no silent casts
    (docs/conventions.md, "Signs, ratios, dtypes")."""
    rewards = torch.tensor([1.0, 3.0, 2.0, 4.0], dtype=torch.float32)
    group_ids = torch.tensor([0, 0, 1, 1])
    assert grpo_advantages(rewards, group_ids, GroupNormConfig()).dtype == torch.float32
    assert rloo_advantages(rewards, group_ids).dtype == torch.float32
    rpp = reinforce_pp_advantages(rewards, config=ReinforcePPConfig(batch_norm=True))
    assert rpp.dtype == torch.float32
    token_rewards = torch.ones((2, 3), dtype=torch.float32)
    values = torch.zeros((2, 3), dtype=torch.float32)
    mask = torch.tensor([[True, True, True], [True, True, False]])
    adv, ret = gae(token_rewards, values, config=GAEConfig(gamma=0.9, lam=0.9), response_mask=mask)
    assert adv.dtype == torch.float32
    assert ret.dtype == torch.float32
    per_seq = torch.tensor([1.0, -1.0], dtype=torch.float32)
    assert broadcast_to_tokens(per_seq, mask).dtype == torch.float32
    assert whiten(token_rewards + values, mask).dtype == torch.float32
