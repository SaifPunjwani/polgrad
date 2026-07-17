"""Enforces the KL estimator semantics of docs/derivations/kl.md: estimator values and
signs, exact/biased expectations against closed-form categorical KL, the as-loss
pathwise gradients (k1 zero-mean, k2 == reverse_kl_grad_surrogate, k3 systematic bias),
kl_in_reward detachment, and mask invariance."""

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
from polgrad.kl import (
    KLEstimator,
    KLLossConfig,
    kl_estimate,
    kl_in_reward,
    kl_loss,
    reverse_kl_grad_surrogate,
)

KINDS = tuple(KLEstimator)
MODES = tuple(Aggregation)
NORM_LEN = 5


def _norm_len(mode: Aggregation) -> int | None:
    return NORM_LEN if mode is Aggregation.TOKEN_SUM_NORM else None


class KLBatch(NamedTuple):
    """Local [B, T] float64 batch: logprob streams, rewards, right-padded mask."""

    logprobs: torch.Tensor
    old_logprobs: torch.Tensor
    ref_logprobs: torch.Tensor
    rewards: torch.Tensor
    response_mask: torch.Tensor


@st.composite
def kl_batches(draw: st.DrawFn) -> KLBatch:
    """Like strategies.logprob_batches but draws a KLBatch: a rewards stream for
    kl_in_reward instead of advantages, and no rollout logprobs. Masked positions
    hold MASKED_JUNK."""
    mask = draw(padded_masks())
    b, t = mask.shape

    def fill(low: float, high: float, junk: float) -> torch.Tensor:
        vals = [
            draw(st.floats(low, high, allow_nan=False, allow_infinity=False, width=32))
            for _ in range(b * t)
        ]
        x = torch.tensor(vals, dtype=torch.float64).reshape(b, t)
        return torch.where(mask, x, torch.full_like(x, junk))

    logprobs = fill(-8.0, -0.0625, MASKED_JUNK)
    old = torch.where(mask, logprobs + fill(-2.0, 2.0, 0.0), torch.full_like(logprobs, MASKED_JUNK))
    ref = torch.where(mask, logprobs + fill(-2.0, 2.0, 0.0), torch.full_like(logprobs, MASKED_JUNK))
    rewards = fill(-3.0, 3.0, MASKED_JUNK)
    return KLBatch(logprobs, old, ref, rewards, mask)


def _categorical_kl(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Closed-form KL(p || q) for categorical distributions."""
    return (p * (p / q).log()).sum()


def test_kl_estimate_golden_values() -> None:
    """Hand-derived estimator values (docs/derivations/kl.md, estimator table).
    delta = ref - lp = [-1.0, 0.5]."""
    lp = torch.tensor([[-0.5, -1.5]], dtype=torch.float64)
    ref = torch.tensor([[-1.5, -1.0]], dtype=torch.float64)
    mask = torch.ones((1, 2), dtype=torch.bool)
    # k1 = lp - ref = [-0.5 - (-1.5), -1.5 - (-1.0)] = [1.0, -0.5], exact in fp64
    assert torch.equal(
        kl_estimate(lp, ref, KLEstimator.K1, response_mask=mask),
        torch.tensor([[1.0, -0.5]], dtype=torch.float64),
    )
    # k2 = delta^2 / 2 = [1.0/2, 0.25/2] = [0.5, 0.125], exact in fp64
    assert torch.equal(
        kl_estimate(lp, ref, KLEstimator.K2, response_mask=mask),
        torch.tensor([[0.5, 0.125]], dtype=torch.float64),
    )
    # k3 = e^delta - 1 - delta = [e^-1 - 1 + 1, e^0.5 - 1 - 0.5] = [e^-1, e^0.5 - 1.5]
    assert_close(
        kl_estimate(lp, ref, KLEstimator.K3, response_mask=mask),
        torch.tensor([[math.exp(-1.0), math.exp(0.5) - 1.5]], dtype=torch.float64),
        rtol=1e-13,
        atol=0.0,
    )
    # abs = |delta| = [1.0, 0.5], exact in fp64
    assert torch.equal(
        kl_estimate(lp, ref, KLEstimator.ABS, response_mask=mask),
        torch.tensor([[1.0, 0.5]], dtype=torch.float64),
    )


@pytest.mark.parametrize("kind", [KLEstimator.K2, KLEstimator.K3, KLEstimator.ABS])
@given(batch=kl_batches())
def test_k2_k3_abs_are_pointwise_nonnegative(kind: KLEstimator, batch: KLBatch) -> None:
    """k2, k3, and abs are pointwise >= 0 (docs/derivations/kl.md: delta^2/2 >= 0,
    e^delta >= 1 + delta, |delta| >= 0)."""
    k = kl_estimate(batch.logprobs, batch.ref_logprobs, kind, response_mask=batch.response_mask)
    assert bool((k >= 0).all())


@pytest.mark.parametrize("kind", KINDS)
@given(batch=kl_batches())
def test_kl_estimate_masked_positions_are_zero_in_value_and_gradient(
    kind: KLEstimator, batch: KLBatch
) -> None:
    """kl_estimate is exactly 0 at masked positions, and so is its gradient w.r.t.
    logprobs (docs/conventions.md, masked positions)."""
    mask = batch.response_mask
    leaf = batch.logprobs.clone().requires_grad_(True)
    k = kl_estimate(leaf, batch.ref_logprobs, kind, response_mask=mask)
    assert torch.equal(k[~mask], torch.zeros_like(k[~mask]))
    (grad,) = torch.autograd.grad(k.sum(), leaf)
    assert torch.equal(grad[~mask], torch.zeros_like(grad[~mask]))


@pytest.mark.parametrize("kind", KINDS)
@given(batch=kl_batches())
def test_kl_estimate_mask_invariance(kind: KLEstimator, batch: KLBatch) -> None:
    """Perturbing masked logprobs/ref_logprobs leaves kl_estimate bitwise unchanged
    (docs/conventions.md, masked positions)."""
    mask = batch.response_mask
    lp2 = torch.where(mask, batch.logprobs, batch.logprobs - 11.5)
    ref2 = torch.where(mask, batch.ref_logprobs, batch.ref_logprobs + 3.25)
    assert torch.equal(
        kl_estimate(batch.logprobs, batch.ref_logprobs, kind, response_mask=mask),
        kl_estimate(lp2, ref2, kind, response_mask=mask),
    )


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("kind", KINDS)
@given(batch=kl_batches())
def test_kl_loss_equals_aggregate_of_kl_estimate(
    mode: Aggregation, kind: KLEstimator, batch: KLBatch
) -> None:
    """kl_loss is aggregate(kl_estimate(...)) bitwise, every kind x aggregation
    (docs/derivations/kl.md)."""
    mask = batch.response_mask
    norm_len = _norm_len(mode)
    loss = kl_loss(
        batch.logprobs, batch.ref_logprobs, kind, mode, response_mask=mask, norm_len=norm_len
    )
    per_token = kl_estimate(batch.logprobs, batch.ref_logprobs, kind, response_mask=mask)
    assert torch.equal(loss, aggregate(per_token, mask, mode, norm_len=norm_len))


@pytest.mark.parametrize("mode", MODES)
@given(batch=kl_batches())
def test_k2_as_loss_gradient_equals_reverse_kl_grad_surrogate(
    mode: Aggregation, batch: KLBatch
) -> None:
    """grad(kl_loss(K2, agg)) == grad(reverse_kl_grad_surrogate(agg)) exactly, for every
    aggregation (docs/derivations/kl.md, k2 pathwise gradient; cross-module
    obligation 2, tests/test_cross.py)."""
    mask = batch.response_mask
    norm_len = _norm_len(mode)
    leaf_a = batch.logprobs.clone().requires_grad_(True)
    (grad_k2,) = torch.autograd.grad(
        kl_loss(
            leaf_a, batch.ref_logprobs, KLEstimator.K2, mode, response_mask=mask, norm_len=norm_len
        ),
        leaf_a,
    )
    leaf_b = batch.logprobs.clone().requires_grad_(True)
    (grad_surrogate,) = torch.autograd.grad(
        reverse_kl_grad_surrogate(
            leaf_b, batch.ref_logprobs, mode, response_mask=mask, norm_len=norm_len
        ),
        leaf_b,
    )
    assert torch.equal(grad_k2, grad_surrogate)


@given(batch=kl_batches())
def test_kl_estimate_gradients_match_analytic_formulas(batch: KLBatch) -> None:
    """The pathwise per-token gradients match the closed forms of
    docs/derivations/kl.md: dk1/dlp = 1, dk2/dlp = lp - ref, dk3/dlp = 1 - e^delta,
    d|delta|/dlp = sign(lp - ref), all masked to 0."""
    mask = batch.response_mask
    zero = torch.zeros((), dtype=torch.float64)
    delta = torch.where(mask, batch.ref_logprobs - batch.logprobs, zero)
    analytic = {
        KLEstimator.K1: torch.where(mask, torch.ones_like(delta), zero),
        KLEstimator.K2: torch.where(mask, batch.logprobs - batch.ref_logprobs, zero),
        KLEstimator.K3: torch.where(mask, -torch.expm1(delta), zero),
        KLEstimator.ABS: torch.where(mask, (batch.logprobs - batch.ref_logprobs).sign(), zero),
    }
    for kind, expected in analytic.items():
        leaf = batch.logprobs.clone().requires_grad_(True)
        k = kl_estimate(leaf, batch.ref_logprobs, kind, response_mask=mask)
        (grad,) = torch.autograd.grad(k.sum(), leaf)
        assert_close(grad, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("mode", [Aggregation.TOKEN_MEAN, Aggregation.SEQ_MEAN_TOKEN_MEAN])
@pytest.mark.parametrize("kind", KINDS)
def test_fp64_gradcheck_kl_loss(kind: KLEstimator, mode: Aggregation) -> None:
    """torch.autograd.gradcheck of kl_loss on a small ragged fp64 batch; |delta| is kept
    away from the abs kink (docs/derivations/kl.md)."""
    lp = torch.tensor(
        [[-0.7, -1.2, -3.0], [-0.4, -2.5, -1.1]], dtype=torch.float64, requires_grad=True
    )
    ref = torch.tensor([[-0.2, -2.0, MASKED_JUNK], [-0.9, -1.6, -0.4]], dtype=torch.float64)
    mask = torch.tensor([[True, True, False], [True, True, True]])

    def fn(leaf: torch.Tensor) -> torch.Tensor:
        return kl_loss(leaf, ref, kind, mode, response_mask=mask)

    assert torch.autograd.gradcheck(fn, (lp,))


@given(batch=kl_batches())
def test_reverse_kl_grad_surrogate_gradient_is_score_function_sample(batch: KLBatch) -> None:
    """The surrogate's gradient is w * (lp - ref) per token: the aggregation-weighted
    score-function sample of grad KL(pi || ref) (docs/derivations/kl.md)."""
    mask = batch.response_mask
    weights = effective_token_weights(mask, Aggregation.TOKEN_MEAN)
    leaf = batch.logprobs.clone().requires_grad_(True)
    (grad,) = torch.autograd.grad(
        reverse_kl_grad_surrogate(
            leaf, batch.ref_logprobs, Aggregation.TOKEN_MEAN, response_mask=mask
        ),
        leaf,
    )
    zero = torch.zeros((), dtype=torch.float64)
    expected = weights * torch.where(mask, batch.logprobs - batch.ref_logprobs, zero)
    assert_close(grad, expected, rtol=1e-12, atol=1e-12)


def test_reverse_kl_grad_surrogate_golden_value() -> None:
    """Hand-derived surrogate value: sg[lp - ref] * lp = (-0.5 - (-1.5)) * (-0.5) with a
    single token and TOKEN_MEAN weight 1 gives exactly -0.5."""
    lp = torch.tensor([[-0.5]], dtype=torch.float64)
    ref = torch.tensor([[-1.5]], dtype=torch.float64)
    mask = torch.ones((1, 1), dtype=torch.bool)
    value = reverse_kl_grad_surrogate(lp, ref, Aggregation.TOKEN_MEAN, response_mask=mask)
    assert torch.equal(value, torch.tensor(-0.5, dtype=torch.float64))


def _tabular_setup() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Tabular softmax policy pi = [0.4, 0.3, 0.2, 0.1] vs reference q = reversed.

    Returns (theta leaf, log pi (differentiable), pi detached, q).
    """
    theta = torch.log(torch.tensor([0.4, 0.3, 0.2, 0.1], dtype=torch.float64)).requires_grad_(True)
    logp = torch.log_softmax(theta, dim=0)
    pi = torch.softmax(theta, dim=0).detach()
    q = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float64)
    return theta, logp, pi, q


def _exact_expectation_grad(kind: KLEstimator) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exact E_pi[grad_theta k(x)] by enumerating the categorical support.

    Returns (grad, pi, q). The expectation weights pi are detached constants, so
    autograd of sum_x pi_x * k(x) w.r.t. theta is exactly the expected as-loss gradient.
    """
    theta, logp, pi, q = _tabular_setup()
    mask = torch.ones((4, 1), dtype=torch.bool)
    k = kl_estimate(logp.unsqueeze(1), q.log().unsqueeze(1), kind, response_mask=mask)
    expectation = (pi.unsqueeze(1) * k).sum()
    (grad,) = torch.autograd.grad(expectation, theta)
    return grad, pi, q


def test_k1_as_loss_expected_gradient_is_zero() -> None:
    """E_pi[grad k1-as-loss] = E_pi[grad log pi] = 0: k1 as a loss optimizes nothing
    (docs/derivations/kl.md, k1 pathwise gradient). Exact enumeration, no MC."""
    grad, _, _ = _exact_expectation_grad(KLEstimator.K1)
    assert_close(grad, torch.zeros(4, dtype=torch.float64), rtol=0.0, atol=1e-12)


def test_k2_as_loss_expected_gradient_equals_analytic_grad_kl() -> None:
    """E_pi[grad k2-as-loss] equals the analytic grad_theta KL(pi || q) =
    pi_j * (log(pi_j/q_j) - KL): the k2 pathwise gradient is the unbiased
    score-function estimator (docs/derivations/kl.md). Exact enumeration, no MC."""
    grad, pi, q = _exact_expectation_grad(KLEstimator.K2)
    kl = _categorical_kl(pi, q)
    analytic = pi * ((pi / q).log() - kl)
    assert_close(grad, analytic, rtol=1e-12, atol=1e-12)


def test_k3_as_loss_expected_gradient_is_pi_minus_q_not_grad_kl() -> None:
    """E_pi[grad k3-as-loss] = pi - q exactly (docs/derivations/kl.md, k3 pathwise
    gradient), which differs systematically from the analytic grad KL; cf. arXiv
    2512.21852 and arXiv 2510.01555. Exact enumeration; the MC companion is
    test_k3_as_loss_gradient_bias_mc_gap_vs_analytic_grad_kl."""
    grad, pi, q = _exact_expectation_grad(KLEstimator.K3)
    assert_close(grad, pi - q, rtol=1e-12, atol=1e-12)
    kl = _categorical_kl(pi, q)
    analytic = pi * ((pi / q).log() - kl)
    # For pi = [.4,.3,.2,.1], q = reversed, the smallest per-component gap is ~0.072.
    assert bool(((pi - q - analytic).abs() > 0.05).all())


def test_k2_expected_value_bias_demonstrated_on_tabular_policy() -> None:
    """E_pi[k2] != KL on a concrete tabular pair: the third-order bias term of
    docs/derivations/kl.md is ~0.065 for pi = [.4,.3,.2,.1] vs q = reversed.
    Exact enumeration, no MC."""
    _, logp, pi, q = _tabular_setup()
    mask = torch.ones((4, 1), dtype=torch.bool)
    k2 = kl_estimate(
        logp.detach().unsqueeze(1), q.log().unsqueeze(1), KLEstimator.K2, response_mask=mask
    )
    expected_k2 = (pi.unsqueeze(1) * k2).sum()
    kl = _categorical_kl(pi, q)
    assert (expected_k2 - kl).item() > 0.05


def test_mc_k1_and_k3_match_closed_form_categorical_kl(gen: torch.Generator) -> None:
    """MC certification (docs/derivations/kl.md, expectations): the sample means of k1 and k3 under
    x ~ pi match closed-form categorical KL(pi || q) within an inline CLT tolerance of
    z * sample_std / sqrt(n) with z = 4 (two-sided miss probability ~6e-5), on one
    seeded draw of n = 300000 samples."""
    p = torch.tensor([0.5, 0.25, 0.15, 0.1], dtype=torch.float64)
    q = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float64)
    n = 300_000
    x = torch.multinomial(p, n, replacement=True, generator=gen)
    lp = p.log()[x].unsqueeze(1)
    lq = q.log()[x].unsqueeze(1)
    mask = torch.ones((n, 1), dtype=torch.bool)
    kl = _categorical_kl(p, q)
    for kind in (KLEstimator.K1, KLEstimator.K3):
        samples = kl_estimate(lp, lq, kind, response_mask=mask).squeeze(1)
        tol = 4.0 * samples.std().item() / math.sqrt(n)
        assert abs(samples.mean().item() - kl.item()) < tol


def test_var_k3_below_var_k1_for_near_identical_policies(gen: torch.Generator) -> None:
    """var(k3) < var(k1) when pi is close to ref: k1 fluctuates
    at order delta while k3 fluctuates at order delta^2 (docs/derivations/kl.md).
    Verified on one seeded MC draw."""
    logits = torch.tensor([0.3, -0.2, 0.5, 0.0, -0.4], dtype=torch.float64)
    p = torch.softmax(logits, dim=0)
    q = torch.softmax(logits + 0.03 * torch.tensor([1.0, -1.0, 0.5, -0.5, 0.0]), dim=0)
    n = 200_000
    x = torch.multinomial(p, n, replacement=True, generator=gen)
    lp = p.log()[x].unsqueeze(1)
    lq = q.log()[x].unsqueeze(1)
    mask = torch.ones((n, 1), dtype=torch.bool)
    k1 = kl_estimate(lp, lq, KLEstimator.K1, response_mask=mask)
    k3 = kl_estimate(lp, lq, KLEstimator.K3, response_mask=mask)
    assert k3.var().item() < k1.var().item()


def test_k3_as_loss_gradient_bias_mc_gap_vs_analytic_grad_kl(gen: torch.Generator) -> None:
    """MC demonstration of the systematic k3-as-loss gradient bias on a tabular policy
    (docs/derivations/kl.md; cf. arXiv 2512.21852 "A Comedy of Estimators" and arXiv
    2510.01555 "Rethinking KL Regularization in RLHF"): the MC gradient of
    kl_loss(K3, TOKEN_MEAN) matches its expectation pi - q within a per-component CLT
    tolerance, while pi - q sits more than 10 tolerances away from the analytic
    grad_theta KL(pi || q)."""
    theta, logp_all, pi, q = _tabular_setup()
    n = 1_000_000
    x = torch.multinomial(pi, n, replacement=True, generator=gen)
    lp = logp_all[x].unsqueeze(1)
    lq = q.log()[x].unsqueeze(1)
    mask = torch.ones((n, 1), dtype=torch.bool)
    loss = kl_loss(lp, lq, KLEstimator.K3, Aggregation.TOKEN_MEAN, response_mask=mask)
    (grad,) = torch.autograd.grad(loss, theta)
    # Closed-form per-sample pathwise gradient: (1 - e^delta) * (onehot(x) - pi).
    coeff = -torch.expm1((lq - lp).detach().squeeze(1))
    onehot = torch.zeros((n, 4), dtype=torch.float64)
    onehot[torch.arange(n), x] = 1.0
    per_sample = coeff.unsqueeze(1) * (onehot - pi)
    mc_mean = per_sample.mean(dim=0)
    tol = 4.0 * per_sample.std(dim=0) / math.sqrt(n)
    assert_close(grad, mc_mean, rtol=0.0, atol=1e-8)
    assert bool(((mc_mean - (pi - q)).abs() < tol).all())
    kl = _categorical_kl(pi, q)
    analytic = pi * ((pi / q).log() - kl)
    assert bool((((pi - q) - analytic).abs() > 10.0 * tol).all())


def test_kl_in_reward_golden_values() -> None:
    """Hand-derived kl_in_reward outputs (docs/derivations/kl.md, reward placement).
    K1: k = old - ref = [0.5, -1.0]; r - 0.5*k = [2 - 0.25, -1 + 0.5] = [1.75, -0.5]."""
    rewards = torch.tensor([[2.0, -1.0]], dtype=torch.float64)
    old = torch.tensor([[-1.0, -2.0]], dtype=torch.float64)
    ref = torch.tensor([[-1.5, -1.0]], dtype=torch.float64)
    mask = torch.ones((1, 2), dtype=torch.bool)
    out_k1 = kl_in_reward(rewards, old, ref, KLEstimator.K1, 0.5, response_mask=mask)
    assert torch.equal(out_k1, torch.tensor([[1.75, -0.5]], dtype=torch.float64))
    # K3: delta = ref - old = [-0.5, 1.0]; k3 = [e^-0.5 - 0.5, e - 2];
    # r - 0.5*k3 = [2 - 0.5*(e^-0.5 - 0.5), -1 - 0.5*(e - 2)]
    out_k3 = kl_in_reward(rewards, old, ref, KLEstimator.K3, 0.5, response_mask=mask)
    expected = rewards - 0.5 * torch.tensor(
        [[math.exp(-0.5) - 0.5, math.e - 2.0]], dtype=torch.float64
    )
    assert_close(out_k3, expected, rtol=1e-13, atol=0.0)
    # Masked positions come back exactly 0.
    short_mask = torch.tensor([[True, False]])
    out_masked = kl_in_reward(rewards, old, ref, KLEstimator.K1, 0.5, response_mask=short_mask)
    assert torch.equal(out_masked, torch.tensor([[1.75, 0.0]], dtype=torch.float64))


def test_kl_in_reward_is_detached() -> None:
    """kl_in_reward returns a detached tensor even when its inputs carry grad
    (docs/derivations/kl.md, KL in the reward: the penalty uses the sampling policy,
    no gradient)."""
    rewards = torch.tensor([[1.0, 2.0]], dtype=torch.float64, requires_grad=True)
    old = torch.tensor([[-1.0, -2.0]], dtype=torch.float64, requires_grad=True)
    ref = torch.tensor([[-1.5, -1.0]], dtype=torch.float64)
    mask = torch.ones((1, 2), dtype=torch.bool)
    out = kl_in_reward(rewards, old, ref, KLEstimator.K2, 0.1, response_mask=mask)
    assert not out.requires_grad
    assert out.grad_fn is None


@pytest.mark.parametrize("kind", KINDS)
@given(batch=kl_batches())
def test_kl_in_reward_mask_invariance(kind: KLEstimator, batch: KLBatch) -> None:
    """Perturbing masked rewards/old/ref leaves kl_in_reward bitwise unchanged, and
    masked outputs are exactly 0 (docs/conventions.md)."""
    mask = batch.response_mask
    rewards = batch.rewards
    out = kl_in_reward(
        rewards, batch.old_logprobs, batch.ref_logprobs, kind, 0.2, response_mask=mask
    )
    perturbed = kl_in_reward(
        torch.where(mask, rewards, rewards + 9.0),
        torch.where(mask, batch.old_logprobs, batch.old_logprobs - 4.5),
        torch.where(mask, batch.ref_logprobs, batch.ref_logprobs + 2.5),
        kind,
        0.2,
        response_mask=mask,
    )
    assert torch.equal(out, perturbed)
    assert torch.equal(out[~mask], torch.zeros_like(out[~mask]))


def test_kl_functions_preserve_input_dtype() -> None:
    """kl_estimate/kl_loss/kl_in_reward/reverse_kl_grad_surrogate preserve the input
    dtype with no silent casts (docs/conventions.md, dtypes)."""
    for dtype in (torch.float32, torch.float64):
        lp = torch.tensor([[-0.5, -1.5]], dtype=dtype)
        ref = torch.tensor([[-1.5, -1.0]], dtype=dtype)
        rewards = torch.tensor([[1.0, 2.0]], dtype=dtype)
        mask = torch.ones((1, 2), dtype=torch.bool)
        assert kl_estimate(lp, ref, KLEstimator.K3, response_mask=mask).dtype == dtype
        assert (
            kl_loss(lp, ref, KLEstimator.K2, Aggregation.TOKEN_MEAN, response_mask=mask).dtype
            == dtype
        )
        assert (
            kl_in_reward(rewards, lp, ref, KLEstimator.K1, 0.1, response_mask=mask).dtype == dtype
        )
        assert (
            reverse_kl_grad_surrogate(lp, ref, Aggregation.TOKEN_MEAN, response_mask=mask).dtype
            == dtype
        )


def test_kl_loss_config_is_frozen_data() -> None:
    """KLLossConfig is inert frozen data: construction with norm_len=None is always
    legal (the norm_len requirement is enforced at call time,
    docs/derivations/aggregation.md); the TOKEN_SUM_NORM requirement fires when
    kl_loss is called."""
    config = KLLossConfig(kind=KLEstimator.K2, coef=0.05)
    assert config.aggregation is None
    assert config.norm_len is None
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.coef = 1.0  # type: ignore[misc]
    strict = KLLossConfig(
        kind=KLEstimator.K1, coef=1.0, aggregation=Aggregation.TOKEN_SUM_NORM, norm_len=None
    )
    lp = torch.tensor([[-0.5]], dtype=torch.float64)
    ref = torch.tensor([[-1.0]], dtype=torch.float64)
    mask = torch.ones((1, 1), dtype=torch.bool)
    assert strict.aggregation is not None
    with pytest.raises(ValueError, match="norm_len is required"):
        kl_loss(
            lp, ref, strict.kind, strict.aggregation, response_mask=mask, norm_len=strict.norm_len
        )


def test_reverse_kl_grad_surrogate_requires_norm_len_for_token_sum_norm() -> None:
    """The norm_len requirement propagates through reverse_kl_grad_surrogate."""
    lp = torch.tensor([[-0.5]], dtype=torch.float64)
    ref = torch.tensor([[-1.0]], dtype=torch.float64)
    mask = torch.ones((1, 1), dtype=torch.bool)
    with pytest.raises(ValueError, match="norm_len is required"):
        reverse_kl_grad_surrogate(lp, ref, Aggregation.TOKEN_SUM_NORM, response_mask=mask)


def test_kl_estimate_validation_errors() -> None:
    """Shape/dtype/mask/finiteness violations raise ValueError naming the argument;
    non-finite values at MASKED positions do not raise (mask invariance extends to
    validation)."""
    lp = torch.tensor([[-0.5, -1.5]], dtype=torch.float64)
    ref = torch.tensor([[-1.5, -1.0]], dtype=torch.float64)
    mask = torch.ones((1, 2), dtype=torch.bool)
    with pytest.raises(ValueError, match=r"logprobs must be 2-D"):
        kl_estimate(lp.squeeze(0), ref, KLEstimator.K1, response_mask=mask)
    with pytest.raises(ValueError, match=r"identical shapes"):
        kl_estimate(lp, ref[:, :1], KLEstimator.K1, response_mask=mask)
    with pytest.raises(ValueError, match=r"dtype torch\.bool"):
        kl_estimate(lp, ref, KLEstimator.K1, response_mask=torch.ones((1, 2)))
    with pytest.raises(ValueError, match=r"zero response tokens"):
        kl_estimate(lp, ref, KLEstimator.K1, response_mask=torch.zeros((1, 2), dtype=torch.bool))
    bad = torch.tensor([[float("nan"), -1.5]], dtype=torch.float64)
    with pytest.raises(ValueError, match="non-finite"):
        kl_estimate(bad, ref, KLEstimator.K1, response_mask=mask)
    # NaN confined to a masked position is legal and does not perturb the output.
    ragged = torch.tensor([[True, False]])
    junk = torch.tensor([[-0.5, float("nan")]], dtype=torch.float64)
    out = kl_estimate(junk, ref, KLEstimator.K3, response_mask=ragged)
    clean = kl_estimate(lp, ref, KLEstimator.K3, response_mask=ragged)
    assert torch.equal(out, clean)
