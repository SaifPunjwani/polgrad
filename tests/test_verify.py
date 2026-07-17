"""Enforces docs/derivations/goldens.md for polgrad.verify.gradcheck and polgrad.verify.goldens:
gradcheck_loss passes on representative configs; check_gradient_formula validates the
correct hand-derived PG gradient and raises on a deliberately wrong derivation;
SoftmaxBandit's closed forms match autograd on the exact expected objective and direct
categorical KL (plus MC certification via verify.mc); every golden case is satisfied by
polgrad.losses.policy_loss to 1e-12. Derivations: docs/derivations/goldens.md."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st
from strategies import MASKED_JUNK, padded_masks
from torch.testing import assert_close

from polgrad.aggregate import Aggregation, effective_token_weights
from polgrad.kl import KLEstimator, KLLossConfig
from polgrad.losses import (
    ClipConfig,
    ISCorrectionConfig,
    PolicyLossConfig,
    RatioKind,
    SurrogateKind,
    policy_loss,
)
from polgrad.verify.goldens import GoldenCase, SoftmaxBandit, golden_cases
from polgrad.verify.gradcheck import check_gradient_formula, gradcheck_loss
from polgrad.verify.mc import clt_tolerance, mc_mean

DOCS_PAGE = Path(__file__).resolve().parent.parent / "docs" / "derivations" / "goldens.md"


def _gen(seed: int = 7) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


# --- gradcheck_loss (docs/derivations/goldens.md) --------------------------------------

REPRESENTATIVE_CONFIGS = [
    pytest.param(
        PolicyLossConfig(
            ratio=RatioKind.TOKEN,
            surrogate=SurrogateKind.PG_CLIP,
            clip=ClipConfig(eps_low=0.2, eps_high=0.3),
            aggregation=Aggregation.TOKEN_MEAN,
        ),
        id="pg_clip-token-token_mean",
    ),
    pytest.param(
        PolicyLossConfig(
            ratio=RatioKind.TOKEN,
            surrogate=SurrogateKind.PG_CLIP,
            clip=ClipConfig(eps_low=0.2, eps_high=0.3, ratio_cap=2.0),
            aggregation=Aggregation.SEQ_MEAN_TOKEN_MEAN,
        ),
        id="pg_clip-dual_clip-token-seq_mean_token_mean",
    ),
    pytest.param(
        PolicyLossConfig(
            ratio=RatioKind.SEQUENCE,
            surrogate=SurrogateKind.PG_CLIP,
            clip=ClipConfig(eps_low=0.2, eps_high=0.3),
            aggregation=Aggregation.SEQ_MEAN_TOKEN_MEAN,
        ),
        id="pg_clip-sequence-seq_mean_token_mean",
    ),
    pytest.param(
        PolicyLossConfig(
            ratio=RatioKind.SEQUENCE_TOKEN,
            surrogate=SurrogateKind.PG_CLIP,
            clip=ClipConfig(eps_low=0.2, eps_high=0.3),
            aggregation=Aggregation.TOKEN_MEAN,
        ),
        id="pg_clip-sequence_token-token_mean",
    ),
    pytest.param(
        PolicyLossConfig(
            ratio=RatioKind.TOKEN,
            surrogate=SurrogateKind.PG,
            clip=None,
            aggregation=Aggregation.TOKEN_SUM_NORM,
            norm_len=4,
        ),
        id="pg-token-token_sum_norm",
    ),
    pytest.param(
        PolicyLossConfig(
            ratio=RatioKind.SEQUENCE,
            surrogate=SurrogateKind.PG,
            clip=None,
            aggregation=Aggregation.SEQ_MEAN_TOKEN_SUM,
        ),
        id="pg-sequence-seq_mean_token_sum",
    ),
    pytest.param(
        PolicyLossConfig(
            ratio=RatioKind.TOKEN,
            surrogate=SurrogateKind.REINFORCE,
            clip=None,
            aggregation=Aggregation.TOKEN_MEAN,
        ),
        id="reinforce-token-token_mean",
    ),
    pytest.param(
        PolicyLossConfig(
            ratio=RatioKind.TOKEN,
            surrogate=SurrogateKind.CISPO,
            clip=ClipConfig(eps_low=None, eps_high=0.3),
            aggregation=Aggregation.TOKEN_MEAN,
        ),
        id="cispo-one_sided-token-token_mean",
    ),
    pytest.param(
        PolicyLossConfig(
            ratio=RatioKind.SEQUENCE,
            surrogate=SurrogateKind.CISPO,
            clip=ClipConfig(eps_low=0.15, eps_high=0.3),
            aggregation=Aggregation.SEQ_MEAN_TOKEN_MEAN,
        ),
        id="cispo-two_sided-sequence-seq_mean_token_mean",
    ),
    pytest.param(
        PolicyLossConfig(
            ratio=RatioKind.TOKEN,
            surrogate=SurrogateKind.PG_CLIP,
            clip=ClipConfig(eps_low=0.2, eps_high=0.3),
            aggregation=Aggregation.TOKEN_MEAN,
            is_correction=ISCorrectionConfig(cap=1.5, level="token"),
            kl=KLLossConfig(kind=KLEstimator.K3, coef=0.07),
        ),
        id="pg_clip-token-tis-kl_k3",
    ),
    pytest.param(
        PolicyLossConfig(
            ratio=RatioKind.TOKEN,
            surrogate=SurrogateKind.PG,
            clip=None,
            aggregation=Aggregation.TOKEN_MEAN,
            kl=KLLossConfig(kind=KLEstimator.ABS, coef=0.05),
        ),
        id="pg-token-kl_abs",
    ),
]


@pytest.mark.parametrize("config", REPRESENTATIVE_CONFIGS)
def test_gradcheck_loss_passes_for_representative_configs(config: PolicyLossConfig) -> None:
    """gradcheck_loss completes without raising on ragged fp64 batches for
    representative SurrogateKind x RatioKind x Aggregation configs, including
    dual-clip, TIS + KL composition, and the stop-gradient surrogates whose frozen
    equivalents are derived in docs/derivations/losses.md."""
    gradcheck_loss(config, batch_shapes=((2, 3), (3, 5)), generator=_gen())


def test_gradcheck_loss_full_mask_when_ragged_false() -> None:
    """ragged=False draws all-true masks and still passes for a clipped config."""
    config = PolicyLossConfig(
        ratio=RatioKind.TOKEN,
        surrogate=SurrogateKind.PG_CLIP,
        clip=ClipConfig(eps_low=0.2, eps_high=0.3),
        aggregation=Aggregation.SEQ_MEAN_TOKEN_MEAN,
    )
    gradcheck_loss(config, batch_shapes=((2, 2),), generator=_gen(11), ragged=False)


def test_gradcheck_loss_validation_errors() -> None:
    """Empty or non-positive batch_shapes raise ValueError, and an invalid config
    propagates policy_loss's config-validation ValueError."""
    config = PolicyLossConfig(
        ratio=RatioKind.TOKEN,
        surrogate=SurrogateKind.PG,
        clip=None,
        aggregation=Aggregation.TOKEN_MEAN,
    )
    with pytest.raises(ValueError, match="batch_shapes"):
        gradcheck_loss(config, batch_shapes=(), generator=_gen())
    with pytest.raises(ValueError, match="batch_shapes"):
        gradcheck_loss(config, batch_shapes=((0, 3),), generator=_gen())
    bad = PolicyLossConfig(
        ratio=RatioKind.TOKEN,
        surrogate=SurrogateKind.PG_CLIP,
        clip=None,
        aggregation=Aggregation.TOKEN_MEAN,
    )
    with pytest.raises(ValueError, match="PG_CLIP requires"):
        gradcheck_loss(bad, batch_shapes=((2, 2),), generator=_gen())


# --- check_gradient_formula (docs/derivations/goldens.md) ------------------------------
#
# Target: PG/TOKEN/TOKEN_MEAN loss L = sum_t w_t * (-r_t * A_t) with w_t = m_t / N.
# Correct derivation (docs/derivations/losses.md, PG gradient): dL/dlp_t = -w_t A_t r_t.
# Deliberately wrong derivation: -w_t A_t (a REINFORCE gradient — drops the ratio
# factor r_t, the classic error when differentiating exp(lp - old_lp)).

PG_CONFIG = PolicyLossConfig(
    ratio=RatioKind.TOKEN,
    surrogate=SurrogateKind.PG,
    clip=None,
    aggregation=Aggregation.TOKEN_MEAN,
)
CGF_MASK = torch.tensor([[True, True, True], [True, True, False]])
CGF_LP = torch.tensor([[-0.9, -1.6, -0.3], [-0.6, -1.1, MASKED_JUNK]], dtype=torch.float64)
CGF_OLP = torch.tensor([[-1.0, -1.4, -0.5], [-0.5, -1.3, MASKED_JUNK]], dtype=torch.float64)
CGF_ADV = torch.tensor([[1.5, -2.0, 0.8], [-1.0, 2.2, MASKED_JUNK]], dtype=torch.float64)


def _pg_loss(lp: torch.Tensor, olp: torch.Tensor, adv: torch.Tensor) -> torch.Tensor:
    return policy_loss(
        PG_CONFIG, logprobs=lp, old_logprobs=olp, advantages=adv, response_mask=CGF_MASK
    ).loss


def _pg_grad(lp: torch.Tensor, olp: torch.Tensor, adv: torch.Tensor) -> torch.Tensor:
    weights = effective_token_weights(CGF_MASK, Aggregation.TOKEN_MEAN)
    zero = torch.zeros((), dtype=torch.float64)
    ratio = torch.exp(torch.where(CGF_MASK, lp - olp, zero))
    return torch.where(CGF_MASK, -(weights * adv * ratio), zero)


def _pg_grad_missing_ratio(lp: torch.Tensor, olp: torch.Tensor, adv: torch.Tensor) -> torch.Tensor:
    weights = effective_token_weights(CGF_MASK, Aggregation.TOKEN_MEAN)
    zero = torch.zeros((), dtype=torch.float64)
    return torch.where(CGF_MASK, -(weights * adv), zero)


def test_check_gradient_formula_accepts_correct_pg_derivation() -> None:
    """The hand-derived PG gradient -w_t A_t r_t matches central finite differences of
    the PG loss (docs/derivations/losses.md, PG gradient)."""
    check_gradient_formula(_pg_loss, _pg_grad, (CGF_LP, CGF_OLP, CGF_ADV), atol=1e-8, rtol=1e-6)


def test_check_gradient_formula_raises_on_wrong_derivation() -> None:
    """A deliberately wrong derivation — the REINFORCE gradient -w_t A_t, which drops
    the ratio factor r_t — is rejected against finite differences of the PG loss
    (the check catches wrong derivations, not autograd inconsistency)."""
    with pytest.raises(AssertionError, match="finite differences"):
        check_gradient_formula(
            _pg_loss, _pg_grad_missing_ratio, (CGF_LP, CGF_OLP, CGF_ADV), atol=1e-8, rtol=1e-6
        )


@st.composite
def pg_batches(draw: st.DrawFn) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Like strategies.logprob_batches but a plain (lp, olp, adv, mask) 4-tuple with
    tighter |gaps| <= 1.5, matching check_gradient_formula's positional inputs. Masked
    positions hold MASKED_JUNK."""
    mask = draw(padded_masks(max_b=4, max_t=5))
    b, t = mask.shape

    def fill(low: float, high: float, junk: float) -> torch.Tensor:
        vals = [
            draw(st.floats(low, high, allow_nan=False, allow_infinity=False, width=32))
            for _ in range(b * t)
        ]
        x = torch.tensor(vals, dtype=torch.float64).reshape(b, t)
        return torch.where(mask, x, torch.full_like(x, junk))

    lp = fill(-8.0, -0.0625, MASKED_JUNK)
    olp = torch.where(mask, lp + fill(-1.5, 1.5, 0.0), torch.full_like(lp, MASKED_JUNK))
    adv = fill(-3.0, 3.0, MASKED_JUNK)
    return lp, olp, adv, mask


@given(batch=pg_batches())
def test_check_gradient_formula_pg_derivation_property(
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    """The PG gradient formula -w_t A_t r_t survives finite differences on random
    ragged batches (docs/derivations/losses.md, PG gradient)."""
    lp, olp, adv, mask = batch

    def loss(x: torch.Tensor, o: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return policy_loss(
            PG_CONFIG, logprobs=x, old_logprobs=o, advantages=a, response_mask=mask
        ).loss

    def grad(x: torch.Tensor, o: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        weights = effective_token_weights(mask, Aggregation.TOKEN_MEAN)
        zero = torch.zeros((), dtype=torch.float64)
        ratio = torch.exp(torch.where(mask, x - o, zero))
        return torch.where(mask, -(weights * a * ratio), zero)

    check_gradient_formula(loss, grad, (lp, olp, adv), atol=1e-6, rtol=1e-6)


def test_check_gradient_formula_validation_errors() -> None:
    """Empty inputs, non-floating inputs[0], invalid eps/atol/rtol, non-scalar fn
    output, and wrong-shape analytic output raise ValueError."""
    x = torch.tensor([1.0, 2.0], dtype=torch.float64)

    def cube(v: torch.Tensor) -> torch.Tensor:
        return (v**3).sum()

    def cube_grad(v: torch.Tensor) -> torch.Tensor:
        return 3.0 * v**2

    with pytest.raises(ValueError, match="inputs"):
        check_gradient_formula(cube, cube_grad, (), atol=1e-8, rtol=1e-6)
    with pytest.raises(ValueError, match="floating-point"):
        check_gradient_formula(cube, cube_grad, (torch.tensor([1, 2]),), atol=1e-8, rtol=1e-6)
    with pytest.raises(ValueError, match="eps"):
        check_gradient_formula(cube, cube_grad, (x,), eps=0.0, atol=1e-8, rtol=1e-6)
    with pytest.raises(ValueError, match="atol"):
        check_gradient_formula(cube, cube_grad, (x,), atol=-1.0, rtol=1e-6)
    with pytest.raises(ValueError, match="scalar"):
        check_gradient_formula(cube_grad, cube_grad, (x,), atol=1e-8, rtol=1e-6)
    with pytest.raises(ValueError, match="shape"):
        check_gradient_formula(cube, lambda v: (3.0 * v**2)[:1], (x,), atol=1e-8, rtol=1e-6)
    check_gradient_formula(cube, cube_grad, (x,), atol=1e-8, rtol=1e-6)


# --- SoftmaxBandit closed forms (docs/derivations/goldens.md) --------------------------


@st.composite
def theta_pairs(draw: st.DrawFn) -> tuple[torch.Tensor, torch.Tensor]:
    """Pairs of same-length [K] float64 logit/value vectors, K in [2, 8]."""
    k = draw(st.integers(2, 8))
    fl = st.floats(-4.0, 4.0, allow_nan=False, allow_infinity=False, width=32)
    first = torch.tensor([draw(fl) for _ in range(k)], dtype=torch.float64)
    second = torch.tensor([draw(fl) for _ in range(k)], dtype=torch.float64)
    return first, second


@given(pair=theta_pairs())
def test_exact_policy_gradient_matches_autograd_on_expected_objective(
    pair: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """exact_policy_gradient equals torch.autograd on the exact expected objective
    J(theta) = sum_k softmax(theta)_k A_k (docs/derivations/goldens.md, bandit policy
    gradient)."""
    theta, advantages = pair
    leaf = theta.clone().requires_grad_(True)
    objective = (torch.softmax(leaf, dim=0) * advantages).sum()
    (autograd_grad,) = torch.autograd.grad(objective, leaf)
    bandit = SoftmaxBandit(theta, torch.zeros_like(theta))
    assert_close(bandit.exact_policy_gradient(advantages), autograd_grad, rtol=1e-12, atol=1e-12)


@given(pair=theta_pairs())
def test_exact_policy_gradient_sums_to_zero_and_is_shift_invariant(
    pair: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """The closed form pi_j (A_j - mean_pi(A)) sums to zero over arms and is invariant
    to a constant advantage shift (docs/derivations/goldens.md, bandit policy
    gradient)."""
    theta, advantages = pair
    bandit = SoftmaxBandit(theta, torch.zeros_like(theta))
    grad = bandit.exact_policy_gradient(advantages)
    assert abs(float(grad.sum())) <= 1e-12
    shifted = bandit.exact_policy_gradient(advantages + 3.7)
    assert_close(shifted, grad, rtol=1e-12, atol=1e-12)


@given(pair=theta_pairs())
def test_exact_kl_matches_direct_categorical_kl(pair: tuple[torch.Tensor, torch.Tensor]) -> None:
    """exact_kl equals the direct categorical KL sum_k p_k (log p_k - log q_k), is
    non-negative, and is exactly 0 against itself (docs/derivations/goldens.md, bandit
    KL)."""
    theta, other = pair
    bandit = SoftmaxBandit(theta, torch.zeros_like(theta))
    log_p = torch.log_softmax(theta, dim=0)
    log_q = torch.log_softmax(other, dim=0)
    direct = float((log_p.exp() * (log_p - log_q)).sum())
    assert math.isclose(bandit.exact_kl(other), direct, rel_tol=1e-12, abs_tol=1e-12)
    assert bandit.exact_kl(other) >= -1e-15
    assert bandit.exact_kl(theta) == 0.0


def test_exact_policy_gradient_matches_mc_score_function_estimate() -> None:
    """MC certification: the REINFORCE estimator A(a) * (delta_aj - pi_j) of
    dJ/dtheta_j, sampled through SoftmaxBandit.sample, matches exact_policy_gradient
    within 4 standard errors per coordinate (docs/derivations/goldens.md, score-function
    identity; verifies on one seeded run of n = 16384 per arm)."""
    theta = torch.tensor([0.4, -0.6, 0.1, -0.2], dtype=torch.float64)
    advantages = torch.tensor([1.0, -0.5, 0.25, 2.0], dtype=torch.float64)
    bandit = SoftmaxBandit(theta, torch.zeros_like(theta))
    pi = bandit.probs()
    exact = bandit.exact_policy_gradient(advantages)
    generator = _gen(19)
    for j in range(bandit.num_arms):

        def sample_coordinate(k: int, gen: torch.Generator, arm: int = j) -> torch.Tensor:
            actions = bandit.sample(k, gen).actions
            indicator = (actions == arm).to(torch.float64)
            return advantages[actions] * (indicator - pi[arm])

        mean, std_err = mc_mean(sample_coordinate, 16384, generator)
        assert abs(mean - float(exact[j])) <= 4.0 * std_err + 1e-12


def test_softmax_bandit_sample_contract() -> None:
    """sample(n, generator) returns [n, 1] streams with old == ref == rollout ==
    sg[logprobs], an all-true bool mask, arm rewards indexed by actions, and a
    logprobs stream whose gradient w.r.t. theta is counts_j - n * pi_j
    (docs/derivations/goldens.md, sampling contract)."""
    theta = torch.tensor([0.3, -0.2, 0.5], dtype=torch.float64, requires_grad=True)
    rewards = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    bandit = SoftmaxBandit(theta, rewards)
    n = 32
    batch = bandit.sample(n, _gen(5))
    assert batch.logprobs.shape == (n, 1)
    assert batch.response_mask.shape == (n, 1)
    assert batch.response_mask.dtype == torch.bool
    assert bool(batch.response_mask.all())
    assert batch.actions.shape == (n,)
    assert batch.actions.dtype == torch.long
    assert bool((batch.actions >= 0).all()) and bool((batch.actions < 3).all())
    assert torch.equal(batch.old_logprobs, batch.logprobs.detach())
    assert torch.equal(batch.ref_logprobs, batch.old_logprobs)
    assert torch.equal(batch.rollout_logprobs, batch.old_logprobs)
    assert not batch.old_logprobs.requires_grad
    assert torch.equal(batch.rewards, rewards[batch.actions])
    expected_lp = torch.log_softmax(theta.detach(), dim=0)[batch.actions].unsqueeze(1)
    assert torch.equal(batch.logprobs.detach(), expected_lp)
    assert batch.logprobs.requires_grad
    (grad,) = torch.autograd.grad(batch.logprobs.sum(), theta)
    counts = torch.bincount(batch.actions, minlength=3).to(torch.float64)
    pi = torch.softmax(theta.detach(), dim=0)
    assert_close(grad, counts - n * pi, rtol=1e-12, atol=1e-12)


def test_softmax_bandit_sample_reward_mean_matches_closed_form_mc() -> None:
    """MC certification via verify.mc: the sample-mean reward over n = 32768 draws
    matches the closed form sum_k pi_k r_k within clt_tolerance(std, n) for the exact
    reward standard deviation (docs/derivations/goldens.md, sampling contract;
    verifies on one seeded run)."""
    theta = torch.tensor([0.2, -0.4, 0.9, 0.0], dtype=torch.float64)
    rewards = torch.tensor([0.0, 1.0, 0.25, 2.0], dtype=torch.float64)
    bandit = SoftmaxBandit(theta, rewards)
    pi = bandit.probs()
    target = float((pi * rewards).sum())
    exact_std = math.sqrt(float((pi * rewards**2).sum()) - target**2)
    n = 32768

    def sample_rewards(k: int, gen: torch.Generator) -> torch.Tensor:
        return bandit.sample(k, gen).rewards

    mean, _ = mc_mean(sample_rewards, n, _gen(23))
    assert abs(mean - target) <= clt_tolerance(exact_std, n)


def test_softmax_bandit_validation_errors() -> None:
    """Constructor and method shape/finiteness rules raise ValueError naming the
    argument (docs/conventions.md error rules)."""
    theta = torch.tensor([0.1, -0.1], dtype=torch.float64)
    with pytest.raises(ValueError, match="theta"):
        SoftmaxBandit(torch.zeros((2, 2)), torch.zeros(2))
    with pytest.raises(ValueError, match="arm_rewards"):
        SoftmaxBandit(theta, torch.zeros(3))
    with pytest.raises(ValueError, match="at least 2 arms"):
        SoftmaxBandit(torch.zeros(1), torch.zeros(1))
    with pytest.raises(ValueError, match="theta"):
        SoftmaxBandit(torch.tensor([0.0, float("inf")]), torch.zeros(2))
    bandit = SoftmaxBandit(theta, torch.zeros(2))
    with pytest.raises(ValueError, match="n must be"):
        bandit.sample(0, _gen())
    with pytest.raises(ValueError, match="advantages"):
        bandit.exact_policy_gradient(torch.zeros(3))
    with pytest.raises(ValueError, match="other_theta"):
        bandit.exact_kl(torch.zeros((2, 2)))


# --- golden cases (docs/derivations/goldens.md) ------------------------------------------


def _golden_params() -> list[object]:
    return [pytest.param(case, id=case.name) for case in golden_cases()]


@pytest.mark.parametrize("case", _golden_params())
def test_golden_cases_satisfied_by_policy_loss(case: GoldenCase) -> None:
    """policy_loss reproduces each hand-derived expected_loss and
    expected_grad_logprobs to 1e-12 (docs/derivations/goldens.md, section named by the
    case)."""
    leaf = case.logprobs.clone().requires_grad_(True)
    result = policy_loss(
        case.config,
        logprobs=leaf,
        old_logprobs=case.old_logprobs,
        advantages=case.advantages,
        response_mask=case.response_mask,
    )
    (grad,) = torch.autograd.grad(result.loss, leaf)
    assert abs(float(result.loss.detach()) - case.expected_loss) <= 1e-12
    assert_close(grad, case.expected_grad_logprobs, rtol=0.0, atol=1e-12)


def test_golden_cases_cover_contract_branches() -> None:
    """golden_cases() contains every case docs/derivations/goldens.md lists — inside
    clip, clipped high (A > 0), clipped low (A < 0), dual-clip (A < 0, r > c), and a
    2-token ragged case — and every derivation anchor names an existing section of
    docs/derivations/goldens.md."""
    cases = golden_cases()
    names = {case.name for case in cases}
    required = {
        "pg_clip_inside_band",
        "pg_clip_clipped_high_positive_advantage",
        "pg_clip_clipped_low_negative_advantage",
        "dual_clip_negative_advantage_above_cap",
        "pg_clip_two_token_ragged_token_mean",
    }
    assert required <= names
    page = DOCS_PAGE.read_text()
    for case in cases:
        assert case.derivation == f"docs/derivations/goldens.md#{case.name}"
        assert f"### {case.name}" in page


def test_golden_one_token_cases_report_expected_clip_branch() -> None:
    """The four 1-token goldens land on the documented PG_CLIP branch: inside-band sets
    no mask, clipped-high and the dual-clip cap set clipped_high, clipped-low sets
    clipped_low (docs/derivations/goldens.md)."""
    expectations = {
        "pg_clip_inside_band": (False, False),
        "pg_clip_clipped_high_positive_advantage": (False, True),
        "pg_clip_clipped_low_negative_advantage": (True, False),
        "dual_clip_negative_advantage_above_cap": (False, True),
    }
    by_name = {case.name: case for case in golden_cases()}
    for name, (low, high) in expectations.items():
        case = by_name[name]
        result = policy_loss(
            case.config,
            logprobs=case.logprobs,
            old_logprobs=case.old_logprobs,
            advantages=case.advantages,
            response_mask=case.response_mask,
        )
        assert bool(result.clipped_low.all()) == low and bool(result.clipped_low.any()) == low
        assert bool(result.clipped_high.all()) == high and bool(result.clipped_high.any()) == high


def test_golden_ragged_cases_are_mask_invariant() -> None:
    """Perturbing the masked position of the ragged goldens leaves loss, per-token
    objective, and gradient bitwise unchanged (docs/conventions.md, masked positions;
    the shipped goldens already hold junk there)."""
    ragged = [case for case in golden_cases() if not bool(case.response_mask.all())]
    assert ragged
    for case in ragged:
        mask = case.response_mask

        def run(shift: float, case: GoldenCase = case) -> tuple[torch.Tensor, ...]:
            def bump(x: torch.Tensor) -> torch.Tensor:
                return torch.where(case.response_mask, x, x + shift)

            leaf = bump(case.logprobs).requires_grad_(True)
            result = policy_loss(
                case.config,
                logprobs=leaf,
                old_logprobs=bump(case.old_logprobs),
                advantages=bump(case.advantages),
                response_mask=case.response_mask,
            )
            (grad,) = torch.autograd.grad(result.loss, leaf)
            return result.loss, result.per_token_objective, grad

        base, perturbed = run(0.0), run(-9.5)
        assert torch.equal(base[0], perturbed[0])
        assert torch.equal(base[1], perturbed[1])
        assert torch.equal(base[2], perturbed[2])
        assert not bool(mask.all())
