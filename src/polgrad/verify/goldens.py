"""Analytic golden references: exact softmax-bandit quantities and hand-derived cases.

:class:`SoftmaxBandit` is a K-arm bandit with tabular policy ``π = softmax(θ)`` — small
enough that the policy gradient and the KL to another policy have closed forms, derived
step by step in ``docs/derivations/goldens.md``, so Monte Carlo and autograd results can
be certified against exact numbers. :func:`golden_cases` returns hand-derived
:func:`polgrad.losses.policy_loss` cases (every expected number worked line by line on
the same docs page) covering each PG_CLIP branch, dual-clip, and ragged aggregation.
All tensors are ``float64``; masked positions of golden inputs deliberately hold junk so
mask handling is load-bearing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from polgrad._validation import check_1d, check_finite, check_same_shape
from polgrad.aggregate import Aggregation
from polgrad.losses import ClipConfig, PolicyLossConfig, RatioKind, SurrogateKind

__all__ = ["BanditBatch", "GoldenCase", "SoftmaxBandit", "golden_cases"]

# Masked positions of golden inputs hold this junk value; a correct loss never reads it.
_MASKED_JUNK = 123.0


@dataclass(frozen=True)
class BanditBatch:
    """A sampled bandit batch in polgrad's ``[B, T]`` convention with ``T = 1``.

    Every response is a single token (the chosen arm), so per-token and per-sequence
    semantics coincide and closed-form bandit quantities certify full-pipeline losses.

    Attributes:
        logprobs: ``[B, 1]``, ``log π_θ(a_i)`` — differentiable in ``θ``.
        old_logprobs: ``[B, 1]`` sampling-time constant; equals ``sg[logprobs]``
            because the bandit samples exactly from the current policy.
        ref_logprobs: ``[B, 1]``; equals ``old_logprobs`` (reference = current policy
            at sampling time).
        rollout_logprobs: ``[B, 1]``; equals ``old_logprobs`` (exact sampling: no
            rollout/trainer mismatch).
        response_mask: ``[B, 1]`` bool, all ``True``.
        actions: ``[B]`` long, the sampled arm indices.
        rewards: ``[B]``, the per-arm rewards of the sampled arms.

    References:
        docs/derivations/goldens.md;
        tests/test_verify.py::test_softmax_bandit_sample_contract.
    """

    logprobs: Tensor
    old_logprobs: Tensor
    ref_logprobs: Tensor
    rollout_logprobs: Tensor
    response_mask: Tensor
    actions: Tensor
    rewards: Tensor


class SoftmaxBandit:
    """K-arm bandit with tabular policy ``π = softmax(θ)`` and fixed per-arm rewards.

    ``π_k = e^{θ_k} / Σ_j e^{θ_j}``. The instance holds ``theta`` by reference, so a
    training loop that updates the same tensor in place is immediately reflected in
    :meth:`sample` and :meth:`probs`.

    Args:
        theta: ``[K]`` logits, ``K >= 2``; may be an autograd leaf.
        arm_rewards: ``[K]`` deterministic reward of each arm.

    Raises:
        ValueError: If either tensor is not 1-D, shapes differ, values are non-finite,
            or ``K < 2``.

    References:
        docs/derivations/goldens.md;
        tests/test_verify.py::test_exact_policy_gradient_matches_autograd_on_expected_objective.
    """

    def __init__(self, theta: Tensor, arm_rewards: Tensor) -> None:
        check_1d("theta", theta)
        check_finite("theta", theta)
        check_1d("arm_rewards", arm_rewards)
        check_same_shape("theta", theta, "arm_rewards", arm_rewards)
        check_finite("arm_rewards", arm_rewards)
        if theta.shape[0] < 2:
            raise ValueError(f"theta must have at least 2 arms; got shape {tuple(theta.shape)}")
        self.theta = theta
        self.arm_rewards = arm_rewards

    @property
    def num_arms(self) -> int:
        """Number of arms ``K``."""
        return int(self.theta.shape[0])

    def probs(self) -> Tensor:
        """Current arm probabilities ``π = softmax(θ)`` as a ``[K]`` tensor.

        Returns:
            Detached probabilities in ``theta``'s dtype.

        References:
            docs/derivations/goldens.md;
            tests/test_verify.py::test_softmax_bandit_sample_reward_mean_matches_closed_form_mc.
        """
        # detach: reported probabilities are reference data, not a gradient path into θ.
        return torch.softmax(self.theta.detach(), dim=0)

    def exact_policy_gradient(self, advantages: Tensor) -> Tensor:
        """Closed-form ascent gradient of ``J(θ) = E_{a~π_θ}[A_a] = Σ_k π_k A_k``.

        ``∂J/∂θ_j = π_j (A_j - Σ_k π_k A_k)``, from the softmax Jacobian
        ``∂π_k/∂θ_j = π_k (δ_kj - π_j)`` (derived in docs/derivations/goldens.md).
        This is a gradient of the objective to **ascend**; polgrad losses are its
        negation.

        Args:
            advantages: ``[K]`` per-arm advantages ``A_k``.

        Returns:
            ``[K]`` detached gradient ``∇_θ J``.

        Raises:
            ValueError: If ``advantages`` is not 1-D of shape ``[K]`` or non-finite.

        References:
            docs/derivations/goldens.md;
            tests/test_verify.py::test_exact_policy_gradient_matches_autograd_on_expected_objective,
            tests/test_verify.py::test_exact_policy_gradient_matches_mc_score_function_estimate.
        """
        check_1d("advantages", advantages)
        check_same_shape("theta", self.theta, "advantages", advantages)
        check_finite("advantages", advantages)
        # detach: the closed form is a reference value, not a gradient path into θ.
        pi = torch.softmax(self.theta.detach(), dim=0)
        mean_advantage = (pi * advantages).sum()
        return pi * (advantages - mean_advantage)

    def exact_kl(self, other_theta: Tensor) -> float:
        """Categorical ``KL(π_θ ‖ π_θ') = Σ_k π_k (log π_k - log π'_k)``.

        Computed via ``log_softmax`` in explicit ``float64`` (verification helpers
        upcast; docs/conventions.md).

        Args:
            other_theta: ``[K]`` logits of the other policy ``π' = softmax(θ')``.

        Returns:
            The KL divergence as a Python float (``>= 0``, ``0`` iff ``π = π'``).

        Raises:
            ValueError: If ``other_theta`` is not 1-D of shape ``[K]`` or non-finite.

        References:
            docs/derivations/goldens.md;
            tests/test_verify.py::test_exact_kl_matches_direct_categorical_kl.
        """
        check_1d("other_theta", other_theta)
        check_same_shape("theta", self.theta, "other_theta", other_theta)
        check_finite("other_theta", other_theta)
        log_p = torch.log_softmax(self.theta.detach().to(torch.float64), dim=0)
        log_q = torch.log_softmax(other_theta.detach().to(torch.float64), dim=0)
        return float((log_p.exp() * (log_p - log_q)).sum())

    def sample(self, n: int, generator: torch.Generator) -> BanditBatch:
        """Draw ``n`` actions ``a_i ~ π_θ`` and package them as a ``[n, 1]`` batch.

        ``logprobs[i, 0] = log softmax(θ)_{a_i}`` is differentiable in ``θ``;
        ``old_logprobs``, ``ref_logprobs``, and ``rollout_logprobs`` are the identical
        detached values, because sampling is exact and on-policy with the reference
        equal to the current policy (see :class:`BanditBatch`).

        Args:
            n: Number of draws; must be ``>= 1``.
            generator: Explicit RNG (docs/conventions.md determinism rules).

        Returns:
            A :class:`BanditBatch` of ``n`` single-token sequences.

        Raises:
            ValueError: If ``n < 1``.

        References:
            docs/derivations/goldens.md;
            tests/test_verify.py::test_softmax_bandit_sample_contract,
            tests/test_verify.py::test_softmax_bandit_sample_reward_mean_matches_closed_form_mc.
        """
        if n < 1:
            raise ValueError(f"n must be >= 1; got {n}")
        actions = torch.multinomial(self.probs(), n, replacement=True, generator=generator)
        log_pi = torch.log_softmax(self.theta, dim=0)
        logprobs = log_pi.index_select(0, actions).unsqueeze(1)
        # detach: behavior/reference/rollout streams are sampling-time constants; exact
        # sampling from π_θ makes all three coincide with the current logprob values.
        constant = logprobs.detach()
        return BanditBatch(
            logprobs=logprobs,
            old_logprobs=constant,
            ref_logprobs=constant.clone(),
            rollout_logprobs=constant.clone(),
            response_mask=torch.ones((n, 1), dtype=torch.bool),
            actions=actions,
            rewards=self.arm_rewards.detach().index_select(0, actions),
        )


@dataclass(frozen=True)
class GoldenCase:
    """A hand-derived :func:`polgrad.losses.policy_loss` input/output pair.

    Attributes:
        name: Case identifier; equals the section heading on the derivation page.
        config: Loss specification the case exercises.
        logprobs: ``[B, T]`` float64; junk at masked positions.
        old_logprobs: ``[B, T]`` float64; junk at masked positions.
        advantages: ``[B]`` or ``[B, T]`` float64.
        response_mask: ``[B, T]`` bool.
        expected_loss: Hand-derived scalar loss.
        expected_grad_logprobs: ``[B, T]`` hand-derived ``d loss / d logprobs``.
        derivation: Anchor into docs/derivations/goldens.md showing the arithmetic.

    References:
        docs/derivations/goldens.md;
        tests/test_verify.py::test_golden_cases_satisfied_by_policy_loss.
    """

    name: str
    config: PolicyLossConfig
    logprobs: Tensor
    old_logprobs: Tensor
    advantages: Tensor
    response_mask: Tensor
    expected_loss: float
    expected_grad_logprobs: Tensor
    derivation: str


def _one_token_case(
    name: str,
    clip: ClipConfig,
    lp: float,
    olp: float,
    advantage: float,
    expected_loss: float,
    expected_grad: float,
) -> GoldenCase:
    f64 = torch.float64
    return GoldenCase(
        name=name,
        config=PolicyLossConfig(
            ratio=RatioKind.TOKEN,
            surrogate=SurrogateKind.PG_CLIP,
            clip=clip,
            aggregation=Aggregation.TOKEN_MEAN,
        ),
        logprobs=torch.tensor([[lp]], dtype=f64),
        old_logprobs=torch.tensor([[olp]], dtype=f64),
        advantages=torch.tensor([advantage], dtype=f64),
        response_mask=torch.ones((1, 1), dtype=torch.bool),
        expected_loss=expected_loss,
        expected_grad_logprobs=torch.tensor([[expected_grad]], dtype=f64),
        derivation=f"docs/derivations/goldens.md#{name}",
    )


def _ragged_case(name: str, aggregation: Aggregation, loss: float, grad_10: float) -> GoldenCase:
    f64 = torch.float64
    junk = _MASKED_JUNK
    return GoldenCase(
        name=name,
        config=PolicyLossConfig(
            ratio=RatioKind.TOKEN,
            surrogate=SurrogateKind.PG_CLIP,
            clip=ClipConfig(eps_low=0.2, eps_high=0.3),
            aggregation=aggregation,
        ),
        logprobs=torch.tensor([[-0.3, -1.2], [-0.5, junk]], dtype=f64),
        old_logprobs=torch.tensor([[-0.7, -0.8], [-0.6, junk]], dtype=f64),
        advantages=torch.tensor([[1.0, -2.0], [-1.0, junk]], dtype=f64),
        response_mask=torch.tensor([[True, True], [True, False]]),
        expected_loss=loss,
        expected_grad_logprobs=torch.tensor([[0.0, 0.0], [grad_10, 0.0]], dtype=f64),
        derivation=f"docs/derivations/goldens.md#{name}",
    )


def golden_cases() -> tuple[GoldenCase, ...]:
    """Return the hand-derived policy-loss golden cases.

    Covers, with ``ε_lo = 0.2``, ``ε_hi = 0.3``: a 1-token PG_CLIP ratio inside the
    clip band, above ``1 + ε_hi`` with ``A > 0``, below ``1 - ε_lo`` with ``A < 0``;
    the dual-clip branch (``A < 0``, ``r > c = 2``); and a 2-token ragged batch mixing
    all three branches under ``TOKEN_MEAN`` and ``SEQ_MEAN_TOKEN_MEAN``. Every
    ``expected_loss``/``expected_grad_logprobs`` value is worked line by line in
    docs/derivations/goldens.md. Tensors are constructed fresh on each call.

    Returns:
        Tuple of :class:`GoldenCase`.

    References:
        docs/derivations/goldens.md;
        tests/test_verify.py::test_golden_cases_satisfied_by_policy_loss,
        tests/test_verify.py::test_golden_cases_cover_contract_branches.
    """
    clip = ClipConfig(eps_low=0.2, eps_high=0.3)
    dual = ClipConfig(eps_low=0.2, eps_high=0.3, ratio_cap=2.0)
    return (
        _one_token_case(
            "pg_clip_inside_band",
            clip,
            lp=-0.7,
            olp=-0.9,
            advantage=1.5,
            expected_loss=-1.5 * math.exp(0.2),
            expected_grad=-1.5 * math.exp(0.2),
        ),
        _one_token_case(
            "pg_clip_clipped_high_positive_advantage",
            clip,
            lp=-0.2,
            olp=-0.7,
            advantage=2.0,
            expected_loss=-2.6,
            expected_grad=0.0,
        ),
        _one_token_case(
            "pg_clip_clipped_low_negative_advantage",
            clip,
            lp=-1.5,
            olp=-1.0,
            advantage=-1.0,
            expected_loss=0.8,
            expected_grad=0.0,
        ),
        _one_token_case(
            "dual_clip_negative_advantage_above_cap",
            dual,
            lp=-0.4,
            olp=-1.4,
            advantage=-1.5,
            expected_loss=3.0,
            expected_grad=0.0,
        ),
        _ragged_case(
            "pg_clip_two_token_ragged_token_mean",
            Aggregation.TOKEN_MEAN,
            loss=(0.3 + math.exp(0.1)) / 3.0,
            grad_10=math.exp(0.1) / 3.0,
        ),
        _ragged_case(
            "pg_clip_two_token_ragged_seq_mean_token_mean",
            Aggregation.SEQ_MEAN_TOKEN_MEAN,
            loss=0.3 / 4.0 + math.exp(0.1) / 2.0,
            grad_10=math.exp(0.1) / 2.0,
        ),
    )
