"""KL estimators and KL penalties for RL post-training.

Sign convention (stated once in ``docs/derivations/kl.md``; code follows it): the target
is ``KL(π‖ref) = E_π[log π - log ref]`` with samples drawn from ``π``. Following
Schulman, "Approximating KL Divergence" (http://joschu.net/blog/kl-approx.html), with
``r = ref(x)/π(x)`` define ``δ_t = ref_logprobs_t - logprobs_t = log r``:

- ``k1 = -δ = logprobs - ref_logprobs``; ``E_π[k1] = KL`` exactly.
- ``k2 = δ²/2``; ``E_π[k2] ≈ KL`` with a third-order bias derived in the docs page.
- ``k3 = exp(δ) - 1 - δ``; ``E_π[k3] = KL`` exactly (since ``E_π[r] = 1``).
- ``abs = |δ|``; **not** an estimator of KL — ``E_π[|δ|] ≥ KL`` with equality only when
  ``δ`` has almost-surely constant sign. Included solely for conformance with verl's
  ``kl_penalty("abs")``.

Scope exclusion: verl's ``kl_penalty("full")`` needs full-vocabulary logprobs and is out
of scope for polgrad, which operates on sampled-token logprob streams only.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

import torch
from torch import Tensor

from polgrad._validation import (
    check_2d,
    check_finite,
    check_logprob_streams,
    check_same_shape,
)
from polgrad.aggregate import Aggregation, aggregate

__all__ = [
    "KLEstimator",
    "KLLossConfig",
    "kl_estimate",
    "kl_in_reward",
    "kl_loss",
    "reverse_kl_grad_surrogate",
]


class KLEstimator(enum.Enum):
    """Per-token estimator of ``KL(π‖ref)`` from sampled-token logprobs.

    See the module docstring for the definitions and ``docs/derivations/kl.md`` for the
    expectations, variances, and as-loss pathwise gradients of each member.
    """

    K1 = "k1"
    K2 = "k2"
    K3 = "k3"
    ABS = "abs"


@dataclass(frozen=True)
class KLLossConfig:
    """Configuration of an as-loss KL penalty term (owned by ``kl.py``).

    ``losses.policy_loss`` adds ``coef · kl_loss(logprobs, ref_logprobs, kind,
    aggregation or PolicyLossConfig.aggregation, ...)`` to the surrogate.

    Attributes:
        kind: Which per-token estimator to aggregate.
        coef: Penalty coefficient multiplying the aggregated KL term.
        aggregation: Aggregation for the KL term; ``None`` inherits
            ``PolicyLossConfig.aggregation``.
        norm_len: Fixed generation budget for the KL term; ``None`` inherits
            ``PolicyLossConfig.norm_len``. The effective value is required iff the
            effective aggregation is ``Aggregation.TOKEN_SUM_NORM``; constructing with
            ``norm_len=None`` is always legal — the requirement is enforced when
            :func:`kl_loss` is called.

    References:
        docs/derivations/kl.md;
        tests/test_kl.py::test_kl_loss_config_is_frozen_data.
    """

    kind: KLEstimator
    coef: float
    aggregation: Aggregation | None = None
    norm_len: int | None = None


def kl_estimate(
    logprobs: Tensor,
    ref_logprobs: Tensor,
    kind: KLEstimator,
    *,
    response_mask: Tensor,
) -> Tensor:
    """Per-token estimator of ``KL(π‖ref)``, differentiable in ``logprobs``.

    With ``δ_t = ref_logprobs_t - logprobs_t``:

    ``k1 = -δ``, ``k2 = δ²/2``, ``k3 = exp(δ) - 1 - δ``, ``abs = |δ|``.

    ``k2``, ``k3`` and ``abs`` are pointwise non-negative; ``E_π[k1] = E_π[k3] = KL``
    exactly, ``E_π[k2] = KL + O(E[δ³])``, and ``abs`` does not estimate KL (module
    docstring; derivations in docs/derivations/kl.md).

    Args:
        logprobs: ``[B, T]`` sampled-token logprobs of the current policy θ
            (differentiable stream).
        ref_logprobs: ``[B, T]`` sampled-token logprobs of the frozen reference policy.
        kind: Estimator to compute.
        response_mask: ``[B, T]`` bool mask of real response tokens.

    Returns:
        ``[B, T]`` tensor in the input dtype, exactly ``0`` at masked positions; the
        gradient w.r.t. ``logprobs`` is also exactly ``0`` there.

    Raises:
        ValueError: On shape/mask violations, on non-finite values at response
            positions, or on an unknown ``kind``.

    References:
        Schulman, "Approximating KL Divergence", http://joschu.net/blog/kl-approx.html;
        docs/derivations/kl.md;
        tests/test_kl.py::test_kl_estimate_golden_values,
        tests/test_kl.py::test_mc_k1_and_k3_match_closed_form_categorical_kl.
    """
    check_logprob_streams("logprobs", logprobs, "ref_logprobs", ref_logprobs, response_mask)
    zero = torch.zeros((), dtype=logprobs.dtype, device=logprobs.device)
    # Masked positions are zeroed before the nonlinearity so padding junk can reach
    # neither exp()/pow() in the forward pass nor their backward formulas.
    delta = torch.where(response_mask, ref_logprobs - logprobs, zero)
    if kind is KLEstimator.K1:
        estimate = -delta
    elif kind is KLEstimator.K2:
        estimate = delta.pow(2) / 2
    elif kind is KLEstimator.K3:
        estimate = torch.expm1(delta) - delta
    elif kind is KLEstimator.ABS:
        estimate = delta.abs()
    else:
        raise ValueError(f"unknown KLEstimator: {kind!r}")
    return torch.where(response_mask, estimate, zero)


def kl_loss(
    logprobs: Tensor,
    ref_logprobs: Tensor,
    kind: KLEstimator,
    aggregation: Aggregation,
    *,
    response_mask: Tensor,
    norm_len: int | None = None,
) -> Tensor:
    """Scalar as-loss KL penalty: ``aggregate(kl_estimate(...), aggregation)``.

    The pathwise (as-loss) gradients are, per token: ``∇k1 = ∇logprobs`` (zero-mean
    on-policy — k1-as-loss optimizes nothing), ``∇k2 = (logprobs - ref_logprobs) ·
    ∇logprobs`` (the unbiased score-function sample of ``∇KL(π‖ref)``; identical by
    construction to :func:`reverse_kl_grad_surrogate`), and ``∇k3 = (1 - exp(δ)) ·
    ∇logprobs`` (biased for ``∇KL`` in general). Derivations in docs/derivations/kl.md.

    Args:
        logprobs: ``[B, T]`` current-policy sampled-token logprobs (differentiable).
        ref_logprobs: ``[B, T]`` frozen reference logprobs.
        kind: Per-token estimator.
        aggregation: Aggregation mode reducing the estimate to a scalar.
        response_mask: ``[B, T]`` bool mask of real response tokens.
        norm_len: Required iff ``aggregation`` is ``Aggregation.TOKEN_SUM_NORM``.

    Returns:
        Scalar tensor in the input dtype, differentiable w.r.t. ``logprobs``.

    Raises:
        ValueError: Propagated from :func:`kl_estimate` and :func:`aggregate`.

    References:
        docs/derivations/kl.md;
        tests/test_kl.py::test_kl_loss_equals_aggregate_of_kl_estimate,
        tests/test_kl.py::test_k2_as_loss_gradient_equals_reverse_kl_grad_surrogate,
        tests/test_kl.py::test_k3_as_loss_gradient_bias_mc_gap_vs_analytic_grad_kl.
    """
    per_token = kl_estimate(logprobs, ref_logprobs, kind, response_mask=response_mask)
    return aggregate(per_token, response_mask, aggregation, norm_len=norm_len)


def kl_in_reward(
    token_rewards: Tensor,
    old_logprobs: Tensor,
    ref_logprobs: Tensor,
    kind: KLEstimator,
    coef: float,
    *,
    response_mask: Tensor,
) -> Tensor:
    """Fold a KL penalty into per-token rewards: ``r_t ← r_t - coef · k(t)``.

    The penalty ``k`` is computed from **old_logprobs** (the sampling policy) against
    ``ref_logprobs``; this is the reward-shaping placement of the KL term (PPO-RLHF
    style), evaluated at rollout time with no gradient path.

    Args:
        token_rewards: ``[B, T]`` per-token rewards.
        old_logprobs: ``[B, T]`` sampling-policy sampled-token logprobs (constant).
        ref_logprobs: ``[B, T]`` frozen reference logprobs.
        kind: Per-token estimator used as the penalty.
        coef: Penalty coefficient.
        response_mask: ``[B, T]`` bool mask of real response tokens.

    Returns:
        ``[B, T]`` detached modified rewards, exactly ``0`` at masked positions.

    Raises:
        ValueError: On shape/mask violations or non-finite response-position values.

    References:
        docs/derivations/kl.md;
        tests/test_kl.py::test_kl_in_reward_golden_values,
        tests/test_kl.py::test_kl_in_reward_is_detached.
    """
    check_2d("token_rewards", token_rewards)
    check_same_shape("token_rewards", token_rewards, "old_logprobs", old_logprobs)
    penalty = kl_estimate(old_logprobs, ref_logprobs, kind, response_mask=response_mask)
    check_finite("token_rewards", token_rewards[response_mask])
    zero = torch.zeros((), dtype=token_rewards.dtype, device=token_rewards.device)
    shaped = torch.where(response_mask, token_rewards - coef * penalty, zero)
    # detach: the sampling-time KL penalty is reward shaping — a constant target for the
    # advantage estimator — and must not open a gradient path into any logprob stream.
    return shaped.detach()


def reverse_kl_grad_surrogate(
    logprobs: Tensor,
    ref_logprobs: Tensor,
    aggregation: Aggregation,
    *,
    response_mask: Tensor,
    norm_len: int | None = None,
) -> Tensor:
    """Surrogate whose gradient is the unbiased per-sample reverse-KL policy gradient.

    Computes ``agg( sg[logprobs - ref_logprobs] · logprobs )``. Its pathwise gradient
    per token is ``(logprobs - ref_logprobs) · ∇logprobs``, the score-function sample of
    ``∇KL(π‖ref)`` whose on-policy expectation equals the analytic gradient
    (docs/derivations/kl.md). It is identical — gradient-exactly — to using ``k2`` as a
    loss under the same aggregation.

    Args:
        logprobs: ``[B, T]`` current-policy sampled-token logprobs (differentiable).
        ref_logprobs: ``[B, T]`` frozen reference logprobs.
        aggregation: Aggregation mode.
        response_mask: ``[B, T]`` bool mask of real response tokens.
        norm_len: Required iff ``aggregation`` is ``Aggregation.TOKEN_SUM_NORM``.

    Returns:
        Scalar tensor in the input dtype, differentiable w.r.t. ``logprobs``.

    Raises:
        ValueError: On shape/mask violations, non-finite response-position values, or a
            missing/non-positive ``norm_len`` under ``TOKEN_SUM_NORM``.

    References:
        docs/derivations/kl.md;
        tests/test_kl.py::test_k2_as_loss_gradient_equals_reverse_kl_grad_surrogate,
        tests/test_kl.py::test_k2_as_loss_expected_gradient_equals_analytic_grad_kl.
    """
    check_logprob_streams("logprobs", logprobs, "ref_logprobs", ref_logprobs, response_mask)
    zero = torch.zeros((), dtype=logprobs.dtype, device=logprobs.device)
    # detach: sg[logprobs - ref_logprobs] is the score-function coefficient; holding it
    # constant makes the pathwise gradient equal the REINFORCE-style sample of ∇KL(π‖ref).
    coeff = torch.where(response_mask, logprobs - ref_logprobs, zero).detach()
    return aggregate(coeff * logprobs, response_mask, aggregation, norm_len=norm_len)
