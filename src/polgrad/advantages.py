"""Sequence- and token-level advantage estimators for LLM RL post-training.

Group-relative normalization (GRPO / Dr.GRPO), leave-one-out baselines (RLOO),
REINFORCE++ batch baselines, and Generalized Advantage Estimation over right-padded
response batches, plus the whitening and per-token broadcasting helpers. All functions
are pure and follow the shape/mask conventions of ``docs/conventions.md``; derivations
live in ``docs/derivations/advantages.md``. This module imports only
``polgrad._validation``.
"""

# Contract section 6 allows Unicode math in docstrings; RUF002 flags some of those
# characters (gamma, sigma, minus sign) as confusables, so it is exempted here.

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

from polgrad._validation import (
    check_1d,
    check_2d,
    check_finite,
    check_mask,
    check_same_shape,
)

__all__ = [
    "GAEConfig",
    "GroupNormConfig",
    "ReinforcePPConfig",
    "broadcast_to_tokens",
    "gae",
    "grpo_advantages",
    "reinforce_pp_advantages",
    "rloo_advantages",
    "whiten",
]


@dataclass(frozen=True)
class GroupNormConfig:
    """Knobs for group-relative reward normalization.

    Attributes:
        center: Subtract the per-group mean.
        scale: ``"std"`` divides by the per-group standard deviation plus ``eps``
            (GRPO); ``"none"`` skips the division (Dr.GRPO).
        eps: Added to the standard deviation before dividing. The GRPO paper is silent
            on this value; TRL uses 1e-4 (the default here), verl 1e-6, and the released
            Dr.GRPO code 1e-8.
        unbiased: Bessel-corrected standard deviation (divide by G − 1). Matches the
            ``torch.Tensor.std`` default used by the TRL, verl, and Dr.GRPO code.
    """

    center: bool = True
    scale: Literal["std", "none"] = "std"
    eps: float = 1e-4
    unbiased: bool = True


@dataclass(frozen=True)
class GAEConfig:
    """Discount and trace parameters for Generalized Advantage Estimation.

    Attributes:
        gamma: Discount factor γ applied to bootstrapped values.
        lam: GAE trace parameter λ weighting the exponential average of n-step
            advantage estimators.
    """

    gamma: float
    lam: float


@dataclass(frozen=True)
class ReinforcePPConfig:
    """Knobs for the REINFORCE++ advantage estimators.

    Attributes:
        batch_norm: After baseline subtraction, divide by the global batch standard
            deviation (Bessel-corrected) plus ``eps``.
        eps: Added to the batch standard deviation before dividing.
    """

    batch_norm: bool
    eps: float = 1e-8


def _check_reward_batch(name: str, rewards: Tensor) -> None:
    check_1d(name, rewards)
    if rewards.numel() == 0:
        raise ValueError(f"{name} must contain at least one sequence; got shape (0,)")
    check_finite(name, rewards)


def _check_group_ids(rewards: Tensor, group_ids: Tensor) -> None:
    check_1d("group_ids", group_ids)
    check_same_shape("rewards", rewards, "group_ids", group_ids)
    dtype = group_ids.dtype
    if dtype.is_floating_point or dtype.is_complex or dtype == torch.bool:
        raise ValueError(f"group_ids must have an integer dtype; got {dtype}")
    if int(group_ids.min()) < 0:
        raise ValueError(f"group_ids must be non-negative; got minimum {int(group_ids.min())}")


def _grouped_stats(rewards: Tensor, group_ids: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return (unique group labels, inverse index [B], integer counts, per-group sums)."""
    unique_ids, inverse = torch.unique(group_ids, return_inverse=True)
    n_groups = int(unique_ids.numel())
    counts = torch.bincount(inverse, minlength=n_groups)
    sums = torch.zeros(n_groups, dtype=rewards.dtype, device=rewards.device)
    sums.index_add_(0, inverse, rewards)
    return unique_ids, inverse, counts, sums


def _group_centered(rewards: Tensor, inverse: Tensor, counts: Tensor, sums: Tensor) -> Tensor:
    """Per-group mean subtraction shared by GRPO and REINFORCE++-baseline."""
    means = sums / counts.to(rewards.dtype)
    return rewards - means[inverse]


def grpo_advantages(rewards: Tensor, group_ids: Tensor, config: GroupNormConfig) -> Tensor:
    """Group-normalized sequence advantages (GRPO; ``scale="none"`` gives Dr.GRPO).

    With per-group mean μ_g and standard deviation σ_g over the rewards sharing a
    group id (Bessel-corrected iff ``config.unbiased``):

        A_i = (r_i − μ_{g(i)}) / (σ_{g(i)} + ε)     center=True,  scale="std"   (GRPO)
        A_i =  r_i − μ_{g(i)}                       center=True,  scale="none"  (Dr.GRPO)
        A_i =  r_i / (σ_{g(i)} + ε)                 center=False, scale="std"

    The per-group 1/(σ_g + ε) factor is the difficulty bias Dr.GRPO removes: groups
    whose rewards have low spread are up-weighted relative to high-spread groups even
    when their centered rewards are identical (docs/derivations/advantages.md,
    "Group normalization and the Dr.GRPO difficulty bias").

    Args:
        rewards: ``[B]`` scalar reward per sequence.
        group_ids: ``[B]`` non-negative integers mapping rows to prompt groups.
        config: Centering/scaling knobs; see :class:`GroupNormConfig`.

    Returns:
        ``[B]`` advantages, same dtype as ``rewards``.

    Raises:
        ValueError: On shape/dtype violations, non-finite rewards, negative group ids,
            an unknown ``config.scale``, or — with ``scale="std"`` — any group of size
            1, whose standard deviation is undefined. Frameworks silently emit 0/ε for
            such groups; polgrad raises instead (recorded in the conformance deviation
            docs, not imitated).

    References:
        GRPO: "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open
        Language Models" (arXiv 2402.03300). Dr.GRPO: "Understanding R1-Zero-Like
        Training: A Critical Perspective" (arXiv 2503.20783).
        docs/derivations/advantages.md. Enforced by
        tests/test_advantages.py::test_grpo_closed_form_single_group and
        ::test_dr_grpo_difficulty_bias_exact_std_factor.
    """
    _check_reward_batch("rewards", rewards)
    _check_group_ids(rewards, group_ids)
    if config.scale not in ("std", "none"):
        raise ValueError(f"config.scale must be 'std' or 'none'; got {config.scale!r}")
    unique_ids, inverse, counts, sums = _grouped_stats(rewards, group_ids)
    centered = _group_centered(rewards, inverse, counts, sums)
    out = centered if config.center else rewards.clone()
    if config.scale == "std":
        if bool((counts == 1).any()):
            singles = unique_ids[counts == 1].tolist()
            raise ValueError(
                "grpo_advantages with scale='std' requires every group to contain at "
                f"least 2 rewards; group id(s) {singles} have a single reward"
            )
        counts_f = counts.to(rewards.dtype)
        sq = torch.zeros_like(sums)
        sq.index_add_(0, inverse, centered * centered)
        denom = counts_f - 1 if config.unbiased else counts_f
        std = (sq / denom).sqrt()
        out = out / (std[inverse] + config.eps)
    return out


def rloo_advantages(rewards: Tensor, group_ids: Tensor) -> Tensor:
    """Leave-one-out baselined sequence advantages (RLOO).

    For a group g of size G with reward sum S = Σ_j r_j:

        A_i = r_i − mean_{j≠i}(r_j) = r_i − (S − r_i)/(G − 1) = (G/(G − 1))·(r_i − S/G)

    The two forms are algebraically identical (proof in
    docs/derivations/advantages.md, "The RLOO identity"); this function computes the
    leave-one-out form.

    Args:
        rewards: ``[B]`` scalar reward per sequence.
        group_ids: ``[B]`` non-negative integers mapping rows to prompt groups.

    Returns:
        ``[B]`` advantages, same dtype as ``rewards``.

    Raises:
        ValueError: On shape/dtype violations, non-finite rewards, negative group ids,
            or any group of size 1, for which the leave-one-out baseline is undefined.

    References:
        "Back to Basics: Revisiting REINFORCE Style Optimization for Learning from
        Human Feedback in LLMs" (arXiv 2402.14740). docs/derivations/advantages.md.
        Enforced by tests/test_advantages.py::test_rloo_closed_form_two_groups and
        ::test_rloo_two_form_identity_exact_on_dyadic_inputs.
    """
    _check_reward_batch("rewards", rewards)
    _check_group_ids(rewards, group_ids)
    unique_ids, inverse, counts, sums = _grouped_stats(rewards, group_ids)
    if bool((counts == 1).any()):
        singles = unique_ids[counts == 1].tolist()
        raise ValueError(
            "rloo_advantages requires every group to contain at least 2 rewards for "
            f"the leave-one-out baseline; group id(s) {singles} have a single reward"
        )
    counts_f = counts.to(rewards.dtype)
    loo_baseline = (sums[inverse] - rewards) / (counts_f[inverse] - 1)
    return rewards - loo_baseline


def reinforce_pp_advantages(
    rewards: Tensor, group_ids: Tensor | None = None, *, config: ReinforcePPConfig
) -> Tensor:
    """REINFORCE++ sequence advantages with a batch or per-group mean baseline.

    With ``group_ids=None`` the baseline is the global batch mean (REINFORCE++); with
    ``group_ids`` it is the per-group mean (REINFORCE++-baseline):

        A_i = r_i − b_i,   b_i = mean(r)  or  μ_{g(i)}

    If ``config.batch_norm``, the centered advantages are then divided by the global
    batch standard deviation (Bessel-corrected) plus ``config.eps``:

        A_i ← A_i / (std(A) + ε)

    Unlike GRPO's per-group σ_g, this is a single global scalar and therefore rescales
    every advantage uniformly (docs/derivations/advantages.md, "REINFORCE++ baseline
    variants").

    Args:
        rewards: ``[B]`` scalar reward per sequence.
        group_ids: Optional ``[B]`` non-negative integers mapping rows to prompt
            groups; ``None`` selects the global batch-mean baseline.
        config: Normalization knobs; see :class:`ReinforcePPConfig`.

    Returns:
        ``[B]`` advantages, same dtype as ``rewards``.

    Raises:
        ValueError: On shape/dtype violations, non-finite rewards, negative group ids,
            or ``batch_norm=True`` with fewer than 2 rewards (Bessel-corrected batch
            standard deviation undefined).

    References:
        "REINFORCE++: Stabilizing Critic-Free Policy Optimization with Global
        Advantage Normalization" (arXiv 2501.03262). docs/derivations/advantages.md.
        Enforced by tests/test_advantages.py::test_reinforce_pp_global_baseline_closed_form
        and ::test_reinforce_pp_batch_norm_closed_form.
    """
    _check_reward_batch("rewards", rewards)
    if group_ids is None:
        adv = rewards - rewards.mean()
    else:
        _check_group_ids(rewards, group_ids)
        _, inverse, counts, sums = _grouped_stats(rewards, group_ids)
        adv = _group_centered(rewards, inverse, counts, sums)
    if config.batch_norm:
        n = int(rewards.numel())
        if n < 2:
            raise ValueError(
                "reinforce_pp_advantages with batch_norm=True requires at least 2 "
                f"rewards for the Bessel-corrected batch std; got shape {tuple(rewards.shape)}"
            )
        mu = adv.mean()
        var = (adv - mu).pow(2).sum() / (n - 1)
        adv = adv / (var.sqrt() + config.eps)
    return adv


def _check_right_padded(response_mask: Tensor) -> None:
    if response_mask.shape[1] < 2:
        return
    revived = response_mask[:, 1:] & ~response_mask[:, :-1]
    if bool(revived.any()):
        rows = torch.nonzero(revived.any(dim=1)).flatten().tolist()
        raise ValueError(
            f"response_mask must be right-padded for gae; rows {rows} have a real "
            "token after a padded position"
        )


def gae(
    token_rewards: Tensor, values: Tensor, *, config: GAEConfig, response_mask: Tensor
) -> tuple[Tensor, Tensor]:
    """Generalized Advantage Estimation over right-padded response batches.

    For a row with L real tokens, with V_L := 0 (terminal bootstrap):

        δ_t = r_t + γ·V_{t+1}·1[t+1 < L] − V_t
        A_t = δ_t + γλ·A_{t+1},  A_L = 0        (⇔ A_t = Σ_{l≥0} (γλ)^l δ_{t+l})
        R_t = A_t + V_t

    Computed as a single O(T) reverse scan vectorized over the batch; padded positions
    contribute nothing to any real position and are exactly 0 in both outputs. At
    γ = λ = 1 the deltas telescope, so A_t = Σ_{s≥t} r_s − V_t (reward-to-go minus
    value) and R_t = Σ_{s≥t} r_s.

    Args:
        token_rewards: ``[B, T]`` per-token rewards (KL-in-reward shaping, if any,
            already applied).
        values: ``[B, T]`` value predictions V(s_t).
        config: γ and λ; see :class:`GAEConfig`.
        response_mask: ``[B, T]`` bool, right-padded, ≥ 1 real token per row.

    Returns:
        ``(advantages, returns)``, both ``[B, T]`` in the input dtype and exactly 0 at
        masked positions.

    Raises:
        ValueError: On shape/dtype/mask violations, non-finite inputs, or a
            ``response_mask`` that is not right-padded (a real token after a padded
            position would make the reverse scan ambiguous).

    References:
        "High-Dimensional Continuous Control Using Generalized Advantage Estimation"
        (arXiv 1506.02438). docs/derivations/advantages.md. Enforced by
        tests/test_advantages.py::test_gae_matches_slow_oracle,
        ::test_gae_closed_form_ragged_batch, and
        ::test_gae_gamma_lambda_one_reduces_to_reward_to_go.
    """
    check_2d("token_rewards", token_rewards)
    check_2d("values", values)
    check_same_shape("token_rewards", token_rewards, "values", values)
    check_finite("token_rewards", token_rewards)
    check_finite("values", values)
    check_mask(response_mask, like=token_rewards)
    _check_right_padded(response_mask)
    b, t_len = token_rewards.shape
    mask_f = response_mask.to(token_rewards.dtype)
    zero_col = torch.zeros(b, dtype=token_rewards.dtype, device=token_rewards.device)
    last = zero_col
    cols: list[Tensor] = []
    for t in range(t_len - 1, -1, -1):
        # the t+1 gate is 0 at the last real token, bootstrapping V = 0 at the terminal
        # state and resetting the λ-trace so padded junk never reaches real positions
        next_gate = mask_f[:, t + 1] if t + 1 < t_len else zero_col
        next_values = values[:, t + 1] if t + 1 < t_len else zero_col
        delta = token_rewards[:, t] + config.gamma * next_gate * next_values - values[:, t]
        last = delta + config.gamma * config.lam * next_gate * last
        cols.append(last)
    cols.reverse()
    scan = torch.stack(cols, dim=1)
    zero = torch.zeros_like(scan)
    advantages = torch.where(response_mask, scan, zero)
    returns = torch.where(response_mask, scan + values, zero)
    return advantages, returns


def broadcast_to_tokens(per_seq: Tensor, response_mask: Tensor) -> Tensor:
    """Broadcast per-sequence values across their response tokens.

        out[i, t] = per_seq[i] · m[i, t]

    Args:
        per_seq: ``[B]`` per-sequence values (e.g. sequence-level advantages).
        response_mask: ``[B, T]`` bool, ≥ 1 real token per row.

    Returns:
        ``[B, T]`` in the input dtype, exactly 0 at masked positions.

    Raises:
        ValueError: On shape/dtype/mask violations, non-finite ``per_seq``, or a batch
            size mismatch between ``per_seq`` and ``response_mask``.

    References:
        docs/derivations/advantages.md, "Broadcasting and whitening". Enforced by
        tests/test_advantages.py::test_broadcast_to_tokens_closed_form.
    """
    check_1d("per_seq", per_seq)
    if per_seq.numel() == 0:
        raise ValueError("per_seq must contain at least one sequence; got shape (0,)")
    check_finite("per_seq", per_seq)
    check_mask(response_mask, like=response_mask)
    if response_mask.shape[0] != per_seq.shape[0]:
        raise ValueError(
            "per_seq and response_mask must agree in batch size; got per_seq shape "
            f"{tuple(per_seq.shape)} vs response_mask shape {tuple(response_mask.shape)}"
        )
    expanded = per_seq.unsqueeze(1).expand(-1, response_mask.shape[1])
    return torch.where(response_mask, expanded, torch.zeros_like(expanded))


def whiten(
    x: Tensor, response_mask: Tensor, *, shift_mean: bool = True, eps: float = 1e-8
) -> Tensor:
    """Whiten a per-token tensor over its response tokens.

    With masked mean μ = Σ m·x / Σ m and Bessel-corrected masked variance
    σ² = Σ m·(x − μ)² / (Σ m − 1):

        out = (x − μ) / √(σ² + ε)            shift_mean=True
        out = (x − μ) / √(σ² + ε) + μ        shift_mean=False

    ``shift_mean=False`` restores the mean after scaling (the convention of the
    masked-whitening helpers in common PPO implementations). Masked positions are
    exactly 0 in the output and never influence μ or σ².

    Args:
        x: ``[B, T]`` per-token values (typically GAE advantages).
        response_mask: ``[B, T]`` bool, ≥ 1 real token per row.
        shift_mean: Keep the result zero-mean (``True``) or restore μ (``False``).
        eps: Added to the variance before the square root.

    Returns:
        ``[B, T]`` in the input dtype, exactly 0 at masked positions.

    Raises:
        ValueError: On shape/dtype/mask violations, non-finite ``x``, or fewer than 2
            response tokens in total (Bessel-corrected variance undefined).

    References:
        docs/derivations/advantages.md, "Broadcasting and whitening". Enforced by
        tests/test_advantages.py::test_whiten_closed_form and
        ::test_whiten_masked_moments.
    """
    check_2d("x", x)
    check_finite("x", x)
    check_mask(response_mask, like=x)
    n_tokens = int(response_mask.sum())
    if n_tokens < 2:
        raise ValueError(
            "whiten requires at least 2 response tokens for the Bessel-corrected "
            f"variance; response_mask has {n_tokens} true token(s)"
        )
    mask_f = response_mask.to(x.dtype)
    n = mask_f.sum()
    mean = (x * mask_f).sum() / n
    centered = (x - mean) * mask_f
    var = (centered * centered).sum() / (n - 1)
    out = (x - mean) * torch.rsqrt(var + eps)
    if not shift_mean:
        out = out + mean
    return torch.where(response_mask, out, torch.zeros_like(out))
