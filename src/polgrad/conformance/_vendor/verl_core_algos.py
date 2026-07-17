# =============================================================================
# VENDORED CODE - DO NOT EDIT. Used only by polgrad's conformance harness.
#
# Upstream repository: https://github.com/volcengine/verl
#   (GitHub redirects this URL to https://github.com/verl-project/verl after the
#   project rename; the raw files below were fetched through that redirect.)
# Commit: 74a718a492092312f1004fe25369975137388849
#   (HEAD of the default branch `main` at fetch time, 2026-07-16.)
# Source paths at that commit:
#   verl/trainer/ppo/core_algos.py
#     upstream file SHA256: 9114f9e16c87e4c9ebf2fa016baf733c9bbc819766b53c8968aaa9e8abcd7916
#   verl/utils/torch_functional.py (three helpers the kept losses call)
#     upstream file SHA256: cb37d045dec2161d637e2633032cc97c97334e5547cea62dace42630c9683259
# Upstream license: Apache-2.0 (upstream notice retained below; attribution in
#   the repo-root NOTICE file).
#
# Functions kept (bodies byte-identical to upstream):
#   from verl/utils/torch_functional.py:
#     clip_by_value, masked_sum, masked_mean
#   from verl/trainer/ppo/core_algos.py:
#     agg_loss            - loss aggregation modes ("token-mean",
#                           "seq-mean-token-sum", "seq-mean-token-sum-norm",
#                           "seq-mean-token-mean")
#     compute_policy_loss - clipped PPO objective including the dual-clip branch
#     compute_value_loss  - clipped PPO value loss
#     kl_penalty          - KL penalty dispatch including the "+" straight-through
#                           variants
#     kl_penalty_forward  - k1 / abs / k2(mse) / k3(low_var_kl) estimators
#
# Functions dropped (not pure functions of tensors with minimal edits):
#   - All advantage estimators (compute_gae_advantage_return,
#     compute_grpo_outcome_advantage, compute_rloo_outcome_advantage, ...):
#     attached to a framework registry via @register_adv_est and dependent on
#     numpy / omegaconf / verl config objects.
#   - compute_policy_loss_vanilla, _dppo_tv, _dppo_kl, _gspo, _sapo, _gpg,
#     _clip_cov, _kl_cov, _geo_mean, _cispo, _bypass_mode, _reinforce: each
#     asserts on and reads a verl ActorConfig/DictConfig instance and expands
#     `config.global_batch_info`; the kept `compute_policy_loss` computes the
#     same vanilla clipped objective (including dual-clip) from plain arguments.
#   - compute_entropy_loss (needs full-vocab logits via entropy_from_logits),
#     compute_rewards, compute_pf_ppo_reweight_data (verl DataProto),
#     AdaptiveKLController / FixedKLController / get_kl_controller, and the
#     policy-loss / advantage-estimator registry machinery.
#
# Edits made (all of them; everything else is verbatim):
#   1. Replaced the upstream import block (numpy, omegaconf, verl.* imports) and
#      upstream `__all__` with the minimal imports the kept functions use:
#      `from types import SimpleNamespace`, `from typing import Optional`,
#      `import torch`.
#   2. Replaced `import verl.utils.torch_functional as verl_F` with a local
#      `verl_F = SimpleNamespace(...)` built from clip_by_value / masked_sum /
#      masked_mean vendored verbatim from verl/utils/torch_functional.py at the
#      same commit, so `verl_F.masked_mean` etc. inside the kept bodies resolve
#      unchanged.
#   3. Removed the framework deprecation decorator
#      `@deprecated("verl.trainer.ppo.core_algos.compute_policy_loss_vanilla")`
#      from `compute_policy_loss`.
#   4. Added the `# fmt: off` directive so formatters preserve upstream layout.
#
# Link rot inherited from upstream (docstrings below cite TRL PPOTrainer line
# links on `main`; that file has since been removed from TRL main, so the links
# are dead — they are upstream's bytes and are preserved verbatim under the
# body hash). Working permalinks to the referenced code at TRL v0.11.4:
#   compute_policy_loss docstring's ...ppo_trainer.py#L1122 ->
#     https://github.com/huggingface/trl/blob/714cd42f67cb2c8bef91546dbb2258e326c03c89/trl/trainer/ppo_trainer.py#L1234-L1240
#   compute_value_loss docstring's ...ppo_trainer.py#L1151 ->
#     https://github.com/huggingface/trl/blob/714cd42f67cb2c8bef91546dbb2258e326c03c89/trl/trainer/ppo_trainer.py#L1223-L1232
#   kl_penalty_forward docstring's ...ppo_trainer.py#L1104 ->
#     https://github.com/huggingface/trl/blob/714cd42f67cb2c8bef91546dbb2258e326c03c89/trl/trainer/ppo_trainer.py#L1150-L1165
#
# Self-integrity: SHA256 of every byte of this file below the END-OF-HEADER
# line, exactly as committed (recomputed and checked by
# tests/test_vendor.py::test_verl_vendored_body_sha256_matches_header):
#   BODY_SHA256: d335dc60a28544ce25ad104b4e3364b2c2ae85b042022e5c55222cf85b5e2d0f
# fmt: off
# === END VENDOR HEADER ===
# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Core functions to implement PPO algorithms.
The function implemented in this file should be used by trainer with different distributed strategies to
implement PPO-like algorithms.
"""

from types import SimpleNamespace
from typing import Optional

import torch


def clip_by_value(x: torch.Tensor, tensor_min: torch.Tensor, tensor_max: torch.Tensor) -> torch.Tensor:
    """Clip tensor values to a range defined by tensor bounds.

    Extension of torch.clamp that supports tensor-valued min/max bounds
    instead of only scalar bounds.

    Args:
        x: Input tensor to clip.
        tensor_min: Minimum bound tensor (broadcastable to x).
        tensor_max: Maximum bound tensor (broadcastable to x).

    Returns:
        torch.Tensor: Clipped tensor with values in [tensor_min, tensor_max].

    See Also:
        https://github.com/pytorch/pytorch/issues/2793#issuecomment-428784713
    """
    clipped = torch.max(torch.min(x, tensor_max), tensor_min)
    return clipped


def masked_sum(values: torch.Tensor, mask: torch.Tensor, axis: int | tuple[int, ...] | None = None) -> torch.Tensor:
    """Compute sum of tensor values where mask is True.

    NaN values outside the mask are replaced with zeros to prevent
    contaminating the sum.

    Args:
        values: Input tensor containing values to sum.
        mask: Boolean or numeric mask tensor (same shape as values).
            Non-zero values indicate elements to include.
        axis: Dimension(s) along which to sum. None sums all elements.

    Returns:
        torch.Tensor: Sum of masked values, reduced along specified axis.
    """
    # If NaNs exist out of mask, replace NaNs in values with a value that
    # won't affect the sum (e.g., 0 for masked regions)
    valid_values = torch.where(mask.bool(), values, 0.0)
    return (valid_values * mask).sum(axis=axis)


def masked_mean(values, mask, axis=None):
    """
    Compute the mean of `values` over elements selected by `mask`.

    Args:
        values (Tensor): Input tensor.
        mask (Tensor): Boolean or numeric mask of the same shape as `values`.
        axis (int or tuple of int, optional): Dimension(s) along which to compute the mean.
            Defaults to None (over all elements).

    Returns:
        Tensor: Masked mean, with shape equal to `values` reduced over `axis`.
    """
    s = masked_sum(values, mask, axis)
    return s / (mask.sum(axis=axis) + 1e-8)


verl_F = SimpleNamespace(clip_by_value=clip_by_value, masked_sum=masked_sum, masked_mean=masked_mean)


def agg_loss(
    loss_mat: torch.Tensor,
    loss_mask: torch.Tensor,
    loss_agg_mode: str,
    dp_size: int = 1,
    batch_num_tokens: Optional[int] = None,
    global_batch_size: Optional[int] = None,
    loss_scale_factor: Optional[int] = None,
):
    """
    Aggregate the loss across global batch to ensure the loss is invariant to fsdp/megatron parallelism.

    NOTE: The returned loss has different behaviors for different backend:
    - FSDP: the loss is directly used for backward.
    - Megatron: the loss should be scaled by `num_microbatches` and `cp_size` for pp schedule.

    Args:
        loss_mat: micro batch loss matrix, (bs, response_length)
        loss_mask: micro batch loss mask, (bs, response_length)
        loss_agg_mode: method to aggregate the loss matrix into a scalar
        dp_size: data parallel size
        batch_num_tokens: number of valid tokens in global batch
        global_batch_size: global batch size
        loss_scale_factor: scale factor for "seq-mean-token-sum-norm" mode. If None, uses loss_mask.shape[-1].
            Set this to a constant value to ensure consistent normalization throughout training.

    Returns:
        loss: `a scalar torch.Tensor`
            aggregated loss
    """
    if loss_agg_mode == "token-mean":
        if batch_num_tokens is None:
            if dp_size > 1:
                raise ValueError("(global) batch_num_tokens is required when dp_size > 1")
            batch_num_tokens = loss_mask.sum()
        loss = verl_F.masked_sum(loss_mat, loss_mask) / batch_num_tokens * dp_size
    elif loss_agg_mode in ["seq-mean-token-sum", "seq-mean-token-sum-norm"]:
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)  # token-sum
        seq_mask = (torch.sum(loss_mask, dim=-1) > 0).float()  # exclude fully masked sequences
        if global_batch_size is None:
            if dp_size > 1:
                raise ValueError("global_batch_size is required when dp_size > 1")
            global_batch_size = seq_mask.sum()
        loss = verl_F.masked_sum(seq_losses, seq_mask) / global_batch_size * dp_size  # seq-mean
        if loss_agg_mode == "seq-mean-token-sum-norm":
            if loss_scale_factor is None:
                horizon = loss_mask.shape[-1]
                loss_scale_factor = horizon
            loss /= loss_scale_factor
    elif loss_agg_mode == "seq-mean-token-mean":
        seq_mask = torch.sum(loss_mask, dim=-1)  # per-sequence token count
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1) / (seq_mask + 1e-8)  # token-mean
        seq_mask = (seq_mask > 0).float()  # exclude fully masked sequences
        if global_batch_size is None:
            if dp_size > 1:
                raise ValueError("global_batch_size is required when dp_size > 1")
            global_batch_size = seq_mask.sum()
        loss = verl_F.masked_sum(seq_losses, seq_mask) / global_batch_size * dp_size  # seq-mean
    else:
        raise ValueError(f"Invalid loss_agg_mode: {loss_agg_mode}")

    return loss


def compute_policy_loss(
    old_log_prob,
    log_prob,
    advantages,
    response_mask,
    cliprange=None,
    cliprange_low=None,
    cliprange_high=None,
    clip_ratio_c=3.0,
    loss_agg_mode: str = "token-mean",
):
    """
    Compute the clipped policy objective and related metrics for PPO.

    Adapted from
    https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        cliprange (float, optional):
            Clipping parameter ε for standard PPO. See https://arxiv.org/abs/1707.06347.
            Defaults to None (must be provided).
        cliprange_low (float, optional):
            Lower clip range for dual-clip PPO. Defaults to same as `cliprange`.
        cliprange_high (float, optional):
            Upper clip range for dual-clip PPO. Defaults to same as `cliprange`.
        clip_ratio_c (float, optional):
            Lower bound of the ratio for dual-clip PPO. See https://arxiv.org/pdf/1912.09729.
            Defaults to 3.0.
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".
    """
    assert clip_ratio_c > 1.0, (
        "The lower bound of the clip_ratio_c for dual-clip PPO should be greater than 1.0,"
        + f" but get the value: {clip_ratio_c}."
    )

    negative_approx_kl = log_prob - old_log_prob
    # Clamp negative_approx_kl for stability
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    pg_losses1 = -advantages * ratio
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange
    pg_losses2 = -advantages * torch.clamp(
        ratio, 1 - cliprange_low, 1 + cliprange_high
    )  # - clip(ratio, 1-cliprange, 1+cliprange) * A
    clip_pg_losses1 = torch.maximum(
        pg_losses1, pg_losses2
    )  # max(-ratio * A, -clip(ratio, 1-cliprange, 1+cliprange) * A)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)

    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    pg_clipfrac_lower = verl_F.masked_mean(
        torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), response_mask
    )

    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower


def compute_value_loss(
    vpreds: torch.Tensor,
    returns: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    cliprange_value: float,
    loss_agg_mode: str = "token-mean",
    dp_size: int = 1,
    batch_num_tokens: Optional[int] = None,
    global_batch_size: Optional[int] = None,
    loss_scale_factor: Optional[int] = None,
):
    """
    Compute the clipped value-function loss for PPO.

    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1151

    Args:
        vpreds (torch.FloatTensor):
            Predicted values from the value head, shape (batch_size, response_length).
        values (torch.FloatTensor):
            Old (baseline) values from the value head, shape (batch_size, response_length).
        returns (torch.FloatTensor):
            Ground-truth returns, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the value loss calculation.
        cliprange_value (float):
            Clip range for value prediction updates.
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".
        dp_size (int, optional):
            Data parallel size, forwarded to `agg_loss` for global-batch normalization. Defaults to 1.
        batch_num_tokens (Optional[int], optional):
            Number of valid tokens in the global batch, forwarded to `agg_loss`. Defaults to None
            (normalize by the local micro-batch token count).
        global_batch_size (Optional[int], optional):
            Global batch size, forwarded to `agg_loss` for the seq-mean modes. Defaults to None.
        loss_scale_factor (Optional[int], optional):
            Scale factor for the "seq-mean-token-sum-norm" mode, forwarded to `agg_loss`. Defaults to None.

    Returns:
        vf_loss (torch.FloatTensor):
            A scalar tensor containing the aggregated value-function loss.
        vf_clipfrac (float):
            Fraction of elements where the clipped loss was used.
    """
    vpredclipped = verl_F.clip_by_value(vpreds, values - cliprange_value, values + cliprange_value)
    vf_losses1 = (vpreds - returns) ** 2
    vf_losses2 = (vpredclipped - returns) ** 2
    clipped_vf_losses = torch.max(vf_losses1, vf_losses2)
    vf_loss = 0.5 * agg_loss(
        loss_mat=clipped_vf_losses,
        loss_mask=response_mask,
        loss_agg_mode=loss_agg_mode,
        dp_size=dp_size,
        batch_num_tokens=batch_num_tokens,
        global_batch_size=global_batch_size,
        loss_scale_factor=loss_scale_factor,
    )
    vf_clipfrac = verl_F.masked_mean(torch.gt(vf_losses2, vf_losses1).float(), response_mask)
    return vf_loss, vf_clipfrac


def kl_penalty(logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty) -> torch.FloatTensor:
    """Compute KL divergence given logprob and ref_logprob. Optionally using straight through to bind k2 on other
    kl penalty compute method for unbiased KL gradient estimation.
    See more description in http://joschu.net/blog/kl-approx.html

    Args:
        logprob:
        ref_logprob:

    Returns:
        kl_estimate
    """
    # Strip the optional '+' suffix so e.g. "k3+" dispatches to "k3".
    base_kl_penalty = kl_penalty[:-1] if kl_penalty.endswith("+") else kl_penalty
    forward_score = kl_penalty_forward(logprob, ref_logprob, base_kl_penalty)
    if not kl_penalty.endswith("+") or kl_penalty in ("mse", "k2"):
        return forward_score

    """
    The expectation of k1 and k3 estimator is the expected value of KL, but the expected gradient of k1 and k3
    estimator is not the expected gradient of KL. On the other hand k2 estimator gives right gradient estimator, 
    so we use a straight through trick here if the kl_penalty method ends with '+', e.g., k3+. 
    """
    backward_score = 0.5 * (logprob - ref_logprob).square()

    return backward_score - backward_score.detach() + forward_score.detach()


def kl_penalty_forward(logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty) -> torch.FloatTensor:
    """Compute KL divergence given logprob and ref_logprob.
    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1104
    See more description in http://joschu.net/blog/kl-approx.html

    Args:
        logprob:
        ref_logprob:

    Returns:
        kl_estimate
    """
    if kl_penalty in ("kl", "k1"):
        return logprob - ref_logprob

    if kl_penalty == "abs":
        return (logprob - ref_logprob).abs()

    if kl_penalty in ("mse", "k2"):
        return 0.5 * (logprob - ref_logprob).square()

    # J. Schulman. Approximating kl divergence, 2020.
    # # URL http://joschu.net/blog/kl-approx.html.
    if kl_penalty in ("low_var_kl", "k3"):
        kl = ref_logprob - logprob
        # For numerical stability
        kl = torch.clamp(kl, min=-20, max=20)
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1).contiguous()
        return torch.clamp(kld, min=-10, max=10)

    if kl_penalty == "full":
        # so, here logprob and ref_logprob should contain the logits for every token in vocabulary
        raise NotImplementedError

    raise NotImplementedError
