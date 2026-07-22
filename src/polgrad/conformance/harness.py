"""Conformance harness: measure how framework losses deviate from polgrad semantics.

``VENDORED`` maps ``(framework, variant)`` to wrappers around the vendored framework
loss functions (``polgrad.conformance._vendor``) and around the TRL reimplementation in
this module. Every wrapper uses the same calling convention — keyword tensors
``(logprobs, old_logprobs, advantages, response_mask)`` with ``response_mask`` boolean
``[B, T]`` and ``advantages`` ``[B, T]`` — and returns a scalar loss differentiable in
``logprobs``. Wrapper hyperparameters are pinned module constants (``_CLIP_EPS``,
``_VERL_CLIP_RATIO_C``) so that fixtures, tests, and reports agree on one setting.

Registered keys:

- ``("verl", "pg_clip_token_mean" | "pg_clip_seq_mean_token_mean" |
  "pg_clip_seq_mean_token_sum" | "pg_clip_seq_mean_token_sum_norm")``:
  ``compute_policy_loss`` (dual-clip always active upstream, ``clip_ratio_c = 3.0``)
  with the matching ``agg_loss`` mode.
- ``("openrlhf", "pg_clip_token_mean" | "pg_clip_seq_mean_token_mean")``:
  ``PolicyLoss`` with ``token_level_loss`` True / False (no dual clip by default).
- ``("trl", "grpo" | "bnpo" | "dr_grpo")``: the ``loss_type`` values of TRL's
  ``GRPOTrainer`` via :func:`_trl_grpo_loss`, a faithful **reimplementation** (TRL's
  loss is entangled with trainer state and cannot be vendored as a pure function; see
  the provenance in that function's docstring). It is never presented as vendored code.
- ``("verl", "gspo" | "cispo")``: verl's ``compute_policy_loss_gspo`` and
  ``compute_policy_loss_cispo`` at the same pinned verl commit, via
  :func:`_verl_gspo_loss` and :func:`_verl_cispo_loss` — faithful
  **reimplementations**, never presented as vendored code (both upstream functions
  assert on and read a verl ``ActorConfig``, so they were dropped from the vendored
  file; every config default frozen here is documented in the function docstrings).
  Their aggregation calls the vendored ``verl_core_algos.agg_loss``, which is
  byte-identical to upstream at that commit.

:func:`compare_losses` runs two such callables on seeded random inputs and reports the
worst relative loss/gradient differences; :func:`deviation_report` compares a polgrad
:class:`~polgrad.losses.PolicyLossConfig` against a ``VENDORED`` entry. Randomness
enters only through the explicit ``torch.Generator`` argument (docs/conventions.md).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

import torch
from torch import Tensor

from polgrad.conformance._vendor import openrlhf_loss, verl_core_algos
from polgrad.losses import PolicyLossConfig, policy_loss

__all__ = [
    "VENDORED",
    "DeviationReport",
    "compare_losses",
    "deviation_report",
]

LossFn = Callable[..., Tensor]

# Shared PPO clip width for every VENDORED wrapper (the upstream defaults of all three
# frameworks: verl cliprange, OpenRLHF clip_eps_low/high, TRL GRPOConfig.epsilon).
_CLIP_EPS = 0.2
# verl's compute_policy_loss applies dual-clip unconditionally with this upstream
# default; polgrad configs compared against verl must set ClipConfig.ratio_cap to it.
_VERL_CLIP_RATIO_C = 3.0
# Relative-difference floor: denominators are clamped to this so exact-zero losses and
# gradients compare as zero difference instead of dividing by zero.
_REL_FLOOR = 1e-12


@dataclass(frozen=True)
class DeviationReport:
    """Worst-case loss/gradient disagreement between two loss callables.

    Per case ``i`` with scalar losses ``a_i, b_i`` and gradients ``g^a_i, g^b_i``
    (w.r.t. ``logprobs``):

    - loss rel diff: ``|a_i - b_i| / max(|a_i|, |b_i|, 1e-12)``
    - grad rel diff: ``‖g^a_i - g^b_i‖₂ / max(‖g^a_i‖₂, ‖g^b_i‖₂, 1e-12)``
    - grad cosine: ``⟨g^a_i, g^b_i⟩ / (‖g^a_i‖₂ · ‖g^b_i‖₂)``; defined as ``1.0`` when
      both gradients are zero and ``0.0`` when exactly one is.

    Attributes:
        max_loss_rel_diff: Maximum loss rel diff over all cases.
        max_grad_rel_diff: Maximum grad rel diff over all cases.
        grad_cosine_min: Minimum grad cosine over all cases.
        n_cases: Number of seeded cases compared.
        worst_case_seed: Seed of the case maximizing ``max(loss rel, grad rel)``
            (the first such case on ties); rebuild it with
            ``torch.Generator().manual_seed(worst_case_seed)`` and the harness sampler.
        notes: Free-form provenance lines (config, framework/variant, caveats).

    References:
        docs/conventions.md (determinism rules);
        tests/test_conformance.py::test_deviation_report_summary_mentions_key_figures.
    """

    max_loss_rel_diff: float
    max_grad_rel_diff: float
    grad_cosine_min: float
    n_cases: int
    worst_case_seed: int
    notes: tuple[str, ...]

    def summary(self) -> str:
        """One header line of the worst-case metrics, followed by the notes."""
        header = (
            f"conformance over {self.n_cases} seeded cases: "
            f"max loss rel diff {self.max_loss_rel_diff:.3e}, "
            f"max grad rel diff {self.max_grad_rel_diff:.3e}, "
            f"min grad cosine {self.grad_cosine_min:.9f}, "
            f"worst-case seed {self.worst_case_seed}"
        )
        return "\n".join((header, *self.notes))


def _trl_grpo_loss(
    logprobs: Tensor,
    old_logprobs: Tensor,
    advantages: Tensor,
    response_mask: Tensor,
    *,
    loss_type: str,
    importance_sampling_level: str = "token",
    epsilon_low: float = _CLIP_EPS,
    epsilon_high: float = _CLIP_EPS,
    max_completion_length: int | None = None,
) -> Tensor:
    """Faithful reimplementation of TRL ``GRPOTrainer._compute_loss`` (policy term).

    Provenance (this is a reimplementation, not vendored code — TRL's loss reads
    trainer state such as ``self.beta``, entropy masking, gradient-accumulation
    normalizers, and vLLM correction buffers, so it cannot be extracted as a pure
    function):

    - Upstream: https://github.com/huggingface/trl, version v1.8.0,
      commit ``95809b942eb5d11d0b06d749510d88be99230b73``.
    - Source: ``trl/trainer/grpo_trainer.py``, ``GRPOTrainer._compute_loss``; permalink
      https://github.com/huggingface/trl/blob/95809b942eb5d11d0b06d749510d88be99230b73/trl/trainer/grpo_trainer.py#L2857-L3016
      (upstream file SHA256
      ``52d9a6c1e298df35d0da4a6fa17874d750ee627f6ac15393c8860d74d1ba4917``).
    - The reproduced arithmetic is Copyright 2020-2026 The HuggingFace Team. All
      rights reserved. Licensed under the Apache License, Version 2.0. See also the
      TRL entry in the repository-level ``NOTICE`` file.

    Reimplemented scope, keeping upstream variable names (``coef_1``, ``coef_2``,
    ``per_token_loss``) and upstream arithmetic verbatim:

    - ``importance_sampling_level``: ``"token"`` (``log_importance_weights =
      log_ratio``) and ``"sequence"`` (masked per-row mean of ``log_ratio``,
      broadcast); ``coef_1 = exp(log_importance_weights)``,
      ``coef_2 = clamp(coef_1, 1 - ε_low, 1 + ε_high)``,
      ``per_token_loss = -min(coef_1 · A, coef_2 · A)``.
    - ``loss_type="grpo"``: ``mean_b(Σ_t m·x / clamp(L_b, min=1))``;
      ``loss_type="bnpo"``: ``Σ m·x / clamp(Σ m, min=1)``;
      ``loss_type="dr_grpo"``: ``Σ m·x / (B · max_completion_length)``.

    Omitted trainer-state features (each defaults off upstream or reads state that has
    no pure-function equivalent): the ``beta``-scaled k3 KL term, entropy bonus and
    ``top_entropy_quantile`` masking, ``off_policy_mask_threshold``, the vLLM
    importance-sampling correction, the ``delta`` two-sided cap, the
    ``current_gradient_accumulation_steps`` normalizer (equal to 1 without
    accumulation), and the ``dapo``/``cispo``/``sapo``/``luspo``/``vespo`` loss types
    whose normalizers read ``num_items_in_batch`` or other trainer state.

    Args:
        logprobs: ``[B, T]`` current-policy logprobs (differentiable).
        old_logprobs: ``[B, T]`` behavior-policy logprobs.
        advantages: ``[B]`` (unsqueezed to ``[B, 1]`` exactly as upstream) or
            ``[B, T]``.
        response_mask: ``[B, T]`` bool mask (TRL's ``completion_mask``).
        loss_type: ``"grpo"``, ``"bnpo"``, or ``"dr_grpo"``.
        importance_sampling_level: ``"token"`` or ``"sequence"``.
        epsilon_low: TRL ``GRPOConfig.epsilon`` (upstream default 0.2).
        epsilon_high: TRL ``GRPOConfig.epsilon_high`` (defaults to ``epsilon``).
        max_completion_length: Fixed budget dividing the ``dr_grpo`` loss; required for
            that loss type.

    Returns:
        Scalar loss tensor, differentiable in ``logprobs``.

    Raises:
        ValueError: On an unknown ``loss_type`` or ``importance_sampling_level``, or a
            missing ``max_completion_length`` under ``loss_type="dr_grpo"``.

    References:
        TRL v1.8.0 permalink above; docs/derivations/aggregation.md (the matching
        polgrad modes);
        tests/test_conformance.py::test_trl_grpo_agrees_with_polgrad_seq_mean_token_mean_on_fixtures,
        tests/test_conformance.py::test_trl_reimplementation_provenance_documented.
    """
    mask = response_mask.to(logprobs.dtype)
    advantages_2d = advantages.unsqueeze(1) if advantages.dim() == 1 else advantages
    log_ratio = logprobs - old_logprobs
    if importance_sampling_level == "token":
        log_importance_weights = log_ratio
    elif importance_sampling_level == "sequence":
        seq_mean = (log_ratio * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)
        log_importance_weights = seq_mean.unsqueeze(-1)
    else:
        raise ValueError(
            f"unknown importance_sampling_level: {importance_sampling_level!r}; "
            f"TRL v1.8.0 supports 'token' and 'sequence'"
        )
    coef_1 = torch.exp(log_importance_weights)
    coef_2 = torch.clamp(coef_1, 1 - epsilon_low, 1 + epsilon_high)
    per_token_loss1 = coef_1 * advantages_2d
    per_token_loss2 = coef_2 * advantages_2d
    per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
    if loss_type == "grpo":
        return ((per_token_loss * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)).mean()
    if loss_type == "bnpo":
        return (per_token_loss * mask).sum() / mask.sum().clamp(min=1.0)
    if loss_type == "dr_grpo":
        if max_completion_length is None:
            raise ValueError("loss_type='dr_grpo' requires max_completion_length; got None")
        return (per_token_loss * mask).sum() / (per_token_loss.size(0) * max_completion_length)
    raise ValueError(
        f"unknown loss_type: {loss_type!r}; this reimplementation covers "
        f"'grpo', 'bnpo', and 'dr_grpo'"
    )


def _verl_gspo_loss(
    logprobs: Tensor,
    old_logprobs: Tensor,
    advantages: Tensor,
    response_mask: Tensor,
    *,
    clip_ratio_low: float = _CLIP_EPS,
    clip_ratio_high: float = _CLIP_EPS,
) -> Tensor:
    """Faithful reimplementation of verl ``compute_policy_loss_gspo`` (policy term).

    Provenance (this is a reimplementation, not vendored code — the upstream function
    asserts ``isinstance(config, ActorConfig)`` and reads its clip fields and
    ``config.global_batch_info``, so it was dropped from the vendored file; see the
    drop list in ``polgrad/conformance/_vendor/verl_core_algos.py``):

    - Upstream: https://github.com/volcengine/verl (redirects to
      https://github.com/verl-project/verl), commit
      ``74a718a492092312f1004fe25369975137388849`` (the vendored verl pin).
    - Source: ``verl/trainer/ppo/core_algos.py``, ``compute_policy_loss_gspo``;
      permalink
      https://github.com/verl-project/verl/blob/74a718a492092312f1004fe25369975137388849/verl/trainer/ppo/core_algos.py#L1538-L1611
      (upstream file SHA256
      ``9114f9e16c87e4c9ebf2fa016baf733c9bbc819766b53c8968aaa9e8abcd7916``, the same
      hash recorded in the vendored file's header).
    - The reproduced arithmetic is Copyright 2024 Bytedance Ltd. and/or its
      affiliates. Licensed under the Apache License, Version 2.0. See also the verl
      entry in the repository-level ``NOTICE`` file.

    Frozen config defaults (each read from ``config`` upstream; pinned to the defaults
    of ``ActorConfig`` in ``verl/workers/config/actor.py`` at the same commit, which
    ``verl/trainer/config/actor/actor.yaml`` mirrors):

    - ``clip_ratio_low = 0.2`` and ``clip_ratio_high = 0.2`` (``ActorConfig`` fields;
      actor.yaml lines 39/42). Both default non-``None``, so upstream's fallback to
      ``config.clip_ratio`` (default 0.2, actor.yaml line 36) never fires.
    - ``config.global_batch_info = {}`` (the ``ActorConfig`` dataclass default,
      ``field(default_factory=dict)``). Upstream expands it into ``agg_loss`` and the
      trainer populates it per training step with ``dp_size`` / ``batch_num_tokens`` /
      ``global_batch_size`` for parallelism-invariant aggregation — runtime batch
      state with no pure-function equivalent. Frozen at the empty default, ``agg_loss``
      runs with ``dp_size=1`` and batch statistics from the local batch, the same
      single-process semantics as the vendored ``compute_policy_loss`` wrappers.

    Reimplemented scope, keeping upstream variable names (``negative_approx_kl``,
    ``log_seq_importance_ratio``, ``pg_losses1``/``pg_losses2``) and arithmetic
    verbatim: the GSPO-token combined ratio ``s_{i,t}`` with its exact stop-gradient
    structure (arXiv 2507.18071 eq. 14) ``log s_{i,t} = log_prob - sg[log_prob] +
    sg[mean masked log-ratio]``, clamped at ``max=10.0``; the clipped surrogate
    ``max(-A·s, -A·clamp(s, 1-ε_lo, 1+ε_hi))``; aggregation through the vendored
    ``agg_loss`` with the **hardcoded** upstream mode ``"seq-mean-token-mean"`` (the
    upstream ``loss_agg_mode`` parameter is ignored by the loss, so this
    reimplementation does not take one).

    Omitted upstream surface (nothing else is): the ``rollout_is_weights`` multiplier
    (defaults ``None`` upstream) and the returned ``pg_metrics`` dict of detached
    diagnostics (``pg_clipfrac``, ``ppo_kl``, ``pg_clipfrac_lower``) — only
    ``pg_loss`` is reproduced.

    Args:
        logprobs: ``[B, T]`` current-policy logprobs (upstream ``log_prob``,
            differentiable).
        old_logprobs: ``[B, T]`` behavior-policy logprobs (upstream ``old_log_prob``).
        advantages: ``[B, T]`` per-token advantages, as upstream.
        response_mask: ``[B, T]`` bool mask (converted to the logprob dtype for
            upstream's float-mask arithmetic).
        clip_ratio_low: Frozen ``ActorConfig.clip_ratio_low`` (upstream default 0.2).
        clip_ratio_high: Frozen ``ActorConfig.clip_ratio_high`` (upstream default 0.2).

    Returns:
        Scalar loss tensor, differentiable in ``logprobs``.

    References:
        verl permalink above; arXiv 2507.18071 eq. 13-14 (GSPO-token);
        docs/derivations/losses.md (GSPO gradients from the sg[] algebra);
        tests/test_conformance.py::test_verl_gspo_agrees_with_polgrad_sequence_ratio_on_seeded_batches,
        tests/test_conformance.py::test_verl_gspo_gradient_is_token_local_gspo_token_form,
        tests/test_conformance.py::test_verl_reimplementation_provenance_documented.
    """
    log_prob = logprobs
    old_log_prob = old_logprobs
    response_mask = response_mask.to(logprobs.dtype)

    negative_approx_kl = log_prob - old_log_prob
    # sequence-level importance ratio: s_i(θ) = exp[(1/|y_i|) Σ_t log-ratio_t]
    seq_lengths = torch.sum(response_mask, dim=-1).clamp(min=1)
    negative_approx_kl_seq = torch.sum(negative_approx_kl * response_mask, dim=-1) / seq_lengths
    # detach (upstream): the GSPO-token combined ratio freezes both the sequence weight
    # (sg[s_i]) and the token denominator (sg[log_prob]), leaving the token-local
    # gradient sg[s_i]·∇logprob_t (arXiv 2507.18071 eq. 14):
    # log(s_i,t(θ)) = sg[log(s_i(θ))] + log_prob - sg[log_prob]
    log_seq_importance_ratio = (
        log_prob - log_prob.detach() + negative_approx_kl_seq.detach().unsqueeze(-1)
    )
    log_seq_importance_ratio = torch.clamp(
        log_seq_importance_ratio, max=10.0
    )  # clamp for numerical stability (upstream literal)
    seq_importance_ratio = torch.exp(log_seq_importance_ratio)

    pg_losses1 = -advantages * seq_importance_ratio
    pg_losses2 = -advantages * torch.clamp(
        seq_importance_ratio, 1 - clip_ratio_low, 1 + clip_ratio_high
    )
    pg_losses = torch.maximum(pg_losses1, pg_losses2)

    # upstream hardcodes "seq-mean-token-mean" (its loss_agg_mode parameter is unused
    # by the loss); global_batch_info frozen to the ActorConfig default {}.
    pg_loss: Tensor = verl_core_algos.agg_loss(
        loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode="seq-mean-token-mean"
    )
    return pg_loss


def _verl_cispo_loss(
    logprobs: Tensor,
    old_logprobs: Tensor,
    advantages: Tensor,
    response_mask: Tensor,
    *,
    clip_ratio_low: float = _CLIP_EPS,
    clip_ratio_high: float = _CLIP_EPS,
    loss_agg_mode: str = "token-mean",
) -> Tensor:
    """Faithful reimplementation of verl ``compute_policy_loss_cispo`` (policy term).

    Provenance (this is a reimplementation, not vendored code — the upstream function
    asserts ``isinstance(config, ActorConfig)`` and reads its clip fields and
    ``config.global_batch_info``, so it was dropped from the vendored file; see the
    drop list in ``polgrad/conformance/_vendor/verl_core_algos.py``):

    - Upstream: https://github.com/volcengine/verl (redirects to
      https://github.com/verl-project/verl), commit
      ``74a718a492092312f1004fe25369975137388849`` (the vendored verl pin).
    - Source: ``verl/trainer/ppo/core_algos.py``, ``compute_policy_loss_cispo``;
      permalink
      https://github.com/verl-project/verl/blob/74a718a492092312f1004fe25369975137388849/verl/trainer/ppo/core_algos.py#L2006-L2064
      (upstream file SHA256
      ``9114f9e16c87e4c9ebf2fa016baf733c9bbc819766b53c8968aaa9e8abcd7916``, the same
      hash recorded in the vendored file's header).
    - The reproduced arithmetic is Copyright 2024 Bytedance Ltd. and/or its
      affiliates. Licensed under the Apache License, Version 2.0. See also the verl
      entry in the repository-level ``NOTICE`` file.

    Frozen config defaults (each read from ``config`` upstream; pinned to the defaults
    of ``ActorConfig`` in ``verl/workers/config/actor.py`` at the same commit, which
    ``verl/trainer/config/actor/actor.yaml`` mirrors):

    - ``clip_ratio_low = 0.2`` and ``clip_ratio_high = 0.2`` (``ActorConfig`` fields;
      actor.yaml lines 39/42). Both default non-``None``, so upstream's fallback to
      ``config.clip_ratio`` (default 0.2, actor.yaml line 36) never fires. Note the
      resulting IS-weight clamp is **two-sided** — the CISPO paper (arXiv 2506.13585)
      imposes no lower bound in its experiments.
    - ``loss_agg_mode = "token-mean"`` (the upstream parameter default, equal to
      ``ActorConfig.loss_agg_mode``).
    - ``config.global_batch_info = {}`` (the ``ActorConfig`` dataclass default,
      ``field(default_factory=dict)``); frozen exactly as in :func:`_verl_gspo_loss`
      (the trainer-populated contents are runtime batch state).

    Reimplemented scope, keeping upstream variable names (``negative_approx_kl``,
    ``ratio``, ``clipped_ratio_sg``, ``pg_losses``) and arithmetic verbatim: the
    log-ratio clamp at ``±20.0``; ``clipped_ratio = clamp(ratio, 1-ε_lo, 1+ε_hi)``
    detached in full (upstream ``clipped_ratio_sg``), so the gradient flows only
    through the REINFORCE factor ``log_prob`` in
    ``pg_losses = -clipped_ratio_sg · advantages · log_prob`` (arXiv 2506.13585
    eq. 5); aggregation through the vendored ``agg_loss``.

    Omitted upstream surface (nothing else is): the ``rollout_is_weights`` multiplier
    (defaults ``None`` upstream) and the returned ``pg_metrics`` dict of detached
    diagnostics (``pg_clipfrac``, ``ppo_kl``, ``pg_clipfrac_lower``) — only
    ``pg_loss`` is reproduced.

    Args:
        logprobs: ``[B, T]`` current-policy logprobs (upstream ``log_prob``,
            differentiable).
        old_logprobs: ``[B, T]`` behavior-policy logprobs (upstream ``old_log_prob``).
        advantages: ``[B, T]`` per-token advantages, as upstream.
        response_mask: ``[B, T]`` bool mask (converted to the logprob dtype for
            upstream's float-mask arithmetic).
        clip_ratio_low: Frozen ``ActorConfig.clip_ratio_low`` (upstream default 0.2).
        clip_ratio_high: Frozen ``ActorConfig.clip_ratio_high`` (upstream default 0.2).
        loss_agg_mode: ``agg_loss`` mode (upstream default ``"token-mean"``).

    Returns:
        Scalar loss tensor, differentiable in ``logprobs``.

    References:
        verl permalink above; arXiv 2506.13585 eq. 4-5 (CISPO);
        docs/derivations/losses.md (CISPO stop-gradient semantics);
        tests/test_conformance.py::test_verl_cispo_agrees_with_polgrad_cispo_on_fixtures,
        tests/test_conformance.py::test_verl_reimplementation_provenance_documented.
    """
    log_prob = logprobs
    old_log_prob = old_logprobs
    response_mask = response_mask.to(logprobs.dtype)

    # importance sampling ratio: π_θ / π_θ_old
    negative_approx_kl = log_prob - old_log_prob
    negative_approx_kl = torch.clamp(
        negative_approx_kl, min=-20.0, max=20.0
    )  # clamp for numerical stability (upstream literals)
    ratio = torch.exp(negative_approx_kl)

    clipped_ratio = torch.clamp(ratio, 1 - clip_ratio_low, 1 + clip_ratio_high)
    # detach (upstream ``clipped_ratio_sg``): CISPO stop-gradients the clipped IS
    # weight; gradients flow only through log_prob in the loss term (arXiv 2506.13585
    # eq. 5).
    clipped_ratio_sg = clipped_ratio.detach()

    # CISPO loss (to minimize): L = -sg[clip(ratio)] * A * log_prob
    pg_losses = -clipped_ratio_sg * advantages * log_prob

    # global_batch_info frozen to the ActorConfig default {}.
    pg_loss: Tensor = verl_core_algos.agg_loss(
        loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode
    )
    return pg_loss


def _verl_pg_clip(loss_agg_mode: str) -> LossFn:
    """Wrapper factory over vendored verl ``compute_policy_loss`` + ``agg_loss``.

    Pins ``cliprange = 0.2`` and the upstream default ``clip_ratio_c = 3.0`` (verl
    applies the dual-clip floor unconditionally); converts the boolean mask to the
    logprob dtype as verl's float-mask convention expects.
    """

    def fn(
        *, logprobs: Tensor, old_logprobs: Tensor, advantages: Tensor, response_mask: Tensor
    ) -> Tensor:
        outputs = verl_core_algos.compute_policy_loss(
            old_logprobs,
            logprobs,
            advantages,
            response_mask.to(logprobs.dtype),
            cliprange=_CLIP_EPS,
            clip_ratio_c=_VERL_CLIP_RATIO_C,
            loss_agg_mode=loss_agg_mode,
        )
        loss: Tensor = outputs[0]
        return loss

    return fn


def _openrlhf_pg_clip(token_level_loss: bool) -> LossFn:
    """Wrapper factory over vendored OpenRLHF ``PolicyLoss`` (no dual clip upstream).

    Pins ``clip_eps_low = clip_eps_high = 0.2`` (the upstream defaults); converts the
    boolean mask to the logprob dtype for the float-mask arithmetic inside.
    """

    def fn(
        *, logprobs: Tensor, old_logprobs: Tensor, advantages: Tensor, response_mask: Tensor
    ) -> Tensor:
        module = openrlhf_loss.PolicyLoss(
            clip_eps_low=_CLIP_EPS, clip_eps_high=_CLIP_EPS, token_level_loss=token_level_loss
        )
        outputs = module.forward(
            logprobs, old_logprobs, advantages, response_mask.to(logprobs.dtype)
        )
        loss: Tensor = outputs[0]
        return loss

    return fn


def _trl_variant(loss_type: str) -> LossFn:
    """Wrapper factory over the TRL reimplementation for one ``loss_type``.

    For ``dr_grpo`` the fixed budget ``max_completion_length`` is pinned to the padded
    width ``T`` of the batch (the wrapper takes only the four keyword tensors), so
    polgrad ``TOKEN_SUM_NORM`` with ``norm_len = T`` reproduces it.
    """

    def fn(
        *, logprobs: Tensor, old_logprobs: Tensor, advantages: Tensor, response_mask: Tensor
    ) -> Tensor:
        budget = int(response_mask.shape[1]) if loss_type == "dr_grpo" else None
        return _trl_grpo_loss(
            logprobs,
            old_logprobs,
            advantages,
            response_mask,
            loss_type=loss_type,
            max_completion_length=budget,
        )

    return fn


def _verl_gspo(
    *, logprobs: Tensor, old_logprobs: Tensor, advantages: Tensor, response_mask: Tensor
) -> Tensor:
    """Wrapper over the verl GSPO reimplementation (pinned defaults only)."""
    return _verl_gspo_loss(logprobs, old_logprobs, advantages, response_mask)


def _verl_cispo(
    *, logprobs: Tensor, old_logprobs: Tensor, advantages: Tensor, response_mask: Tensor
) -> Tensor:
    """Wrapper over the verl CISPO reimplementation (pinned defaults only)."""
    return _verl_cispo_loss(logprobs, old_logprobs, advantages, response_mask)


VENDORED: dict[tuple[str, str], LossFn] = {
    ("verl", "pg_clip_token_mean"): _verl_pg_clip("token-mean"),
    ("verl", "pg_clip_seq_mean_token_mean"): _verl_pg_clip("seq-mean-token-mean"),
    ("verl", "pg_clip_seq_mean_token_sum"): _verl_pg_clip("seq-mean-token-sum"),
    ("verl", "pg_clip_seq_mean_token_sum_norm"): _verl_pg_clip("seq-mean-token-sum-norm"),
    ("verl", "gspo"): _verl_gspo,
    ("verl", "cispo"): _verl_cispo,
    ("openrlhf", "pg_clip_token_mean"): _openrlhf_pg_clip(token_level_loss=True),
    ("openrlhf", "pg_clip_seq_mean_token_mean"): _openrlhf_pg_clip(token_level_loss=False),
    ("trl", "grpo"): _trl_variant("grpo"),
    ("trl", "bnpo"): _trl_variant("bnpo"),
    ("trl", "dr_grpo"): _trl_variant("dr_grpo"),
}

# VENDORED keys whose callables are labeled reimplementations, not vendored code (the
# TRL loss reads trainer state; verl's GSPO/CISPO read a verl ActorConfig).
_REIMPLEMENTED: frozenset[tuple[str, str]] = frozenset(
    {
        ("trl", "grpo"),
        ("trl", "bnpo"),
        ("trl", "dr_grpo"),
        ("verl", "gspo"),
        ("verl", "cispo"),
    }
)


def _sample_case(
    shape: tuple[int, int], generator: torch.Generator, dtype: torch.dtype
) -> dict[str, Tensor]:
    """Draw one random comparison case in the harness input distribution.

    Bounds follow the shared test strategy bounds (tests/strategies.py): right-padded
    masks with at least one true token per row, ``logprobs`` uniform in
    ``[-8, -0.05]``, log-ratio gaps uniform in ``[-2, 2]`` (so ratios stay within
    ``e^±2`` and verl's ``±20`` log-ratio clamp never binds), per-token advantages
    uniform in ``[-3, 3]``. Positions outside the mask hold values from the same
    ranges, which doubles as an implicit mask-invariance stress on every wrapper.
    """
    b, t = shape
    lengths = torch.randint(1, t + 1, (b,), generator=generator)
    response_mask = torch.arange(t).unsqueeze(0) < lengths.unsqueeze(1)
    logprobs = torch.rand((b, t), generator=generator, dtype=dtype) * 7.95 - 8.0
    gap = torch.rand((b, t), generator=generator, dtype=dtype) * 4.0 - 2.0
    advantages = torch.rand((b, t), generator=generator, dtype=dtype) * 6.0 - 3.0
    return {
        "logprobs": logprobs,
        "old_logprobs": logprobs + gap,
        "advantages": advantages,
        "response_mask": response_mask,
    }


def _loss_and_grad(fn: LossFn, name: str, case: dict[str, Tensor]) -> tuple[float, Tensor]:
    """Evaluate a wrapper on one case; return the scalar loss and grad w.r.t. logprobs."""
    logprobs = case["logprobs"].detach().clone().requires_grad_(True)
    loss = fn(
        logprobs=logprobs,
        old_logprobs=case["old_logprobs"],
        advantages=case["advantages"],
        response_mask=case["response_mask"],
    )
    if not isinstance(loss, Tensor) or loss.dim() != 0:
        raise ValueError(f"{name} must return a scalar loss tensor; got {loss!r}")
    (grad,) = torch.autograd.grad(loss, logprobs)
    # detach: the gradient is compared as data; no higher-order path is needed.
    return float(loss.detach()), grad.detach()


def compare_losses(
    fn_a: LossFn,
    fn_b: LossFn,
    *,
    n_cases: int,
    shapes: Sequence[tuple[int, int]],
    generator: torch.Generator,
    dtype: torch.dtype = torch.float64,
) -> DeviationReport:
    """Compare two loss callables on seeded random inputs.

    For each of ``n_cases`` cases a fresh seed is drawn from ``generator``, inputs are
    sampled with :func:`_sample_case` over ``shapes`` (cycled in order), and both
    callables are evaluated with the keyword-tensor convention. The report collects the
    per-case metrics defined on :class:`DeviationReport`.

    Args:
        fn_a: First loss callable (keyword tensors → scalar loss).
        fn_b: Second loss callable.
        n_cases: Number of cases; must be ``>= 1``.
        shapes: Non-empty sequence of ``(B, T)`` shapes, each with ``B, T >= 1``.
        generator: Explicit RNG; equal seeds give bitwise-identical reports.
        dtype: Floating dtype of the sampled inputs.

    Returns:
        :class:`DeviationReport` with empty ``notes``.

    Raises:
        ValueError: If ``n_cases < 1``, ``shapes`` is empty or holds a non-positive
            dimension, ``dtype`` is not floating point, or a callable returns a
            non-scalar.

    References:
        docs/conventions.md (determinism rules);
        tests/test_conformance.py::test_compare_losses_zero_deviation_for_identical_fn,
        tests/test_conformance.py::test_compare_losses_detects_scale_deviation.
    """
    if n_cases < 1:
        raise ValueError(f"n_cases must be >= 1; got {n_cases}")
    shape_list: list[tuple[int, int]] = []
    for shape in shapes:
        if len(shape) != 2 or shape[0] < 1 or shape[1] < 1:
            raise ValueError(f"every shape must be (B, T) with B, T >= 1; got {shape}")
        shape_list.append((int(shape[0]), int(shape[1])))
    if not shape_list:
        raise ValueError("shapes must be a non-empty sequence of (B, T) pairs")
    if not dtype.is_floating_point:
        raise ValueError(f"dtype must be floating point; got {dtype}")

    max_loss_rel = 0.0
    max_grad_rel = 0.0
    cosine_min = 1.0
    worst_seed = 0
    worst_score = -1.0
    for index in range(n_cases):
        seed = int(torch.randint(0, 2**31 - 1, (1,), generator=generator).item())
        case_shape = shape_list[index % len(shape_list)]
        case = _sample_case(case_shape, torch.Generator().manual_seed(seed), dtype)
        loss_a, grad_a = _loss_and_grad(fn_a, "fn_a", case)
        loss_b, grad_b = _loss_and_grad(fn_b, "fn_b", case)

        loss_rel = abs(loss_a - loss_b) / max(abs(loss_a), abs(loss_b), _REL_FLOOR)
        norm_a = float(grad_a.norm())
        norm_b = float(grad_b.norm())
        grad_rel = float((grad_a - grad_b).norm()) / max(norm_a, norm_b, _REL_FLOOR)
        if norm_a <= _REL_FLOOR and norm_b <= _REL_FLOOR:
            cosine = 1.0
        elif norm_a <= _REL_FLOOR or norm_b <= _REL_FLOOR:
            cosine = 0.0
        else:
            cosine = float((grad_a * grad_b).sum()) / (norm_a * norm_b)

        max_loss_rel = max(max_loss_rel, loss_rel)
        max_grad_rel = max(max_grad_rel, grad_rel)
        cosine_min = min(cosine_min, cosine)
        score = max(loss_rel, grad_rel)
        if score > worst_score:
            worst_score = score
            worst_seed = seed

    return DeviationReport(
        max_loss_rel_diff=max_loss_rel,
        max_grad_rel_diff=max_grad_rel,
        grad_cosine_min=cosine_min,
        n_cases=n_cases,
        worst_case_seed=worst_seed,
        notes=(),
    )


def deviation_report(
    config: PolicyLossConfig,
    framework: str,
    variant: str,
    *,
    n_cases: int = 64,
    shapes: Sequence[tuple[int, int]] = ((4, 8),),
    generator: torch.Generator,
    dtype: torch.dtype = torch.float64,
) -> DeviationReport:
    """Compare a polgrad policy-loss config against a ``VENDORED`` framework loss.

    Runs ``compare_losses(polgrad, VENDORED[(framework, variant)], ...)`` where the
    polgrad side is ``policy_loss(config, ...).loss`` under the harness keyword-tensor
    convention, and attaches provenance notes to the report.

    Args:
        config: polgrad loss specification to evaluate.
        framework: Registered framework name (``"verl"``, ``"openrlhf"``, ``"trl"``).
        variant: Registered variant name (module docstring lists the keys).
        n_cases: Number of seeded comparison cases.
        shapes: ``(B, T)`` input shapes, cycled across cases.
        generator: Explicit RNG; equal seeds give bitwise-identical reports.
        dtype: Floating dtype of the sampled inputs.

    Returns:
        :class:`DeviationReport` with notes naming the config and framework variant
        (and the reimplementation caveat for the TRL and verl gspo/cispo entries).

    Raises:
        ValueError: If ``(framework, variant)`` is not in ``VENDORED``, or propagated
            from :func:`compare_losses` and :func:`~polgrad.losses.policy_loss`.

    References:
        docs/conventions.md (determinism rules);
        tests/test_conformance.py::test_deviation_report_verl_token_mean_matches_polgrad,
        tests/test_conformance.py::test_deviation_report_unknown_key_raises_value_error.
    """
    key = (framework, variant)
    if key not in VENDORED:
        known = ", ".join(f"{fw}/{var}" for fw, var in sorted(VENDORED))
        raise ValueError(f"unknown VENDORED entry {framework}/{variant}; registered: {known}")

    def polgrad_fn(
        *, logprobs: Tensor, old_logprobs: Tensor, advantages: Tensor, response_mask: Tensor
    ) -> Tensor:
        return policy_loss(
            config,
            logprobs=logprobs,
            old_logprobs=old_logprobs,
            advantages=advantages,
            response_mask=response_mask,
        ).loss

    report = compare_losses(
        polgrad_fn, VENDORED[key], n_cases=n_cases, shapes=shapes, generator=generator, dtype=dtype
    )
    notes: tuple[str, ...] = (
        f"polgrad: surrogate={config.surrogate.value}, ratio={config.ratio.value}, "
        f"aggregation={config.aggregation.value}, clip={config.clip}",
        f"framework: {framework}/{variant}",
    )
    if framework == "trl":
        notes = (
            *notes,
            "TRL entry is a faithful reimplementation (trl v1.8.0), not vendored code.",
        )
    elif key in _REIMPLEMENTED:
        notes = (
            *notes,
            f"verl {variant} entry is a faithful reimplementation "
            f"(commit 74a718a4), not vendored code.",
        )
    return replace(report, notes=notes)
