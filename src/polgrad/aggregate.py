"""Aggregation semantics: how a per-token RL objective becomes a scalar loss.

Every mode is expressible as a linear functional ``L = Σ_{b,t} w_{b,t} · x_{b,t}`` of the
per-token values; :func:`effective_token_weights` returns the closed-form weights ``w``
and :func:`aggregate` is exactly that inner product. The per-mode weight derivations,
equal-length collapse theorems, and the micro-batch (gradient-accumulation) weight
algebra live in ``docs/derivations/aggregation.md``.
"""

from __future__ import annotations

import enum
from collections.abc import Sequence
from typing import Literal

import torch
from torch import Tensor

from polgrad._validation import check_2d, check_mask

__all__ = [
    "Aggregation",
    "aggregate",
    "effective_token_weights",
    "microbatch_token_weights",
]


class Aggregation(enum.Enum):
    """Reduction of a masked per-token objective ``x`` to a scalar loss.

    With mask ``m ∈ {0, 1}^{B, T}``, row token counts ``L_b = Σ_t m_{b,t}``, and total
    token count ``N = Σ_{b,t} m_{b,t}``:

    Attributes:
        TOKEN_MEAN: ``Σ_{b,t} m·x / N`` — verl ``"token-mean"``; TRL adopted this
            global token-level normalization in v0.16.0 (PR #2881; see issue #2995 for
            the ensuing debate — resolved for KL logging only in PR #3004). Current TRL
            exposes it as ``loss_type="bnpo"`` (local batch); TRL's present default is
            ``loss_type="dapo"``, i.e. TOKEN_MEAN computed over the global
            gradient-accumulated batch (see the micro-batch weight algebra in
            docs/derivations/aggregation.md).
        SEQ_MEAN_TOKEN_MEAN: ``mean_b( Σ_t m·x / L_b )`` — the GRPO paper equation; TRL
            GRPOTrainer default in v0.14-v0.15, changed in v0.16.0.
        SEQ_MEAN_TOKEN_SUM: ``mean_b( Σ_t m·x )``.
        TOKEN_SUM_NORM: ``Σ_{b,t} m·x / (B · norm_len)`` — Dr.GRPO, where ``norm_len``
            is the fixed generation budget.

    References:
        docs/derivations/aggregation.md;
        tests/test_aggregate.py::test_aggregate_golden_values.
    """

    TOKEN_MEAN = "token_mean"
    SEQ_MEAN_TOKEN_MEAN = "seq_mean_token_mean"
    SEQ_MEAN_TOKEN_SUM = "seq_mean_token_sum"
    TOKEN_SUM_NORM = "token_sum_norm"


def effective_token_weights(
    response_mask: Tensor, mode: Aggregation, *, norm_len: int | None = None
) -> Tensor:
    """Return the weight each token carries in the aggregated loss.

    ``aggregate(x, m, mode) = Σ_{b,t} w_{b,t} · x_{b,t}`` with the closed forms

    - ``TOKEN_MEAN``:          ``w = m / N``
    - ``SEQ_MEAN_TOKEN_MEAN``: ``w = m / (B · L_b)``
    - ``SEQ_MEAN_TOKEN_SUM``:  ``w = m / B``
    - ``TOKEN_SUM_NORM``:      ``w = m / (B · norm_len)``

    derived in docs/derivations/aggregation.md. Weights are returned in
    ``torch.float64`` (documented choice: the mask carries no floating dtype to
    preserve, and the aggregate/weights identity is asserted at full precision);
    :func:`aggregate` casts them to the per-token dtype.

    Args:
        response_mask: ``[B, T]`` bool mask of real response tokens; every row must
            contain at least one true token.
        mode: Aggregation mode.
        norm_len: Fixed generation budget; required iff ``mode`` is
            ``Aggregation.TOKEN_SUM_NORM`` and ignored otherwise, so callers may pass a
            shared config value (the requirement is enforced at call time, so configs
            may carry ``None``; see docs/derivations/aggregation.md).

    Returns:
        ``[B, T]`` float64 weights, exactly ``0`` at masked positions.

    Raises:
        ValueError: If the mask is invalid, or if ``norm_len`` is ``None`` or
            non-positive while ``mode`` is ``TOKEN_SUM_NORM``.

    References:
        docs/derivations/aggregation.md;
        tests/test_aggregate.py::test_aggregate_equals_effective_weights_inner_product,
        tests/test_aggregate.py::test_effective_weights_golden_closed_forms.
    """
    check_mask(response_mask, like=response_mask)
    m = response_mask.to(torch.float64)
    batch = float(response_mask.shape[0])
    if mode is Aggregation.TOKEN_MEAN:
        return m / m.sum()
    if mode is Aggregation.SEQ_MEAN_TOKEN_MEAN:
        return m / (batch * m.sum(dim=1, keepdim=True))
    if mode is Aggregation.SEQ_MEAN_TOKEN_SUM:
        return m / batch
    if mode is Aggregation.TOKEN_SUM_NORM:
        if norm_len is None:
            raise ValueError(
                "norm_len is required when mode is Aggregation.TOKEN_SUM_NORM; got None"
            )
        if norm_len < 1:
            raise ValueError(f"norm_len must be a positive int; got {norm_len}")
        return m / (batch * float(norm_len))
    raise ValueError(f"unknown Aggregation mode: {mode!r}")


def aggregate(
    per_token: Tensor,
    response_mask: Tensor,
    mode: Aggregation,
    *,
    norm_len: int | None = None,
) -> Tensor:
    """Reduce a per-token objective to a scalar loss under the given aggregation mode.

    Computes ``L = Σ_{b,t} w_{b,t} · x_{b,t}`` where
    ``w = effective_token_weights(response_mask, mode, norm_len=norm_len)``. Masked
    positions are zeroed before the weighted sum, so their values reach neither the
    output nor the gradient, and ``∂L/∂x_{b,t} = w_{b,t}`` exactly.

    Args:
        per_token: ``[B, T]`` per-token objective values.
        response_mask: ``[B, T]`` bool mask of real response tokens.
        mode: Aggregation mode.
        norm_len: Fixed generation budget; required iff ``mode`` is
            ``Aggregation.TOKEN_SUM_NORM`` and ignored otherwise.

    Returns:
        Scalar tensor in the dtype of ``per_token``, differentiable w.r.t. ``per_token``.

    Raises:
        ValueError: If shapes or the mask are invalid, or if ``norm_len`` is missing or
            non-positive while ``mode`` is ``TOKEN_SUM_NORM``.

    References:
        docs/derivations/aggregation.md;
        tests/test_aggregate.py::test_aggregate_equals_effective_weights_inner_product,
        tests/test_aggregate.py::test_aggregate_gradient_equals_effective_weights.
    """
    check_2d("per_token", per_token)
    check_mask(response_mask, like=per_token)
    weights = effective_token_weights(response_mask, mode, norm_len=norm_len).to(per_token.dtype)
    zero = torch.zeros((), dtype=per_token.dtype, device=per_token.device)
    masked = torch.where(response_mask, per_token, zero)
    return (weights * masked).sum()


def microbatch_token_weights(
    response_mask: Tensor,
    mode: Aggregation,
    microbatch_sizes: Sequence[int],
    *,
    norm_len: int | None = None,
    loss_scale: Literal["mean", "sum"] = "mean",
) -> Tensor:
    """Effective per-token weights when the batch is aggregated in micro-batches.

    The batch is split into consecutive micro-batches of the given row counts, each
    aggregated with ``mode``, and the micro-batch losses are combined by averaging
    (``loss_scale="mean"``, the gradient-accumulation/DDP default) or summing
    (``loss_scale="sum"``). Writing ``w^c`` for the weights of chunk ``c`` under
    ``mode``, a token in chunk ``c`` carries weight ``w^c / K`` (mean over ``K``
    chunks) or ``w^c`` (sum). This is the closed form of gradient-accumulation
    inequivalence: for ``TOKEN_MEAN`` the result differs from the full-batch weights
    unless every micro-batch holds the same token count (docs/derivations/aggregation.md
    derives the per-mode equivalence conditions).

    Args:
        response_mask: ``[B, T]`` bool mask of real response tokens.
        mode: Aggregation mode applied inside each micro-batch.
        microbatch_sizes: Consecutive micro-batch row counts; entries must be positive
            and sum to ``B``.
        norm_len: Fixed generation budget; required iff ``mode`` is
            ``Aggregation.TOKEN_SUM_NORM`` and ignored otherwise.
        loss_scale: ``"mean"`` averages micro-batch losses; ``"sum"`` adds them.

    Returns:
        ``[B, T]`` float64 weights, exactly ``0`` at masked positions.

    Raises:
        ValueError: If the mask is invalid, ``microbatch_sizes`` has a non-positive
            entry or does not sum to ``B``, ``loss_scale`` is unknown, or ``norm_len``
            is missing or non-positive while ``mode`` is ``TOKEN_SUM_NORM``.

    References:
        docs/derivations/aggregation.md;
        tests/test_aggregate.py::test_microbatch_weights_match_simulated_loop_autograd,
        tests/test_aggregate.py::test_token_mean_microbatch_mean_deviates_for_unequal_token_counts.
    """
    check_mask(response_mask, like=response_mask)
    sizes = tuple(microbatch_sizes)
    if any(size < 1 for size in sizes):
        raise ValueError(f"microbatch_sizes entries must be >= 1; got {sizes}")
    batch = response_mask.shape[0]
    if sum(sizes) != batch:
        raise ValueError(
            f"microbatch_sizes must sum to the batch size {batch}; "
            f"got {sizes} summing to {sum(sizes)}"
        )
    if loss_scale not in ("mean", "sum"):
        raise ValueError(f"loss_scale must be 'mean' or 'sum'; got {loss_scale!r}")
    chunks: list[Tensor] = []
    start = 0
    for size in sizes:
        chunk_mask = response_mask[start : start + size]
        chunks.append(effective_token_weights(chunk_mask, mode, norm_len=norm_len))
        start += size
    weights = torch.cat(chunks, dim=0)
    if loss_scale == "mean":
        weights = weights / len(sizes)
    return weights
