"""Input validation and tiny shared numeric helpers used across polgrad.

Single source of error-message style. See ``docs/conventions.md`` for the shape and
masking rules these helpers enforce. Logprob tensors are validated for finiteness only:
real frameworks emit slightly positive logprob values from numerics, so rejecting
``> 0`` would reject real data. The per-site message parameters of
:func:`broadcast_advantages` exist because callers' error strings are pinned by tests;
the wording differences are preserved deliberately, not drift.
"""

from __future__ import annotations

import torch
from torch import Tensor

__all__ = [
    "broadcast_advantages",
    "check_1d",
    "check_2d",
    "check_finite",
    "check_logprob_streams",
    "check_mask",
    "check_same_shape",
    "std_or_zero",
]


def check_1d(name: str, x: Tensor) -> None:
    """Raise ``ValueError`` unless ``x`` is 1-D ``[B]``."""
    if x.dim() != 1:
        raise ValueError(f"{name} must be 1-D [B]; got shape {tuple(x.shape)}")


def check_2d(name: str, x: Tensor) -> None:
    """Raise ``ValueError`` unless ``x`` is 2-D ``[B, T]``."""
    if x.dim() != 2:
        raise ValueError(f"{name} must be 2-D [B, T]; got shape {tuple(x.shape)}")


def check_same_shape(name_a: str, a: Tensor, name_b: str, b: Tensor) -> None:
    """Raise ``ValueError`` unless ``a`` and ``b`` have identical shapes."""
    if a.shape != b.shape:
        raise ValueError(
            f"{name_a} and {name_b} must have identical shapes; "
            f"got {tuple(a.shape)} vs {tuple(b.shape)}"
        )


def check_finite(name: str, x: Tensor) -> None:
    """Raise ``ValueError`` if ``x`` contains NaN or infinite values."""
    if not bool(torch.isfinite(x).all()):
        raise ValueError(f"{name} contains non-finite values")


def check_mask(response_mask: Tensor, *, like: Tensor) -> None:
    """Validate a response mask against a reference tensor.

    Enforces the masking rules of ``docs/conventions.md``: ``response_mask`` must be a
    2-D boolean tensor with the same shape as ``like``, and every row must contain at
    least one response token.

    Raises:
        ValueError: On dtype, dimensionality, or shape mismatch, or if any row of the
            mask has zero response tokens.
    """
    if response_mask.dtype != torch.bool:
        raise ValueError(f"response_mask must have dtype torch.bool; got {response_mask.dtype}")
    check_2d("response_mask", response_mask)
    if response_mask.shape != like.shape:
        raise ValueError(
            f"response_mask shape {tuple(response_mask.shape)} does not match "
            f"input shape {tuple(like.shape)}"
        )
    row_counts = response_mask.sum(dim=1)
    if int(row_counts.min()) == 0:
        empty = torch.nonzero(row_counts == 0).flatten().tolist()
        raise ValueError(f"response_mask has rows with zero response tokens: rows {empty}")


def check_logprob_streams(
    name_a: str,
    a: Tensor,
    name_b: str,
    b: Tensor,
    response_mask: Tensor,
    *,
    check_b_2d: bool = False,
    finite_suffix: str = "",
) -> None:
    """Shared shape/mask/finiteness validation for a pair of logprob streams.

    Finiteness is checked only at response positions: masked padding never affects any
    output, so non-finite junk there must not raise (mask invariance extends to
    validation). ``check_b_2d`` additionally names ``b`` in its own rank error before
    the shape comparison; ``finite_suffix`` (e.g. ``" (response positions)"``) is
    appended to the finiteness labels. Both parameters preserve the callers' historical
    error strings.
    """
    check_2d(name_a, a)
    if check_b_2d:
        check_2d(name_b, b)
    check_same_shape(name_a, a, name_b, b)
    check_mask(response_mask, like=a)
    check_finite(name_a + finite_suffix, a[response_mask])
    check_finite(name_b + finite_suffix, b[response_mask])


def broadcast_advantages(
    advantages: Tensor,
    like: Tensor,
    response_mask: Tensor,
    *,
    like_name: str,
    zero_masked: bool = False,
    finite_label_2d: str = "advantages (response positions)",
    batch_mismatch_template: str = "advantages [B] must have B = {b} rows; got shape {adv_shape}",
    shape_mismatch_template: str | None = None,
) -> Tensor:
    """Validate ``[B]`` or ``[B, T]`` advantages and return them as ``[B, T]``.

    A ``[B]`` input is expanded across its row's tokens; a ``[B, T]`` input must match
    ``like``'s shape and be finite at response positions. With ``zero_masked=True`` the
    result is exactly 0 at masked positions, so padded advantage junk reaches neither
    forward values nor backward formulas (mask invariance); with ``zero_masked=False``
    the raw broadcast view is returned for callers that intersect with the mask
    themselves. The message templates take ``{b}``, ``{adv_shape}``, and ``{like_shape}``
    placeholders; ``shape_mismatch_template=None`` uses :func:`check_same_shape` with
    ``like_name``.
    """
    if advantages.dim() == 1:
        if advantages.shape[0] != like.shape[0]:
            raise ValueError(
                batch_mismatch_template.format(b=like.shape[0], adv_shape=tuple(advantages.shape))
            )
        check_finite("advantages", advantages)
        expanded = advantages.unsqueeze(1).expand_as(like)
    elif advantages.dim() == 2:
        if shape_mismatch_template is None:
            check_same_shape("advantages", advantages, like_name, like)
        elif advantages.shape != like.shape:
            raise ValueError(
                shape_mismatch_template.format(
                    like_shape=tuple(like.shape), adv_shape=tuple(advantages.shape)
                )
            )
        check_finite(finite_label_2d, advantages[response_mask])
        expanded = advantages
    else:
        raise ValueError(f"advantages must be [B] or [B, T]; got shape {tuple(advantages.shape)}")
    if not zero_masked:
        return expanded
    zero = torch.zeros((), dtype=expanded.dtype, device=expanded.device)
    return torch.where(response_mask, expanded, zero)


def std_or_zero(values: Tensor) -> float:
    """Bessel-corrected std; a single observation has no spread estimate, reported as 0.0."""
    if values.numel() < 2:
        return 0.0
    return float(values.std())
