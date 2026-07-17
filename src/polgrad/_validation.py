"""Input validation shared across polgrad.

Single source of error-message style. See ``docs/conventions.md`` for the shape and
masking rules these helpers enforce. Logprob tensors are validated for finiteness only:
real frameworks emit slightly positive logprob values from numerics, so rejecting
``> 0`` would reject real data.
"""

from __future__ import annotations

import torch
from torch import Tensor

__all__ = [
    "check_1d",
    "check_2d",
    "check_finite",
    "check_mask",
    "check_same_shape",
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
