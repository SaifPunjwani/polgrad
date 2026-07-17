"""Enforces the validation rules of docs/conventions.md (shapes, mask dtype, row
coverage, finiteness)."""

from __future__ import annotations

import pytest
import torch

from polgrad._validation import check_2d, check_finite, check_mask, check_same_shape


def test_check_2d_rejects_wrong_rank() -> None:
    with pytest.raises(ValueError, match=r"must be 2-D \[B, T\]; got shape \(3,\)"):
        check_2d("logprobs", torch.zeros(3))


def test_check_same_shape_names_both_arguments() -> None:
    with pytest.raises(ValueError, match=r"logprobs and old_logprobs .* \(2, 3\) vs \(2, 4\)"):
        check_same_shape("logprobs", torch.zeros(2, 3), "old_logprobs", torch.zeros(2, 4))


def test_check_mask_rejects_non_bool() -> None:
    with pytest.raises(ValueError, match=r"dtype torch\.bool"):
        check_mask(torch.ones(2, 3), like=torch.zeros(2, 3))


def test_check_mask_rejects_empty_row() -> None:
    mask = torch.tensor([[True, True], [False, False]])
    with pytest.raises(ValueError, match=r"zero response tokens: rows \[1\]"):
        check_mask(mask, like=torch.zeros(2, 2))


def test_check_mask_accepts_valid_ragged_mask() -> None:
    mask = torch.tensor([[True, False], [True, True]])
    check_mask(mask, like=torch.zeros(2, 2))


def test_check_finite_rejects_nan_and_inf() -> None:
    for bad in (float("nan"), float("inf")):
        with pytest.raises(ValueError, match="non-finite"):
            check_finite("advantages", torch.tensor([[0.0, bad]]))
