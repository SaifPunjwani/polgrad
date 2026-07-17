"""Enforces the aggregation semantics of docs/derivations/aggregation.md: the
aggregate/effective-weights identity, the autograd cross-check, the equal-length
collapse theorems, the micro-batch weight algebra, and mask invariance."""

from __future__ import annotations

import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st
from strategies import MASKED_JUNK, padded_masks
from torch.testing import assert_close

from polgrad.aggregate import (
    Aggregation,
    aggregate,
    effective_token_weights,
    microbatch_token_weights,
)

MODES = tuple(Aggregation)
NORM_LEN = 7


def _norm_len(mode: Aggregation) -> int | None:
    return NORM_LEN if mode is Aggregation.TOKEN_SUM_NORM else None


@st.composite
def masked_values(draw: st.DrawFn, *, min_b: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    """[B, T] float64 values with MASKED_JUNK at padded positions, plus the mask."""
    mask = draw(padded_masks(min_b=min_b))
    b, t = mask.shape
    vals = [
        draw(st.floats(-5.0, 5.0, allow_nan=False, allow_infinity=False, width=32))
        for _ in range(b * t)
    ]
    x = torch.tensor(vals, dtype=torch.float64).reshape(b, t)
    return torch.where(mask, x, torch.full_like(x, MASKED_JUNK)), mask


@st.composite
def masked_values_with_partition(
    draw: st.DrawFn,
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, ...]]:
    """Masked values plus a drawn partition of the batch into micro-batch sizes."""
    x, mask = draw(masked_values(min_b=2))
    sizes: list[int] = []
    remaining = mask.shape[0]
    while remaining > 0:
        size = draw(st.integers(1, remaining))
        sizes.append(size)
        remaining -= size
    return x, mask, tuple(sizes)


@st.composite
def equal_length_values(draw: st.DrawFn) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Full [B, L] batches where every row has exactly L response tokens."""
    b = draw(st.integers(1, 6))
    length = draw(st.integers(1, 8))
    vals = [
        draw(st.floats(-5.0, 5.0, allow_nan=False, allow_infinity=False, width=32))
        for _ in range(b * length)
    ]
    x = torch.tensor(vals, dtype=torch.float64).reshape(b, length)
    return x, torch.ones((b, length), dtype=torch.bool), length


@pytest.mark.parametrize("mode", MODES)
@given(data=masked_values())
def test_aggregate_equals_effective_weights_inner_product(
    mode: Aggregation, data: tuple[torch.Tensor, torch.Tensor]
) -> None:
    """Core identity of docs/derivations/aggregation.md: aggregate(x, m, mode) equals
    (effective_token_weights(m, mode) * x).sum() bitwise on ragged masks."""
    x, mask = data
    norm_len = _norm_len(mode)
    weights = effective_token_weights(mask, mode, norm_len=norm_len)
    assert torch.equal(aggregate(x, mask, mode, norm_len=norm_len), (weights * x).sum())


@pytest.mark.parametrize("mode", MODES)
@given(data=masked_values())
def test_aggregate_gradient_equals_effective_weights(
    mode: Aggregation, data: tuple[torch.Tensor, torch.Tensor]
) -> None:
    """Autograd cross-check of the same identity: d aggregate / d per_token equals
    effective_token_weights exactly (docs/derivations/aggregation.md)."""
    x, mask = data
    norm_len = _norm_len(mode)
    leaf = x.clone().requires_grad_(True)
    (grad,) = torch.autograd.grad(aggregate(leaf, mask, mode, norm_len=norm_len), leaf)
    assert torch.equal(grad, effective_token_weights(mask, mode, norm_len=norm_len))


def _scalar(v: float) -> torch.Tensor:
    return torch.tensor(v, dtype=torch.float64)


def test_aggregate_golden_values() -> None:
    """Hand-derived aggregate values for every mode on a 2x2 ragged batch
    (docs/derivations/aggregation.md, closed forms)."""
    x = torch.tensor([[1.0, 2.0], [3.0, MASKED_JUNK]], dtype=torch.float64)
    mask = torch.tensor([[True, True], [True, False]])
    scalar = _scalar
    # TOKEN_MEAN: N = 3 response tokens, (1 + 2 + 3) / 3 = 2.0
    assert_close(aggregate(x, mask, Aggregation.TOKEN_MEAN), scalar(2.0), rtol=1e-14, atol=0.0)
    # SEQ_MEAN_TOKEN_MEAN: row means (1 + 2)/2 = 1.5 and 3/1 = 3.0; (1.5 + 3.0)/2 = 2.25
    assert_close(
        aggregate(x, mask, Aggregation.SEQ_MEAN_TOKEN_MEAN), scalar(2.25), rtol=1e-14, atol=0.0
    )
    # SEQ_MEAN_TOKEN_SUM: row sums 3 and 3; (3 + 3)/2 = 3.0
    assert_close(
        aggregate(x, mask, Aggregation.SEQ_MEAN_TOKEN_SUM), scalar(3.0), rtol=1e-14, atol=0.0
    )
    # TOKEN_SUM_NORM, norm_len = 4: (1 + 2 + 3) / (B * norm_len) = 6 / (2 * 4) = 0.75
    assert_close(
        aggregate(x, mask, Aggregation.TOKEN_SUM_NORM, norm_len=4),
        scalar(0.75),
        rtol=1e-14,
        atol=0.0,
    )


def test_effective_weights_golden_closed_forms() -> None:
    """Closed-form weights per mode on a 2x2 ragged mask, hand-derived in
    docs/derivations/aggregation.md (lengths L = [2, 1], B = 2, N = 3)."""
    mask = torch.tensor([[True, True], [True, False]])
    ones = torch.tensor([[1.0, 1.0], [1.0, 0.0]], dtype=torch.float64)
    # TOKEN_MEAN: m / N = m / 3
    assert torch.equal(effective_token_weights(mask, Aggregation.TOKEN_MEAN), ones / 3.0)
    # SEQ_MEAN_TOKEN_MEAN: m / (B * L_b) -> row 0: 1/(2*2) = 0.25, row 1: 1/(2*1) = 0.5
    assert torch.equal(
        effective_token_weights(mask, Aggregation.SEQ_MEAN_TOKEN_MEAN),
        torch.tensor([[0.25, 0.25], [0.5, 0.0]], dtype=torch.float64),
    )
    # SEQ_MEAN_TOKEN_SUM: m / B = m / 2
    assert torch.equal(
        effective_token_weights(mask, Aggregation.SEQ_MEAN_TOKEN_SUM),
        torch.tensor([[0.5, 0.5], [0.5, 0.0]], dtype=torch.float64),
    )
    # TOKEN_SUM_NORM, norm_len = 4: m / (B * norm_len) = m / 8
    assert torch.equal(
        effective_token_weights(mask, Aggregation.TOKEN_SUM_NORM, norm_len=4),
        torch.tensor([[0.125, 0.125], [0.125, 0.0]], dtype=torch.float64),
    )


@pytest.mark.parametrize("mode", MODES)
@given(mask=padded_masks())
def test_effective_weights_zero_at_masked_positions(mode: Aggregation, mask: torch.Tensor) -> None:
    """Masked-position outputs are exactly 0 (docs/conventions.md)."""
    weights = effective_token_weights(mask, mode, norm_len=_norm_len(mode))
    assert weights.dtype == torch.float64
    assert torch.equal(weights[~mask], torch.zeros_like(weights[~mask]))


@pytest.mark.parametrize("mode", [Aggregation.TOKEN_MEAN, Aggregation.SEQ_MEAN_TOKEN_MEAN])
@given(mask=padded_masks())
def test_effective_weights_of_mean_modes_sum_to_one(mode: Aggregation, mask: torch.Tensor) -> None:
    """TOKEN_MEAN and SEQ_MEAN_TOKEN_MEAN weights are convex: they sum to 1
    (docs/derivations/aggregation.md, weight normalization)."""
    total = effective_token_weights(mask, mode).sum()
    assert_close(total, torch.tensor(1.0, dtype=torch.float64), rtol=1e-12, atol=0.0)


@given(data=equal_length_values())
def test_equal_length_token_mean_collapses_to_seq_mean_token_mean(
    data: tuple[torch.Tensor, torch.Tensor, int],
) -> None:
    """Equal-length collapse: all rows of length L imply N = B*L, so TOKEN_MEAN ==
    SEQ_MEAN_TOKEN_MEAN, bitwise (docs/derivations/aggregation.md)."""
    x, mask, _ = data
    assert torch.equal(
        effective_token_weights(mask, Aggregation.TOKEN_MEAN),
        effective_token_weights(mask, Aggregation.SEQ_MEAN_TOKEN_MEAN),
    )
    assert torch.equal(
        aggregate(x, mask, Aggregation.TOKEN_MEAN),
        aggregate(x, mask, Aggregation.SEQ_MEAN_TOKEN_MEAN),
    )


@given(data=equal_length_values())
def test_equal_length_seq_mean_token_sum_is_length_times_token_mean(
    data: tuple[torch.Tensor, torch.Tensor, int],
) -> None:
    """Equal-length collapse with exact constant L: SEQ_MEAN_TOKEN_SUM == L * TOKEN_MEAN
    (docs/derivations/aggregation.md)."""
    x, mask, length = data
    assert_close(
        aggregate(x, mask, Aggregation.SEQ_MEAN_TOKEN_SUM),
        length * aggregate(x, mask, Aggregation.TOKEN_MEAN),
        rtol=1e-12,
        atol=1e-12,
    )


@given(data=equal_length_values(), norm_len=st.integers(1, 9))
def test_equal_length_token_sum_norm_is_length_over_norm_len_times_token_mean(
    data: tuple[torch.Tensor, torch.Tensor, int], norm_len: int
) -> None:
    """Equal-length collapse with exact constant L/norm_len: TOKEN_SUM_NORM ==
    (L/norm_len) * TOKEN_MEAN (docs/derivations/aggregation.md)."""
    x, mask, length = data
    assert_close(
        aggregate(x, mask, Aggregation.TOKEN_SUM_NORM, norm_len=norm_len),
        (length / norm_len) * aggregate(x, mask, Aggregation.TOKEN_MEAN),
        rtol=1e-12,
        atol=1e-12,
    )


def _simulated_microbatch_grad(
    x: torch.Tensor,
    mask: torch.Tensor,
    mode: Aggregation,
    sizes: tuple[int, ...],
    norm_len: int | None,
    loss_scale: str,
) -> torch.Tensor:
    """Explicit micro-batch loop: aggregate each chunk, combine, differentiate."""
    leaf = x.clone().requires_grad_(True)
    losses = []
    start = 0
    for size in sizes:
        losses.append(
            aggregate(
                leaf[start : start + size], mask[start : start + size], mode, norm_len=norm_len
            )
        )
        start += size
    stacked = torch.stack(losses)
    total = stacked.mean() if loss_scale == "mean" else stacked.sum()
    (grad,) = torch.autograd.grad(total, leaf)
    return grad


@pytest.mark.parametrize("mode", MODES)
@given(data=masked_values_with_partition())
def test_microbatch_weights_match_simulated_loop_autograd(
    mode: Aggregation, data: tuple[torch.Tensor, torch.Tensor, tuple[int, ...]]
) -> None:
    """microbatch_token_weights (mean combine) equals the autograd gradient of an
    explicit simulated micro-batch loop (docs/derivations/aggregation.md, micro-batch
    algebra; cross-module obligation 4, tests/test_cross.py)."""
    x, mask, sizes = data
    norm_len = _norm_len(mode)
    weights = microbatch_token_weights(mask, mode, sizes, norm_len=norm_len, loss_scale="mean")
    grad = _simulated_microbatch_grad(x, mask, mode, sizes, norm_len, "mean")
    assert_close(weights, grad, rtol=1e-15, atol=0.0)


@pytest.mark.parametrize("mode", MODES)
@given(data=masked_values_with_partition())
def test_microbatch_weights_sum_scale_match_simulated_loop_bitwise(
    mode: Aggregation, data: tuple[torch.Tensor, torch.Tensor, tuple[int, ...]]
) -> None:
    """With loss_scale="sum" the micro-batch weights equal the simulated-loop autograd
    gradient bitwise (docs/derivations/aggregation.md, micro-batch algebra)."""
    x, mask, sizes = data
    norm_len = _norm_len(mode)
    weights = microbatch_token_weights(mask, mode, sizes, norm_len=norm_len, loss_scale="sum")
    grad = _simulated_microbatch_grad(x, mask, mode, sizes, norm_len, "sum")
    assert torch.equal(weights, grad)


def test_microbatch_weights_golden_values() -> None:
    """Hand-derived micro-batch weights: lengths [2, 1], sizes [1, 1], TOKEN_MEAN.
    Chunk 0 has N_1 = 2 tokens -> weight 1/2 each; chunk 1 has N_2 = 1 -> weight 1.
    Mean combine divides by K = 2 (docs/derivations/aggregation.md)."""
    mask = torch.tensor([[True, True], [True, False]])
    mean_weights = microbatch_token_weights(mask, Aggregation.TOKEN_MEAN, [1, 1])
    # mean: [[1/2/2, 1/2/2], [1/2, 0]] = [[0.25, 0.25], [0.5, 0.0]]
    assert torch.equal(mean_weights, torch.tensor([[0.25, 0.25], [0.5, 0.0]], dtype=torch.float64))
    sum_weights = microbatch_token_weights(mask, Aggregation.TOKEN_MEAN, [1, 1], loss_scale="sum")
    # sum: [[1/2, 1/2], [1, 0]]
    assert torch.equal(sum_weights, torch.tensor([[0.5, 0.5], [1.0, 0.0]], dtype=torch.float64))


def test_token_mean_microbatch_mean_deviates_for_unequal_token_counts() -> None:
    """Gradient-accumulation inequivalence (docs/derivations/aggregation.md): TOKEN_MEAN
    micro-batched with unequal chunk token counts differs from the full batch. Full
    batch: every token weighs 1/4. Chunks of N_1 = 1 and N_2 = 3 tokens give weights
    1/(1*2) = 1/2 and 1/(3*2) = 1/6."""
    mask = torch.tensor([[True, False, False], [True, True, True]])
    micro = microbatch_token_weights(mask, Aggregation.TOKEN_MEAN, [1, 1])
    expected = torch.tensor([[0.5, 0.0, 0.0], [1 / 6, 1 / 6, 1 / 6]], dtype=torch.float64)
    assert_close(micro, expected, rtol=1e-15, atol=0.0)
    full = effective_token_weights(mask, Aggregation.TOKEN_MEAN)
    assert not torch.equal(micro, full)


def test_token_mean_microbatch_mean_matches_full_batch_for_equal_token_counts() -> None:
    """TOKEN_MEAN micro-batching is exact when every chunk holds the same token count:
    chunks of N_c = 4 tokens with K = 2 give 1/(4*2) = 1/8 = full-batch 1/N with N = 8
    (docs/derivations/aggregation.md)."""
    mask = torch.ones((4, 2), dtype=torch.bool)
    micro = microbatch_token_weights(mask, Aggregation.TOKEN_MEAN, [2, 2])
    full = effective_token_weights(mask, Aggregation.TOKEN_MEAN)
    assert torch.equal(micro, full)


@pytest.mark.parametrize(
    "mode",
    [Aggregation.SEQ_MEAN_TOKEN_MEAN, Aggregation.SEQ_MEAN_TOKEN_SUM, Aggregation.TOKEN_SUM_NORM],
)
def test_per_sequence_modes_microbatch_mean_matches_full_batch_for_equal_row_counts(
    mode: Aggregation,
) -> None:
    """For the per-sequence-denominator modes, micro-batching with equal chunk ROW
    counts is exact even on ragged batches: w = m/(B_c * d_b) per chunk, averaged over
    K chunks, equals m/(B * d_b) when K * B_c = B (docs/derivations/aggregation.md)."""
    mask = torch.tensor(
        [
            [True, False, False],
            [True, True, True],
            [True, True, False],
            [True, True, True],
        ]
    )
    norm_len = _norm_len(mode)
    micro = microbatch_token_weights(mask, mode, [2, 2], norm_len=norm_len)
    full = effective_token_weights(mask, mode, norm_len=norm_len)
    assert torch.equal(micro, full)


@pytest.mark.parametrize("mode", MODES)
@given(data=masked_values())
def test_aggregate_mask_invariance(
    mode: Aggregation, data: tuple[torch.Tensor, torch.Tensor]
) -> None:
    """Perturbing per-token values at masked positions leaves aggregate bitwise
    unchanged (docs/conventions.md, masked positions)."""
    x, mask = data
    norm_len = _norm_len(mode)
    perturbed = torch.where(mask, x, x - 57.25)
    assert torch.equal(
        aggregate(x, mask, mode, norm_len=norm_len),
        aggregate(perturbed, mask, mode, norm_len=norm_len),
    )


@pytest.mark.parametrize("mode", MODES)
@given(mask=padded_masks(min_b=2))
def test_microbatch_weights_zero_at_masked_positions(mode: Aggregation, mask: torch.Tensor) -> None:
    """Micro-batch weights are exactly 0 at masked positions (docs/conventions.md)."""
    sizes = [1, mask.shape[0] - 1]
    weights = microbatch_token_weights(mask, mode, sizes, norm_len=_norm_len(mode))
    assert torch.equal(weights[~mask], torch.zeros_like(weights[~mask]))


def test_aggregate_preserves_input_dtype() -> None:
    """aggregate returns the per-token dtype with no silent casts
    (docs/conventions.md, dtypes)."""
    mask = torch.tensor([[True, True], [True, False]])
    for dtype in (torch.float32, torch.float64):
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=dtype)
        assert aggregate(x, mask, Aggregation.TOKEN_MEAN).dtype == dtype


def test_norm_len_required_for_token_sum_norm_at_call_time() -> None:
    """norm_len=None with TOKEN_SUM_NORM raises at call time in all three entry points
    (docs/derivations/aggregation.md: configs may carry norm_len=None)."""
    mask = torch.tensor([[True, True], [True, False]])
    x = torch.zeros((2, 2), dtype=torch.float64)
    with pytest.raises(ValueError, match="norm_len is required"):
        aggregate(x, mask, Aggregation.TOKEN_SUM_NORM)
    with pytest.raises(ValueError, match="norm_len is required"):
        effective_token_weights(mask, Aggregation.TOKEN_SUM_NORM)
    with pytest.raises(ValueError, match="norm_len is required"):
        microbatch_token_weights(mask, Aggregation.TOKEN_SUM_NORM, [1, 1])


def test_norm_len_must_be_positive() -> None:
    """A non-positive norm_len is rejected."""
    mask = torch.tensor([[True, True]])
    with pytest.raises(ValueError, match="positive int; got 0"):
        effective_token_weights(mask, Aggregation.TOKEN_SUM_NORM, norm_len=0)


def test_norm_len_ignored_for_other_modes() -> None:
    """Passing norm_len with a mode other than TOKEN_SUM_NORM is legal and ignored, so
    callers may forward a shared config value (docs/derivations/aggregation.md)."""
    mask = torch.tensor([[True, True], [True, False]])
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
    assert torch.equal(
        aggregate(x, mask, Aggregation.TOKEN_MEAN, norm_len=3),
        aggregate(x, mask, Aggregation.TOKEN_MEAN),
    )


def test_microbatch_sizes_validation() -> None:
    """microbatch_sizes must be positive and sum to B; loss_scale must be known."""
    mask = torch.ones((4, 2), dtype=torch.bool)
    with pytest.raises(ValueError, match=r"sum to the batch size 4"):
        microbatch_token_weights(mask, Aggregation.TOKEN_MEAN, [1, 2])
    with pytest.raises(ValueError, match=r"entries must be >= 1"):
        microbatch_token_weights(mask, Aggregation.TOKEN_MEAN, [0, 4])
    with pytest.raises(ValueError, match=r"loss_scale must be 'mean' or 'sum'"):
        microbatch_token_weights(
            mask,
            Aggregation.TOKEN_MEAN,
            [2, 2],
            loss_scale="avg",  # type: ignore[arg-type]
        )


def test_aggregate_rejects_invalid_shapes_and_masks() -> None:
    """Shape/dtype/mask violations raise ValueError naming the argument
    (docs/conventions.md, errors)."""
    x = torch.zeros((2, 3), dtype=torch.float64)
    good = torch.ones((2, 3), dtype=torch.bool)
    with pytest.raises(ValueError, match=r"per_token must be 2-D"):
        aggregate(torch.zeros(3, dtype=torch.float64), good, Aggregation.TOKEN_MEAN)
    with pytest.raises(ValueError, match=r"dtype torch\.bool"):
        aggregate(x, torch.ones((2, 3)), Aggregation.TOKEN_MEAN)
    with pytest.raises(ValueError, match=r"does not match"):
        aggregate(x, torch.ones((2, 2), dtype=torch.bool), Aggregation.TOKEN_MEAN)
    empty_row = torch.tensor([[True, True, True], [False, False, False]])
    with pytest.raises(ValueError, match=r"zero response tokens"):
        aggregate(x, empty_row, Aggregation.TOKEN_MEAN)
