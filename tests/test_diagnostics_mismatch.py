"""Enforces docs/diagnostics/mismatch.md: gap statistics, KL estimates between the
trainer and rollout streams, sequence log-ratios, catastrophic-token indexing, the
perplexity ratio, mask invariance, and validation errors."""

from __future__ import annotations

import math

import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st
from strategies import MASKED_JUNK, LogprobBatch, padded_masks

from polgrad.diagnostics.mismatch import MismatchReport, logprob_mismatch


@st.composite
def logprob_batches(
    draw: st.DrawFn, *, max_b: int = 8, max_t: int = 12, max_gap: float = 2.0
) -> LogprobBatch:
    """Local mirror of strategies.logprob_batches with 64-bit float bounds.

    The shared strategy passes width=32 bounds that this Hypothesis version rejects
    (-0.05 is not float32-representable); strategies.py may not be edited by module
    agents, so the batches are generated locally with the same shape and junk rules.
    """
    mask = draw(padded_masks(max_b=max_b, max_t=max_t))
    b, t = mask.shape

    def fill(low: float, high: float) -> torch.Tensor:
        vals = [
            draw(st.floats(low, high, allow_nan=False, allow_infinity=False)) for _ in range(b * t)
        ]
        out = torch.tensor(vals, dtype=torch.float64).reshape(b, t)
        return torch.where(mask, out, torch.full_like(out, MASKED_JUNK))

    def near(base: torch.Tensor) -> torch.Tensor:
        gap = fill(-max_gap, max_gap)
        return torch.where(mask, base + gap, torch.full_like(base, MASKED_JUNK))

    logprobs = fill(-8.0, -0.05)
    old_logprobs = near(logprobs)
    ref_logprobs = near(logprobs)
    rollout_logprobs = near(old_logprobs)
    advantages = fill(-3.0, 3.0)
    return LogprobBatch(logprobs, old_logprobs, ref_logprobs, rollout_logprobs, advantages, mask)


def _perturb_masked(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Replace masked positions with a different junk value than the strategy used."""
    return torch.where(mask, x, torch.full_like(x, -55.25))


def _golden_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gaps Δ = trainer - rollout of [0.2, -0.6 | 1.0] under mask [[T, T], [T, F]]."""
    trainer = torch.tensor([[-1.0, -2.0], [-1.5, 123.0]], dtype=torch.float64)
    rollout = torch.tensor([[-1.2, -1.4], [-2.5, -123.0]], dtype=torch.float64)
    mask = torch.tensor([[True, True], [True, False]])
    return trainer, rollout, mask


def test_golden_case_hand_derived() -> None:
    """Hand-derived numbers of docs/diagnostics/mismatch.md for Δ = [0.2, -0.6, 1.0]:
    moments, quantiles, k1/k2/k3 of KL(rollout‖trainer) plus the trainer→rollout k3,
    sequence stats, catastrophic indices at gap 0.7, and ppl_ratio = exp(-0.2)."""
    trainer, rollout, mask = _golden_inputs()
    report = logprob_mismatch(trainer, rollout, mask, catastrophic_gap=0.7)
    assert report.n_tokens == 3
    assert report.gap_mean == pytest.approx(0.2, rel=1e-12)
    assert report.gap_std == pytest.approx(0.8, rel=1e-12)
    assert report.gap_abs_max == pytest.approx(1.0, rel=1e-12)
    # Linear-interpolation quantiles of sorted [-0.6, 0.2, 1.0] at h = 2q.
    expected_q = (-0.584, -0.52, -0.2, 0.2, 0.6, 0.92, 0.984)
    assert report.gap_quantiles == pytest.approx(expected_q, rel=1e-9)
    # Rollout‖trainer direction (tokens ~ rollout): k1 = mean -Δ; k2 = mean Δ²/2;
    # k3 = mean exp(Δ) - 1 - Δ; the trainer→rollout k3 formula swaps the sign of Δ.
    assert report.kl_k1 == pytest.approx(-0.2, rel=1e-12)
    assert report.kl_k2 == pytest.approx((0.02 + 0.18 + 0.5) / 3, rel=1e-12)
    expected_k3 = (
        (math.exp(0.2) - 1 - 0.2) + (math.exp(-0.6) - 1 + 0.6) + (math.exp(1.0) - 1 - 1.0)
    ) / 3
    assert report.kl_k3 == pytest.approx(expected_k3, rel=1e-9)
    expected_k3_tr = (
        (math.exp(-0.2) - 1 + 0.2) + (math.exp(0.6) - 1 - 0.6) + (math.exp(-1.0) - 1 + 1.0)
    ) / 3
    assert report.kl_k3_trainer_rollout == pytest.approx(expected_k3_tr, rel=1e-9)
    # Sequence log-ratios: row sums [-0.4, 1.0].
    assert report.seq_log_ratio_mean == pytest.approx(0.3, rel=1e-12)
    assert report.seq_log_ratio_std == pytest.approx(math.sqrt(0.98), rel=1e-12)
    assert report.seq_log_ratio_abs_max == pytest.approx(1.0, rel=1e-12)
    # Only (b=1, t=0) has |Δ| = 1.0 > 0.7.
    assert report.catastrophic_count == 1
    assert report.catastrophic_indices.dtype == torch.long
    assert report.catastrophic_indices.tolist() == [[1, 0]]
    assert report.ppl_ratio == pytest.approx(math.exp(-0.2), rel=1e-12)


def test_identical_streams_zero_gaps() -> None:
    """Equal streams give zero gaps, zero KL estimates, no catastrophic tokens, and
    ppl_ratio == 1 exactly."""
    trainer = torch.tensor([[-0.7, -1.9], [-2.4, 123.0]], dtype=torch.float64)
    mask = torch.tensor([[True, True], [True, False]])
    report = logprob_mismatch(trainer, trainer.clone(), mask)
    assert report.gap_mean == 0.0
    assert report.gap_std == 0.0
    assert report.gap_abs_max == 0.0
    assert report.gap_quantiles == (0.0,) * 7
    assert report.kl_k1 == 0.0 and report.kl_k2 == 0.0 and report.kl_k3 == 0.0
    assert report.kl_k3_trainer_rollout == 0.0
    assert report.seq_log_ratio_abs_max == 0.0
    assert report.catastrophic_count == 0
    assert report.catastrophic_indices.shape == (0, 2)
    assert report.ppl_ratio == 1.0


@given(batch=logprob_batches())
def test_kl_k1_equals_neg_gap_mean_and_ppl_ratio_is_exp_neg(batch) -> None:  # type: ignore[no-untyped-def]
    """Internal consistency: k1 of KL(rollout‖trainer) is minus the mean gap, and
    ppl_ratio == exp(-gap_mean) (docs/diagnostics/mismatch.md)."""
    report = logprob_mismatch(batch.old_logprobs, batch.rollout_logprobs, batch.response_mask)
    assert report.kl_k1 == pytest.approx(-report.gap_mean, rel=1e-9, abs=1e-12)
    assert report.ppl_ratio == pytest.approx(math.exp(-report.gap_mean), rel=1e-9)


@given(batch=logprob_batches())
def test_k3_estimates_nonnegative(batch) -> None:  # type: ignore[no-untyped-def]
    """k3 = exp(δ) - 1 - δ ≥ 0 pointwise, so both direction k3 fields are nonnegative
    (docs/derivations/kl.md), as is k2."""
    report = logprob_mismatch(batch.old_logprobs, batch.rollout_logprobs, batch.response_mask)
    assert report.kl_k2 >= 0.0
    assert report.kl_k3 >= -1e-12
    assert report.kl_k3_trainer_rollout >= -1e-12


def test_mc_kl_fields_match_closed_form_categorical_kl(gen: torch.Generator) -> None:
    """MC calibration on a tabular pair where the two KL directions differ measurably:
    with tokens sampled from the ROLLOUT categorical, kl_k1 and kl_k3 must match the
    closed-form KL(rollout‖trainer) — positive sign, magnitude within a CLT tolerance
    of 4·std/√n — and kl_k3_trainer_rollout must match its true expectation
    χ²(rollout‖trainer) − KL(rollout‖trainer). A direction mislabel (evaluating the
    trainer‖rollout formulas on rollout samples) would flip kl_k1's sign to
    −KL(rollout‖trainer) and fail both assertions."""
    rollout_p = torch.tensor([0.5, 0.25, 0.15, 0.1], dtype=torch.float64)
    trainer_q = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float64)
    kl_rollout_trainer = float((rollout_p * (rollout_p / trainer_q).log()).sum())
    kl_trainer_rollout = float((trainer_q * (trainer_q / rollout_p).log()).sum())
    chi2_rollout_trainer = float((rollout_p * rollout_p / trainer_q).sum()) - 1.0
    # The construction must separate the directions, or the test could not catch a
    # mislabel: here KL(rollout‖trainer) ≈ 0.618, KL(trainer‖rollout) ≈ 0.557.
    assert abs(kl_rollout_trainer - kl_trainer_rollout) > 0.05
    n = 300_000
    x = torch.multinomial(rollout_p, n, replacement=True, generator=gen)
    rollout_logprobs = rollout_p.log()[x].unsqueeze(1)
    trainer_logprobs = trainer_q.log()[x].unsqueeze(1)
    mask = torch.ones((n, 1), dtype=torch.bool)
    report = logprob_mismatch(trainer_logprobs, rollout_logprobs, mask)
    delta = (trainer_logprobs - rollout_logprobs).squeeze(1)
    for value, samples, target in (
        (report.kl_k1, -delta, kl_rollout_trainer),
        (report.kl_k3, torch.expm1(delta) - delta, kl_rollout_trainer),
        (
            report.kl_k3_trainer_rollout,
            torch.expm1(-delta) + delta,
            chi2_rollout_trainer - kl_rollout_trainer,
        ),
    ):
        tol = 4.0 * float(samples.std()) / math.sqrt(n)
        assert value > 0.0
        assert abs(value - target) < tol
    # The mislabeled direction is far outside the same tolerance (sign flip).
    k1_tol = 4.0 * float(delta.std()) / math.sqrt(n)
    assert abs(report.kl_k1 - (-kl_rollout_trainer)) > 100 * k1_tol


@given(batch=logprob_batches(), gap=st.floats(0.1, 3.0, allow_nan=False))
def test_catastrophic_indices_consistency(batch, gap: float) -> None:  # type: ignore[no-untyped-def]
    """catastrophic_indices is a [N, 2] long tensor listing exactly the response
    positions with |Δ| > catastrophic_gap (strict), in row-major order."""
    report = logprob_mismatch(
        batch.old_logprobs, batch.rollout_logprobs, batch.response_mask, catastrophic_gap=gap
    )
    delta = batch.old_logprobs - batch.rollout_logprobs
    expected = [
        [b, t]
        for b in range(batch.response_mask.shape[0])
        for t in range(batch.response_mask.shape[1])
        if bool(batch.response_mask[b, t]) and abs(float(delta[b, t])) > gap
    ]
    assert report.catastrophic_indices.dtype == torch.long
    assert report.catastrophic_indices.shape == (len(expected), 2)
    assert report.catastrophic_indices.tolist() == expected
    assert report.catastrophic_count == len(expected)


@given(batch=logprob_batches())
def test_quantiles_monotone_and_bounded(batch) -> None:  # type: ignore[no-untyped-def]
    """Gap quantiles are nondecreasing and lie within [-gap_abs_max, gap_abs_max]."""
    report = logprob_mismatch(batch.old_logprobs, batch.rollout_logprobs, batch.response_mask)
    assert len(report.gap_quantiles) == 7
    for lo, hi in zip(report.gap_quantiles, report.gap_quantiles[1:], strict=False):
        assert lo <= hi + 1e-12
    assert report.gap_quantiles[0] >= -report.gap_abs_max - 1e-12
    assert report.gap_quantiles[-1] <= report.gap_abs_max + 1e-12


@given(batch=logprob_batches())
def test_seq_log_ratio_stats_match_recomputation(batch) -> None:  # type: ignore[no-untyped-def]
    """Sequence log-ratio statistics match an independent per-row recomputation."""
    report = logprob_mismatch(batch.old_logprobs, batch.rollout_logprobs, batch.response_mask)
    rows = torch.tensor(
        [
            float((batch.old_logprobs[b] - batch.rollout_logprobs[b])[batch.response_mask[b]].sum())
            for b in range(batch.response_mask.shape[0])
        ],
        dtype=torch.float64,
    )
    assert report.seq_log_ratio_mean == pytest.approx(float(rows.mean()), rel=1e-9, abs=1e-12)
    assert report.seq_log_ratio_abs_max == pytest.approx(
        float(rows.abs().max()), rel=1e-9, abs=1e-12
    )
    if rows.numel() < 2:
        assert report.seq_log_ratio_std == 0.0
    else:
        assert report.seq_log_ratio_std == pytest.approx(float(rows.std()), rel=1e-9, abs=1e-12)


def test_single_token_stds_are_zero() -> None:
    """A single observation has no spread estimate: gap_std and seq_log_ratio_std are
    reported as 0.0, never NaN."""
    report = logprob_mismatch(
        torch.tensor([[-1.0]], dtype=torch.float64),
        torch.tensor([[-1.5]], dtype=torch.float64),
        torch.tensor([[True]]),
    )
    assert report.n_tokens == 1
    assert report.gap_std == 0.0
    assert report.seq_log_ratio_std == 0.0
    assert report.gap_mean == pytest.approx(0.5, rel=1e-12)


@given(batch=logprob_batches())
def test_mask_invariance_mismatch(batch) -> None:  # type: ignore[no-untyped-def]
    """Masked inputs never affect the report (docs/conventions.md): every field is
    bitwise-equal after perturbing masked positions."""
    mask = batch.response_mask
    base = logprob_mismatch(batch.old_logprobs, batch.rollout_logprobs, mask)
    perturbed = logprob_mismatch(
        _perturb_masked(batch.old_logprobs, mask),
        _perturb_masked(batch.rollout_logprobs, mask),
        mask,
    )
    for field in (
        "n_tokens",
        "gap_mean",
        "gap_std",
        "gap_abs_max",
        "gap_quantiles",
        "kl_k1",
        "kl_k2",
        "kl_k3",
        "kl_k3_trainer_rollout",
        "seq_log_ratio_mean",
        "seq_log_ratio_std",
        "seq_log_ratio_abs_max",
        "catastrophic_count",
        "ppl_ratio",
    ):
        assert getattr(base, field) == getattr(perturbed, field), field
    assert torch.equal(base.catastrophic_indices, perturbed.catastrophic_indices)


def test_validation_errors() -> None:
    """Invalid shapes, masks, non-finite response values, and non-positive
    catastrophic_gap raise ValueError naming the argument (docs/conventions.md)."""
    trainer = torch.zeros(2, 3, dtype=torch.float64)
    mask = torch.ones(2, 3, dtype=torch.bool)
    with pytest.raises(ValueError, match=r"trainer_logprobs must be 2-D"):
        logprob_mismatch(torch.zeros(3), trainer, mask)
    with pytest.raises(ValueError, match=r"trainer_logprobs and rollout_logprobs"):
        logprob_mismatch(trainer, torch.zeros(2, 4, dtype=torch.float64), mask)
    with pytest.raises(ValueError, match=r"dtype torch\.bool"):
        logprob_mismatch(trainer, trainer, torch.ones(2, 3))
    with pytest.raises(ValueError, match=r"zero response tokens"):
        logprob_mismatch(
            trainer, trainer, torch.tensor([[True, True, True], [False, False, False]])
        )
    bad = trainer.clone()
    bad[0, 1] = float("nan")
    with pytest.raises(ValueError, match=r"non-finite"):
        logprob_mismatch(bad, trainer, mask)
    for bad_gap in (0.0, -1.0):
        with pytest.raises(ValueError, match=r"catastrophic_gap must be > 0"):
            logprob_mismatch(trainer, trainer, mask, catastrophic_gap=bad_gap)


def test_summary_is_compact_multiline() -> None:
    """summary() is a compact human-readable multi-line string."""
    trainer, rollout, mask = _golden_inputs()
    report = logprob_mismatch(trainer, rollout, mask, catastrophic_gap=0.7)
    text = report.summary()
    assert isinstance(report, MismatchReport)
    assert text.count("\n") == 5
    assert "n_tokens=3" in text
    assert "ppl_ratio" in text
    assert "catastrophic tokens: 1" in text
    assert "q50" in text
