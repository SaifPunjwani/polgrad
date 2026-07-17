"""Enforces docs/diagnostics/ess.md: ESS formula, unnormalized sequence weights, null
calibration (identical policies and the Normal log-weight law), sliding-window
semantics, mask invariance, and validation errors."""

from __future__ import annotations

import math

import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st
from strategies import logprob_batches

from polgrad.diagnostics.ess import ESSReport, importance_ess, sliding_ess

_LN2 = math.log(2.0)


def _perturb_masked(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Replace masked positions with a different junk value than the strategy used."""
    return torch.where(mask, x, torch.full_like(x, -55.25))


def _golden_batch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gaps new-old of [[0.5, -0.5], [ln 2, junk]] under mask [[T, T], [T, F]]."""
    old = torch.tensor([[-1.0, -1.5], [-2.0, 123.0]], dtype=torch.float64)
    new = torch.tensor([[-0.5, -2.0], [-2.0 + _LN2, 123.0]], dtype=torch.float64)
    mask = torch.tensor([[True, True], [True, False]])
    return new, old, mask


def test_identical_policies_ess_ratio_is_exactly_one() -> None:
    """Null calibration (docs/diagnostics/ess.md): equal streams give ess_ratio == 1
    exactly, at both levels."""
    lp = torch.tensor([[-0.3, -2.7, 123.0], [-1.1, 123.0, 123.0]], dtype=torch.float64)
    mask = torch.tensor([[True, True, False], [True, False, False]])
    for level in ("sequence", "token"):
        report = importance_ess(lp, lp.clone(), mask, level=level)  # type: ignore[arg-type]
        assert report.ess_ratio == 1.0
        assert report.ess == float(report.n)
        assert report.log_weight_mean == 0.0
        assert report.log_weight_min == 0.0
        assert report.log_weight_max == 0.0


def test_identical_policies_sliding_ess_is_exactly_one() -> None:
    """Null calibration: equal streams give ESS/window == 1 exactly in every window."""
    lp = torch.tensor([[-0.5], [-1.0], [-1.5], [-2.0]], dtype=torch.float64)
    mask = torch.ones(4, 1, dtype=torch.bool)
    out = sliding_ess(lp, lp.clone(), mask, window=2, step=1)
    assert torch.equal(out, torch.ones(3, dtype=torch.float64))


def test_sequence_level_golden_case() -> None:
    """Hand-derived (docs/diagnostics/ess.md): weights [1, 2] give ESS = 9/5 = 1.8,
    ess_ratio = 0.9, and log-weight stats over [0, ln 2]."""
    new, old, mask = _golden_batch()
    report = importance_ess(new, old, mask, level="sequence")
    assert report.level == "sequence"
    assert report.n == 2
    assert report.ess == pytest.approx(1.8, rel=1e-12)
    assert report.ess_ratio == pytest.approx(0.9, rel=1e-12)
    assert report.log_weight_mean == pytest.approx(_LN2 / 2, rel=1e-12)
    assert report.log_weight_std == pytest.approx(_LN2 / math.sqrt(2.0), rel=1e-12)
    assert report.log_weight_min == pytest.approx(0.0, abs=1e-12)
    assert report.log_weight_max == pytest.approx(_LN2, rel=1e-12)


def test_token_level_golden_case() -> None:
    """Hand-derived (docs/diagnostics/ess.md): pooled token weights
    [e^0.5, e^-0.5, 2] give ESS = (Σw)²/Σw² with Σw² = e + 1/e + 4."""
    new, old, mask = _golden_batch()
    report = importance_ess(new, old, mask, level="token")
    weights = [math.exp(0.5), math.exp(-0.5), 2.0]
    expected_ess = sum(weights) ** 2 / sum(w * w for w in weights)
    assert report.n == 3
    assert report.ess == pytest.approx(expected_ess, rel=1e-12)
    assert report.ess_ratio == pytest.approx(expected_ess / 3, rel=1e-12)
    assert report.log_weight_max == pytest.approx(_LN2, rel=1e-12)
    assert report.log_weight_min == pytest.approx(-0.5, rel=1e-12)


def test_sequence_weights_are_unnormalized_sums() -> None:
    """Sequence weights exponentiate the SUMMED gap, not the
    length-normalized GSPO exponent — equal per-token gaps on unequal lengths give
    unequal weights (docs/diagnostics/ess.md)."""
    old = torch.zeros(2, 2, dtype=torch.float64)
    new = torch.ones(2, 2, dtype=torch.float64)
    mask = torch.tensor([[True, False], [True, True]])
    report = importance_ess(new, old, mask, level="sequence")
    assert report.log_weight_min == pytest.approx(1.0, rel=1e-12)
    assert report.log_weight_max == pytest.approx(2.0, rel=1e-12)
    # A length-normalized exponent would give equal weights and ess_ratio == 1.
    assert report.ess_ratio < 1.0


@given(batch=logprob_batches(), level=st.sampled_from(["token", "sequence"]))
def test_ess_ratio_bounds(batch, level) -> None:  # type: ignore[no-untyped-def]
    """ESS/n lies in (0, 1] (Cauchy-Schwarz; docs/diagnostics/ess.md), both levels."""
    report = importance_ess(batch.logprobs, batch.old_logprobs, batch.response_mask, level=level)
    assert 0.0 < report.ess_ratio <= 1.0 + 1e-12
    assert report.ess == pytest.approx(report.ess_ratio * report.n, rel=1e-12)


@given(batch=logprob_batches(), shift=st.floats(-3.0, 3.0, allow_nan=False))
def test_token_level_shift_invariance(batch, shift) -> None:  # type: ignore[no-untyped-def]
    """Token-level ESS is invariant to a constant log-weight shift, since it rescales
    every weight by exp(shift) (docs/diagnostics/ess.md scale-invariance)."""
    base = importance_ess(batch.logprobs, batch.old_logprobs, batch.response_mask, level="token")
    shifted_new = torch.where(batch.response_mask, batch.logprobs + shift, batch.logprobs)
    shifted = importance_ess(shifted_new, batch.old_logprobs, batch.response_mask, level="token")
    assert shifted.ess == pytest.approx(base.ess, rel=1e-9)


@given(batch=logprob_batches(), level=st.sampled_from(["token", "sequence"]))
def test_mask_invariance_importance_ess(batch, level) -> None:  # type: ignore[no-untyped-def]
    """Masked inputs never affect the report (docs/conventions.md): all fields are
    bitwise-equal after perturbing masked positions."""
    mask = batch.response_mask
    base = importance_ess(batch.logprobs, batch.old_logprobs, mask, level=level)
    perturbed = importance_ess(
        _perturb_masked(batch.logprobs, mask),
        _perturb_masked(batch.old_logprobs, mask),
        mask,
        level=level,
    )
    assert base == perturbed  # ESSReport holds only scalars; dataclass eq is bitwise


@given(batch=logprob_batches(max_b=8, max_t=6), window=st.integers(2, 8), step=st.integers(1, 3))
def test_mask_invariance_sliding_ess(batch, window, step) -> None:  # type: ignore[no-untyped-def]
    """Masked inputs never affect the sliding ESS trace (docs/conventions.md)."""
    mask = batch.response_mask
    if window > mask.shape[0]:
        window = mask.shape[0]
    if window < 2:
        return
    base = sliding_ess(batch.logprobs, batch.old_logprobs, mask, window=window, step=step)
    perturbed = sliding_ess(
        _perturb_masked(batch.logprobs, mask),
        _perturb_masked(batch.old_logprobs, mask),
        mask,
        window=window,
        step=step,
    )
    assert torch.equal(base, perturbed)


@given(batch=logprob_batches(max_b=8, max_t=6))
def test_sliding_ess_matches_importance_ess_full_window(batch) -> None:  # type: ignore[no-untyped-def]
    """A single window covering the whole batch reproduces importance_ess(...).ess_ratio."""
    b = batch.response_mask.shape[0]
    if b < 2:
        return
    report = importance_ess(
        batch.logprobs, batch.old_logprobs, batch.response_mask, level="sequence"
    )
    out = sliding_ess(batch.logprobs, batch.old_logprobs, batch.response_mask, window=b, step=1)
    assert out.shape == (1,)
    assert float(out[0]) == pytest.approx(report.ess_ratio, rel=1e-12)


@given(
    b=st.integers(2, 12),
    window=st.integers(2, 12),
    step=st.integers(1, 4),
)
def test_sliding_ess_window_and_step_shapes(b: int, window: int, step: int) -> None:
    """Output shape is [(B - window)//step + 1] (docs/diagnostics/ess.md, sliding
    windows)."""
    if window > b:
        return
    lp = -torch.rand(b, 1, generator=torch.Generator().manual_seed(b), dtype=torch.float64)
    mask = torch.ones(b, 1, dtype=torch.bool)
    out = sliding_ess(lp, torch.zeros_like(lp), mask, window=window, step=step)
    assert out.shape == ((b - window) // step + 1,)
    assert bool((out > 0).all()) and bool((out <= 1.0 + 1e-12).all())


def test_sliding_ess_golden_case() -> None:
    """Hand-derived (docs/diagnostics/ess.md): log-weights [0, ln2, 0, ln2] give
    ESS/window = 0.9 for window 2 and [8/9, 25/27] for window 3."""
    new = torch.tensor([[0.0], [_LN2], [0.0], [_LN2]], dtype=torch.float64)
    old = torch.zeros(4, 1, dtype=torch.float64)
    mask = torch.ones(4, 1, dtype=torch.bool)
    out2 = sliding_ess(new, old, mask, window=2, step=1)
    assert torch.allclose(out2, torch.full((3,), 0.9, dtype=torch.float64), rtol=1e-12, atol=0.0)
    out3 = sliding_ess(new, old, mask, window=3, step=1)
    expected3 = torch.tensor([8.0 / 9.0, 25.0 / 27.0], dtype=torch.float64)
    assert torch.allclose(out3, expected3, rtol=1e-12, atol=0.0)
    out_step2 = sliding_ess(new, old, mask, window=2, step=2)
    assert torch.allclose(
        out_step2, torch.full((2,), 0.9, dtype=torch.float64), rtol=1e-12, atol=0.0
    )


def test_sliding_ess_preserves_dtype() -> None:
    """sliding_ess keeps the input dtype (docs/conventions.md, no silent casts)."""
    for dtype in (torch.float32, torch.float64):
        lp = -torch.rand(4, 2, generator=torch.Generator().manual_seed(0)).to(dtype)
        mask = torch.ones(4, 2, dtype=torch.bool)
        out = sliding_ess(lp, torch.zeros_like(lp), mask, window=2)
        assert out.dtype == dtype


def test_importance_ess_matches_direct_simulation() -> None:
    """Verifies on a seeded case that importance_ess computes exactly the simulated
    quantity (Σw)²/Σw² with w = exp(z) (docs/diagnostics/ess.md)."""
    g = torch.Generator().manual_seed(0)
    z = torch.randn(512, 1, generator=g, dtype=torch.float64) * 0.8
    mask = torch.ones(512, 1, dtype=torch.bool)
    report = importance_ess(z, torch.zeros_like(z), mask, level="sequence")
    w = torch.exp(z.flatten())
    direct = float(w.sum() ** 2 / (w * w).sum())
    assert report.ess == pytest.approx(direct, rel=1e-10)
    # Pooling the same weights at token level ([1, n] row) gives the same ESS.
    row = z.reshape(1, -1)
    token = importance_ess(
        row, torch.zeros_like(row), torch.ones_like(row, dtype=torch.bool), level="token"
    )
    assert token.ess == pytest.approx(report.ess, rel=1e-12)


def test_mc_calibration_mean_ess_ratio_approaches_exp_neg_var() -> None:
    """Verifies on 12 seeded runs of n=8192 that iid Normal(0, σ²) log-weight
    perturbations give E[ESS/n] ≈ exp(-σ²) (lognormal-moment algebra in
    docs/diagnostics/ess.md)."""
    for sigma in (0.3, 0.6):
        ratios = []
        for seed in range(12):
            g = torch.Generator().manual_seed(seed)
            z = torch.randn(8192, 1, generator=g, dtype=torch.float64) * sigma
            mask = torch.ones(8192, 1, dtype=torch.bool)
            ratios.append(importance_ess(z, torch.zeros_like(z), mask).ess_ratio)
        mean_ratio = sum(ratios) / len(ratios)
        assert mean_ratio == pytest.approx(math.exp(-(sigma**2)), abs=0.01)


@given(batch=logprob_batches(), level=st.sampled_from(["token", "sequence"]))
def test_report_log_weight_stats_match_recomputation(batch, level) -> None:  # type: ignore[no-untyped-def]
    """ESSReport log-weight statistics match an independent recomputation; a single
    weight reports std == 0.0."""
    mask = batch.response_mask
    report = importance_ess(batch.logprobs, batch.old_logprobs, mask, level=level)
    gaps = (batch.logprobs - batch.old_logprobs)[mask]
    if level == "sequence":
        rows = [
            float(gaps[mask.sum(dim=1)[:i].sum() : mask.sum(dim=1)[: i + 1].sum()].sum())
            for i in range(mask.shape[0])
        ]
        values = torch.tensor(rows, dtype=torch.float64)
    else:
        values = gaps
    assert report.n == values.numel()
    assert report.log_weight_mean == pytest.approx(float(values.mean()), rel=1e-9, abs=1e-12)
    assert report.log_weight_min == pytest.approx(float(values.min()), rel=1e-9, abs=1e-12)
    assert report.log_weight_max == pytest.approx(float(values.max()), rel=1e-9, abs=1e-12)
    if values.numel() < 2:
        assert report.log_weight_std == 0.0
    else:
        assert report.log_weight_std == pytest.approx(float(values.std()), rel=1e-9, abs=1e-12)


def test_window_validation_errors() -> None:
    """sliding_ess raises for window > B, window < 2, step < 1
    (docs/diagnostics/ess.md, sliding windows)."""
    lp = torch.zeros(4, 1, dtype=torch.float64)
    mask = torch.ones(4, 1, dtype=torch.bool)
    with pytest.raises(ValueError, match=r"window must be <= batch size B=4"):
        sliding_ess(lp, lp, mask, window=5)
    with pytest.raises(ValueError, match=r"window must be >= 2"):
        sliding_ess(lp, lp, mask, window=1)
    with pytest.raises(ValueError, match=r"step must be >= 1"):
        sliding_ess(lp, lp, mask, window=2, step=0)


def test_input_validation_errors() -> None:
    """Invalid shapes, masks, non-finite response values, and bad level raise
    ValueError naming the argument (docs/conventions.md)."""
    lp = torch.zeros(2, 3, dtype=torch.float64)
    mask = torch.ones(2, 3, dtype=torch.bool)
    with pytest.raises(ValueError, match=r"logprobs_new must be 2-D"):
        importance_ess(torch.zeros(3), lp, mask)
    with pytest.raises(ValueError, match=r"logprobs_new and logprobs_old"):
        importance_ess(lp, torch.zeros(2, 4, dtype=torch.float64), mask)
    with pytest.raises(ValueError, match=r"dtype torch\.bool"):
        importance_ess(lp, lp, torch.ones(2, 3))
    with pytest.raises(ValueError, match=r"zero response tokens"):
        importance_ess(lp, lp, torch.tensor([[True, True, True], [False, False, False]]))
    bad = lp.clone()
    bad[0, 0] = float("nan")
    with pytest.raises(ValueError, match=r"non-finite"):
        importance_ess(bad, lp, mask)
    with pytest.raises(ValueError, match=r"level must be 'token' or 'sequence'"):
        importance_ess(lp, lp, mask, level="batch")  # type: ignore[arg-type]


def test_masked_positions_may_hold_nonfinite_junk() -> None:
    """Finiteness is enforced on response positions only; masked junk may be anything."""
    lp = torch.tensor([[-1.0, float("inf")], [-2.0, float("nan")]], dtype=torch.float64)
    mask = torch.tensor([[True, False], [True, False]])
    report = importance_ess(lp, torch.where(mask, lp, -lp), mask)
    assert report.ess_ratio == 1.0


def test_summary_is_compact_multiline() -> None:
    """summary() is a compact human-readable multi-line string."""
    new, old, mask = _golden_batch()
    report = importance_ess(new, old, mask)
    text = report.summary()
    assert isinstance(report, ESSReport)
    assert isinstance(text, str)
    assert text.count("\n") == 1
    assert "ESS" in text and "n=2" in text and "log-weights" in text
