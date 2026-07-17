"""Enforces docs/diagnostics/length_bias.md: the regressand/regressor definitions, the
OLS + HC3 closed forms (hand-computed golden case, full-matrix numpy sandwich, scipy
linregress cross-checks), CI coverage of a known slope, the mode-induced weight
structure via polgrad.aggregate.effective_token_weights, mask invariance, and the
degenerate-input ValueErrors."""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st
from scipy import stats
from strategies import MASKED_JUNK

from polgrad.aggregate import Aggregation, effective_token_weights
from polgrad.diagnostics.length_bias import LengthBiasReport, length_bias_probe

_MODES: list[tuple[Aggregation, int | None]] = [
    (Aggregation.TOKEN_MEAN, None),
    (Aggregation.SEQ_MEAN_TOKEN_MEAN, None),
    (Aggregation.SEQ_MEAN_TOKEN_SUM, None),
    (Aggregation.TOKEN_SUM_NORM, 6),
]


@st.composite
def valid_masks(draw: st.DrawFn, *, max_t: int = 8, max_extra: int = 4) -> torch.Tensor:
    """Right-padded masks whose length pattern keeps the probe non-degenerate.

    Two distinct lengths are each held by at least two rows, so the lengths are never
    constant and never leave a single row at a distinct length (the leverage h = 1
    pattern length_bias_probe rejects); extra rows may take any length.
    """
    t = draw(st.integers(2, max_t))
    short = draw(st.integers(1, t - 1))
    long = draw(st.integers(short + 1, t))
    lengths = [short, short, long, long]
    lengths += draw(st.lists(st.integers(1, t), max_size=max_extra))
    mask = torch.zeros((len(lengths), t), dtype=torch.bool)
    for i, length in enumerate(lengths):
        mask[i, :length] = True
    return mask


@st.composite
def probe_cases(draw: st.DrawFn) -> tuple[torch.Tensor, torch.Tensor]:
    """(advantages [B, T], mask) pairs; masked positions hold MASKED_JUNK."""
    mask = draw(valid_masks())
    b, t = mask.shape
    vals = [draw(st.floats(-3.0, 3.0, allow_nan=False, allow_infinity=False)) for _ in range(b * t)]
    adv = torch.tensor(vals, dtype=torch.float64).reshape(b, t)
    return torch.where(mask, adv, torch.full_like(adv, MASKED_JUNK)), mask


def _probe_regressand(
    adv: torch.Tensor, mask: torch.Tensor, mode: Aggregation, norm_len: int | None
) -> tuple[torch.Tensor, torch.Tensor]:
    """The (x, y) of docs/diagnostics/length_bias.md, recomputed from the definition."""
    weights = effective_token_weights(mask, mode, norm_len=norm_len)
    zero = torch.zeros((), dtype=torch.float64)
    y = (torch.where(mask, adv, zero).abs() * weights).sum(dim=1)
    return mask.sum(dim=1).to(torch.float64), y


def _numpy_ols_hc3(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Full-matrix OLS + HC3 sandwich (slope, intercept, slope_se).

    This is the textbook form V = (XᵀX)⁻¹ Xᵀ diag(e²/(1-h)²) X (XᵀX)⁻¹ that the module
    reduces to a simple-regression closed form; agreement checks the reduction algebra.
    """
    design = np.stack([np.ones_like(x), x], axis=1)
    xtx_inv = np.linalg.inv(design.T @ design)
    beta = xtx_inv @ (design.T @ y)
    resid = y - design @ beta
    hat = np.einsum("ij,jk,ik->i", design, xtx_inv, design)
    omega = (resid / (1.0 - hat)) ** 2
    meat = design.T @ (design * omega[:, None])
    cov = xtx_inv @ meat @ xtx_inv
    return float(beta[1]), float(beta[0]), float(np.sqrt(cov[1, 1]))


def _assert_reports_equal(a: LengthBiasReport, b: LengthBiasReport) -> None:
    assert a.slope == b.slope
    assert a.slope_se == b.slope_se
    assert a.ci_low == b.ci_low
    assert a.ci_high == b.ci_high
    assert a.intercept == b.intercept
    assert a.n == b.n
    assert torch.equal(a.per_seq_weight_sum, b.per_seq_weight_sum)
    assert torch.equal(a.per_seq_length, b.per_seq_length)


def _golden_case() -> LengthBiasReport:
    """Lengths [1, 2, 3], SEQ_MEAN_TOKEN_SUM (w = 1/3 per response token).

    Advantage rows [3], [3, 3], [4, 4, 4] give the hand-worked regression of
    docs/diagnostics/length_bias.md:

        y = [3/3, (3+3)/3, (4+4+4)/3] = [1, 2, 4],   x = [1, 2, 3]
        x̄ = 2,  ȳ = 7/3,  Sxx = (-1)² + 0² + 1² = 2
        Sxy = (-1)(1 - 7/3) + 0·(2 - 7/3) + (1)(4 - 7/3) = 4/3 + 5/3 = 3
        slope = Sxy/Sxx = 3/2,   intercept = 7/3 - (3/2)·2 = -2/3
        fitted ŷ = [5/6, 7/3, 23/6],   e = y - ŷ = [1/6, -1/3, 1/6]
        h = 1/3 + (x - 2)²/2 = [5/6, 1/3, 5/6],   1 - h = [1/6, 2/3, 1/6]
        ω = e²/(1 - h)² = [(1/36)/(1/36), (1/9)/(4/9), (1/36)/(1/36)] = [1, 1/4, 1]
        Var(slope) = Σ ω·(x - x̄)² / Sxx² = (1·1 + (1/4)·0 + 1·1)/2² = 1/2
        se = 1/√2
    """
    mask = torch.tensor([[True, False, False], [True, True, False], [True, True, True]])
    adv = torch.tensor(
        [[3.0, MASKED_JUNK, MASKED_JUNK], [3.0, 3.0, MASKED_JUNK], [4.0, 4.0, 4.0]],
        dtype=torch.float64,
    )
    return length_bias_probe(adv, mask, agg_mode=Aggregation.SEQ_MEAN_TOKEN_SUM)


def test_hc3_slope_se_matches_hand_computed_golden_case() -> None:
    """Hand-derived golden case (arithmetic in _golden_case and
    docs/diagnostics/length_bias.md): slope 3/2, intercept -2/3, HC3 se 1/√2."""
    report = _golden_case()
    assert report.n == 3
    assert report.slope == pytest.approx(1.5, rel=1e-12)
    assert report.intercept == pytest.approx(-2.0 / 3.0, rel=1e-12)
    assert report.slope_se == pytest.approx(1.0 / math.sqrt(2.0), rel=1e-12)
    assert torch.equal(report.per_seq_length, torch.tensor([1, 2, 3]))
    expected_weight_sums = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64) / 3.0
    assert torch.allclose(report.per_seq_weight_sum, expected_weight_sums, rtol=1e-12, atol=0.0)


def test_ci_uses_normal_975_quantile() -> None:
    """CI endpoints are slope ± z·se with z = Φ⁻¹(0.975); the module's
    torch.special.ndtri constant is cross-checked against scipy.stats.norm.ppf."""
    z = float(stats.norm.ppf(0.975))
    report = _golden_case()
    assert report.ci_high - report.slope == pytest.approx(z * report.slope_se, rel=1e-12)
    assert report.slope - report.ci_low == pytest.approx(z * report.slope_se, rel=1e-12)
    assert report.ci_low == pytest.approx(1.5 - z / math.sqrt(2.0), rel=1e-9)
    assert report.ci_high == pytest.approx(1.5 + z / math.sqrt(2.0), rel=1e-9)


def test_ols_matches_scipy_linregress() -> None:
    """Verifies on 12 seeded cases (3 seeds x 4 modes) that slope and intercept match
    scipy.stats.linregress applied to the probe's (x, y) definition."""
    b, t = 12, 6
    for seed, (mode, norm_len) in itertools.product(range(3), _MODES):
        g = torch.Generator().manual_seed(seed)
        lengths = torch.randint(1, t + 1, (b,), generator=g)
        mask = torch.arange(t).unsqueeze(0) < lengths.unsqueeze(1)
        adv = torch.randn(b, t, generator=g, dtype=torch.float64)
        report = length_bias_probe(adv, mask, agg_mode=mode, norm_len=norm_len)
        x, y = _probe_regressand(adv, mask, mode, norm_len)
        expected = stats.linregress(x.numpy(), y.numpy())
        assert report.slope == pytest.approx(expected.slope, rel=1e-9, abs=1e-15)
        assert report.intercept == pytest.approx(expected.intercept, rel=1e-9, abs=1e-15)


def test_hc3_matches_full_numpy_sandwich() -> None:
    """Verifies on 12 seeded cases that the module's closed-form HC3 slope se equals an
    independent full-matrix sandwich (docs/diagnostics/length_bias.md reduction)."""
    b, t = 16, 7
    for seed, (mode, norm_len) in itertools.product(range(3), _MODES):
        g = torch.Generator().manual_seed(100 + seed)
        lengths = torch.randint(1, t + 1, (b,), generator=g)
        mask = torch.arange(t).unsqueeze(0) < lengths.unsqueeze(1)
        adv = torch.randn(b, t, generator=g, dtype=torch.float64)
        report = length_bias_probe(adv, mask, agg_mode=mode, norm_len=norm_len)
        x, y = _probe_regressand(adv, mask, mode, norm_len)
        slope, intercept, se = _numpy_ols_hc3(x.numpy(), y.numpy())
        assert report.slope == pytest.approx(slope, rel=1e-9, abs=1e-15)
        assert report.intercept == pytest.approx(intercept, rel=1e-9, abs=1e-15)
        assert report.slope_se == pytest.approx(se, rel=1e-9, abs=1e-15)


def test_ci_covers_known_slope_in_about_95_percent_of_runs() -> None:
    """Verifies on 400 seeded runs (B=200, heteroscedastic noise sd 0.04·L_i, true
    slope 0.5) that the 95% HC3 CI covers the true slope at close to nominal rate; the
    binomial sd of the coverage estimate at 0.95 is √(0.95·0.05/400) ≈ 0.011
    (docs/diagnostics/length_bias.md), and the mean estimate recovers the slope."""
    true_slope, true_intercept = 0.5, 2.0
    b, t = 200, 12
    covered = 0
    slopes = []
    for seed in range(400):
        g = torch.Generator().manual_seed(seed)
        lengths = torch.randint(1, t + 1, (b,), generator=g)
        mask = torch.arange(t).unsqueeze(0) < lengths.unsqueeze(1)
        lengths_f = lengths.to(torch.float64)
        noise = torch.randn(b, generator=g, dtype=torch.float64) * (0.04 * lengths_f)
        target = true_intercept + true_slope * lengths_f + noise
        # SEQ_MEAN_TOKEN_SUM gives y_i = |a_i|·L_i/B for [B] advantages, so
        # a_i = B·target_i/L_i makes the probe's regressand equal target_i exactly
        # up to fp rounding (target > 0 with wide margin here).
        adv = b * target / lengths_f
        report = length_bias_probe(adv, mask, agg_mode=Aggregation.SEQ_MEAN_TOKEN_SUM)
        slopes.append(report.slope)
        covered += int(report.ci_low <= true_slope <= report.ci_high)
    coverage = covered / 400
    assert 0.92 <= coverage <= 0.98
    assert sum(slopes) / len(slopes) == pytest.approx(true_slope, abs=2e-3)


def test_mode_induced_weights_and_structural_slopes() -> None:
    """docs/diagnostics/length_bias.md analytic check via effective_token_weights: with lengths
    [1, 2, 3, 3] (B=4, N=9), SEQ_MEAN_TOKEN_MEAN weights are m/(B·L_i) (per-token
    weight ∝ 1/L_i, so |A| ≡ 1 gives constant y_i = 1/B and slope ≈ 0), while
    TOKEN_SUM_NORM weights are the constant m/(B·norm_len) (so y_i = L_i/(B·norm_len)
    and the structural slope is exactly the per-token weight); TOKEN_MEAN and
    SEQ_MEAN_TOKEN_SUM give structural slopes 1/N and 1/B
    (docs/diagnostics/length_bias.md mode table)."""
    mask = torch.tensor(
        [
            [True, False, False],
            [True, True, False],
            [True, True, True],
            [True, True, True],
        ]
    )
    lengths = mask.sum(dim=1).to(torch.float64)
    ones = torch.ones((4, 3), dtype=torch.float64)

    w_smtm = effective_token_weights(mask, Aggregation.SEQ_MEAN_TOKEN_MEAN)
    assert torch.equal(w_smtm, mask.to(torch.float64) / (4.0 * lengths.unsqueeze(1)))
    r_smtm = length_bias_probe(ones, mask, agg_mode=Aggregation.SEQ_MEAN_TOKEN_MEAN)
    assert torch.allclose(
        r_smtm.per_seq_weight_sum, torch.full((4,), 0.25, dtype=torch.float64), rtol=1e-12
    )
    assert abs(r_smtm.slope) < 1e-12

    w_tsn = effective_token_weights(mask, Aggregation.TOKEN_SUM_NORM, norm_len=5)
    assert torch.equal(w_tsn, mask.to(torch.float64) / 20.0)
    r_tsn = length_bias_probe(ones, mask, agg_mode=Aggregation.TOKEN_SUM_NORM, norm_len=5)
    assert r_tsn.slope == pytest.approx(1.0 / 20.0, rel=1e-12)
    assert r_tsn.slope_se == pytest.approx(0.0, abs=1e-12)
    assert r_tsn.intercept == pytest.approx(0.0, abs=1e-12)
    assert torch.allclose(r_tsn.per_seq_weight_sum, lengths / 20.0, rtol=1e-12)

    r_tm = length_bias_probe(ones, mask, agg_mode=Aggregation.TOKEN_MEAN)
    assert r_tm.slope == pytest.approx(1.0 / 9.0, rel=1e-12)
    r_smts = length_bias_probe(ones, mask, agg_mode=Aggregation.SEQ_MEAN_TOKEN_SUM)
    assert r_smts.slope == pytest.approx(1.0 / 4.0, rel=1e-12)


@given(case=probe_cases(), mode=st.sampled_from(_MODES))
def test_mask_invariance(
    case: tuple[torch.Tensor, torch.Tensor], mode: tuple[Aggregation, int | None]
) -> None:
    """Masked advantage values never affect the report (docs/conventions.md): every
    field is bitwise-equal after perturbing masked positions."""
    adv, mask = case
    agg_mode, norm_len = mode
    base = length_bias_probe(adv, mask, agg_mode=agg_mode, norm_len=norm_len)
    perturbed_adv = torch.where(mask, adv, torch.full_like(adv, -55.25))
    perturbed = length_bias_probe(perturbed_adv, mask, agg_mode=agg_mode, norm_len=norm_len)
    _assert_reports_equal(base, perturbed)


@given(mask=valid_masks(), mode=st.sampled_from(_MODES))
def test_seq_advantages_equal_expanded_token_advantages(
    mask: torch.Tensor, mode: tuple[Aggregation, int | None]
) -> None:
    """[B] advantages are broadcast across the row's tokens: the report is identical to
    passing the explicitly expanded [B, T] tensor (docs/diagnostics/length_bias.md)."""
    agg_mode, norm_len = mode
    b, t = mask.shape
    g = torch.Generator().manual_seed(b * 100 + t)
    adv_seq = torch.randn(b, generator=g, dtype=torch.float64)
    from_seq = length_bias_probe(adv_seq, mask, agg_mode=agg_mode, norm_len=norm_len)
    expanded = adv_seq.unsqueeze(1).expand(b, t)
    from_tok = length_bias_probe(expanded, mask, agg_mode=agg_mode, norm_len=norm_len)
    _assert_reports_equal(from_seq, from_tok)


@given(case=probe_cases(), mode=st.sampled_from(_MODES))
def test_report_per_seq_fields_match_recomputation(
    case: tuple[torch.Tensor, torch.Tensor], mode: tuple[Aggregation, int | None]
) -> None:
    """per_seq_length is Σ_t m (int64) and per_seq_weight_sum is the row sum of
    effective_token_weights, bitwise (docs/diagnostics/length_bias.md)."""
    adv, mask = case
    agg_mode, norm_len = mode
    report = length_bias_probe(adv, mask, agg_mode=agg_mode, norm_len=norm_len)
    assert report.per_seq_length.dtype == torch.int64
    assert torch.equal(report.per_seq_length, mask.sum(dim=1))
    weights = effective_token_weights(mask, agg_mode, norm_len=norm_len)
    assert torch.equal(report.per_seq_weight_sum, weights.sum(dim=1))
    assert report.n == mask.shape[0]


@given(case=probe_cases(), mode=st.sampled_from(_MODES))
def test_doubling_advantages_doubles_the_fit(
    case: tuple[torch.Tensor, torch.Tensor], mode: tuple[Aggregation, int | None]
) -> None:
    """Scale equivariance: y is linear in |A|, so scaling advantages by 2 scales slope,
    se, intercept, and CI endpoints by 2 — exactly, since scaling by a power of two is
    lossless in floating point (docs/diagnostics/length_bias.md)."""
    adv, mask = case
    agg_mode, norm_len = mode
    base = length_bias_probe(adv, mask, agg_mode=agg_mode, norm_len=norm_len)
    doubled = length_bias_probe(2.0 * adv, mask, agg_mode=agg_mode, norm_len=norm_len)
    assert doubled.slope == 2.0 * base.slope
    assert doubled.slope_se == 2.0 * base.slope_se
    assert doubled.intercept == 2.0 * base.intercept
    assert doubled.ci_low == 2.0 * base.ci_low
    assert doubled.ci_high == 2.0 * base.ci_high


def test_degenerate_inputs_raise_value_errors() -> None:
    """docs/diagnostics/length_bias.md (degenerate inputs) / docs/conventions.md:
    degenerate inputs raise ValueError naming the argument, never NaN."""
    mask = torch.tensor(
        [
            [True, False, False],
            [True, True, False],
            [True, True, True],
            [True, True, True],
        ]
    )
    ones = torch.ones((4, 3), dtype=torch.float64)

    with pytest.raises(ValueError, match=r"at least 3 sequences.*got B=2"):
        length_bias_probe(
            torch.ones((2, 2), dtype=torch.float64),
            torch.tensor([[True, False], [True, True]]),
            agg_mode=Aggregation.TOKEN_MEAN,
        )
    with pytest.raises(ValueError, match=r"per_seq_length is constant"):
        length_bias_probe(
            torch.ones((3, 2), dtype=torch.float64),
            torch.ones((3, 2), dtype=torch.bool),
            agg_mode=Aggregation.TOKEN_MEAN,
        )
    # Lengths [1, 1, 1, 3]: the single distinct row has hat value h = 1 exactly
    # (docs/diagnostics/length_bias.md), so HC3's 1/(1-h)² is undefined.
    leverage_mask = torch.tensor(
        [
            [True, False, False],
            [True, False, False],
            [True, False, False],
            [True, True, True],
        ]
    )
    with pytest.raises(ValueError, match=r"leverage h = 1"):
        length_bias_probe(ones, leverage_mask, agg_mode=Aggregation.TOKEN_MEAN)
    with pytest.raises(ValueError, match=r"advantages must be \[B\] or \[B, T\]"):
        length_bias_probe(torch.ones((4, 3, 1)), mask, agg_mode=Aggregation.TOKEN_MEAN)
    with pytest.raises(ValueError, match=r"advantages must be \[B\] or \[B, T\]"):
        length_bias_probe(torch.ones((4, 2)), mask, agg_mode=Aggregation.TOKEN_MEAN)
    with pytest.raises(ValueError, match=r"advantages has shape \(5,\).*B=4"):
        length_bias_probe(torch.ones(5), mask, agg_mode=Aggregation.TOKEN_MEAN)
    with pytest.raises(ValueError, match=r"dtype torch\.bool"):
        length_bias_probe(ones, mask.to(torch.float64), agg_mode=Aggregation.TOKEN_MEAN)
    with pytest.raises(ValueError, match=r"zero response tokens"):
        bad_mask = mask.clone()
        bad_mask[0, 0] = False
        length_bias_probe(ones, bad_mask, agg_mode=Aggregation.TOKEN_MEAN)
    bad_adv = ones.clone()
    bad_adv[2, 1] = float("nan")
    with pytest.raises(ValueError, match=r"non-finite"):
        length_bias_probe(bad_adv, mask, agg_mode=Aggregation.TOKEN_MEAN)


def test_norm_len_required_at_call_time_for_token_sum_norm() -> None:
    """TOKEN_SUM_NORM without norm_len raises at call time, propagated from
    effective_token_weights (docs/derivations/aggregation.md)."""
    mask = torch.tensor([[True, False], [True, False], [True, True], [True, True]])
    adv = torch.ones((4, 2), dtype=torch.float64)
    with pytest.raises(ValueError, match=r"norm_len is required"):
        length_bias_probe(adv, mask, agg_mode=Aggregation.TOKEN_SUM_NORM)
    report = length_bias_probe(adv, mask, agg_mode=Aggregation.TOKEN_SUM_NORM, norm_len=4)
    assert report.n == 4


def test_masked_positions_may_hold_nonfinite_junk() -> None:
    """Finiteness is enforced at response positions only; masked junk may be anything
    (docs/conventions.md)."""
    mask = torch.tensor([[True, False], [True, False], [True, True], [True, True]])
    adv = torch.tensor(
        [[1.0, float("inf")], [1.5, float("nan")], [2.0, 2.0], [3.0, 3.0]], dtype=torch.float64
    )
    report = length_bias_probe(adv, mask, agg_mode=Aggregation.TOKEN_MEAN)
    assert math.isfinite(report.slope)


def test_probe_detaches_from_autograd() -> None:
    """The probe consumes detached advantages: it accepts a requires_grad tensor and
    returns tensors outside any autograd graph (diagnostics never feed gradients)."""
    mask = torch.tensor([[True, False], [True, False], [True, True], [True, True]])
    adv = torch.ones((4, 2), dtype=torch.float64, requires_grad=True)
    report = length_bias_probe(adv, mask, agg_mode=Aggregation.TOKEN_MEAN)
    assert not report.per_seq_weight_sum.requires_grad
    assert not report.per_seq_length.requires_grad
    assert adv.grad is None


def test_summary_is_compact_multiline() -> None:
    """summary() is a compact human-readable multi-line string."""
    report = _golden_case()
    text = report.summary()
    assert isinstance(text, str)
    assert text.count("\n") == 1
    assert "slope=1.5" in text
    assert "n=3" in text
    assert "HC3" in text
