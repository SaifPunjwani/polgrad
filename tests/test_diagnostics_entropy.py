"""Enforces docs/diagnostics/entropy.md: the sampled-token MC entropy estimator, the
exact per-token entropy path, the Theil-Sen slope, CUSUM changepoint localization, the
permutation calibration of the false-positive rate, determinism, and validation
errors."""

from __future__ import annotations

import math

import pytest
import torch
from hypothesis import given
from strategies import logprob_batches, padded_masks

from polgrad.diagnostics.entropy import (
    EntropyReport,
    TrendReport,
    entropy_trend,
    token_entropy_estimate,
)


def _perturb_masked(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Replace masked positions with a different junk value than the strategy used."""
    return torch.where(mask, x, torch.full_like(x, -55.25))


def test_token_entropy_golden_case() -> None:
    """Hand-derived (docs/diagnostics/entropy.md): logprobs [-1, -2 | -4] give pooled
    estimate 7/3 and per-sequence entropies [1.5, 4.0]."""
    logprobs = torch.tensor([[-1.0, -2.0], [-4.0, 123.0]], dtype=torch.float64)
    mask = torch.tensor([[True, True], [True, False]])
    report = token_entropy_estimate(logprobs, mask)
    assert report.n_tokens == 3
    assert report.entropy_estimate == pytest.approx(7.0 / 3.0, rel=1e-12)
    assert torch.equal(report.per_seq_entropy, torch.tensor([1.5, 4.0], dtype=torch.float64))


@given(batch=logprob_batches())
def test_pooled_entropy_is_length_weighted_mean_of_per_seq(batch) -> None:  # type: ignore[no-untyped-def]
    """entropy_estimate == Σ_b L_b·per_seq_b / Σ_b L_b, i.e. pooled and per-sequence
    views are consistent (docs/diagnostics/entropy.md)."""
    report = token_entropy_estimate(batch.logprobs, batch.response_mask)
    lengths = batch.response_mask.sum(dim=1).to(torch.float64)
    weighted = float((lengths * report.per_seq_entropy).sum() / lengths.sum())
    assert report.entropy_estimate == pytest.approx(weighted, rel=1e-9, abs=1e-12)
    assert report.n_tokens == int(lengths.sum())


@given(batch=logprob_batches())
def test_entropy_estimate_matches_negative_masked_mean(batch) -> None:  # type: ignore[no-untyped-def]
    """entropy_estimate == -masked-mean(logprobs), recomputed independently."""
    report = token_entropy_estimate(batch.logprobs, batch.response_mask)
    values = [
        -float(batch.logprobs[b, t])
        for b in range(batch.response_mask.shape[0])
        for t in range(batch.response_mask.shape[1])
        if bool(batch.response_mask[b, t])
    ]
    assert report.entropy_estimate == pytest.approx(sum(values) / len(values), rel=1e-9)


@given(batch=logprob_batches())
def test_mask_invariance_token_entropy(batch) -> None:  # type: ignore[no-untyped-def]
    """Masked inputs never affect the report (docs/conventions.md): bitwise equality
    after perturbing masked positions."""
    base = token_entropy_estimate(batch.logprobs, batch.response_mask)
    perturbed = token_entropy_estimate(
        _perturb_masked(batch.logprobs, batch.response_mask), batch.response_mask
    )
    assert base.n_tokens == perturbed.n_tokens
    assert base.entropy_estimate == perturbed.entropy_estimate
    assert torch.equal(base.per_seq_entropy, perturbed.per_seq_entropy)


@given(mask=padded_masks(max_b=4, max_t=6))
def test_per_seq_entropy_dtype_preserved(mask: torch.Tensor) -> None:
    """per_seq_entropy keeps the input dtype (docs/conventions.md, no silent casts)."""
    for dtype in (torch.float32, torch.float64):
        logprobs = -torch.rand(mask.shape, generator=torch.Generator().manual_seed(1)).to(dtype)
        report = token_entropy_estimate(logprobs, mask)
        assert report.per_seq_entropy.dtype == dtype
        assert report.per_seq_entropy.shape == (mask.shape[0],)


def test_token_entropy_exact_golden_case() -> None:
    """Hand-derived (docs/diagnostics/entropy.md): a uniform distribution over K arms
    has H = log K exactly, so exact entropies [log 2, log 4 | log 3] pool to
    (log 2 + log 4 + log 3)/3 with per-sequence means [(log 2 + log 4)/2, log 3];
    negative junk at the masked position must not trip validation."""
    entropies = torch.tensor(
        [[math.log(2.0), math.log(4.0)], [math.log(3.0), -77.0]], dtype=torch.float64
    )
    mask = torch.tensor([[True, True], [True, False]])
    report = token_entropy_estimate(None, mask, entropies=entropies)
    assert report.estimator == "exact"
    assert report.n_tokens == 3
    expected_pooled = (math.log(2.0) + math.log(4.0) + math.log(3.0)) / 3.0
    assert report.entropy_estimate == pytest.approx(expected_pooled, rel=1e-12)
    expected_per_seq = torch.tensor(
        [(math.log(2.0) + math.log(4.0)) / 2.0, math.log(3.0)], dtype=torch.float64
    )
    assert torch.equal(report.per_seq_entropy, expected_per_seq)


def test_exact_and_mc_agree_on_policy_within_clt_bound() -> None:
    """On-policy the MC path and the exact path estimate the same quantity: sampling
    N tokens from a known categorical p, the pooled -masked-mean(logprobs) has mean
    H(p) and standard error sqrt(Var[-log p(Y)]/N), so it must agree with the analytic
    exact-path value within 4 standard errors (docs/diagnostics/entropy.md)."""
    p = torch.tensor([0.7, 0.2, 0.1], dtype=torch.float64)
    b, t = 64, 32
    n_tokens = b * t
    g = torch.Generator().manual_seed(2024)
    samples = torch.multinomial(p, n_tokens, replacement=True, generator=g).reshape(b, t)
    logprobs = torch.log(p)[samples]
    mask = torch.ones(b, t, dtype=torch.bool)
    entropy = float(-(p * torch.log(p)).sum())
    variance = float((p * torch.log(p) ** 2).sum()) - entropy**2
    mc = token_entropy_estimate(logprobs, mask)
    exact = token_entropy_estimate(
        None, mask, entropies=torch.full((b, t), entropy, dtype=torch.float64)
    )
    assert mc.estimator == "mc_cross_entropy"
    assert exact.estimator == "exact"
    assert exact.entropy_estimate == pytest.approx(entropy, rel=1e-12)
    assert torch.allclose(exact.per_seq_entropy, torch.full((b,), entropy, dtype=torch.float64))
    clt_bound = 4.0 * math.sqrt(variance / n_tokens)
    assert abs(mc.entropy_estimate - exact.entropy_estimate) <= clt_bound


@given(mask=padded_masks())
def test_mask_invariance_exact_path(mask: torch.Tensor) -> None:
    """Masked entropies never affect the exact-path report and never trip validation,
    even when the junk is negative (docs/conventions.md): bitwise equality after
    perturbing masked positions."""
    values = 3.0 * torch.rand(
        mask.shape, generator=torch.Generator().manual_seed(17), dtype=torch.float64
    )
    entropies = torch.where(mask, values, torch.full_like(values, 123.0))
    base = token_entropy_estimate(None, mask, entropies=entropies)
    perturbed = token_entropy_estimate(None, mask, entropies=_perturb_masked(entropies, mask))
    assert base.estimator == perturbed.estimator == "exact"
    assert base.n_tokens == perturbed.n_tokens
    assert base.entropy_estimate == perturbed.entropy_estimate
    assert torch.equal(base.per_seq_entropy, perturbed.per_seq_entropy)


def test_estimator_field_on_both_paths() -> None:
    """estimator is "mc_cross_entropy" without entropies and "exact" with them, both
    surfaced in summary(); when both inputs are given, entropies win and logprobs is
    ignored (docs/diagnostics/entropy.md)."""
    logprobs = torch.tensor([[-1.0, -2.0], [-4.0, 123.0]], dtype=torch.float64)
    entropies = torch.tensor([[0.5, 1.5], [2.5, -9.0]], dtype=torch.float64)
    mask = torch.tensor([[True, True], [True, False]])
    mc = token_entropy_estimate(logprobs, mask)
    assert mc.estimator == "mc_cross_entropy"
    assert "estimator=mc_cross_entropy" in mc.summary()
    exact = token_entropy_estimate(None, mask, entropies=entropies)
    assert exact.estimator == "exact"
    assert "estimator=exact" in exact.summary()
    both = token_entropy_estimate(logprobs, mask, entropies=entropies)
    assert both.estimator == "exact"
    assert both.entropy_estimate == exact.entropy_estimate
    assert torch.equal(both.per_seq_entropy, exact.per_seq_entropy)


def test_exact_entropy_validation_errors() -> None:
    """Both inputs None, logprobs None without entropies, non-2-D or badly masked
    entropies, and negative or non-finite entropies at response positions raise
    ValueError naming the argument (docs/conventions.md); negative junk at masked
    positions must not raise."""
    mask = torch.ones(2, 3, dtype=torch.bool)
    entropies = torch.full((2, 3), 0.5, dtype=torch.float64)
    with pytest.raises(ValueError, match=r"logprobs must be provided when entropies is None"):
        token_entropy_estimate(None, mask)
    with pytest.raises(ValueError, match=r"logprobs must be provided when entropies is None"):
        token_entropy_estimate(None, mask, entropies=None)
    with pytest.raises(ValueError, match=r"entropies must be 2-D"):
        token_entropy_estimate(None, mask, entropies=torch.zeros(3))
    with pytest.raises(ValueError, match=r"dtype torch\.bool"):
        token_entropy_estimate(None, torch.ones(2, 3), entropies=entropies)
    negative = entropies.clone()
    negative[0, 1] = -1e-6
    with pytest.raises(ValueError, match=r"entropies must be >= 0 at response positions"):
        token_entropy_estimate(None, mask, entropies=negative)
    nonfinite = entropies.clone()
    nonfinite[1, 2] = float("inf")
    with pytest.raises(ValueError, match=r"entropies \(response positions\) contains non-finite"):
        token_entropy_estimate(None, mask, entropies=nonfinite)
    partial_mask = torch.tensor([[True, True, False], [True, True, True]])
    masked_junk = entropies.clone()
    masked_junk[0, 2] = -55.25
    report = token_entropy_estimate(None, partial_mask, entropies=masked_junk)
    assert report.n_tokens == 5


def test_entropy_validation_errors() -> None:
    """Invalid shapes, masks, and non-finite response values raise ValueError naming
    the argument (docs/conventions.md)."""
    logprobs = torch.zeros(2, 3, dtype=torch.float64)
    with pytest.raises(ValueError, match=r"logprobs must be 2-D"):
        token_entropy_estimate(torch.zeros(3), torch.ones(3, dtype=torch.bool))
    with pytest.raises(ValueError, match=r"dtype torch\.bool"):
        token_entropy_estimate(logprobs, torch.ones(2, 3))
    with pytest.raises(ValueError, match=r"zero response tokens"):
        token_entropy_estimate(logprobs, torch.tensor([[True, True, True], [False, False, False]]))
    bad = logprobs.clone()
    bad[1, 2] = float("inf")
    with pytest.raises(ValueError, match=r"non-finite"):
        token_entropy_estimate(bad, torch.ones(2, 3, dtype=torch.bool))


def test_entropy_summary_is_compact_multiline() -> None:
    """summary() is a compact human-readable multi-line string."""
    logprobs = torch.tensor([[-1.0, -2.0], [-4.0, 123.0]], dtype=torch.float64)
    mask = torch.tensor([[True, True], [True, False]])
    report = token_entropy_estimate(logprobs, mask)
    text = report.summary()
    assert isinstance(report, EntropyReport)
    assert text.count("\n") == 1
    assert "n_tokens=3" in text and "per-sequence" in text


def test_theil_sen_recovers_exact_linear_slope(gen: torch.Generator) -> None:
    """On an exactly linear window every pairwise slope equals the true slope, so the
    Theil-Sen median recovers it (docs/diagnostics/entropy.md)."""
    steps = torch.arange(24, dtype=torch.float64)
    series = 5.0 - 0.1 * steps
    report = entropy_trend(series, window=24, n_perm=99, alpha=0.05, generator=gen)
    assert report.slope == pytest.approx(-0.1, rel=1e-9)


def test_theil_sen_robust_to_minority_corruption(gen: torch.Generator) -> None:
    """Corrupting 3 of 30 points leaves a minority of corrupted pairs, so the median
    pairwise slope stays near the true slope (docs/diagnostics/entropy.md)."""
    steps = torch.arange(30, dtype=torch.float64)
    series = 5.0 - 0.1 * steps
    series[3] += 5.0
    series[11] -= 5.0
    series[22] += 5.0
    report = entropy_trend(series, window=30, n_perm=99, alpha=0.05, generator=gen)
    assert report.slope == pytest.approx(-0.1, abs=0.05)


def test_cusum_detects_mean_shift_and_localizes_changepoint(gen: torch.Generator) -> None:
    """A level shift from 3.0 to 1.0 after global step 29 is detected and the CUSUM
    argmax reports 29 as the last pre-change step, for two window sizes
    (docs/diagnostics/entropy.md)."""
    noise_gen = torch.Generator().manual_seed(42)
    series = torch.cat([torch.full((30,), 3.0), torch.full((30,), 1.0)]).to(torch.float64)
    series = series + 1e-3 * torch.randn(60, generator=noise_gen, dtype=torch.float64)
    for window in (60, 40):
        report = entropy_trend(series, window=window, n_perm=199, alpha=0.05, generator=gen)
        assert report.cusum_stat > report.threshold
        assert report.changepoint_index == 29


def test_no_changepoint_iff_stat_at_most_threshold(gen: torch.Generator) -> None:
    """changepoint_index is None exactly when cusum_stat <= threshold, across null and
    shifted seeded series."""
    data_gen = torch.Generator().manual_seed(7)
    for shift in (0.0, 0.2, 0.5, 1.0, 3.0):
        series = torch.randn(40, generator=data_gen, dtype=torch.float64)
        series[20:] += shift
        report = entropy_trend(series, window=40, n_perm=199, alpha=0.05, generator=gen)
        detected = report.changepoint_index is not None
        assert detected == (report.cusum_stat > report.threshold)
        if detected:
            assert 0 <= report.changepoint_index < 40


def test_trend_false_positive_rate_calibrated() -> None:
    """MC calibration (docs/diagnostics/entropy.md): under an iid Gaussian null the
    changepoint FPR equals m/(n_perm+1) = alpha; verified on 500 seeded runs within
    4 binomial standard errors."""
    alpha, n_perm, window, runs = 0.05, 199, 48, 500
    g = torch.Generator().manual_seed(1234)
    hits = 0
    for _ in range(runs):
        series = torch.randn(window, generator=g, dtype=torch.float64)
        report = entropy_trend(series, window=window, n_perm=n_perm, alpha=alpha, generator=g)
        hits += report.changepoint_index is not None
    fpr = hits / runs
    stderr = math.sqrt(alpha * (1 - alpha) / runs)
    assert abs(fpr - alpha) <= 4 * stderr


def test_trend_constant_series_reports_no_changepoint(gen: torch.Generator) -> None:
    """A constant window has zero slope and zero CUSUM everywhere, so no changepoint is
    reported and no NaN appears."""
    series = torch.full((16,), 2.5, dtype=torch.float64)
    report = entropy_trend(series, window=16, n_perm=99, alpha=0.05, generator=gen)
    assert report.slope == 0.0
    assert report.cusum_stat == 0.0
    assert report.changepoint_index is None


def test_trend_determinism_same_seed() -> None:
    """The same generator seed yields an identical TrendReport (explicit-RNG rule of
    docs/conventions.md)."""
    series = torch.randn(32, generator=torch.Generator().manual_seed(3), dtype=torch.float64)
    report_a = entropy_trend(
        series, window=32, n_perm=199, alpha=0.05, generator=torch.Generator().manual_seed(9)
    )
    report_b = entropy_trend(
        series, window=32, n_perm=199, alpha=0.05, generator=torch.Generator().manual_seed(9)
    )
    assert report_a == report_b  # TrendReport holds only scalars; dataclass eq is bitwise


def test_trend_window_uses_trailing_values() -> None:
    """Only the trailing `window` values enter the statistics: rewriting the prefix
    leaves the report unchanged (same generator seed)."""
    series = torch.randn(30, generator=torch.Generator().manual_seed(5), dtype=torch.float64)
    altered = series.clone()
    altered[:20] = 99.0
    report_a = entropy_trend(
        series, window=10, n_perm=99, alpha=0.05, generator=torch.Generator().manual_seed(11)
    )
    report_b = entropy_trend(
        altered, window=10, n_perm=99, alpha=0.05, generator=torch.Generator().manual_seed(11)
    )
    assert report_a == report_b


def test_trend_validation_errors(gen: torch.Generator) -> None:
    """Window bounds, alpha range, n_perm, the alpha-too-small calibration rule, and
    input shape/finiteness raise ValueError (docs/diagnostics/entropy.md, masking and
    validation)."""
    series = torch.zeros(10, dtype=torch.float64)
    with pytest.raises(ValueError, match=r"entropy_per_step must be 1-D"):
        entropy_trend(torch.zeros(2, 5), window=4, generator=gen)
    with pytest.raises(ValueError, match=r"non-finite"):
        entropy_trend(torch.tensor([0.0, float("nan")]), window=2, generator=gen)
    with pytest.raises(ValueError, match=r"window must be >= 2"):
        entropy_trend(series, window=1, generator=gen)
    with pytest.raises(ValueError, match=r"window must be <= len\(entropy_per_step\)=10"):
        entropy_trend(series, window=11, generator=gen)
    with pytest.raises(ValueError, match=r"n_perm must be >= 1"):
        entropy_trend(series, window=4, n_perm=0, generator=gen)
    with pytest.raises(ValueError, match=r"alpha must be in \(0, 1\)"):
        entropy_trend(series, window=4, alpha=0.0, generator=gen)
    with pytest.raises(ValueError, match=r"alpha must be in \(0, 1\)"):
        entropy_trend(series, window=4, alpha=1.5, generator=gen)
    with pytest.raises(ValueError, match=r"too small for n_perm"):
        entropy_trend(series, window=4, n_perm=99, alpha=0.001, generator=gen)


def test_trend_summary_is_compact_multiline(gen: torch.Generator) -> None:
    """summary() is a compact human-readable multi-line string."""
    series = torch.randn(20, generator=torch.Generator().manual_seed(2), dtype=torch.float64)
    report = entropy_trend(series, window=20, n_perm=99, alpha=0.05, generator=gen)
    text = report.summary()
    assert isinstance(report, TrendReport)
    assert text.count("\n") == 2
    assert "slope" in text and "CUSUM" in text and "changepoint" in text
