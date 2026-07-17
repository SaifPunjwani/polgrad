"""Enforces docs/diagnostics/clipping.md: quadrant clip fractions on constructed and
generic cases, the killed-gradient census verified against autograd of
polgrad.losses.policy_loss, the dual-clip pathology demonstration, mask invariance,
and validation errors."""

from __future__ import annotations

import math
from typing import NamedTuple

import pytest
import torch
from hypothesis import assume, given
from hypothesis import strategies as st
from strategies import MASKED_JUNK, padded_masks

from polgrad.aggregate import Aggregation
from polgrad.diagnostics.clipping import ClipReport, clip_report
from polgrad.losses import ClipConfig, PolicyLossConfig, RatioKind, SurrogateKind, policy_loss

_FRAC_FIELDS = (
    "frac_pos_adv_clipped_high",
    "frac_pos_adv_clipped_low",
    "frac_neg_adv_clipped_high",
    "frac_neg_adv_clipped_low",
)


class ClipCase(NamedTuple):
    """A generated batch plus the ClipConfig it is diagnosed under (float64)."""

    logprobs: torch.Tensor
    old_logprobs: torch.Tensor
    advantages: torch.Tensor
    response_mask: torch.Tensor
    clip: ClipConfig


@st.composite
def clip_cases(draw: st.DrawFn, *, max_gap: float = 2.0) -> ClipCase:
    """Like strategies.logprob_batches but a ClipCase: only the old-logprob and
    advantage streams the clip diagnostics need, bundled with a drawn ClipConfig
    (eps_low, eps_high, optional dual-clip cap)."""
    mask = draw(padded_masks())
    b, t = mask.shape

    def fill(low: float, high: float) -> torch.Tensor:
        vals = [
            draw(st.floats(low, high, allow_nan=False, allow_infinity=False)) for _ in range(b * t)
        ]
        out = torch.tensor(vals, dtype=torch.float64).reshape(b, t)
        return torch.where(mask, out, torch.full_like(out, MASKED_JUNK))

    logprobs = fill(-8.0, -0.05)
    gap = fill(-max_gap, max_gap)
    old_logprobs = torch.where(mask, logprobs + gap, torch.full_like(logprobs, MASKED_JUNK))
    advantages = fill(-3.0, 3.0)
    eps_low = draw(st.floats(0.05, 0.5, allow_nan=False))
    eps_high = draw(st.floats(0.05, 0.6, allow_nan=False))
    ratio_cap = draw(st.one_of(st.none(), st.floats(1.05, 4.0, allow_nan=False)))
    clip = ClipConfig(eps_low, eps_high, ratio_cap)
    return ClipCase(logprobs, old_logprobs, advantages, mask, clip)


def _used_ratio(case: ClipCase) -> torch.Tensor:
    """The token ratio policy_loss would use, with junk at masked positions."""
    zero = torch.zeros((), dtype=torch.float64)
    lp = torch.where(case.response_mask, case.logprobs, zero)
    olp = torch.where(case.response_mask, case.old_logprobs, zero)
    ratio = torch.exp(lp - olp)
    return torch.where(case.response_mask, ratio, torch.full_like(ratio, MASKED_JUNK))


def _assume_ratio_off_bounds(case: ClipCase, ratio: torch.Tensor) -> None:
    """Exclude the non-differentiable tie points where r sits exactly on a bound."""
    response_ratio = ratio[case.response_mask]
    assert case.clip.eps_low is not None and case.clip.eps_high is not None
    bounds = [1.0 - case.clip.eps_low, 1.0 + case.clip.eps_high]
    if case.clip.ratio_cap is not None:
        bounds.append(case.clip.ratio_cap)
    for bound in bounds:
        assume(float((response_ratio - bound).abs().min()) > 1e-6)


def _snap_tiny_advantages(advantages: torch.Tensor) -> torch.Tensor:
    """Snap |A| <= 1e-6 to exactly 0 (a valid tie-free input: A == 0 kills the
    gradient exactly on both sides, while a tiny nonzero A could make the branch
    comparison r·A vs bound·A collapse to a floating-point tie)."""
    zero = torch.zeros((), dtype=advantages.dtype)
    return torch.where(advantages.abs() <= 1e-6, zero, advantages)


def _constructed_case() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One token per quadrant of band (0.8, 1.3), plus a dual-clip hit and an A == 0.

    (row, t): (0,0) A=+1, r=1.5 pos-high; (0,1) A=+1, r=0.5 pos-low;
    (0,2) A=-1, r=0.5 neg-low; (0,3) A=-1, r=1.5 neg-high (below cap 3.0);
    (1,0) A=-1, r=3.5 neg-high and above cap; (1,1) A=0, r=1.0; (1,2), (1,3) masked
    junk (negative ratio junk: positivity is enforced at response positions only).
    """
    ratio = torch.tensor([[1.5, 0.5, 0.5, 1.5], [3.5, 1.0, -123.0, 123.0]], dtype=torch.float64)
    advantages = torch.tensor(
        [[1.0, 1.0, -1.0, -1.0], [-1.0, 0.0, 123.0, -123.0]], dtype=torch.float64
    )
    mask = torch.tensor([[True, True, True, True], [True, True, False, False]])
    return ratio, advantages, mask


def test_quadrant_fractions_constructed_case_dual_clip() -> None:
    """Hand-derived numbers of docs/diagnostics/clipping.md, N = 6 response tokens:
    pos-high 1/6, pos-low 1/6, neg-low 1/6, neg-high 2/6 (r = 1.5 and 3.5 both > 1.3);
    killed with cap 3.0 = {pos-high, neg-low, r=3.5 > cap, A=0} -> 4/6."""
    ratio, advantages, mask = _constructed_case()
    report = clip_report(ratio, advantages, mask, ClipConfig(0.2, 0.3, ratio_cap=3.0))
    assert report.eps_low == 0.2 and report.eps_high == 0.3
    assert report.frac_pos_adv_clipped_high == pytest.approx(1 / 6)
    assert report.frac_pos_adv_clipped_low == pytest.approx(1 / 6)
    assert report.frac_neg_adv_clipped_high == pytest.approx(2 / 6)
    assert report.frac_neg_adv_clipped_low == pytest.approx(1 / 6)
    assert report.gradient_killed_mask.dtype == torch.bool
    assert report.gradient_killed_mask.tolist() == [
        [True, False, True, False],
        [True, True, False, False],
    ]
    assert report.gradient_killed_frac == pytest.approx(4 / 6)


def test_quadrant_fractions_constructed_case_without_dual_clip() -> None:
    """Same tensors with ratio_cap=None: quadrant fractions are unchanged (they count
    band crossings, not killed gradients) but the r=3.5, A=-1 token now flows gradient,
    so killed drops to 3/6 — the PPO pathology (docs/diagnostics/clipping.md)."""
    ratio, advantages, mask = _constructed_case()
    report = clip_report(ratio, advantages, mask, ClipConfig(0.2, 0.3, ratio_cap=None))
    assert report.frac_pos_adv_clipped_high == pytest.approx(1 / 6)
    assert report.frac_pos_adv_clipped_low == pytest.approx(1 / 6)
    assert report.frac_neg_adv_clipped_high == pytest.approx(2 / 6)
    assert report.frac_neg_adv_clipped_low == pytest.approx(1 / 6)
    assert report.gradient_killed_mask.tolist() == [
        [True, False, True, False],
        [False, True, False, False],
    ]
    assert report.gradient_killed_frac == pytest.approx(3 / 6)


@given(case=clip_cases())
def test_quadrant_fractions_match_python_oracle(case: ClipCase) -> None:
    """Quadrant fractions, the killed mask, and gradient_killed_frac match a per-token
    Python oracle that transcribes the docs/diagnostics/clipping.md definitions
    directly."""
    ratio = _used_ratio(case)
    report = clip_report(ratio, case.advantages, case.response_mask, case.clip)
    assert case.clip.eps_low is not None and case.clip.eps_high is not None
    low, high = 1.0 - case.clip.eps_low, 1.0 + case.clip.eps_high
    cap = case.clip.ratio_cap
    counts = dict.fromkeys([*_FRAC_FIELDS, "killed"], 0)
    n = 0
    b_size, t_size = case.response_mask.shape
    for b in range(b_size):
        for t in range(t_size):
            if not bool(case.response_mask[b, t]):
                assert not bool(report.gradient_killed_mask[b, t])
                continue
            n += 1
            r, a = float(ratio[b, t]), float(case.advantages[b, t])
            counts["frac_pos_adv_clipped_high"] += a > 0 and r > high
            counts["frac_pos_adv_clipped_low"] += a > 0 and r < low
            counts["frac_neg_adv_clipped_high"] += a < 0 and r > high
            counts["frac_neg_adv_clipped_low"] += a < 0 and r < low
            killed = (
                (a > 0 and r > high)
                or (a < 0 and r < low)
                or (cap is not None and a < 0 and r > cap)
                or a == 0
            )
            counts["killed"] += killed
            assert bool(report.gradient_killed_mask[b, t]) == killed
    for field in _FRAC_FIELDS:
        assert getattr(report, field) == counts[field] / n
    assert report.gradient_killed_frac == counts["killed"] / n


@given(case=clip_cases())
def test_gradient_killed_matches_policy_loss_autograd(case: ClipCase) -> None:
    """gradient_killed_mask == (per-token autograd gradient of PG_CLIP policy_loss
    w.r.t. logprobs is exactly 0) & mask, on generic inputs with tie points excluded.
    Under TOKEN_MEAN every response token carries weight 1/N > 0, so the aggregate
    gradient at token t is 0 iff the per-token PG_CLIP gradient is 0
    (docs/diagnostics/clipping.md)."""
    _assume_ratio_off_bounds(case, _used_ratio(case))
    advantages = _snap_tiny_advantages(case.advantages)
    logprobs = case.logprobs.clone().requires_grad_(True)
    config = PolicyLossConfig(
        ratio=RatioKind.TOKEN,
        surrogate=SurrogateKind.PG_CLIP,
        clip=case.clip,
        aggregation=Aggregation.TOKEN_MEAN,
    )
    result = policy_loss(
        config,
        logprobs=logprobs,
        old_logprobs=case.old_logprobs,
        advantages=advantages,
        response_mask=case.response_mask,
    )
    (grad,) = torch.autograd.grad(result.loss, logprobs)
    report = clip_report(result.ratio, advantages, case.response_mask, case.clip)
    assert torch.equal(report.gradient_killed_mask, (grad == 0) & case.response_mask)


def test_pathology_neg_adv_high_ratio_flows_gradient_without_dual_clip() -> None:
    """The PPO pathology (docs/diagnostics/clipping.md): A=-1, r=2.0 above the band
    (0.8, 1.2) is reported clipped-high but NOT gradient-killed, and autograd confirms
    the per-token gradient is w·r·|A| = 2.0/2 = 1.0 — growing with r. Adding dual-clip
    cap c=1.5 < r kills it exactly."""
    old_logprobs = torch.zeros(1, 2, dtype=torch.float64)
    # r = exp(logprobs - 0) = [2.0, 1.1]: one token above the band, one inside it.
    target_ratio = torch.tensor([[2.0, 1.1]], dtype=torch.float64)
    advantages = torch.tensor([[-1.0, -1.0]], dtype=torch.float64)
    mask = torch.tensor([[True, True]])
    for ratio_cap, expect_killed in ((None, False), (1.5, True)):
        clip = ClipConfig(0.2, 0.2, ratio_cap=ratio_cap)
        logprobs = torch.log(target_ratio).requires_grad_(True)
        config = PolicyLossConfig(
            ratio=RatioKind.TOKEN,
            surrogate=SurrogateKind.PG_CLIP,
            clip=clip,
            aggregation=Aggregation.TOKEN_MEAN,
        )
        result = policy_loss(
            config,
            logprobs=logprobs,
            old_logprobs=old_logprobs,
            advantages=advantages,
            response_mask=mask,
        )
        (grad,) = torch.autograd.grad(result.loss, logprobs)
        report = clip_report(result.ratio, advantages, mask, clip)
        assert report.frac_neg_adv_clipped_high == pytest.approx(1 / 2)
        assert bool(report.gradient_killed_mask[0, 0]) is expect_killed
        assert not bool(report.gradient_killed_mask[0, 1])
        if expect_killed:
            assert float(grad[0, 0]) == 0.0
        else:
            # loss = mean(-min(rA, clip(r)A)) picks rA = -r; d(-rA)/dlogprob = r|A|.
            assert float(grad[0, 0]) == pytest.approx(2.0 / 2)
        assert float(grad[0, 1]) == pytest.approx(1.1 / 2)


def test_identical_policies_nothing_clipped_killed_only_at_zero_advantage() -> None:
    """Null calibration: identical policies give r = 1 inside any band, so every
    quadrant fraction is 0 and the gradient is killed exactly where A == 0."""
    ratio = torch.ones(2, 3, dtype=torch.float64)
    advantages = torch.tensor([[1.5, 0.0, -2.0], [-0.5, 3.0, 123.0]], dtype=torch.float64)
    mask = torch.tensor([[True, True, True], [True, True, False]])
    report = clip_report(ratio, advantages, mask, ClipConfig(0.2, 0.3, ratio_cap=2.0))
    for field in _FRAC_FIELDS:
        assert getattr(report, field) == 0.0
    assert report.gradient_killed_mask.tolist() == [
        [False, True, False],
        [False, False, False],
    ]
    assert report.gradient_killed_frac == pytest.approx(1 / 5)


def test_zero_advantages_kill_every_response_token() -> None:
    """A == 0 everywhere: quadrant fractions are 0 (no advantage sign) and the
    gradient is killed on every response token, none elsewhere."""
    ratio = torch.tensor([[2.0, 0.1], [1.0, 123.0]], dtype=torch.float64)
    advantages = torch.zeros(2, 2, dtype=torch.float64)
    mask = torch.tensor([[True, True], [True, False]])
    report = clip_report(ratio, advantages, mask, ClipConfig(0.2, 0.2))
    for field in _FRAC_FIELDS:
        assert getattr(report, field) == 0.0
    assert torch.equal(report.gradient_killed_mask, mask)
    assert report.gradient_killed_frac == 1.0


def test_sequence_advantages_broadcast_matches_explicit_broadcast() -> None:
    """[B] advantages behave exactly like the same values explicitly expanded to
    [B, T] (the docs/derivations/losses.md broadcast semantics, mirrored by
    clip_report)."""
    ratio = torch.tensor([[1.5, 0.9], [0.5, -123.0]], dtype=torch.float64)
    per_seq = torch.tensor([2.0, -1.0], dtype=torch.float64)
    mask = torch.tensor([[True, True], [True, False]])
    clip = ClipConfig(0.2, 0.3, ratio_cap=2.5)
    from_seq = clip_report(ratio, per_seq, mask, clip)
    from_tokens = clip_report(ratio, per_seq.unsqueeze(1).expand(2, 2), mask, clip)
    for field in (*_FRAC_FIELDS, "eps_low", "eps_high", "gradient_killed_frac"):
        assert getattr(from_seq, field) == getattr(from_tokens, field)
    assert torch.equal(from_seq.gradient_killed_mask, from_tokens.gradient_killed_mask)


@given(case=clip_cases())
def test_mask_invariance_clipping(case: ClipCase) -> None:
    """Masked inputs never affect the report (docs/conventions.md): every field is
    bitwise-equal after perturbing masked positions (including a negative masked
    ratio), and gradient_killed_mask is False at masked positions."""
    mask = case.response_mask
    ratio = _used_ratio(case)
    base = clip_report(ratio, case.advantages, mask, case.clip)
    perturbed = clip_report(
        torch.where(mask, ratio, torch.full_like(ratio, -7.25)),
        torch.where(mask, case.advantages, torch.full_like(case.advantages, 55.0)),
        mask,
        case.clip,
    )
    for field in (*_FRAC_FIELDS, "eps_low", "eps_high", "gradient_killed_frac"):
        assert getattr(base, field) == getattr(perturbed, field), field
    assert torch.equal(base.gradient_killed_mask, perturbed.gradient_killed_mask)
    assert not bool(base.gradient_killed_mask[~mask].any())


def test_validation_errors() -> None:
    """Missing or invalid clip epsilons, a non-finite or <= 1 ratio_cap, bad shapes,
    bad masks, non-finite response values, and non-positive response ratios raise
    ValueError naming the argument (docs/conventions.md)."""
    ratio = torch.ones(2, 3, dtype=torch.float64)
    advantages = torch.ones(2, 3, dtype=torch.float64)
    mask = torch.ones(2, 3, dtype=torch.bool)
    with pytest.raises(ValueError, match=r"eps_low and eps_high non-None"):
        clip_report(ratio, advantages, mask, ClipConfig(None, 0.2))
    with pytest.raises(ValueError, match=r"eps_low and eps_high non-None"):
        clip_report(ratio, advantages, mask, ClipConfig(0.2, None))
    with pytest.raises(ValueError, match=r"eps_high must be a positive finite float"):
        clip_report(ratio, advantages, mask, ClipConfig(0.2, -0.1))
    with pytest.raises(ValueError, match=r"eps_low must be a positive finite float"):
        clip_report(ratio, advantages, mask, ClipConfig(0.0, 0.2))
    for bad_cap in (1.0, 0.5, math.inf):
        with pytest.raises(ValueError, match=r"ratio_cap must be a finite float > 1"):
            clip_report(ratio, advantages, mask, ClipConfig(0.2, 0.2, ratio_cap=bad_cap))
    with pytest.raises(ValueError, match=r"ratio must be 2-D"):
        clip_report(torch.ones(3), advantages, mask, ClipConfig(0.2, 0.2))
    with pytest.raises(ValueError, match=r"dtype torch\.bool"):
        clip_report(ratio, advantages, torch.ones(2, 3), ClipConfig(0.2, 0.2))
    with pytest.raises(ValueError, match=r"zero response tokens"):
        bad_mask = torch.tensor([[True, True, True], [False, False, False]])
        clip_report(ratio, advantages, bad_mask, ClipConfig(0.2, 0.2))
    with pytest.raises(ValueError, match=r"advantages must be \[B\] or \[B, T\]"):
        clip_report(ratio, torch.ones(2, 3, 1), mask, ClipConfig(0.2, 0.2))
    with pytest.raises(ValueError, match=r"advantages \[B\] must have B = 2"):
        clip_report(ratio, torch.ones(3), mask, ClipConfig(0.2, 0.2))
    with pytest.raises(ValueError, match=r"advantages and ratio"):
        clip_report(ratio, torch.ones(2, 4), mask, ClipConfig(0.2, 0.2))
    with pytest.raises(ValueError, match=r"ratio must be strictly positive"):
        bad_ratio = ratio.clone()
        bad_ratio[0, 1] = 0.0
        clip_report(bad_ratio, advantages, mask, ClipConfig(0.2, 0.2))
    with pytest.raises(ValueError, match=r"ratio \(response positions\) contains non-finite"):
        bad_ratio = ratio.clone()
        bad_ratio[1, 2] = float("nan")
        clip_report(bad_ratio, advantages, mask, ClipConfig(0.2, 0.2))
    with pytest.raises(ValueError, match=r"advantages \(response positions\) contains non-finite"):
        bad_adv = advantages.clone()
        bad_adv[0, 0] = float("inf")
        clip_report(ratio, bad_adv, mask, ClipConfig(0.2, 0.2))


def test_summary_is_compact_multiline() -> None:
    """summary() is a compact human-readable multi-line string stating the band and
    which quadrants kill the gradient."""
    ratio, advantages, mask = _constructed_case()
    report = clip_report(ratio, advantages, mask, ClipConfig(0.2, 0.3, ratio_cap=3.0))
    text = report.summary()
    assert isinstance(report, ClipReport)
    assert text.count("\n") == 3
    assert "[0.8, 1.3]" in text
    assert "gradient killed on 0.6667 of response tokens" in text
    assert "dual-clip" in text
