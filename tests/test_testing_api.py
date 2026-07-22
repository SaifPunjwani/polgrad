"""Tests for the public conformance-testing API (polgrad.testing) and its pytest plugin.

The GRPO-style losses below are written inline, independently of polgrad's
implementation, so agreement is evidence rather than tautology. The pytester test
exercises the ``pytest11`` entry point (``polgrad = "polgrad._pytest_plugin"``) in a
fresh pytest run.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch
from torch import Tensor

from polgrad.aggregate import Aggregation
from polgrad.losses import ClipConfig, PolicyLossConfig, RatioKind, SurrogateKind
from polgrad.testing import assert_conforms, random_batches

pytest_plugins = ["pytester"]

GRPO_CONFIG = PolicyLossConfig(
    ratio=RatioKind.TOKEN,
    surrogate=SurrogateKind.PG_CLIP,
    clip=ClipConfig(eps_low=0.2, eps_high=0.2),
    aggregation=Aggregation.SEQ_MEAN_TOKEN_MEAN,
)


def _grpo_style(eps: float = 0.2, token_mean: bool = False) -> Callable[..., Tensor]:
    """Hand-written GRPO-style clipped surrogate (independent reference math).

    ``-min(r·A, clip(r, 1-eps, 1+eps)·A)`` with ``r = exp(logprobs - old_logprobs)``
    and ``[B]`` advantages broadcast across the row, aggregated as the sequence mean of
    per-sequence token means (or, with ``token_mean=True``, the wrong flat token mean).
    """

    def fn(
        *, logprobs: Tensor, old_logprobs: Tensor, advantages: Tensor, response_mask: Tensor
    ) -> Tensor:
        mask = response_mask.to(logprobs.dtype)
        ratio = torch.exp(logprobs - old_logprobs)
        adv = advantages.unsqueeze(1)
        clipped = torch.clamp(ratio, 1.0 - eps, 1.0 + eps)
        per_token = -torch.minimum(ratio * adv, clipped * adv)
        if token_mean:
            return (per_token * mask).sum() / mask.sum()
        return ((per_token * mask).sum(-1) / mask.sum(-1)).mean()

    return fn


def test_assert_conforms_passes_for_correct_grpo_fn() -> None:
    """A correct hand-written GRPO-style fn conforms to the explicit config."""
    report = assert_conforms(_grpo_style(), GRPO_CONFIG)
    assert report.n_cases == 64
    assert report.max_loss_rel_diff <= 1e-9
    assert report.max_grad_rel_diff <= 1e-9
    assert report.grad_cosine_min >= 1.0 - 1e-9
    assert any("certifies" in note for note in report.notes)
    # kl=None in the config: nothing was stripped, so no stripping note appears.
    assert not any("stripped" in note for note in report.notes)


def test_assert_conforms_registry_key_strips_kl_and_passes() -> None:
    """The ALGORITHMS-key path works: "grpo" certifies the surrogate without its KL.

    ``ALGORITHMS["grpo"].loss.kl`` expects ``ref_logprobs``; assert_conforms strips it
    via dataclasses.replace and records the stripping in the report notes.
    """
    report = assert_conforms(_grpo_style(), "grpo")
    assert report.max_loss_rel_diff <= 1e-9
    assert report.max_grad_rel_diff <= 1e-9
    assert any("stripped" in note for note in report.notes)
    assert any("'grpo'" in note for note in report.notes)


def test_assert_conforms_fails_for_wrong_aggregation() -> None:
    """Flat token-mean aggregation deviates from SEQ_MEAN_TOKEN_MEAN with a clear message."""
    with pytest.raises(AssertionError) as excinfo:
        assert_conforms(_grpo_style(token_mean=True), GRPO_CONFIG, n_cases=16)
    message = str(excinfo.value)
    assert "polgrad conformance violation" in message
    assert "max loss rel diff" in message
    assert "seq_mean_token_mean" in message  # config provenance from the report notes


def test_assert_conforms_fails_for_subtly_wrong_clip() -> None:
    """A clip width of 0.21 instead of 0.20 is caught at the default tolerances."""
    with pytest.raises(AssertionError) as excinfo:
        assert_conforms(_grpo_style(eps=0.21), GRPO_CONFIG, n_cases=16)
    message = str(excinfo.value)
    assert "max grad rel diff" in message
    assert "worst-case seed" in message


def test_assert_conforms_unknown_key_raises_value_error() -> None:
    with pytest.raises(ValueError, match="unknown algorithm name"):
        assert_conforms(_grpo_style(), "gropo")


def test_assert_conforms_rejects_tis_configs() -> None:
    """grpo_tis needs rollout_logprobs, which the four-tensor convention cannot carry."""
    with pytest.raises(ValueError, match="is_correction"):
        assert_conforms(_grpo_style(), "grpo_tis")


def test_random_batches_same_seed_is_bitwise_identical() -> None:
    shapes = ((4, 8), (2, 12))
    first = list(random_batches(6, shapes, seed=7))
    second = list(random_batches(6, shapes, seed=7))
    assert len(first) == len(second) == 6
    for batch_a, batch_b in zip(first, second, strict=True):
        assert batch_a.keys() == batch_b.keys()
        for key in batch_a:
            assert torch.equal(batch_a[key], batch_b[key])


def test_random_batches_different_seed_differs() -> None:
    batch_a = next(iter(random_batches(1, ((4, 8),), seed=0)))
    batch_b = next(iter(random_batches(1, ((4, 8),), seed=1)))
    assert not torch.equal(batch_a["logprobs"], batch_b["logprobs"])


def test_random_batches_masks_shapes_and_bounds() -> None:
    """Masks are right-padded bool with >= 1 token per row; values obey the documented bounds."""
    shapes = ((4, 8), (2, 12))
    batches = list(random_batches(8, shapes, seed=3, max_gap=2.0))
    assert len(batches) == 8
    for index, batch in enumerate(batches):
        b, t = shapes[index % len(shapes)]
        mask = batch["response_mask"]
        assert mask.shape == (b, t)
        assert mask.dtype == torch.bool
        lengths = mask.sum(dim=1)
        assert (lengths >= 1).all()
        # Right-padded: every row is a prefix of true positions.
        rebuilt = torch.arange(t).unsqueeze(0) < lengths.unsqueeze(1)
        assert torch.equal(mask, rebuilt)
        logprobs = batch["logprobs"]
        assert logprobs.dtype == torch.float64
        assert logprobs.shape == (b, t)
        assert (logprobs >= -8.0).all()
        assert (logprobs <= -0.0625).all()
        for key in ("old_logprobs", "ref_logprobs"):
            assert ((batch[key] - logprobs).abs() <= 2.0).all()
        assert ((batch["rollout_logprobs"] - batch["old_logprobs"]).abs() <= 2.0).all()
        advantages = batch["advantages"]
        assert advantages.shape == (b,)
        assert (advantages.abs() <= 3.0).all()


def test_random_batches_validates_eagerly() -> None:
    """Argument errors raise at call time, not on first iteration."""
    with pytest.raises(ValueError, match="n_cases"):
        random_batches(0, ((4, 8),), seed=0)
    with pytest.raises(ValueError, match="shapes"):
        random_batches(1, (), seed=0)
    with pytest.raises(ValueError, match="max_gap"):
        random_batches(1, ((4, 8),), seed=0, max_gap=-1.0)
    with pytest.raises(ValueError, match="dtype"):
        random_batches(1, ((4, 8),), seed=0, dtype=torch.int64)


def test_polgrad_batches_fixture_available_in_fresh_run(pytester: pytest.Pytester) -> None:
    """The pytest11 entry point provides polgrad_batches in a fresh pytest run."""
    pytester.makepyfile(
        """
        def test_uses_polgrad_batches(polgrad_batches):
            batches = list(polgrad_batches(2, [(2, 4)], seed=0))
            assert len(batches) == 2
            assert batches[0]["response_mask"].shape == (2, 4)
            assert batches[0]["advantages"].shape == (2,)
        """
    )
    result = pytester.runpytest()
    result.assert_outcomes(passed=1)
