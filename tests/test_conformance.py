"""Tests for the conformance harness, fixtures, and deviation registry: where a
vendored framework's semantics match polgrad's, the two must agree to fp64 tolerance;
where they deviate, the deviation registry demonstrates each entry by test.

Fixture replay: ``tools/record_fixtures.py`` recorded every ``VENDORED`` wrapper's loss
and gradient on seeded inputs into ``tests/fixtures/*.json``; the agreement tests here
feed the recorded inputs to polgrad live and compare against the recorded framework
outputs, and the deviation tests assert the exact analytic factors registered in
``polgrad.conformance.deviations.DEVIATIONS``. No external framework package is
imported (the vendored copies under ``polgrad.conformance._vendor`` are part of this
repository).
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch
from hypothesis import given
from strategies import LogprobBatch, logprob_batches
from torch import Tensor
from torch.testing import assert_close

from polgrad.aggregate import Aggregation
from polgrad.conformance import harness
from polgrad.conformance.deviations import DEVIATIONS
from polgrad.conformance.harness import VENDORED, compare_losses, deviation_report
from polgrad.losses import ClipConfig, PolicyLossConfig, RatioKind, SurrogateKind, policy_loss
from polgrad.registry import ALGORITHMS

FIXTURES = Path(__file__).resolve().parent / "fixtures"
RTOL = 1e-12
ATOL = 1e-14

VERL_COMMIT = "74a718a492092312f1004fe25369975137388849"
OPENRLHF_COMMIT = "bc71bb19464aca306b33080b2d2bb45d154e2f49"
TRL_COMMIT = "95809b942eb5d11d0b06d749510d88be99230b73"

# OpenRLHF and TRL apply no dual-clip by default; verl's compute_policy_loss always
# applies the dual-clip floor with its default clip_ratio_c = 3.0, so polgrad configs
# compared against verl must set ratio_cap to match.
CLIP = ClipConfig(eps_low=0.2, eps_high=0.2)
CLIP_VERL = ClipConfig(eps_low=0.2, eps_high=0.2, ratio_cap=3.0)

# verl's GSPO/CISPO clip both sides with the pinned ActorConfig defaults
# clip_ratio_low = clip_ratio_high = 0.2 (no dual clip in either function); the paper
# values the registry ships differ (GSPO: 3e-4/4e-4; CISPO: one-sided, eps_low=None),
# so the agreement tests dataclasses.replace the registry configs with this clip.
CLIP_VERL_GSPO_CISPO = ClipConfig(eps_low=0.2, eps_high=0.2)


def _config(
    aggregation: Aggregation, clip: ClipConfig, norm_len: int | None = None
) -> PolicyLossConfig:
    return PolicyLossConfig(
        ratio=RatioKind.TOKEN,
        surrogate=SurrogateKind.PG_CLIP,
        clip=clip,
        aggregation=aggregation,
        norm_len=norm_len,
    )


def _load(name: str) -> dict[str, Any]:
    with (FIXTURES / name).open() as fh:
        data: dict[str, Any] = json.load(fh)
    return data


def _tensors(case: dict[str, Any]) -> dict[str, Tensor]:
    inputs = case["inputs"]
    return {
        "logprobs": torch.tensor(inputs["logprobs"], dtype=torch.float64),
        "old_logprobs": torch.tensor(inputs["old_logprobs"], dtype=torch.float64),
        "advantages": torch.tensor(inputs["advantages"], dtype=torch.float64),
        "response_mask": torch.tensor(inputs["response_mask"], dtype=torch.bool),
    }


def _recorded(case: dict[str, Any]) -> tuple[float, Tensor]:
    outputs = case["outputs"]
    return float(outputs["loss"]), torch.tensor(outputs["grad_logprobs"], dtype=torch.float64)


def _polgrad_loss_and_grad(
    config: PolicyLossConfig, tensors: dict[str, Tensor]
) -> tuple[float, Tensor]:
    logprobs = tensors["logprobs"].clone().requires_grad_(True)
    loss = policy_loss(
        config,
        logprobs=logprobs,
        old_logprobs=tensors["old_logprobs"],
        advantages=tensors["advantages"],
        response_mask=tensors["response_mask"],
    ).loss
    (grad,) = torch.autograd.grad(loss, logprobs)
    return float(loss.detach()), grad


def _assert_agreement(fixture: str, variant: str, config: PolicyLossConfig) -> None:
    """Replay every fixture case of a variant and assert loss+grad match polgrad."""
    cases = _load(fixture)["variants"][variant]
    assert cases, f"{fixture} has no cases for {variant}"
    for case in cases:
        loss, grad = _polgrad_loss_and_grad(config, _tensors(case))
        recorded_loss, recorded_grad = _recorded(case)
        assert math.isclose(loss, recorded_loss, rel_tol=RTOL, abs_tol=ATOL)
        assert_close(grad, recorded_grad, rtol=RTOL, atol=ATOL)


def test_fixture_files_declare_pinned_provenance() -> None:
    """Every fixture file records its upstream pin; TRL is labeled a reimplementation."""
    verl = _load("verl_losses.json")["provenance"]
    assert verl["kind"] == "vendored"
    assert verl["upstream_commit"] == VERL_COMMIT
    openrlhf = _load("openrlhf_losses.json")["provenance"]
    assert openrlhf["kind"] == "vendored"
    assert openrlhf["upstream_commit"] == OPENRLHF_COMMIT
    trl = _load("trl_reimpl_losses.json")["provenance"]
    assert trl["kind"] == "reimplementation"
    assert trl["upstream_version"] == "v1.8.0"
    assert trl["upstream_commit"] == TRL_COMMIT
    assert TRL_COMMIT in trl["permalink"]
    verl_reimpl = _load("verl_reimpl_losses.json")["provenance"]
    assert verl_reimpl["kind"] == "reimplementation"
    assert verl_reimpl["upstream_commit"] == VERL_COMMIT
    for variant, fragment in (
        ("gspo", "core_algos.py#L1538-L1611"),
        ("cispo", "core_algos.py#L2006-L2064"),
    ):
        assert VERL_COMMIT in verl_reimpl["permalinks"][variant]
        assert verl_reimpl["permalinks"][variant].endswith(fragment)


def test_verl_token_mean_pg_clip_agrees_with_polgrad_on_fixtures() -> None:
    """polgrad TOKEN_MEAN PG_CLIP (ratio_cap 3.0) equals verl token-mean on fixtures.

    Agreement case (matching semantics must agree): identical inputs, fp64 tolerance,
    loss and gradient.
    """
    _assert_agreement(
        "verl_losses.json", "pg_clip_token_mean", _config(Aggregation.TOKEN_MEAN, CLIP_VERL)
    )


def test_verl_seq_mean_token_sum_agrees_with_polgrad_on_fixtures() -> None:
    """polgrad SEQ_MEAN_TOKEN_SUM equals verl seq-mean-token-sum on fixtures.

    verl's divisor is global_batch_size = number of rows with a response token, which
    equals B for polgrad-valid masks, so the two reductions coincide exactly.
    """
    _assert_agreement(
        "verl_losses.json",
        "pg_clip_seq_mean_token_sum",
        _config(Aggregation.SEQ_MEAN_TOKEN_SUM, CLIP_VERL),
    )


def test_openrlhf_token_mean_policy_loss_agrees_with_polgrad_on_fixtures() -> None:
    """polgrad TOKEN_MEAN PG_CLIP equals OpenRLHF PolicyLoss (token level) on fixtures.

    Agreement case, matching semantics to fp64 tolerance (OpenRLHF applies no dual
    clip by default).
    """
    _assert_agreement(
        "openrlhf_losses.json", "pg_clip_token_mean", _config(Aggregation.TOKEN_MEAN, CLIP)
    )


def test_trl_bnpo_agrees_with_polgrad_token_mean_on_fixtures() -> None:
    """polgrad TOKEN_MEAN PG_CLIP equals the TRL reimplementation with loss_type=bnpo."""
    _assert_agreement("trl_reimpl_losses.json", "bnpo", _config(Aggregation.TOKEN_MEAN, CLIP))


def test_trl_grpo_agrees_with_polgrad_seq_mean_token_mean_on_fixtures() -> None:
    """polgrad SEQ_MEAN_TOKEN_MEAN PG_CLIP equals the TRL reimplementation loss_type=grpo."""
    _assert_agreement(
        "trl_reimpl_losses.json", "grpo", _config(Aggregation.SEQ_MEAN_TOKEN_MEAN, CLIP)
    )


def test_trl_dr_grpo_agrees_with_polgrad_token_sum_norm_on_fixtures() -> None:
    """polgrad TOKEN_SUM_NORM (norm_len = T) equals the TRL reimplementation dr_grpo.

    The TRL wrapper pins max_completion_length to the padded width T, and TRL divides
    by B * max_completion_length — exactly polgrad's Σ/(B·norm_len).
    """
    for case in _load("trl_reimpl_losses.json")["variants"]["dr_grpo"]:
        t = int(case["shape"][1])
        loss, grad = _polgrad_loss_and_grad(
            _config(Aggregation.TOKEN_SUM_NORM, CLIP, norm_len=t), _tensors(case)
        )
        recorded_loss, recorded_grad = _recorded(case)
        assert math.isclose(loss, recorded_loss, rel_tol=RTOL, abs_tol=ATOL)
        assert_close(grad, recorded_grad, rtol=RTOL, atol=ATOL)


def test_verl_cispo_agrees_with_polgrad_cispo_on_fixtures() -> None:
    """polgrad CISPO (registry config at verl's pinned clip) equals the reimplementation.

    Agreement case to fp64 tolerance, loss and gradient, on the recorded fixtures. The
    registry ``cispo`` entry ships the paper's one-sided IS-weight clip (eps_low=None;
    MiniMax-M1 imposes no lower bound, arXiv 2506.13585), while verl's pinned
    ActorConfig defaults clamp two-sided with clip_ratio_low = clip_ratio_high = 0.2 —
    a config-level (not semantic) difference, bridged by dataclasses.replace. verl's
    default token-mean aggregation equals polgrad Aggregation.TOKEN_MEAN exactly, and
    its ±20 log-ratio clamp never binds on the fixture distribution
    (|log-ratio gap| <= 2).
    """
    config = replace(ALGORITHMS["cispo"].loss, clip=CLIP_VERL_GSPO_CISPO)
    _assert_agreement("verl_reimpl_losses.json", "cispo", config)


def test_verl_gspo_agrees_with_polgrad_sequence_ratio_on_seeded_batches() -> None:
    """polgrad GSPO (RatioKind.SEQUENCE at verl's pinned clip) matches the
    reimplementation on row-constant advantages, up to the registered row epsilon.

    Two pinned-default differences from the GSPO paper, both handled here:

    - clip widths: the paper sets 3e-4/4e-4 (the registry ``gspo`` values); verl's
      ActorConfig pins clip_ratio_low = clip_ratio_high = 0.2 — config-level, bridged
      by dataclasses.replace.
    - aggregation: verl hardcodes agg_loss('seq-mean-token-mean'), whose rows divide
      by (L_b + 1e-8) instead of L_b (DEVIATIONS[1]); the loss assertion therefore
      compares the reimplementation against the epsilon-deflated prediction from
      polgrad's per-token-objective row sums, and the gradient against polgrad's
      row-rescaled by L_b/(L_b + 1e-8), both to fp64.

    Advantages are row-constant (the papers' per-sequence group-normalized setting):
    verl's ratio is the GSPO-token sg[] form, whose gradient equals
    RatioKind.SEQUENCE's exactly in that case; per-token advantages are covered by
    test_verl_gspo_gradient_is_token_local_gspo_token_form. The upstream
    clamp(max=10.0) on the combined log ratio never binds on the sampled distribution.
    """
    config = replace(ALGORITHMS["gspo"].loss, clip=CLIP_VERL_GSPO_CISPO)
    saw_gap = False
    for seed in (11, 23, 47, 91):
        case = harness._sample_case((4, 8), torch.Generator().manual_seed(seed), torch.float64)
        advantages = case["advantages"][:, :1].expand_as(case["logprobs"]).contiguous()

        logprobs = case["logprobs"].clone().requires_grad_(True)
        result = policy_loss(
            config,
            logprobs=logprobs,
            old_logprobs=case["old_logprobs"],
            advantages=advantages,
            response_mask=case["response_mask"],
        )
        (polgrad_grad,) = torch.autograd.grad(result.loss, logprobs)

        verl_logprobs = case["logprobs"].clone().requires_grad_(True)
        verl_loss = VENDORED[("verl", "gspo")](
            logprobs=verl_logprobs,
            old_logprobs=case["old_logprobs"],
            advantages=advantages,
            response_mask=case["response_mask"],
        )
        (verl_grad,) = torch.autograd.grad(verl_loss, verl_logprobs)

        lengths = case["response_mask"].sum(dim=1).to(torch.float64)
        batch = float(case["response_mask"].shape[0])
        row_sums = result.per_token_objective.detach().sum(dim=1)
        predicted = float((row_sums / (lengths + 1e-8)).sum() / batch)
        assert math.isclose(predicted, float(verl_loss.detach()), rel_tol=RTOL, abs_tol=ATOL)
        scale = (lengths / (lengths + 1e-8)).unsqueeze(1)
        assert_close(polgrad_grad * scale, verl_grad, rtol=RTOL, atol=ATOL)
        exact = float(result.loss.detach())
        if not math.isclose(exact, float(verl_loss.detach()), rel_tol=RTOL, abs_tol=ATOL):
            saw_gap = True
    assert saw_gap, "epsilon deflation never separated from the exact value at fp64"


def test_verl_gspo_gradient_is_token_local_gspo_token_form() -> None:
    """verl's gspo ratio is the paper's GSPO-token eq. 14 form, not the eq. 7 one.

    Demonstrates DEVIATIONS[3] on the fixtures' per-token advantages: the
    reimplementation's recorded loss equals the epsilon-deflated prediction from
    polgrad's row sums (RatioKind.SEQUENCE and SEQUENCE_TOKEN share the loss value),
    while its recorded gradient equals RatioKind.SEQUENCE_TOKEN's — row-rescaled by
    L_b/(L_b + 1e-8), the agg_loss epsilon of DEVIATIONS[1] — and differs from
    RatioKind.SEQUENCE's row-coupled gradient.
    """
    token_config = replace(ALGORITHMS["gspo_token"].loss, clip=CLIP_VERL_GSPO_CISPO)
    sequence_config = replace(ALGORITHMS["gspo"].loss, clip=CLIP_VERL_GSPO_CISPO)
    for case in _load("verl_reimpl_losses.json")["variants"]["gspo"]:
        tensors = _tensors(case)
        recorded_loss, recorded_grad = _recorded(case)
        lengths = tensors["response_mask"].sum(dim=1).to(torch.float64)
        batch = float(tensors["response_mask"].shape[0])
        scale = (lengths / (lengths + 1e-8)).unsqueeze(1)

        logprobs = tensors["logprobs"].clone().requires_grad_(True)
        token_result = policy_loss(
            token_config,
            logprobs=logprobs,
            old_logprobs=tensors["old_logprobs"],
            advantages=tensors["advantages"],
            response_mask=tensors["response_mask"],
        )
        (token_grad,) = torch.autograd.grad(token_result.loss, logprobs)
        row_sums = token_result.per_token_objective.detach().sum(dim=1)
        predicted = float((row_sums / (lengths + 1e-8)).sum() / batch)
        assert math.isclose(predicted, recorded_loss, rel_tol=RTOL, abs_tol=ATOL)
        assert_close(token_grad * scale, recorded_grad, rtol=RTOL, atol=ATOL)

        sequence_loss, sequence_grad = _polgrad_loss_and_grad(sequence_config, tensors)
        assert math.isclose(
            sequence_loss, float(token_result.loss.detach()), rel_tol=RTOL, abs_tol=ATOL
        )
        rel = float((sequence_grad * scale - recorded_grad).norm()) / float(recorded_grad.norm())
        assert rel > 1e-3, f"sequence-ratio gradient unexpectedly matches: rel diff {rel}"


def test_verl_token_sum_norm_deviates_from_dr_grpo_by_norm_len_over_padded_len() -> None:
    """verl seq-mean-token-sum-norm equals Dr.GRPO TOKEN_SUM_NORM times norm_len/T.

    Demonstrates DEVIATIONS[0]: at the pinned commit verl divides by B and by the
    padded width T (default loss_scale_factor), i.e. verl = Σ/(B·T), while polgrad
    TOKEN_SUM_NORM = Σ/(B·norm_len); hence polgrad = verl · T/norm_len for any fixed
    budget, with equality exactly at norm_len = T. Asserted for loss and gradient on
    every fixture case.
    """
    norm_len = 16
    for case in _load("verl_losses.json")["variants"]["pg_clip_seq_mean_token_sum_norm"]:
        tensors = _tensors(case)
        t = int(case["shape"][1])
        recorded_loss, recorded_grad = _recorded(case)
        assert abs(recorded_loss) > 1e-3, "degenerate fixture: loss too close to zero"

        loss, grad = _polgrad_loss_and_grad(
            _config(Aggregation.TOKEN_SUM_NORM, CLIP_VERL, norm_len=norm_len), tensors
        )
        factor = t / norm_len
        assert math.isclose(loss, recorded_loss * factor, rel_tol=RTOL, abs_tol=ATOL)
        assert_close(grad, recorded_grad * factor, rtol=RTOL, atol=ATOL)
        assert not math.isclose(loss, recorded_loss, rel_tol=1e-3)

        loss_at_t, grad_at_t = _polgrad_loss_and_grad(
            _config(Aggregation.TOKEN_SUM_NORM, CLIP_VERL, norm_len=t), tensors
        )
        assert math.isclose(loss_at_t, recorded_loss, rel_tol=RTOL, abs_tol=ATOL)
        assert_close(grad_at_t, recorded_grad, rtol=RTOL, atol=ATOL)


def _epsilon_deviation_check(fixture: str, variant: str, config: PolicyLossConfig) -> None:
    """Assert the row-wise (L_b + 1e-8) deflation against the exact sequence mean."""
    saw_gap = False
    for case in _load(fixture)["variants"][variant]:
        tensors = _tensors(case)
        result = policy_loss(
            config,
            logprobs=tensors["logprobs"],
            old_logprobs=tensors["old_logprobs"],
            advantages=tensors["advantages"],
            response_mask=tensors["response_mask"],
        )
        row_sums = result.per_token_objective.sum(dim=1)
        lengths = tensors["response_mask"].sum(dim=1).to(torch.float64)
        batch = float(tensors["response_mask"].shape[0])
        predicted = float((row_sums / (lengths + 1e-8)).sum() / batch)
        exact = float((row_sums / lengths).sum() / batch)
        recorded_loss, _ = _recorded(case)
        assert math.isclose(float(result.loss), exact, rel_tol=1e-11, abs_tol=ATOL)
        assert math.isclose(predicted, recorded_loss, rel_tol=1e-11, abs_tol=ATOL)
        if not math.isclose(exact, recorded_loss, rel_tol=RTOL, abs_tol=ATOL):
            saw_gap = True
    assert saw_gap, "epsilon deflation never separated from the exact value at fp64"


def test_verl_seq_mean_token_mean_deviates_by_row_epsilon_factor() -> None:
    """verl seq-mean-token-mean divides rows by (L_b + 1e-8), not L_b.

    Demonstrates DEVIATIONS[1]: the recorded verl loss equals
    ``(1/B)·Σ_b S_b/(L_b + 1e-8)`` (asserted to fp64 tolerance from polgrad's per-token
    objective row sums S_b) and differs at fp64 from the exact SEQ_MEAN_TOKEN_MEAN
    value ``(1/B)·Σ_b S_b/L_b``.
    """
    _epsilon_deviation_check(
        "verl_losses.json",
        "pg_clip_seq_mean_token_mean",
        _config(Aggregation.SEQ_MEAN_TOKEN_MEAN, CLIP_VERL),
    )


def test_openrlhf_sample_level_deviates_by_row_epsilon_factor() -> None:
    """OpenRLHF sample-level aggregation divides rows by (L_b + 1e-8), not L_b.

    Demonstrates DEVIATIONS[2] with the same predicted-vs-exact separation as the verl
    epsilon test, on the OpenRLHF fixtures (no dual clip).
    """
    _epsilon_deviation_check(
        "openrlhf_losses.json",
        "pg_clip_seq_mean_token_mean",
        _config(Aggregation.SEQ_MEAN_TOKEN_MEAN, CLIP),
    )


def test_fixture_outputs_match_live_vendored_wrappers() -> None:
    """Recorded fixture outputs reproduce when replayed through the live wrappers.

    Guards against fixture drift relative to the vendored code committed in-tree.
    """
    for fixture, framework in (
        ("verl_losses.json", "verl"),
        ("verl_reimpl_losses.json", "verl"),
        ("openrlhf_losses.json", "openrlhf"),
        ("trl_reimpl_losses.json", "trl"),
    ):
        data = _load(fixture)
        for variant, cases in data["variants"].items():
            fn = VENDORED[(framework, variant)]
            for case in cases:
                tensors = _tensors(case)
                logprobs = tensors["logprobs"].clone().requires_grad_(True)
                loss = fn(
                    logprobs=logprobs,
                    old_logprobs=tensors["old_logprobs"],
                    advantages=tensors["advantages"],
                    response_mask=tensors["response_mask"],
                )
                (grad,) = torch.autograd.grad(loss, logprobs)
                recorded_loss, recorded_grad = _recorded(case)
                assert math.isclose(float(loss.detach()), recorded_loss, rel_tol=RTOL, abs_tol=ATOL)
                assert_close(grad, recorded_grad, rtol=RTOL, atol=ATOL)


def test_masked_input_perturbation_leaves_losses_bitwise_unchanged() -> None:
    """Perturbing masked positions changes no polgrad or wrapper loss bit.

    Mask-invariance rule of docs/conventions.md, checked bitwise on one fixture case
    per framework for both the polgrad loss and the live wrapper loss.
    """
    checks = (
        ("verl_losses.json", ("verl", "pg_clip_token_mean"), CLIP_VERL),
        ("openrlhf_losses.json", ("openrlhf", "pg_clip_token_mean"), CLIP),
        ("trl_reimpl_losses.json", ("trl", "bnpo"), CLIP),
    )
    for fixture, key, clip in checks:
        case = _load(fixture)["variants"][key[1]][0]
        tensors = _tensors(case)
        mask = tensors["response_mask"]
        perturbed = dict(tensors)
        for name, offset in (("logprobs", 0.625), ("old_logprobs", 0.25), ("advantages", -1.5)):
            perturbed[name] = torch.where(mask, tensors[name], tensors[name] + offset)
        config = _config(Aggregation.TOKEN_MEAN, clip)

        def polgrad_result(inputs: dict[str, Tensor], cfg: PolicyLossConfig = config) -> Any:
            return policy_loss(
                cfg,
                logprobs=inputs["logprobs"],
                old_logprobs=inputs["old_logprobs"],
                advantages=inputs["advantages"],
                response_mask=inputs["response_mask"],
            )

        base, moved = polgrad_result(tensors), polgrad_result(perturbed)
        assert torch.equal(base.loss, moved.loss)
        assert torch.equal(base.per_token_objective, moved.per_token_objective)
        assert torch.equal(base.ratio, moved.ratio)

        fn = VENDORED[key]
        assert torch.equal(
            fn(
                logprobs=tensors["logprobs"],
                old_logprobs=tensors["old_logprobs"],
                advantages=tensors["advantages"],
                response_mask=mask,
            ),
            fn(
                logprobs=perturbed["logprobs"],
                old_logprobs=perturbed["old_logprobs"],
                advantages=perturbed["advantages"],
                response_mask=mask,
            ),
        )


def test_compare_losses_zero_deviation_for_identical_fn() -> None:
    """compare_losses(fn, fn) reports exactly zero diffs and cosine one."""
    fn = VENDORED[("trl", "bnpo")]
    report = compare_losses(
        fn, fn, n_cases=5, shapes=((3, 6), (2, 4)), generator=torch.Generator().manual_seed(0)
    )
    assert report.n_cases == 5
    assert report.max_loss_rel_diff == 0.0
    assert report.max_grad_rel_diff == 0.0
    # the cosine of a nonzero vector with itself rounds to 1 only within one ulp
    assert report.grad_cosine_min == pytest.approx(1.0, abs=1e-12)
    assert report.notes == ()


def test_compare_losses_detects_scale_deviation() -> None:
    """fn_b = 2·fn_a yields rel diffs of exactly 1/2 and cosine one.

    ``|a - 2a| / max(|a|, |2a|) = 1/2`` and likewise for the gradient norms; the
    gradients stay parallel.
    """
    fn = VENDORED[("trl", "bnpo")]

    def doubled(**kwargs: Tensor) -> Tensor:
        return 2.0 * fn(**kwargs)

    report = compare_losses(
        fn, doubled, n_cases=4, shapes=((3, 5),), generator=torch.Generator().manual_seed(1)
    )
    assert report.max_loss_rel_diff == pytest.approx(0.5, abs=1e-12)
    assert report.max_grad_rel_diff == pytest.approx(0.5, abs=1e-12)
    assert report.grad_cosine_min == pytest.approx(1.0, abs=1e-12)


def test_compare_losses_validation_errors() -> None:
    """compare_losses rejects bad n_cases, shapes, dtypes, and non-scalar callables."""
    fn = VENDORED[("trl", "bnpo")]
    gen = torch.Generator().manual_seed(0)
    with pytest.raises(ValueError, match="n_cases"):
        compare_losses(fn, fn, n_cases=0, shapes=((2, 3),), generator=gen)
    with pytest.raises(ValueError, match="shapes"):
        compare_losses(fn, fn, n_cases=1, shapes=(), generator=gen)
    with pytest.raises(ValueError, match="B, T"):
        compare_losses(fn, fn, n_cases=1, shapes=((0, 3),), generator=gen)
    with pytest.raises(ValueError, match="dtype"):
        compare_losses(fn, fn, n_cases=1, shapes=((2, 3),), generator=gen, dtype=torch.int64)

    def non_scalar(**kwargs: Tensor) -> Tensor:
        return kwargs["logprobs"]

    with pytest.raises(ValueError, match="fn_a"):
        compare_losses(non_scalar, fn, n_cases=1, shapes=((2, 3),), generator=gen)


def test_deviation_report_verl_token_mean_matches_polgrad(gen: torch.Generator) -> None:
    """deviation_report shows fp64-level agreement for verl token-mean PG_CLIP.

    Verifies the agreement case on 16 seeded cases across three shapes.
    """
    report = deviation_report(
        _config(Aggregation.TOKEN_MEAN, CLIP_VERL),
        "verl",
        "pg_clip_token_mean",
        n_cases=16,
        shapes=((4, 8), (3, 5), (2, 12)),
        generator=gen,
    )
    assert report.n_cases == 16
    assert report.max_loss_rel_diff < 1e-12
    assert report.max_grad_rel_diff < 1e-11
    assert report.grad_cosine_min > 1.0 - 1e-12
    assert any("verl/pg_clip_token_mean" in note for note in report.notes)


def test_deviation_report_openrlhf_token_mean_matches_polgrad(gen: torch.Generator) -> None:
    """deviation_report shows fp64-level agreement for OpenRLHF token-level PolicyLoss."""
    report = deviation_report(
        _config(Aggregation.TOKEN_MEAN, CLIP),
        "openrlhf",
        "pg_clip_token_mean",
        n_cases=16,
        shapes=((4, 8), (3, 5), (2, 12)),
        generator=gen,
    )
    assert report.max_loss_rel_diff < 1e-12
    assert report.max_grad_rel_diff < 1e-11
    assert report.grad_cosine_min > 1.0 - 1e-12


def test_deviation_report_trl_bnpo_matches_polgrad_and_notes_reimplementation(
    gen: torch.Generator,
) -> None:
    """deviation_report agrees with TRL bnpo and labels the entry a reimplementation."""
    report = deviation_report(
        _config(Aggregation.TOKEN_MEAN, CLIP),
        "trl",
        "bnpo",
        n_cases=16,
        shapes=((4, 8), (3, 5)),
        generator=gen,
    )
    assert report.max_loss_rel_diff < 1e-12
    assert report.max_grad_rel_diff < 1e-11
    assert any("reimplementation" in note for note in report.notes)


def test_deviation_report_unknown_key_raises_value_error(gen: torch.Generator) -> None:
    """deviation_report names the unknown (framework, variant) pair in its ValueError."""
    with pytest.raises(ValueError, match="unknown VENDORED entry nosuch/variant"):
        deviation_report(_config(Aggregation.TOKEN_MEAN, CLIP), "nosuch", "variant", generator=gen)


def test_deviation_report_deterministic_for_equal_seeds() -> None:
    """Equal generator seeds give bitwise-identical DeviationReports."""
    config = _config(Aggregation.TOKEN_MEAN, CLIP_VERL)
    reports = [
        deviation_report(
            config,
            "verl",
            "pg_clip_token_mean",
            n_cases=6,
            shapes=((3, 5),),
            generator=torch.Generator().manual_seed(7),
        )
        for _ in range(2)
    ]
    assert reports[0] == reports[1]


def test_deviation_report_summary_mentions_key_figures(gen: torch.Generator) -> None:
    """summary() names the case count, worst-case seed, and every note line."""
    report = deviation_report(
        _config(Aggregation.TOKEN_MEAN, CLIP),
        "trl",
        "bnpo",
        n_cases=3,
        shapes=((2, 4),),
        generator=gen,
    )
    text = report.summary()
    assert "3 seeded cases" in text
    assert f"worst-case seed {report.worst_case_seed}" in text
    assert "max loss rel diff" in text
    assert "min grad cosine" in text
    for note in report.notes:
        assert note in text


def test_deviations_registry_node_ids_exist() -> None:
    """Every DEVIATIONS entry names a demonstrating test that exists in this module.

    The named tests run in the same suite, so existence here plus a green suite means
    each registered deviation is demonstrated.
    """
    assert DEVIATIONS, "DEVIATIONS must register the pre-registered verl deviation"
    module = sys.modules[__name__]
    prefix = "tests/test_conformance.py::"
    for deviation in DEVIATIONS:
        assert deviation.framework
        assert deviation.version
        assert deviation.component
        assert deviation.description
        assert deviation.demonstrated_by.startswith(prefix), deviation.demonstrated_by
        name = deviation.demonstrated_by.removeprefix(prefix)
        assert callable(getattr(module, name, None)), f"missing demonstrating test {name}"


def test_trl_reimplementation_provenance_documented() -> None:
    """The TRL loss is labeled a reimplementation with version, commit, and permalink."""
    doc = harness._trl_grpo_loss.__doc__
    assert doc is not None
    assert "reimplementation" in doc
    assert "v1.8.0" in doc
    assert TRL_COMMIT in doc
    assert "grpo_trainer.py#L2857-L3016" in doc


def test_verl_reimplementation_provenance_documented() -> None:
    """Both verl reimplementations name the commit, permalink, file hash, and label.

    The frozen upstream file SHA256 is the one already recorded in the vendored
    ``verl_core_algos.py`` header (same file, same commit).
    """
    upstream_sha256 = "9114f9e16c87e4c9ebf2fa016baf733c9bbc819766b53c8968aaa9e8abcd7916"
    for fn, fragment in (
        (harness._verl_gspo_loss, "core_algos.py#L1538-L1611"),
        (harness._verl_cispo_loss, "core_algos.py#L2006-L2064"),
    ):
        doc = fn.__doc__
        assert doc is not None
        assert "reimplementation" in doc
        assert VERL_COMMIT in doc
        assert fragment in doc
        assert upstream_sha256 in doc


@given(batch=logprob_batches(max_b=6, max_t=8, max_gap=1.5))
def test_property_trl_bnpo_equals_polgrad_token_mean_on_generated_batches(
    batch: LogprobBatch,
) -> None:
    """polgrad TOKEN_MEAN PG_CLIP equals TRL bnpo on Hypothesis batches (loss + grad)."""
    config = _config(Aggregation.TOKEN_MEAN, CLIP)
    tensors = {
        "logprobs": batch.logprobs,
        "old_logprobs": batch.old_logprobs,
        "advantages": batch.advantages,
        "response_mask": batch.response_mask,
    }
    polgrad_loss, polgrad_grad = _polgrad_loss_and_grad(config, tensors)
    logprobs = batch.logprobs.clone().requires_grad_(True)
    trl_loss = VENDORED[("trl", "bnpo")](
        logprobs=logprobs,
        old_logprobs=batch.old_logprobs,
        advantages=batch.advantages,
        response_mask=batch.response_mask,
    )
    (trl_grad,) = torch.autograd.grad(trl_loss, logprobs)
    assert math.isclose(polgrad_loss, float(trl_loss.detach()), rel_tol=RTOL, abs_tol=1e-12)
    assert_close(polgrad_grad, trl_grad, rtol=RTOL, atol=1e-12)


@given(batch=logprob_batches(max_b=6, max_t=8, max_gap=1.5))
def test_property_verl_token_sum_norm_factor_on_generated_batches(
    batch: LogprobBatch,
) -> None:
    """polgrad TOKEN_SUM_NORM = verl seq-mean-token-sum-norm · T/norm_len on any batch.

    Hypothesis version of the DEVIATIONS[0] factor over ragged generated batches
    (norm_len fixed at 5; T is the batch's padded width).
    """
    norm_len = 5
    config = _config(Aggregation.TOKEN_SUM_NORM, CLIP_VERL, norm_len=norm_len)
    tensors = {
        "logprobs": batch.logprobs,
        "old_logprobs": batch.old_logprobs,
        "advantages": batch.advantages,
        "response_mask": batch.response_mask,
    }
    polgrad_loss, _ = _polgrad_loss_and_grad(config, tensors)
    verl_loss = VENDORED[("verl", "pg_clip_seq_mean_token_sum_norm")](
        logprobs=batch.logprobs,
        old_logprobs=batch.old_logprobs,
        advantages=batch.advantages,
        response_mask=batch.response_mask,
    )
    t = batch.response_mask.shape[1]
    assert math.isclose(
        polgrad_loss, float(verl_loss.detach()) * t / norm_len, rel_tol=RTOL, abs_tol=1e-12
    )
