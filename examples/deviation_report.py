"""Conformance deviation reports: polgrad reference configs vs framework losses.

Runs ``polgrad.conformance.deviation_report`` for pairs of (polgrad
``PolicyLossConfig``, ``VENDORED`` framework loss) on seeded random ragged batches and
prints each ``DeviationReport.summary()``. The first group pins down which framework
settings reproduce which reference aggregation exactly (differences at float64
rounding); the second group shows measured deviations together with the analytic reason
for each. The documented deviation registry
(``polgrad.conformance.deviations.DEVIATIONS``) is printed at the end.

Run: .venv/bin/python examples/deviation_report.py
"""

from __future__ import annotations

import torch

from polgrad.aggregate import Aggregation
from polgrad.conformance.deviations import DEVIATIONS
from polgrad.conformance.harness import deviation_report
from polgrad.losses import ClipConfig, PolicyLossConfig, RatioKind, SurrogateKind

# Fixed comparison shapes. The verl "seq-mean-token-sum-norm" pair relies on the padded
# width being constant (its default divisor is the current batch's padded width).
SHAPES = ((4, 8), (3, 6))
NORM_SHAPES = ((4, 8),)
N_CASES = 64


def pg_clip(
    aggregation: Aggregation,
    *,
    ratio_cap: float | None = None,
    norm_len: int | None = None,
) -> PolicyLossConfig:
    """PPO-clip token-ratio config at eps 0.2/0.2 (every VENDORED wrapper's setting)."""
    return PolicyLossConfig(
        ratio=RatioKind.TOKEN,
        surrogate=SurrogateKind.PG_CLIP,
        clip=ClipConfig(eps_low=0.2, eps_high=0.2, ratio_cap=ratio_cap),
        aggregation=aggregation,
        norm_len=norm_len,
    )


# (title, expectation, config, framework, variant, shapes). verl's compute_policy_loss
# always applies the dual-clip floor with c=3.0, so configs meant to match verl set
# ratio_cap=3.0; OpenRLHF and TRL have no dual clip.
Pair = tuple[str, str, PolicyLossConfig, str, str, tuple[tuple[int, int], ...]]

MATCHING: tuple[Pair, ...] = (
    (
        "polgrad TOKEN_MEAN + dual-clip cap 3.0  vs  verl 'token-mean'",
        "identical semantics; differences at float64 rounding",
        pg_clip(Aggregation.TOKEN_MEAN, ratio_cap=3.0),
        "verl",
        "pg_clip_token_mean",
        SHAPES,
    ),
    (
        "polgrad TOKEN_MEAN  vs  openrlhf token_level_loss=True",
        "identical semantics; differences at float64 rounding",
        pg_clip(Aggregation.TOKEN_MEAN),
        "openrlhf",
        "pg_clip_token_mean",
        SHAPES,
    ),
    (
        "polgrad SEQ_MEAN_TOKEN_MEAN  vs  trl loss_type='grpo'",
        "identical semantics; differences at float64 rounding",
        pg_clip(Aggregation.SEQ_MEAN_TOKEN_MEAN),
        "trl",
        "grpo",
        SHAPES,
    ),
    (
        "polgrad TOKEN_SUM_NORM (norm_len=8)  vs  trl loss_type='dr_grpo' (budget=8)",
        "identical semantics on (4, 8) batches; differences at float64 rounding",
        pg_clip(Aggregation.TOKEN_SUM_NORM, norm_len=8),
        "trl",
        "dr_grpo",
        NORM_SHAPES,
    ),
)

DEVIATING: tuple[Pair, ...] = (
    (
        "polgrad SEQ_MEAN_TOKEN_MEAN + cap 3.0  vs  verl 'seq-mean-token-mean'",
        "verl divides each row's token sum by (L_i + 1e-8) instead of L_i, an\n"
        "O(1e-8/L_i) relative deflation: grad rel diffs ~1e-8, loss rel diffs of the\n"
        "same order but inflated on cases whose loss is near zero, cosine ~1",
        pg_clip(Aggregation.SEQ_MEAN_TOKEN_MEAN, ratio_cap=3.0),
        "verl",
        "pg_clip_seq_mean_token_mean",
        SHAPES,
    ),
    (
        "polgrad SEQ_MEAN_TOKEN_MEAN  vs  openrlhf token_level_loss=False",
        "same (L_i + 1e-8) row divisor as verl's sequence-mean mode: rel diffs ~1e-8",
        pg_clip(Aggregation.SEQ_MEAN_TOKEN_MEAN),
        "openrlhf",
        "pg_clip_seq_mean_token_mean",
        SHAPES,
    ),
    (
        "polgrad TOKEN_SUM_NORM (norm_len=16) + cap 3.0  vs  verl 'seq-mean-token-sum-norm' (T=8)",
        "verl's default divisor is the padded width of the current batch, not a fixed\n"
        "generation budget: on (4, 8) batches it returns norm_len/T = 2x the Dr.GRPO\n"
        "loss, so loss rel diff = 1/2 with grad cosine exactly 1 (pure rescaling)",
        pg_clip(Aggregation.TOKEN_SUM_NORM, ratio_cap=3.0, norm_len=16),
        "verl",
        "pg_clip_seq_mean_token_sum_norm",
        NORM_SHAPES,
    ),
    (
        "polgrad TOKEN_MEAN without dual clip  vs  verl 'token-mean'",
        "verl always applies the dual-clip floor -max(min(rA, clip(r)A), cA), c=3.0;\n"
        "cases with A<0 and ratio>3 take the floor branch (constant, zero gradient)\n"
        "while plain PG_CLIP keeps the unclipped term",
        pg_clip(Aggregation.TOKEN_MEAN),
        "verl",
        "pg_clip_token_mean",
        SHAPES,
    ),
    (
        "polgrad TOKEN_MEAN  vs  trl loss_type='grpo'",
        "aggregation mismatch: per-token weight m/N vs m/(B*L_i); ragged batches make\n"
        "both the loss and the gradient direction differ",
        pg_clip(Aggregation.TOKEN_MEAN),
        "trl",
        "grpo",
        SHAPES,
    ),
)


def run_group(title: str, pairs: tuple[Pair, ...], seed_base: int) -> None:
    print(f"== {title} ==")
    print()
    for index, (name, expectation, config, framework, variant, shapes) in enumerate(pairs):
        generator = torch.Generator().manual_seed(seed_base + index)
        report = deviation_report(
            config,
            framework,
            variant,
            n_cases=N_CASES,
            shapes=shapes,
            generator=generator,
        )
        print(f"[{name}]")
        print(f"expected: {expectation}")
        print(report.summary())
        print()


def main() -> None:
    run_group("matching pairs", MATCHING, seed_base=100)
    run_group("deviating pairs", DEVIATING, seed_base=200)

    print("== documented deviation registry (polgrad.conformance.deviations) ==")
    print()
    for deviation in DEVIATIONS:
        print(f"[{deviation.framework}] {deviation.component}")
        print(f"  pinned at: {deviation.version}")
        print(f"  {deviation.description}")
        print(f"  demonstrated by: {deviation.demonstrated_by}")
        print()


if __name__ == "__main__":
    main()
