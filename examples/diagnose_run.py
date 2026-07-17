"""Diagnosing a pathological RL run with the polgrad diagnostics suite.

Synthesizes a 200-step training trajectory with four injected pathologies and shows
that each diagnostic in ``polgrad.diagnostics`` isolates its own:

1. rollout <-> trainer logprob mismatch growing linearly over training, with two
   catastrophic single-token outliers injected near the end (``logprob_mismatch``);
2. off-policy drift between the update policy and the sampling policy that shrinks the
   importance-sampling effective sample size (``importance_ess``), plus a stale
   asynchronous rollout queue whose staleness grows with queue position
   (``sliding_ess``);
3. an entropy collapse: the per-step entropy declines gently, then drops by ~0.7 nats
   at step 140 (``token_entropy_estimate`` + ``entropy_trend``);
4. length bias under ``SEQ_MEAN_TOKEN_MEAN``: the reward drifts toward favoring long
   completions, so advantage magnitude grows with response length
   (``length_bias_probe``); ``clip_report`` additionally shows the drift-driven growth
   of clipped and gradient-dead tokens.

Everything is synthetic, seeded, and CPU-only; each report's ``summary()`` is printed,
followed by a closing paragraph of what was detected.

Run: .venv/bin/python examples/diagnose_run.py
"""

from __future__ import annotations

import textwrap

import torch
from torch import Tensor

from polgrad.aggregate import Aggregation
from polgrad.diagnostics.clipping import clip_report
from polgrad.diagnostics.entropy import entropy_trend, token_entropy_estimate
from polgrad.diagnostics.ess import importance_ess, sliding_ess
from polgrad.diagnostics.length_bias import length_bias_probe
from polgrad.diagnostics.mismatch import logprob_mismatch
from polgrad.losses import ClipConfig

SEED = 0
STEPS = 200
BATCH = 32
WIDTH = 24  # padded response width T
MIN_LEN = 4
CHANGEPOINT = 140  # first step of the collapsed-entropy regime
TRACE_STEPS = (0, 50, 100, 150, 199)
CATASTROPHIC_FROM = 190  # steps >= this get two injected catastrophic rollout tokens


def entropy_target(step: int) -> float:
    """Scheduled true entropy: gentle decline, then a 0.68-nat collapse at step 140."""
    if step < CHANGEPOINT:
        return 2.0 - 0.0015 * step
    return 1.1 - 0.001 * (step - CHANGEPOINT)


def mismatch_sigma(step: int) -> float:
    """Rollout-engine logprob noise std, growing linearly over training."""
    return 0.002 + 0.0004 * step


def drift_sigma(step: int) -> float:
    """Per-token log-ratio drift std between update policy and sampling policy."""
    return 0.01 + 0.0015 * step


def length_bias_coef(step: int) -> float:
    """Advantage-magnitude-per-token coefficient: the reward drifts toward length."""
    return 0.04 * step / (STEPS - 1)


def synthesize_step(step: int, gen: torch.Generator) -> dict[str, Tensor]:
    """One step's batch: masks, four logprob streams, and per-sequence advantages."""
    lengths = torch.randint(MIN_LEN, WIDTH + 1, (BATCH,), generator=gen)
    mask = torch.arange(WIDTH).unsqueeze(0) < lengths.unsqueeze(1)
    noise = torch.randn((BATCH, WIDTH), generator=gen, dtype=torch.float64)
    sampled = (-entropy_target(step) + 0.3 * noise).clamp(max=-0.01)

    # The trainer's recompute is taken as ground truth; the rollout engine reports it
    # with kernel/precision noise that grows as the serving stack drifts.
    old_logprobs = sampled
    rollout_noise = torch.randn((BATCH, WIDTH), generator=gen, dtype=torch.float64)
    rollout_logprobs = old_logprobs - mismatch_sigma(step) * rollout_noise
    if step >= CATASTROPHIC_FROM:
        rollout_logprobs = rollout_logprobs.clone()
        rollout_logprobs[0, 0] -= 7.0
        rollout_logprobs[1, 1] -= 7.0

    # The update policy has taken minibatch steps since sampling: off-policy drift.
    drift = torch.randn((BATCH, WIDTH), generator=gen, dtype=torch.float64)
    logprobs_new = old_logprobs + drift_sigma(step) * drift

    # Group-normalized-style per-sequence advantages; the injected pathology makes
    # advantage magnitude grow with response length as the reward drifts toward length.
    magnitude = (
        0.2
        + length_bias_coef(step) * lengths.to(torch.float64)
        + 0.05 * torch.randn((BATCH,), generator=gen, dtype=torch.float64)
    ).abs()
    signs = torch.where(torch.rand((BATCH,), generator=gen) < 0.5, -1.0, 1.0)
    advantages = (magnitude * signs).to(torch.float64)
    return {
        "response_mask": mask,
        "logprobs_new": logprobs_new,
        "old_logprobs": old_logprobs,
        "rollout_logprobs": rollout_logprobs,
        "advantages": advantages,
    }


def synthesize_rollout_queue(gen: torch.Generator) -> dict[str, Tensor]:
    """Asynchronous rollout queue of 128 sequences; staleness grows with queue index."""
    rows = 128
    lengths = torch.randint(MIN_LEN, WIDTH + 1, (rows,), generator=gen)
    mask = torch.arange(WIDTH).unsqueeze(0) < lengths.unsqueeze(1)
    old = (-1.5 + 0.3 * torch.randn((rows, WIDTH), generator=gen, dtype=torch.float64)).clamp(
        max=-0.01
    )
    sigma = 0.02 + 0.28 * torch.arange(rows, dtype=torch.float64).unsqueeze(1) / (rows - 1)
    new = old + sigma * torch.randn((rows, WIDTH), generator=gen, dtype=torch.float64)
    return {"logprobs_new": new, "old_logprobs": old, "response_mask": mask}


def main() -> None:
    gen = torch.Generator().manual_seed(SEED)
    entropy_series: list[float] = []
    ess_trace: dict[int, float] = {}
    gap_std_trace: dict[int, float] = {}
    kept: dict[int, dict[str, Tensor]] = {}

    for step in range(STEPS):
        batch = synthesize_step(step, gen)
        entropy_series.append(
            token_entropy_estimate(batch["old_logprobs"], batch["response_mask"]).entropy_estimate
        )
        if step in TRACE_STEPS:
            ess_trace[step] = importance_ess(
                batch["logprobs_new"], batch["old_logprobs"], batch["response_mask"]
            ).ess_ratio
            gap_std_trace[step] = logprob_mismatch(
                batch["old_logprobs"], batch["rollout_logprobs"], batch["response_mask"]
            ).gap_std
        if step in (0, STEPS - 1):
            kept[step] = batch
    first, last = kept[0], kept[STEPS - 1]

    print(f"synthetic {STEPS}-step run: B={BATCH}, T={WIDTH}, seed={SEED}")
    print()

    print("== 1. rollout <-> trainer logprob mismatch (polgrad.diagnostics.mismatch) ==")
    print("gap std trace:", "  ".join(f"step {s}: {v:.4f}" for s, v in gap_std_trace.items()))
    print(f"(steps >= {CATASTROPHIC_FROM} also carry the two injected catastrophic tokens,")
    print(f"which dominate the step-{STEPS - 1} std)")
    print()
    print("final-step report:")
    final_mismatch = logprob_mismatch(
        last["old_logprobs"], last["rollout_logprobs"], last["response_mask"]
    )
    print(final_mismatch.summary())
    print()

    print("== 2. off-policy drift shrinking ESS (polgrad.diagnostics.ess) ==")
    print(
        "sequence-level ESS/n trace:",
        "  ".join(f"step {s}: {v:.3f}" for s, v in ess_trace.items()),
    )
    print()
    print("final-step report:")
    print(
        importance_ess(last["logprobs_new"], last["old_logprobs"], last["response_mask"]).summary()
    )
    print()
    queue = synthesize_rollout_queue(gen)
    windows = sliding_ess(
        queue["logprobs_new"], queue["old_logprobs"], queue["response_mask"], window=48, step=16
    )
    print("stale rollout queue (128 sequences, staleness grows with queue position),")
    print("sliding ESS/window, window=48, step=16:")
    print("  " + "  ".join(f"{float(v):.3f}" for v in windows))
    print()

    print("== 3. entropy collapse at a changepoint (polgrad.diagnostics.entropy) ==")
    print(
        "entropy trace (nats):",
        "  ".join(f"step {s}: {entropy_series[s]:.3f}" for s in (0, 100, 139, 140, 199)),
    )
    print()
    trend = entropy_trend(
        torch.tensor(entropy_series, dtype=torch.float64),
        window=120,
        n_perm=999,
        alpha=0.05,
        generator=torch.Generator().manual_seed(SEED + 1),
    )
    print(trend.summary())
    print()

    print("== 4. length bias under SEQ_MEAN_TOKEN_MEAN (polgrad.diagnostics.length_bias) ==")
    print("under SEQ_MEAN_TOKEN_MEAN the aggregation-induced (structural) slope is zero,")
    print("so a slope whose CI excludes zero is data-level bias: |A| correlates with length.")
    print()
    for label, batch in (("step 0", first), (f"step {STEPS - 1}", last)):
        probe = length_bias_probe(
            batch["advantages"],
            batch["response_mask"],
            agg_mode=Aggregation.SEQ_MEAN_TOKEN_MEAN,
        )
        print(f"{label}:")
        print(probe.summary())
        print()

    print("== 5. clip pressure from the same drift (polgrad.diagnostics.clipping) ==")
    clip = ClipConfig(eps_low=0.2, eps_high=0.2, ratio_cap=3.0)
    for label, batch in (("step 0", first), (f"step {STEPS - 1}", last)):
        ratio = torch.exp(batch["logprobs_new"] - batch["old_logprobs"])
        report = clip_report(ratio, batch["advantages"], batch["response_mask"], clip)
        print(f"{label}:")
        print(report.summary())
        print()

    final_ess = ess_trace[STEPS - 1]
    change = "not detected" if trend.changepoint_index is None else str(trend.changepoint_index)
    print("== what each diagnostic detected ==")
    closing = (
        "The mismatch report caught the serving/trainer divergence injected in (1): the "
        f"trainer-vs-rollout gap std grew from {gap_std_trace[0]:.4f} at step 0 to "
        f"{gap_std_trace[150]:.4f} at step 150, and the final-step std "
        f"({gap_std_trace[199]:.4f}) is dominated by the two tokens whose rollout logprobs "
        f"were shifted by -7.0, flagged as catastrophic_count="
        f"{final_mismatch.catastrophic_count} -- training silently became off-policy even "
        "though no optimizer step was 'wrong'. The ESS report quantified pathology (2): "
        f"sequence-level ESS/n fell from {ess_trace[0]:.3f} to {final_ess:.3f}, i.e. the "
        f"final update behaves as if it had about {final_ess * BATCH:.0f} of its {BATCH} "
        "sequences, and the sliding-window trace over the stale rollout queue shows the "
        "same decay within a single buffer -- the cue to shrink rollout reuse or refresh "
        "the queue. The entropy trend flagged pathology (3): a negative Theil-Sen slope "
        f"with a CUSUM changepoint at step {change} (the injected collapse begins at step "
        f"{CHANGEPOINT}), separating gradual sharpening from a regime change. The "
        "length-bias probe isolated pathology (4): at step 0 the slope's 95% CI covers "
        "zero, while at the final step it is positive and excludes zero -- longer "
        "completions carry more loss mass per sequence, on top of SEQ_MEAN_TOKEN_MEAN's "
        "structural per-token dilution 1/(B*L_i). The clip report ties (2) back to the "
        "optimizer: by the final step a large share of tokens sits outside the PPO band "
        "and a visible fraction of gradients is dead, so the drift is not just a "
        "statistics problem but an update-geometry problem."
    )
    print(textwrap.fill(closing, width=88))


if __name__ == "__main__":
    main()
