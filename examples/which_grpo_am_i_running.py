"""Which GRPO am I running? One batch, many "GRPO" losses.

Builds a single synthetic ragged batch and evaluates it under four polgrad loss
configurations that all ship under the name "GRPO" somewhere (the GRPO paper equation as
in TRL v0.14-v0.15, the token-mean variant of TRL >= 0.16 and verl, Dr.GRPO's
fixed-budget normalization, and GSPO's sequence-level ratio), plus the vendored
verl/OpenRLHF loss functions and the TRL reimplementation from
``polgrad.conformance.harness.VENDORED``. The printed table shows that identical inputs
produce different losses and different per-sequence gradient weights, and the closing
notes state the exact source of each difference.

Run: .venv/bin/python examples/which_grpo_am_i_running.py
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor

from polgrad.aggregate import Aggregation
from polgrad.conformance.harness import VENDORED
from polgrad.losses import (
    ClipConfig,
    PolicyLossConfig,
    RatioKind,
    SurrogateKind,
    policy_loss,
)

SEED = 0
LENGTHS = (2, 4, 6, 8)  # response-token counts: a deliberately ragged batch
BATCH = len(LENGTHS)
WIDTH = max(LENGTHS)  # padded width T
NORM_LEN = 8  # Dr.GRPO fixed generation budget (here equal to the padded width)

RowFn = Callable[[Tensor], Tensor]


def build_batch() -> dict[str, Tensor]:
    """Seeded ragged batch; log-ratio gaps in [-1, 1] keep every ratio inside [1/e, e]."""
    gen = torch.Generator().manual_seed(SEED)
    mask = torch.arange(WIDTH).unsqueeze(0) < torch.tensor(LENGTHS).unsqueeze(1)
    logprobs = -(torch.rand((BATCH, WIDTH), generator=gen, dtype=torch.float64) * 2.0 + 0.1)
    gap = torch.rand((BATCH, WIDTH), generator=gen, dtype=torch.float64) * 2.0 - 1.0
    advantages = torch.tensor([1.5, -0.8, 0.6, -1.2], dtype=torch.float64)
    return {
        "logprobs": logprobs,
        "old_logprobs": logprobs - gap,  # ratio = exp(logprobs - old_logprobs) = exp(gap)
        "advantages": advantages,
        "response_mask": mask,
    }


def polgrad_row(config: PolicyLossConfig, batch: dict[str, Tensor]) -> RowFn:
    """Loss as a function of ``logprobs`` alone, everything else pinned to the batch."""

    def fn(logprobs: Tensor) -> Tensor:
        return policy_loss(
            config,
            logprobs=logprobs,
            old_logprobs=batch["old_logprobs"],
            advantages=batch["advantages"],
            response_mask=batch["response_mask"],
        ).loss

    return fn


def vendored_row(framework: str, variant: str, batch: dict[str, Tensor]) -> RowFn:
    """VENDORED wrapper as a function of ``logprobs``; advantages broadcast to [B, T]."""
    vendored_fn = VENDORED[(framework, variant)]
    advantages_tok = batch["advantages"].unsqueeze(1).expand(BATCH, WIDTH).contiguous()

    def fn(logprobs: Tensor) -> Tensor:
        return vendored_fn(
            logprobs=logprobs,
            old_logprobs=batch["old_logprobs"],
            advantages=advantages_tok,
            response_mask=batch["response_mask"],
        )

    return fn


def loss_and_grad_mass(fn: RowFn, logprobs: Tensor) -> tuple[float, Tensor]:
    """Scalar loss and the per-sequence gradient mass sum_t |d loss / d logprobs[b, t]|."""
    lp = logprobs.detach().clone().requires_grad_(True)
    loss = fn(lp)
    (grad,) = torch.autograd.grad(loss, lp)
    return float(loss.detach()), grad.abs().sum(dim=1)


def pg_clip_config(aggregation: Aggregation, *, norm_len: int | None = None) -> PolicyLossConfig:
    """Plain PPO-clip token-ratio config at eps 0.2/0.2 under the given aggregation."""
    return PolicyLossConfig(
        ratio=RatioKind.TOKEN,
        surrogate=SurrogateKind.PG_CLIP,
        clip=ClipConfig(eps_low=0.2, eps_high=0.2),
        aggregation=aggregation,
        norm_len=norm_len,
    )


def main() -> None:
    batch = build_batch()
    logprobs = batch["logprobs"]
    n_tokens = int(batch["response_mask"].sum())

    gspo_config = PolicyLossConfig(
        ratio=RatioKind.SEQUENCE,
        surrogate=SurrogateKind.PG_CLIP,
        clip=ClipConfig(eps_low=3e-4, eps_high=4e-4),
        aggregation=Aggregation.SEQ_MEAN_TOKEN_MEAN,
    )
    rows: list[tuple[str, RowFn]] = [
        (
            "polgrad seq-mean-token-mean (GRPO paper; TRL v0.14-0.15)",
            polgrad_row(pg_clip_config(Aggregation.SEQ_MEAN_TOKEN_MEAN), batch),
        ),
        (
            "polgrad token-mean (TRL >= 0.16; verl 'token-mean')",
            polgrad_row(pg_clip_config(Aggregation.TOKEN_MEAN), batch),
        ),
        (
            f"polgrad token-sum-norm (Dr.GRPO, norm_len={NORM_LEN})",
            polgrad_row(pg_clip_config(Aggregation.TOKEN_SUM_NORM, norm_len=NORM_LEN), batch),
        ),
        (
            "polgrad GSPO sequence ratio (eps 3e-4/4e-4)",
            polgrad_row(gspo_config, batch),
        ),
        ("vendored verl 'token-mean'", vendored_row("verl", "pg_clip_token_mean", batch)),
        (
            "vendored verl 'seq-mean-token-mean'",
            vendored_row("verl", "pg_clip_seq_mean_token_mean", batch),
        ),
        (
            "vendored openrlhf token_level_loss=True",
            vendored_row("openrlhf", "pg_clip_token_mean", batch),
        ),
        (
            "vendored openrlhf token_level_loss=False",
            vendored_row("openrlhf", "pg_clip_seq_mean_token_mean", batch),
        ),
        ("reimplemented trl loss_type='grpo'", vendored_row("trl", "grpo", batch)),
        (
            f"reimplemented trl loss_type='dr_grpo' (budget={WIDTH})",
            vendored_row("trl", "dr_grpo", batch),
        ),
    ]

    losses: dict[str, float] = {}
    print("one synthetic batch, every loss below sees the identical tensors")
    print(f"B={BATCH} sequences, response lengths {list(LENGTHS)}, N={n_tokens} tokens,")
    print("per-sequence advantages", batch["advantages"].tolist())
    print()
    header = f"{'configuration':<56} {'loss':>12}  " + "  ".join(
        f"seq{i} L={length}" for i, length in enumerate(LENGTHS)
    )
    print(header)
    print("-" * len(header))
    for name, fn in rows:
        loss, mass = loss_and_grad_mass(fn, logprobs)
        losses[name] = loss
        cells = "  ".join(f"{float(m):8.5f}" for m in mass)
        print(f"{name:<56} {loss:>12.8f}  {cells}")
    print()
    print("(per-sequence columns: gradient mass sum_t |d loss / d logprobs[b, t]|)")
    print()

    seq_mean = losses["polgrad seq-mean-token-mean (GRPO paper; TRL v0.14-0.15)"]
    token_mean = losses["polgrad token-mean (TRL >= 0.16; verl 'token-mean')"]
    dr_grpo = losses[f"polgrad token-sum-norm (Dr.GRPO, norm_len={NORM_LEN})"]
    factor = n_tokens / (BATCH * NORM_LEN)
    checks = [
        (
            "verl 'token-mean' == polgrad token-mean",
            losses["vendored verl 'token-mean'"] - token_mean,
            "same semantics; verl's ever-present dual-clip cap c=3.0 cannot bind here "
            "(max ratio e^1 = 2.72)",
        ),
        (
            "openrlhf token_level_loss=True == polgrad token-mean",
            losses["vendored openrlhf token_level_loss=True"] - token_mean,
            "OpenRLHF's token-level flag is exactly TOKEN_MEAN",
        ),
        (
            "trl 'grpo' == polgrad seq-mean-token-mean",
            losses["reimplemented trl loss_type='grpo'"] - seq_mean,
            "TRL loss_type='grpo' is exactly SEQ_MEAN_TOKEN_MEAN",
        ),
        (
            "verl 'seq-mean-token-mean' ~= polgrad seq-mean-token-mean",
            losses["vendored verl 'seq-mean-token-mean'"] - seq_mean,
            "verl divides each row by (L_i + 1e-8), an O(1e-9) relative deflation",
        ),
        (
            "openrlhf token_level_loss=False ~= polgrad seq-mean-token-mean",
            losses["vendored openrlhf token_level_loss=False"] - seq_mean,
            "same (L_i + 1e-8) row divisor as verl's sequence-mean mode",
        ),
        (
            "trl 'dr_grpo' == polgrad token-sum-norm",
            losses[f"reimplemented trl loss_type='dr_grpo' (budget={WIDTH})"] - dr_grpo,
            f"both divide token sums by B*budget = {BATCH * NORM_LEN}",
        ),
        (
            f"Dr.GRPO == token-mean x N/(B*norm_len) = x {factor}",
            dr_grpo - token_mean * factor,
            "proportional weights: m/(B*norm_len) vs m/N",
        ),
    ]
    print("cross-checks on this batch (measured loss differences):")
    for claim, diff, why in checks:
        print(f"  {claim:<64} |diff| = {abs(diff):.2e}")
        print(f"    source: {why}")
    print()

    gspo_result = policy_loss(
        gspo_config,
        logprobs=batch["logprobs"],
        old_logprobs=batch["old_logprobs"],
        advantages=batch["advantages"],
        response_mask=batch["response_mask"],
    )
    seq_ratios = [round(float(r), 4) for r in gspo_result.ratio[:, 0]]
    killed = (gspo_result.clipped_low | gspo_result.clipped_high).any(dim=1)
    killed_rows = [i for i, k in enumerate(killed.tolist()) if k]
    flowing_rows = [i for i, k in enumerate(killed.tolist()) if not k]

    print("where the differences come from:")
    print(
        "* seq-mean-token-mean (GRPO paper eq.; TRL GRPOTrainer default in v0.14-v0.15) gives\n"
        "  every sequence total gradient weight 1/B, so each token weighs 1/(B*L_i): tokens of\n"
        "  the L=2 sequence pull 4x harder per token than tokens of the L=8 sequence."
    )
    print(
        "* token-mean (verl 'token-mean'; TRL default since v0.16.0, TRL PR #2881) weighs\n"
        "  every token 1/N, so a sequence's gradient mass grows with its length -- compare the\n"
        "  per-sequence columns of the two rows. The two aggregations only coincide when all\n"
        "  sequences have equal length; this batch is ragged, so loss and gradients differ."
    )
    print(
        "* Dr.GRPO (arXiv 2503.20783) divides token sums by the constant B*norm_len instead of\n"
        "  realized token counts. Per-token weights are proportional to token-mean's, so on one\n"
        f"  batch it is exactly N/(B*norm_len) = {factor} times the token-mean loss; across\n"
        "  batches the scale no longer depends on the realized lengths."
    )
    print(
        "* GSPO (arXiv 2507.18071) replaces per-token ratios with one length-normalized sequence\n"
        "  ratio s_i = exp(mean_t log-ratio) and clips it in a band of width ~3e-4. At this\n"
        f"  batch's off-policy gap every s_i is far outside the band: s = {seq_ratios} vs\n"
        "  [0.9997, 1.0004]. A row loses its whole gradient when it crosses on the pessimistic\n"
        f"  side (A>0 with s above the band, or A<0 with s below): rows {killed_rows} here --\n"
        f"  the zero columns above. Rows {flowing_rows} cross on the other side and enter\n"
        "  unclipped. This is a different objective, not a rescaling of the others."
    )
    print(
        "* vendored verl applies the dual-clip floor -max(min(rA, clip(r)A), cA) with c=3.0\n"
        "  unconditionally for A<0; ratios here stay below e = 2.72, so it matches plain PG_CLIP\n"
        "  token-mean to float64 rounding. With ratios above 3 and A<0 it would differ."
    )
    print(
        "* the ~1e-10 gaps of verl/OpenRLHF sequence-mean modes are their (L_i + 1e-8) row\n"
        "  divisor, a documented deviation (polgrad.conformance.deviations.DEVIATIONS)."
    )


if __name__ == "__main__":
    main()
