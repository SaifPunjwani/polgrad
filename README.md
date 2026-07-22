# polgrad

[![CI](https://github.com/SaifPunjwani/polgrad/actions/workflows/ci.yml/badge.svg)](https://github.com/SaifPunjwani/polgrad/actions/workflows/ci.yml)

Reference semantics, conformance testing, and pathology diagnostics for LLM
policy-gradient post-training. Pure functions over tensors: no trainers, no models, no
GPU requirement, no dependency beyond PyTorch.

## The problem

The RL post-training stack does not agree on what "the GRPO loss" is. The same batch of
`(logprobs, old_logprobs, advantages, response_mask)` produces materially different losses
and gradients depending on which framework — and which framework *version* — computes it:

- TRL's `GRPOTrainer` aggregated per-sequence-then-batch (the GRPO paper equation) in
  v0.14–v0.15, then switched its default to global token normalization in v0.16.0
  ([trl#2881](https://github.com/huggingface/trl/pull/2881); see
  [trl#2995](https://github.com/huggingface/trl/issues/2995) for the debate that switch
  sparked — resolved for KL logging only in
  [trl#3004](https://github.com/huggingface/trl/pull/3004)). Both defaults are still in
  the wild, and current TRL defaults to `loss_type="dapo"` — token-mean over the global
  gradient-accumulated batch.
- verl ships four aggregation modes; its Dr.GRPO-style mode divides by the *current padded
  length* where the Dr.GRPO paper's released code divides by a fixed generation budget
  ([arXiv:2503.20783](https://arxiv.org/abs/2503.20783)). polgrad's conformance suite
  measures the exact factor.
- The k3 KL estimator, differentiated as a loss term, does not propagate the gradient of
  the KL it estimates — its pathwise gradient is biased for the stated objective
  ([arXiv:2512.21852](https://arxiv.org/abs/2512.21852),
  [arXiv:2510.01555](https://arxiv.org/abs/2510.01555)) — yet k3-as-loss is a common
  configuration.
- Rollout engines report logprobs that differ from the trainer's recompute, silently
  making "on-policy" training off-policy; frameworks patched in truncated
  importance-sampling corrections ([verl#2953](https://github.com/volcengine/verl/pull/2953)).

These are semantic choices with measurable consequences, scattered across codebases as
implicit defaults. polgrad makes each one an explicit, typed, tested object.

## What disagreement looks like

One ragged batch — 4 sequences of lengths (2, 4, 6, 8), identical tensors into every row
of this table (`examples/which_grpo_am_i_running.py` prints it, measured, on every run):

| configuration | loss | grad mass, L=2 seq | grad mass, L=8 seq |
| --- | --- | --- | --- |
| seq-mean-token-mean (GRPO paper; TRL ≤ 0.15) | 0.24038972 | 0.19568 | 0.29263 |
| token-mean (verl; TRL ≥ 0.16) | 0.51717388 | 0.07827 | 0.46821 |
| token-sum-norm (Dr.GRPO, budget 8) | 0.32323368 | 0.04892 | 0.29263 |
| GSPO sequence-level ratio | 0.15429390 | 0.19550 | 0.00000 |

A 2× loss difference and opposite length-weighting of gradients, from settings frameworks
treat as interchangeable defaults. The same script cross-checks polgrad against the
vendored verl and OpenRLHF implementations (agreement to 0 or ~1e-10, each residual
explained) so the table is evidence, not assertion.

## Install

```sh
pip install polgrad
```

or, for the development version, `pip install git+https://github.com/SaifPunjwani/polgrad`.

Python ≥ 3.10, PyTorch ≥ 2.1, CPU is enough.

## The loss algebra

Every named variant is a frozen config over
`ratio kind × surrogate × clip × aggregation × KL placement`, not a separate code path:

```python
import torch
from polgrad import Aggregation, ClipConfig, PolicyLossConfig, RatioKind, SurrogateKind, policy_loss

grpo_paper = PolicyLossConfig(
    ratio=RatioKind.TOKEN,
    surrogate=SurrogateKind.PG_CLIP,
    clip=ClipConfig(eps_low=0.2, eps_high=0.2),
    aggregation=Aggregation.SEQ_MEAN_TOKEN_MEAN,
)
trl_016_default = PolicyLossConfig(
    ratio=RatioKind.TOKEN,
    surrogate=SurrogateKind.PG_CLIP,
    clip=ClipConfig(eps_low=0.2, eps_high=0.2),
    aggregation=Aggregation.TOKEN_MEAN,
)

B, T = 4, 8
g = torch.Generator().manual_seed(0)
logprobs = -torch.rand((B, T), generator=g, dtype=torch.float64)
old_logprobs = logprobs.detach() + 0.1 * torch.randn((B, T), generator=g, dtype=torch.float64)
advantages = torch.randn((B,), generator=g, dtype=torch.float64)
mask = torch.arange(T).expand(B, T) < torch.tensor([[2], [4], [6], [8]])

for cfg in (grpo_paper, trl_016_default):
    out = policy_loss(cfg, logprobs=logprobs.clone().requires_grad_(), old_logprobs=old_logprobs,
                      advantages=advantages, response_mask=mask)
    print(f"{cfg.aggregation.value:>24}  loss={out.loss.item():+.6f}")
```

Ten algorithms ship as registry entries with every constant traced to its paper or the
paper's released code — and explicit `notes` where the paper is silent:

```python
from polgrad import ALGORITHMS, describe_algorithm
print(describe_algorithm("dapo"))     # clip-higher eps, token-mean, no KL — with sources
print(sorted(ALGORITHMS))
# ['cispo', 'dapo', 'dr_grpo', 'grpo', 'grpo_tis', 'gspo', 'gspo_token', 'ppo', 'reinforce_pp', 'rloo']
```

`polgrad.aggregate.effective_token_weights` returns the closed-form per-token gradient
weight each aggregation mode induces — including the micro-batch/gradient-accumulation
weights, whose inequivalence to full-batch aggregation is computed exactly rather than
discovered in production.

## Diagnostics

Metrics for the failure modes RL post-training actually exhibits, computed from tensors a
training loop already has. Every threshold has a documented null distribution with a
Monte Carlo test calibrating it — no magic constants:

```python
from polgrad.diagnostics import importance_ess, logprob_mismatch, clip_report, entropy_trend

ess = importance_ess(trainer_logprobs, rollout_logprobs, mask)   # ESS/n, with exact null == 1
gap = logprob_mismatch(trainer_logprobs, rollout_logprobs, mask) # k1/k2/k3 drift, catastrophic tokens
print(ess.summary()); print(gap.summary())
```

Entropy diagnostics accept either sampled-token logprobs (the Monte Carlo estimator,
valid on-policy) or exact per-token entropies computed from full logits — the docs derive
when each is valid. `examples/diagnose_run.py` synthesizes a 200-step run with injected
pathologies (rollout↔trainer drift, entropy collapse, length bias) and shows each
detector firing at the injection point.

## Verification as public API

`polgrad.verify` exports the harness the library is tested with, so a new variant gets
falsifiable checks for free: fp64 `gradcheck` over ragged batches, central
finite-difference comparison against a *supplied analytic formula* (this catches wrong
derivations, not just wrong code), a softmax bandit with closed-form policy gradient and
KL, and hand-derived golden cases whose arithmetic is written out in
[docs/derivations/goldens.md](docs/derivations/goldens.md).

## Conformance

`src/polgrad/conformance/_vendor/` contains loss functions vendored verbatim from verl and
OpenRLHF at pinned commits, with SHA256-enforced provenance headers; TRL is represented by
a labeled reimplementation pinned to a version. Recorded fixtures keep CI framework-free.
Demonstrated differences are registered in `polgrad.conformance.DEVIATIONS` — each entry
carries the pytest node id that demonstrates it, and the wording is deliberately neutral:
these are *deviations between published equations and shipped defaults*, not bug reports.
Frameworks make defensible engineering choices; the point is that the choices are
currently invisible.

```python
from polgrad.conformance import DEVIATIONS, deviation_report
for d in DEVIATIONS:
    print(f"{d.framework} {d.version}: {d.description}")
```

A weekly scheduled job re-fetches the tracked upstream loss functions at HEAD and diffs
them against the pins (`tools/check_upstream_drift.py`), so framework churn becomes a
detected, filed event rather than silent rot.

For framework and trainer authors, `polgrad.testing` turns conformance into a test:

```python
from polgrad.testing import assert_conforms

def test_my_grpo_matches_the_paper():
    assert_conforms(my_loss_fn, "grpo")   # seeded batches; compares loss and gradients
```

It installs as a pytest plugin (a `polgrad_batches` fixture ships with the package), and
`assert_conforms` raises with the full deviation report — max loss/gradient differences
and the worst-case seed — when semantics drift.

## Every claim is machine-checked

The derivation pages in `docs/` state each identity and link the test that enforces it.
A selection:

| claim | derivation | enforced by |
| --- | --- | --- |
| `aggregate(x, m, mode)` ≡ `Σ wₜ xₜ` with closed-form weights, bitwise, incl. autograd | [aggregation.md](docs/derivations/aggregation.md) | `tests/test_aggregate.py` |
| micro-batch aggregation weights == autograd of an explicit accumulation loop | [aggregation.md](docs/derivations/aggregation.md) | `tests/test_cross.py` |
| E[k1] = E[k3] = KL(π‖ref) exactly; k2's bias quantified | [kl.md](docs/derivations/kl.md) | `tests/test_kl.py` |
| k2-as-loss gradient ≡ unbiased reverse-KL score-function gradient (bitwise, all aggregations) | [kl.md](docs/derivations/kl.md) | `tests/test_cross.py` |
| k3-as-loss gradient ≠ ∇KL (Monte Carlo, >10 CLT tolerances from the analytic gradient) | [kl.md](docs/derivations/kl.md) | `tests/test_kl.py` |
| PG_CLIP ≡ PG ≡ REINFORCE gradients on-policy (bitwise) | [losses.md](docs/derivations/losses.md) | `tests/test_losses.py` |
| GSPO-token value ≡ GSPO-sequence value; gradient is token-local | [losses.md](docs/derivations/losses.md) | `tests/test_losses.py` |
| clip zero-gradient region == autograd-zero tokens, incl. dual-clip | [clipping.md](docs/diagnostics/clipping.md) | `tests/test_cross.py` |
| RLOO leave-one-out ≡ (G/(G−1))·(r − mean) | [advantages.md](docs/derivations/advantages.md) | `tests/test_advantages.py` |
| ESS/n null: ≡ 1 on-policy; → exp(−σ²) under N(0,σ²) log-weight drift | [ess.md](docs/diagnostics/ess.md) | `tests/test_diagnostics_ess.py` |
| entropy changepoint test has exact permutation level ≤ α | [entropy.md](docs/diagnostics/entropy.md) | `tests/test_diagnostics_entropy.py` |
| all ten registry algorithms optimize a bandit to convergence | [variants.md](docs/derivations/variants.md) | `tests/test_cross.py` |

The suite is 554 tests — analytic goldens, Hypothesis property tests over ragged masked
batches, fp64 gradcheck, and seeded Monte Carlo with CLT-derived tolerances — and runs in
about fourteen seconds serially on a laptop CPU (measured 13.8s; a few seconds with
`pytest -n auto`, the CI invocation).

## What polgrad is not

Not a trainer, not a serving stack, not a framework critique, and not a benchmark. It
never runs your training loop. It defines what the equations say, measures what
implementations do, and gives you instruments for the gap.

## References

Papers whose semantics are implemented and tested: PPO
([1707.06347](https://arxiv.org/abs/1707.06347)), GAE
([1506.02438](https://arxiv.org/abs/1506.02438)), dual-clip PPO
([1912.09729](https://arxiv.org/abs/1912.09729)), GRPO / DeepSeekMath
([2402.03300](https://arxiv.org/abs/2402.03300)), RLOO
([2402.14740](https://arxiv.org/abs/2402.14740)), REINFORCE++
([2501.03262](https://arxiv.org/abs/2501.03262)), Dr.GRPO
([2503.20783](https://arxiv.org/abs/2503.20783)), DAPO
([2503.14476](https://arxiv.org/abs/2503.14476)), CISPO / MiniMax-M1
([2506.13585](https://arxiv.org/abs/2506.13585)), GSPO
([2507.18071](https://arxiv.org/abs/2507.18071)); KL estimators after
[Schulman's approximation note](http://joschu.net/blog/kl-approx.html), with the as-loss
gradient analysis of [2512.21852](https://arxiv.org/abs/2512.21852) and
[2510.01555](https://arxiv.org/abs/2510.01555).

## License

Apache-2.0. Vendored framework code retains its upstream Apache-2.0 attribution; see
`NOTICE`.
