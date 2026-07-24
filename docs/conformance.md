# Conformance, deviations, and drift

polgrad treats framework behavior as *data*, not as a dependency. Three mechanisms
implement that: a comparison harness, a registry of demonstrated deviations, and a
scheduled check that detects when upstream code moves.

## What conformance certifies

A conformance comparison feeds the same seeded batch — `logprobs`, `old_logprobs`,
`advantages`, `response_mask` — to two loss functions and compares two things: the
scalar loss, and $\partial \mathcal{L} / \partial \log \pi$ at every response token.
Both are computed in float64, and relative differences use a floor so that near-zero
references do not manufacture large relative errors.

It does **not** certify advantage estimation (upstream of the loss), KL-in-reward
shaping, data handling, or anything that depends on trainer state. Those exclusions
are stated on every wrapper.

## Comparison targets

Targets come in two forms, and the distinction is kept visible everywhere:

**Vendored code** (`polgrad.conformance._vendor`) is copied verbatim from upstream at a
pinned commit. Each file's header records the repository, commit SHA, source paths,
upstream file SHA256s, the license, the definitions kept, and every edit made (import
stripping only). A self-integrity test recomputes the hash of the vendored body on every
CI run, so the code cannot drift silently in-tree.

**Labeled reimplementations** (in `harness.py`) exist where a framework's loss cannot be
extracted as a pure function — TRL's `GRPOTrainer._compute_loss` reads `self.beta`,
accelerator state, and normalizer buffers; verl's `compute_policy_loss_gspo` and
`compute_policy_loss_cispo` read an `ActorConfig`. These reproduce the upstream
arithmetic with upstream variable names, and each docstring pins the permalink, the
upstream file hash, and every config default that was frozen along with where that
default comes from. They are never presented as vendored code.

Registered pairs, addressable through `polgrad.conformance.VENDORED`:

| framework | variants |
| --- | --- |
| verl | `pg_clip_token_mean`, `pg_clip_seq_mean_token_mean`, `pg_clip_seq_mean_token_sum`, `pg_clip_seq_mean_token_sum_norm`, `gspo`, `cispo` |
| openrlhf | `pg_clip_token_mean`, `pg_clip_seq_mean_token_mean` |
| trl | `grpo`, `bnpo`, `dr_grpo` |

Recorded fixtures (`tests/fixtures/*.json`) hold inputs, outputs, and provenance for
each pair, so CI never imports a framework. Regenerate them with
`python tools/record_fixtures.py`; the seeding is derived from the pair name, so
regeneration is byte-stable.

## Deviations

A `Deviation` is registered only when a test demonstrates it against the target code.
Wording is neutral by policy: a deviation is a difference between a published equation
and a shipped default, not a defect. Frameworks make defensible engineering choices;
the point is that those choices are currently invisible to the person training a model.

```python
from polgrad.conformance import DEVIATIONS

for d in DEVIATIONS:
    print(d.framework, d.component, "->", d.demonstrated_by)
```

Currently registered:

- **verl `agg_loss("seq-mean-token-sum-norm")`** divides by the current padded width
  rather than the fixed generation budget the Dr.GRPO code uses, so the loss equals the
  Dr.GRPO form times $\mathrm{norm\_len}/T_\text{padded}$ — a factor that varies with
  batch padding. Demonstrated by
  `tests/test_conformance.py::test_verl_token_sum_norm_deviates_from_dr_grpo_by_norm_len_over_padded_len`.
- **verl `agg_loss("seq-mean-token-mean")`** and **OpenRLHF
  `aggregate_loss(token_level_loss=False)`** divide each row by $L_b + 10^{-8}$ rather
  than $L_b$, deflating every row by $L_b/(L_b + 10^{-8})$. Demonstrated by
  `test_verl_seq_mean_token_mean_deviates_by_row_epsilon_factor` and
  `test_openrlhf_sample_level_deviates_by_row_epsilon_factor`.
- **verl `compute_policy_loss_gspo`** implements the GSPO-token form (eq. 14: detached
  sequence weight, token-local gradient) rather than the paper's eq. 7 sequence ratio.
  Values always coincide; gradients coincide only when advantages are constant within a
  row. Demonstrated by `test_verl_gspo_gradient_is_token_local_gspo_token_form`.

## Conformance as a test (`polgrad.testing`)

Framework and trainer authors can assert conformance directly:

```python
from polgrad.testing import assert_conforms

def test_my_grpo_matches_the_paper():
    assert_conforms(my_loss_fn, "grpo")
```

`assert_conforms(fn, reference, ...)` accepts a `PolicyLossConfig` or an `ALGORITHMS`
key, runs both functions over seeded ragged batches, and raises `AssertionError`
carrying the full `DeviationReport` — maximum loss and gradient relative differences,
minimum gradient cosine, and the worst-case seed — when they disagree.

Two deliberate behaviors: a reference whose config carries an as-loss KL term is
certified *without* that term (the four-tensor convention has no `ref_logprobs`), and
the report says so; a reference carrying a truncated-importance-sampling correction is
rejected outright rather than silently stripped, since dropping it would change what is
being certified. Use `compare_losses` directly for those cases.

`polgrad.testing.random_batches` generates the batches independently if you want to
drive comparisons yourself; it is deterministic given a seed. The package also
registers a pytest plugin, so a `polgrad_batches` fixture is available in any test
suite that has polgrad installed.

## Drift detection

Pins rot: frameworks change monthly. `tools/check_upstream_drift.py` fetches every
tracked file at both its pinned commit and the current default-branch HEAD, extracts
only the tracked definitions by AST, and reports each as unchanged, changed, or
removed. Comments and docstrings count as drift, because they carry semantics.

```sh
python tools/check_upstream_drift.py           # human-readable report
python tools/check_upstream_drift.py --json    # machine-readable, schema_version 1
```

Exit codes: `0` no drift, `3` drift detected, `1` fetch or parse failure. A weekly
workflow (`.github/workflows/drift.yml`) runs it and files — or comments on — an
`upstream-drift` issue when something changes.

The purpose is to convert rot into an event: instead of a pin quietly becoming
unrepresentative, a change becomes a dated, reviewable finding. Findings are recorded
in `CHANGELOG.md` under **Upstream findings**, including whether the change affects the
definitions polgrad actually models.
