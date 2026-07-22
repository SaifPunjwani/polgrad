# Changelog

Framework-behavior findings (conformance deviations, upstream changes detected against
vendored snapshots) are recorded here alongside library changes, versioned by the pin
they were observed at.

## 0.2.0 — 2026-07-18

- `polgrad.testing`: `assert_conforms` and deterministic `random_batches` for
  framework-side conformance testing, installed as a pytest plugin
  (`polgrad_batches` fixture). Configs with an as-loss KL term are certified without it
  (recorded in the report); truncated-importance-sampling configs are rejected rather
  than silently stripped.
- Upstream drift detection: `tools/check_upstream_drift.py` diffs the tracked verl,
  OpenRLHF, and TRL loss definitions at HEAD against the pinned commits, and a weekly
  workflow files an `upstream-drift` issue on change. At release time TRL's
  `GRPOTrainer._compute_loss` has already drifted from the 1.8.0 pin; verl and OpenRLHF
  are unchanged.
- Conformance targets for verl's GSPO and CISPO (labeled reimplementations at the pinned
  commit, with fixtures). Newly registered deviation: verl's `compute_policy_loss_gspo`
  implements the GSPO-token (eq. 14) form — detached sequence weight, token-local
  gradient — rather than the paper's eq. 7 sequence ratio; values coincide, gradients
  coincide only for row-constant advantages.
- Exact entropy support: `token_entropy_estimate(..., entropies=...)` accepts per-token
  entropies computed from full logits, valid off-policy; `EntropyReport` gains an
  `estimator` field.

## 0.1.0 — 2026-07-16

Initial release.

- Loss algebra: `PG_CLIP` (PPO, incl. dual-clip), `PG`, `REINFORCE`, `CISPO` surrogates ×
  token / sequence (GSPO) / sequence-token (GSPO-token) ratios × four aggregation modes,
  with truncated importance-sampling correction for rollout↔trainer mismatch.
- KL estimators k1/k2/k3 (+ `abs` for conformance) with in-reward and as-loss placements
  and machine-checked pathwise gradients.
- Advantage estimators: GRPO group normalization, Dr.GRPO, RLOO, REINFORCE++, GAE.
- Algorithm registry: ppo, grpo, dr_grpo, dapo, gspo, gspo_token, cispo, rloo,
  reinforce_pp, grpo_tis — every constant traced to its paper or released code.
- Diagnostics: importance-ratio ESS (+ sliding window), rollout↔trainer logprob mismatch,
  clip-fraction quadrant decomposition with exact zero-gradient masks, entropy trend with
  permutation-calibrated changepoint, length-bias probe with HC3 robust errors.
- Verification harness as public API: fp64 gradcheck runners, finite-difference checks of
  analytic formulas, softmax-bandit closed forms, hand-derived golden cases.
- Conformance: vendored verl / OpenRLHF loss functions with pinned provenance, recorded
  fixtures, deviation registry.
