# Changelog

Framework-behavior findings (conformance deviations, upstream changes detected against
vendored snapshots) are recorded here alongside library changes, versioned by the pin
they were observed at.

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
