<!--
Read CONTRIBUTING.md first. Delete the sections that do not apply.
-->

## What this changes

<!-- One or two sentences. Link the issue if there is one. -->

## Checks

```sh
pytest -n auto
ruff check src tests examples tools
ruff format --check src tests examples tools
mypy
```

- [ ] All four pass locally.

## If this adds or changes semantics

- [ ] A derivation page in `docs/` states the per-token objective and its gradient, with
      the algebra written out.
- [ ] fp64 `gradcheck` coverage.
- [ ] At least one hand-derived golden case, with the arithmetic shown.
- [ ] Property tests for every equivalence or invariance the derivation claims.
- [ ] Defaults are traced to a paper or its released code. Where the source is silent,
      the ambiguity is recorded in `notes` rather than filled in with a guess.

## If this adds a diagnostic

- [ ] The null distribution is documented.
- [ ] A Monte Carlo test calibrates the threshold. No constant is asserted without an
      error rate.

## If this touches conformance

- [ ] A test demonstrates the claim against the target code, and the entry in
      `conformance/deviations.py` names that test.
- [ ] Wording is relative to a published equation, not phrased as a framework defect.
- [ ] Vendored updates are real fetches at a pinned commit, with the header's SHA256,
      commit, and edit list updated, and `NOTICE` updated if attribution changed.

## Notes for the reviewer

<!-- Anything you want looked at closely, or a claim you are unsure of. -->
