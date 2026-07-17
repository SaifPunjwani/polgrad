# Contributing

polgrad is a reference-semantics library: the tests and derivation pages are the product
as much as the code. Contributions are held to that bar.

## Setup

```sh
python -m venv .venv && source .venv/bin/activate
pip install torch            # or the CPU index: pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e '.[dev]'
```

## Checks

Everything that CI runs, locally:

```sh
pytest -n auto               # HYPOTHESIS_PROFILE=ci for the full example count
ruff check src tests examples tools
ruff format --check src tests examples tools
mypy
```

All four must pass. There are no known-flaky tests; if a seeded test fails
intermittently, that is a bug worth reporting on its own.

## What a change needs

- **A new loss variant or estimator** needs, in one PR: the config entry, a derivation
  page in `docs/derivations/` showing the per-token objective and its gradient with the
  algebra written out, fp64 `gradcheck` coverage, at least one hand-derived golden case
  with the arithmetic shown, and property tests for whatever equivalences or invariances
  the derivation claims. A variant whose defaults cannot be traced to a paper or its
  released code documents that ambiguity in its `notes` rather than inventing values.
- **A new diagnostic** needs a documented null distribution and a Monte Carlo test
  calibrating it. Thresholds without error rates are not accepted.
- **A conformance claim** (an entry in `conformance/deviations.py`) needs a test that
  demonstrates it against vendored code, and neutral wording: deviations are described
  relative to a paper equation, never as defects of the framework.
- **Vendored code updates** must be real fetches at a pinned commit with the header's
  SHA256 updated; see the header format in `src/polgrad/conformance/_vendor/`.

## Conventions

Shapes, masking, sign conventions, and dtype rules are specified in
`docs/conventions.md` and enforced by tests; read it before writing code. Style is
whatever `ruff` and `mypy --strict` accept, plus: no stop-gradient without a comment
stating its semantic reason, and docstrings reference the derivation page and the test
that enforces each claim.
