# Contributing

polgrad is a reference-semantics library. The tests and the derivation pages are part of
the product, and a change is reviewed against them as well as against the code.

## Reporting

Use the issue forms: a bug report for a wrong value, gradient, or exception; a
conformance finding for a framework whose shipped loss differs from a published
equation; a variant or diagnostic proposal for new semantics. Report vulnerabilities
privately as described in [SECURITY.md](SECURITY.md).

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

- A new loss variant or estimator needs, in one PR: the config entry, a derivation page
  in `docs/derivations/` showing the per-token objective and its gradient with the
  algebra written out, fp64 `gradcheck` coverage, at least one hand-derived golden case
  with the arithmetic shown, and property tests for whatever equivalences or invariances
  the derivation claims. Where a default cannot be traced to a paper or its released
  code, record the ambiguity in `notes` instead of picking a value.
- A new diagnostic needs a documented null distribution and a Monte Carlo test
  calibrating it. Thresholds without error rates are not accepted.
- A conformance claim (an entry in `conformance/deviations.py`) needs a test that
  demonstrates it against vendored code, and neutral wording: describe a deviation
  relative to a paper equation, never as a defect of the framework.
- Vendored code updates must be real fetches at a pinned commit with the header's SHA256
  updated; see the header format in `src/polgrad/conformance/_vendor/`. Update `NOTICE`
  if the attribution changes.

## Conventions

Shapes, masking, sign conventions, and dtype rules are specified in
`docs/conventions.md` and enforced by tests; read it before writing code. Style is
whatever `ruff` and `mypy --strict` accept, plus: no stop-gradient without a comment
stating its semantic reason, and docstrings reference the derivation page and the test
that enforces each claim.

## Licensing

Contributions are licensed under Apache-2.0, matching [LICENSE](LICENSE). If a change
brings in third-party code, it has to be Apache-2.0 compatible, and it goes under
`src/polgrad/conformance/_vendor/` with a provenance header and an entry in `NOTICE`.
