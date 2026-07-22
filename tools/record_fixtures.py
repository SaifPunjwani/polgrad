"""Record conformance fixtures: vendored framework losses on seeded inputs.

Runs every ``polgrad.conformance.harness.VENDORED`` wrapper on deterministic seeded
inputs (the harness input distribution) and writes one JSON file per fixture group to
``tests/fixtures/``, holding the exact float64 inputs, the scalar loss, the gradient
w.r.t. ``logprobs``, and the upstream provenance. Vendored and reimplemented entries
are kept in separate files with separate provenance blocks (verl's gspo/cispo
reimplementations go to ``verl_reimpl_losses.json``, mirroring
``trl_reimpl_losses.json``). ``tests/test_conformance.py`` replays these files against
polgrad live, so CI needs no framework installs and the recorded numbers stay pinned
to the vendored commits.

Seeds are ``crc32("framework:variant:index") & 0x7FFFFFFF``, so re-running the tool on
the same vendored code reproduces the files byte-for-byte (JSON float serialization
round-trips float64 exactly).

Run from the repository root:

    .venv/bin/python tools/record_fixtures.py
"""

from __future__ import annotations

import json
import zlib
from pathlib import Path
from typing import Any

import torch

from polgrad.conformance.harness import _CLIP_EPS, _VERL_CLIP_RATIO_C, VENDORED, _sample_case

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures"

SHAPES: tuple[tuple[int, int], ...] = ((4, 8), (3, 5), (2, 12), (5, 7))
CASES_PER_VARIANT = 4
DTYPE = torch.float64

_VERL_COMMIT = "74a718a492092312f1004fe25369975137388849"

PROVENANCE: dict[str, dict[str, Any]] = {
    "verl": {
        "kind": "vendored",
        "wrapper_module": "polgrad.conformance.harness",
        "vendored_module": "polgrad.conformance._vendor.verl_core_algos",
        "upstream_repo": "https://github.com/volcengine/verl",
        "upstream_commit": _VERL_COMMIT,
        "source_path": "verl/trainer/ppo/core_algos.py",
        "clip_eps_low": _CLIP_EPS,
        "clip_eps_high": _CLIP_EPS,
        "clip_ratio_c": _VERL_CLIP_RATIO_C,
    },
    "verl_reimpl": {
        "kind": "reimplementation",
        "wrapper_module": "polgrad.conformance.harness",
        "reimplementations": {
            "gspo": "polgrad.conformance.harness._verl_gspo_loss",
            "cispo": "polgrad.conformance.harness._verl_cispo_loss",
        },
        "upstream_repo": "https://github.com/volcengine/verl",
        "upstream_commit": _VERL_COMMIT,
        "source_path": "verl/trainer/ppo/core_algos.py",
        "source_sha256": "9114f9e16c87e4c9ebf2fa016baf733c9bbc819766b53c8968aaa9e8abcd7916",
        "permalinks": {
            "gspo": (
                "https://github.com/verl-project/verl/blob/"
                f"{_VERL_COMMIT}/verl/trainer/ppo/core_algos.py#L1538-L1611"
            ),
            "cispo": (
                "https://github.com/verl-project/verl/blob/"
                f"{_VERL_COMMIT}/verl/trainer/ppo/core_algos.py#L2006-L2064"
            ),
        },
        "clip_ratio_low": _CLIP_EPS,
        "clip_ratio_high": _CLIP_EPS,
        "gspo_loss_agg_mode": "seq-mean-token-mean (hardcoded upstream)",
        "cispo_loss_agg_mode": "token-mean (upstream parameter default)",
        "global_batch_info": "frozen to the ActorConfig dataclass default {} (dp_size=1)",
    },
    "openrlhf": {
        "kind": "vendored",
        "wrapper_module": "polgrad.conformance.harness",
        "vendored_module": "polgrad.conformance._vendor.openrlhf_loss",
        "upstream_repo": "https://github.com/OpenRLHF/OpenRLHF",
        "upstream_commit": "bc71bb19464aca306b33080b2d2bb45d154e2f49",
        "source_path": "openrlhf/models/loss.py",
        "clip_eps_low": _CLIP_EPS,
        "clip_eps_high": _CLIP_EPS,
    },
    "trl": {
        "kind": "reimplementation",
        "wrapper_module": "polgrad.conformance.harness",
        "reimplementation": "polgrad.conformance.harness._trl_grpo_loss",
        "upstream_repo": "https://github.com/huggingface/trl",
        "upstream_version": "v1.8.0",
        "upstream_commit": "95809b942eb5d11d0b06d749510d88be99230b73",
        "source_path": "trl/trainer/grpo_trainer.py",
        "source_sha256": "52d9a6c1e298df35d0da4a6fa17874d750ee627f6ac15393c8860d74d1ba4917",
        "permalink": (
            "https://github.com/huggingface/trl/blob/"
            "95809b942eb5d11d0b06d749510d88be99230b73/trl/trainer/grpo_trainer.py#L2857-L3016"
        ),
        "clip_eps_low": _CLIP_EPS,
        "clip_eps_high": _CLIP_EPS,
        "dr_grpo_max_completion_length": "pinned to the padded width T of each case",
    },
}

SHARED_PROVENANCE: dict[str, Any] = {
    "tool": "tools/record_fixtures.py",
    "torch_version": torch.__version__,
    "dtype": "float64",
    "input_bounds": "logprobs in [-8, -0.05]; |log-ratio gap| <= 2; |advantage| <= 3",
    "seed_rule": "crc32('framework:variant:index') & 0x7FFFFFFF",
    "shapes": [list(shape) for shape in SHAPES],
}


# The verl VENDORED variants that are harness reimplementations (upstream reads a verl
# ActorConfig); they are recorded to their own fixture file with their own provenance.
VERL_REIMPLEMENTED_VARIANTS = ("cispo", "gspo")

FIXTURE_FILES: dict[str, str] = {
    "verl": "verl_losses.json",
    "verl_reimpl": "verl_reimpl_losses.json",
    "openrlhf": "openrlhf_losses.json",
    "trl": "trl_reimpl_losses.json",
}


def fixture_group(framework: str, variant: str) -> str:
    """Map a VENDORED key to its fixture file group (vendored vs reimplemented)."""
    if framework == "verl" and variant in VERL_REIMPLEMENTED_VARIANTS:
        return "verl_reimpl"
    return framework


def case_seed(framework: str, variant: str, index: int) -> int:
    """Deterministic per-case seed from the fixture identity."""
    return zlib.crc32(f"{framework}:{variant}:{index}".encode()) & 0x7FFFFFFF


def record_case(framework: str, variant: str, index: int) -> dict[str, Any]:
    """Sample one case, run the vendored wrapper, and serialize inputs and outputs."""
    seed = case_seed(framework, variant, index)
    shape = SHAPES[index % len(SHAPES)]
    case = _sample_case(shape, torch.Generator().manual_seed(seed), DTYPE)
    logprobs = case["logprobs"].clone().requires_grad_(True)
    loss = VENDORED[(framework, variant)](
        logprobs=logprobs,
        old_logprobs=case["old_logprobs"],
        advantages=case["advantages"],
        response_mask=case["response_mask"],
    )
    (grad,) = torch.autograd.grad(loss, logprobs)
    return {
        "seed": seed,
        "shape": list(shape),
        "inputs": {
            "logprobs": case["logprobs"].tolist(),
            "old_logprobs": case["old_logprobs"].tolist(),
            "advantages": case["advantages"].tolist(),
            "response_mask": case["response_mask"].tolist(),
        },
        "outputs": {
            "loss": float(loss.detach()),
            "grad_logprobs": grad.tolist(),
        },
    }


def main() -> None:
    """Record every VENDORED entry and write one fixture file per group."""
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    groups = sorted({fixture_group(framework, variant) for framework, variant in VENDORED})
    for group in groups:
        members = sorted(
            (framework, variant)
            for framework, variant in VENDORED
            if fixture_group(framework, variant) == group
        )
        framework = members[0][0]  # every member of a group shares one framework
        payload: dict[str, Any] = {
            "provenance": {**SHARED_PROVENANCE, "framework": framework, **PROVENANCE[group]},
            "variants": {
                variant: [
                    record_case(framework, variant, index) for index in range(CASES_PER_VARIANT)
                ]
                for _, variant in members
            },
        }
        path = FIXTURES_DIR / FIXTURE_FILES[group]
        path.write_text(json.dumps(payload, indent=1) + "\n")
        n_cases = sum(len(cases) for cases in payload["variants"].values())
        print(f"wrote {path.relative_to(REPO_ROOT)}: {len(members)} variants, {n_cases} cases")


if __name__ == "__main__":
    main()
