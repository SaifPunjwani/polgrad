"""Detect upstream drift in the definitions polgrad pins for conformance.

polgrad vendors loss functions from verl and OpenRLHF at pinned commits
(``src/polgrad/conformance/_vendor/``) and reimplements TRL's
``GRPOTrainer._compute_loss`` against a pinned commit
(``src/polgrad/conformance/harness.py``). This tool fetches each tracked upstream
file twice — at the pinned commit and at the current default-branch HEAD — extracts
only the tracked definitions by AST, and compares their source segments.

Normalization strips trailing whitespace per line and nothing else: comments and
docstrings count as drift because upstream uses them to document semantics.

Exit codes: 0 no drift, 3 drift detected (including a tracked definition removed
at HEAD), 1 fetch or parse error. Set ``GITHUB_TOKEN`` to raise API rate limits.

Run from the repository root:

    python tools/check_upstream_drift.py [--json]
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import sys
import urllib.error
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass

EXIT_NO_DRIFT = 0
EXIT_ERROR = 1
EXIT_DRIFT = 3

SCHEMA_VERSION = 1

_API_ROOT = "https://api.github.com"
_RAW_ROOT = "https://raw.githubusercontent.com"
_TIMEOUT_S = 30.0

_STATUS_UNCHANGED = "unchanged"
_STATUS_CHANGED = "changed"
_STATUS_REMOVED = "removed"


class DriftToolError(RuntimeError):
    """Fetch or parse failure that prevents reaching a drift verdict."""


@dataclass(frozen=True)
class TrackedFile:
    """One upstream file with the definitions polgrad pins from it.

    ``definitions`` entries are top-level function/class names, or
    ``"Class.method"`` for a method.
    """

    repo: str
    path: str
    pinned_commit: str
    definitions: tuple[str, ...]


@dataclass(frozen=True)
class DefinitionResult:
    """Comparison verdict for one tracked definition."""

    name: str
    status: str
    pinned_sha256: str
    head_sha256: str | None


@dataclass(frozen=True)
class FileResult:
    """Comparison verdicts for every tracked definition in one upstream file."""

    repo: str
    path: str
    pinned_commit: str
    head_commit: str
    definitions: tuple[DefinitionResult, ...]


# Pins match the provenance headers in src/polgrad/conformance/_vendor/*.py and the
# _trl_grpo_loss docstring in src/polgrad/conformance/harness.py.
TRACKED_FILES: tuple[TrackedFile, ...] = (
    TrackedFile(
        repo="verl-project/verl",
        path="verl/trainer/ppo/core_algos.py",
        pinned_commit="74a718a492092312f1004fe25369975137388849",
        definitions=(
            "agg_loss",
            "compute_policy_loss",
            "compute_value_loss",
            "kl_penalty",
            "kl_penalty_forward",
        ),
    ),
    TrackedFile(
        repo="verl-project/verl",
        path="verl/utils/torch_functional.py",
        pinned_commit="74a718a492092312f1004fe25369975137388849",
        definitions=("clip_by_value", "masked_sum", "masked_mean"),
    ),
    TrackedFile(
        repo="OpenRLHF/OpenRLHF",
        path="openrlhf/models/loss.py",
        pinned_commit="bc71bb19464aca306b33080b2d2bb45d154e2f49",
        definitions=("aggregate_loss", "PolicyLoss", "ValueLoss"),
    ),
    TrackedFile(
        repo="OpenRLHF/OpenRLHF",
        path="openrlhf/models/utils.py",
        pinned_commit="bc71bb19464aca306b33080b2d2bb45d154e2f49",
        definitions=("masked_mean",),
    ),
    TrackedFile(
        repo="huggingface/trl",
        path="trl/trainer/grpo_trainer.py",
        pinned_commit="95809b942eb5d11d0b06d749510d88be99230b73",
        definitions=("GRPOTrainer._compute_loss",),
    ),
)


def _http_get(url: str, token: str | None) -> bytes:
    """Fetch ``url``, raising :class:`DriftToolError` on any network failure."""
    headers = {"User-Agent": "polgrad-upstream-drift-check"}
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=_TIMEOUT_S) as response:
            return bytes(response.read())
    except urllib.error.HTTPError as exc:
        raise DriftToolError(f"HTTP {exc.code} fetching {url}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise DriftToolError(f"failed to fetch {url}: {exc}") from exc


def resolve_head_commit(repo: str, token: str | None) -> str:
    """Return the commit SHA at the HEAD of ``repo``'s default branch."""
    meta = json.loads(_http_get(f"{_API_ROOT}/repos/{repo}", token))
    if not isinstance(meta, dict) or "default_branch" not in meta:
        raise DriftToolError(f"unexpected repository metadata for {repo}")
    branch = str(meta["default_branch"])
    commit = json.loads(_http_get(f"{_API_ROOT}/repos/{repo}/commits/{branch}", token))
    if not isinstance(commit, dict) or "sha" not in commit:
        raise DriftToolError(f"unexpected commit metadata for {repo}@{branch}")
    return str(commit["sha"])


def fetch_file(repo: str, commit: str, path: str, token: str | None) -> str | None:
    """Return the file content at ``repo``/``commit``/``path``, or None if absent."""
    url = f"{_RAW_ROOT}/{repo}/{commit}/{path}"
    try:
        raw = _http_get(url, token)
    except DriftToolError as exc:
        if "HTTP 404" in str(exc):
            return None
        raise
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise DriftToolError(f"non-UTF-8 content at {url}") from exc


def extract_definition(source: str, name: str, *, filename: str = "<source>") -> str | None:
    """Return the normalized source segment of one definition, or None if absent.

    ``name`` is a top-level function/class name, or ``"Class.method"``. The segment
    spans the definition's decorators through its last line. Normalization strips
    trailing whitespace per line only; comments and docstrings are preserved and
    therefore count as drift.
    """
    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError as exc:
        raise DriftToolError(f"cannot parse {filename}: {exc}") from exc
    body: Sequence[ast.stmt] = tree.body
    node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef | None = None
    for part in name.split("."):
        node = None
        for candidate in body:
            if (
                isinstance(candidate, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
                and candidate.name == part
            ):
                node = candidate
                break
        if node is None:
            return None
        body = node.body
    if node is None:  # unreachable: name.split(".") is never empty
        return None
    start = node.lineno
    if node.decorator_list:
        start = min(start, min(dec.lineno for dec in node.decorator_list))
    end = node.end_lineno if node.end_lineno is not None else node.lineno
    lines = source.splitlines()[start - 1 : end]
    return "\n".join(line.rstrip() for line in lines)


def _segment_sha256(segment: str) -> str:
    return hashlib.sha256(segment.encode("utf-8")).hexdigest()


def compare_definitions(
    pinned_source: str,
    head_source: str | None,
    names: Sequence[str],
    *,
    context: str = "<source>",
) -> tuple[DefinitionResult, ...]:
    """Compare each tracked definition between the pinned and HEAD source.

    ``head_source`` may be None (the file itself is gone at HEAD): every tracked
    definition is then reported as removed. A definition missing at the *pinned*
    commit raises :class:`DriftToolError` — the pin is the ground truth and must
    always resolve.
    """
    results: list[DefinitionResult] = []
    for name in names:
        pinned_segment = extract_definition(pinned_source, name, filename=f"{context}@pinned")
        if pinned_segment is None:
            raise DriftToolError(f"tracked definition {name!r} not found in {context} at the pin")
        pinned_sha = _segment_sha256(pinned_segment)
        head_segment = (
            None
            if head_source is None
            else extract_definition(head_source, name, filename=f"{context}@head")
        )
        if head_segment is None:
            results.append(DefinitionResult(name, _STATUS_REMOVED, pinned_sha, None))
            continue
        status = _STATUS_UNCHANGED if head_segment == pinned_segment else _STATUS_CHANGED
        results.append(DefinitionResult(name, status, pinned_sha, _segment_sha256(head_segment)))
    return tuple(results)


def check_tracked_file(tracked: TrackedFile, head_commit: str, token: str | None) -> FileResult:
    """Fetch one tracked file at its pin and at ``head_commit`` and compare."""
    context = f"{tracked.repo}:{tracked.path}"
    pinned_source = fetch_file(tracked.repo, tracked.pinned_commit, tracked.path, token)
    if pinned_source is None:
        raise DriftToolError(f"{context} not found at pinned commit {tracked.pinned_commit}")
    head_source = fetch_file(tracked.repo, head_commit, tracked.path, token)
    definitions = compare_definitions(
        pinned_source, head_source, tracked.definitions, context=context
    )
    return FileResult(
        repo=tracked.repo,
        path=tracked.path,
        pinned_commit=tracked.pinned_commit,
        head_commit=head_commit,
        definitions=definitions,
    )


def build_report(results: Sequence[FileResult]) -> dict[str, object]:
    """Build the machine-readable report emitted by ``--json``.

    Schema (version 1): ``schema_version`` int, ``drift_detected`` bool, ``files``
    list of ``{repo, path, pinned_commit, head_commit, definitions}`` where each
    definition is ``{name, status, pinned_sha256, head_sha256}`` with status one of
    ``unchanged | changed | removed`` and ``head_sha256`` null when removed.
    """
    files: list[dict[str, object]] = []
    for file_result in results:
        files.append(
            {
                "repo": file_result.repo,
                "path": file_result.path,
                "pinned_commit": file_result.pinned_commit,
                "head_commit": file_result.head_commit,
                "definitions": [
                    {
                        "name": definition.name,
                        "status": definition.status,
                        "pinned_sha256": definition.pinned_sha256,
                        "head_sha256": definition.head_sha256,
                    }
                    for definition in file_result.definitions
                ],
            }
        )
    drift = any(
        definition.status != _STATUS_UNCHANGED
        for file_result in results
        for definition in file_result.definitions
    )
    return {"schema_version": SCHEMA_VERSION, "drift_detected": drift, "files": files}


def format_report(results: Sequence[FileResult]) -> str:
    """Render the human-readable report."""
    lines: list[str] = []
    counts = {_STATUS_UNCHANGED: 0, _STATUS_CHANGED: 0, _STATUS_REMOVED: 0}
    for file_result in results:
        lines.append(f"{file_result.repo} {file_result.path}")
        lines.append(
            f"  pinned {file_result.pinned_commit[:12]}  head {file_result.head_commit[:12]}"
        )
        for definition in file_result.definitions:
            counts[definition.status] += 1
            label = (
                definition.status
                if definition.status == _STATUS_UNCHANGED
                else definition.status.upper()
            )
            lines.append(f"    {definition.name:<40} {label}")
        lines.append("")
    total = sum(counts.values())
    drifted = counts[_STATUS_CHANGED] + counts[_STATUS_REMOVED]
    verdict = "DRIFT DETECTED" if drifted else "no drift"
    lines.append(
        f"{verdict}: {counts[_STATUS_CHANGED]} changed, {counts[_STATUS_REMOVED]} removed, "
        f"{counts[_STATUS_UNCHANGED]} unchanged across {total} tracked definitions"
    )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the drift check; return 0 (no drift), 3 (drift), or 1 (error)."""
    parser = argparse.ArgumentParser(
        prog="check_upstream_drift",
        description="Compare polgrad's pinned upstream definitions against upstream HEAD.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit the machine-readable JSON report instead of the text report",
    )
    args = parser.parse_args(argv)
    token = os.environ.get("GITHUB_TOKEN") or None
    try:
        head_commits = {
            repo: resolve_head_commit(repo, token)
            for repo in sorted({tracked.repo for tracked in TRACKED_FILES})
        }
        results = tuple(
            check_tracked_file(tracked, head_commits[tracked.repo], token)
            for tracked in TRACKED_FILES
        )
    except DriftToolError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR
    report = build_report(results)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(format_report(results))
    return EXIT_DRIFT if report["drift_detected"] else EXIT_NO_DRIFT


if __name__ == "__main__":
    sys.exit(main())
