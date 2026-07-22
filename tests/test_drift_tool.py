"""Unit tests for tools/check_upstream_drift.py: extraction, comparison, JSON schema.

No network. ``tools/`` is not a package, so the module is loaded from its file path
via ``importlib``. Fetching (`_http_get`, `resolve_head_commit`, `fetch_file`) is
exercised only by the tool's live runs and the drift workflow, not here.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import textwrap
from pathlib import Path
from types import ModuleType

import pytest

_TOOL_PATH = Path(__file__).resolve().parents[1] / "tools" / "check_upstream_drift.py"


def _load_tool() -> ModuleType:
    spec = importlib.util.spec_from_file_location("check_upstream_drift", _TOOL_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Registration is required before exec: the tool's dataclasses resolve their
    # module through sys.modules while being processed.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


drift = _load_tool()

_PINNED = textwrap.dedent(
    '''
    import math


    def masked_mean(values, mask):
        """Mean over unmasked entries."""
        return (values * mask).sum() / mask.sum()


    @some.decorator(arg=1)
    def agg_loss(loss_mat, loss_mask, mode):
        # comment carrying semantics
        return masked_mean(loss_mat, loss_mask)


    class GRPOTrainer:
        beta = 0.0

        def _compute_loss(self, model, inputs):
            """Policy term only."""
            return inputs.mean()

        def other_method(self):
            return None
    '''
)


def test_extract_function_segment_spans_decorators_and_body() -> None:
    segment = drift.extract_definition(_PINNED, "agg_loss")
    assert segment is not None
    assert segment.startswith("@some.decorator(arg=1)")
    assert "def agg_loss(loss_mat, loss_mask, mode):" in segment
    assert "# comment carrying semantics" in segment
    assert segment.endswith("return masked_mean(loss_mat, loss_mask)")


def test_extract_strips_trailing_whitespace_only() -> None:
    source = "def f():   \n    x = 1  \n    return x\t\n"
    segment = drift.extract_definition(source, "f")
    assert segment == "def f():\n    x = 1\n    return x"


def test_extract_class_method_by_dotted_name() -> None:
    segment = drift.extract_definition(_PINNED, "GRPOTrainer._compute_loss")
    assert segment is not None
    # The segment keeps the method's original indentation inside the class.
    assert segment.startswith("    def _compute_loss(self, model, inputs):")
    assert '"""Policy term only."""' in segment
    assert "other_method" not in segment


def test_extract_whole_class_segment() -> None:
    segment = drift.extract_definition(_PINNED, "GRPOTrainer")
    assert segment is not None
    assert segment.startswith("class GRPOTrainer:")
    assert "def _compute_loss" in segment
    assert "def other_method" in segment


def test_extract_missing_definition_returns_none() -> None:
    assert drift.extract_definition(_PINNED, "nonexistent") is None
    assert drift.extract_definition(_PINNED, "GRPOTrainer.nonexistent") is None
    # Dotted path through a non-existent class.
    assert drift.extract_definition(_PINNED, "NoSuchClass._compute_loss") is None


def test_extract_unparsable_source_raises() -> None:
    with pytest.raises(drift.DriftToolError, match="cannot parse"):
        drift.extract_definition("def broken(:\n", "broken")


def test_compare_identical_sources_is_unchanged() -> None:
    results = drift.compare_definitions(
        _PINNED, _PINNED, ["masked_mean", "agg_loss", "GRPOTrainer._compute_loss"]
    )
    assert [r.status for r in results] == ["unchanged"] * 3
    for result in results:
        assert result.head_sha256 == result.pinned_sha256


def test_compare_changed_function_body_is_drift() -> None:
    head = _PINNED.replace("return inputs.mean()", "return inputs.sum()")
    results = drift.compare_definitions(_PINNED, head, ["agg_loss", "GRPOTrainer._compute_loss"])
    by_name = {r.name: r for r in results}
    assert by_name["agg_loss"].status == "unchanged"
    changed = by_name["GRPOTrainer._compute_loss"]
    assert changed.status == "changed"
    assert changed.head_sha256 is not None
    assert changed.head_sha256 != changed.pinned_sha256


def test_compare_comment_only_change_is_drift() -> None:
    head = _PINNED.replace("# comment carrying semantics", "# reworded comment")
    (result,) = drift.compare_definitions(_PINNED, head, ["agg_loss"])
    assert result.status == "changed"


def test_compare_docstring_only_change_is_drift() -> None:
    head = _PINNED.replace('"""Mean over unmasked entries."""', '"""Reworded."""')
    (result,) = drift.compare_definitions(_PINNED, head, ["masked_mean"])
    assert result.status == "changed"


def test_compare_trailing_whitespace_change_is_not_drift() -> None:
    head = _PINNED.replace("return inputs.mean()", "return inputs.mean()   ")
    (result,) = drift.compare_definitions(_PINNED, head, ["GRPOTrainer._compute_loss"])
    assert result.status == "unchanged"


def test_compare_removed_definition_is_drift() -> None:
    head = _PINNED.replace("def _compute_loss", "def _compute_loss_renamed")
    (result,) = drift.compare_definitions(_PINNED, head, ["GRPOTrainer._compute_loss"])
    assert result.status == "removed"
    assert result.head_sha256 is None


def test_compare_missing_file_at_head_reports_all_removed() -> None:
    results = drift.compare_definitions(_PINNED, None, ["agg_loss", "masked_mean"])
    assert [r.status for r in results] == ["removed", "removed"]


def test_compare_missing_at_pin_raises() -> None:
    with pytest.raises(drift.DriftToolError, match=r"not found .* at the pin"):
        drift.compare_definitions(_PINNED, _PINNED, ["nonexistent"])


def _synthetic_results(head_source: str) -> list[object]:
    definitions = drift.compare_definitions(
        _PINNED, head_source, ["agg_loss", "GRPOTrainer._compute_loss"], context="r:p"
    )
    return [
        drift.FileResult(
            repo="example/repo",
            path="pkg/mod.py",
            pinned_commit="a" * 40,
            head_commit="b" * 40,
            definitions=definitions,
        )
    ]


def test_json_report_schema_is_stable() -> None:
    head = _PINNED.replace("return inputs.mean()", "return inputs.sum()")
    report = drift.build_report(_synthetic_results(head))

    assert set(report) == {"schema_version", "drift_detected", "files"}
    assert report["schema_version"] == 1
    assert report["drift_detected"] is True

    (file_entry,) = report["files"]
    assert set(file_entry) == {"repo", "path", "pinned_commit", "head_commit", "definitions"}
    assert file_entry["repo"] == "example/repo"
    assert file_entry["pinned_commit"] == "a" * 40
    assert file_entry["head_commit"] == "b" * 40

    for definition in file_entry["definitions"]:
        assert set(definition) == {"name", "status", "pinned_sha256", "head_sha256"}
        assert definition["status"] in {"unchanged", "changed", "removed"}
        assert isinstance(definition["pinned_sha256"], str)
        assert len(definition["pinned_sha256"]) == 64

    # The report must round-trip through JSON unchanged.
    assert json.loads(json.dumps(report)) == report


def test_json_report_no_drift() -> None:
    report = drift.build_report(_synthetic_results(_PINNED))
    assert report["drift_detected"] is False
    (file_entry,) = report["files"]
    statuses = {d["status"] for d in file_entry["definitions"]}
    assert statuses == {"unchanged"}


def test_format_report_verdict_lines() -> None:
    head = _PINNED.replace("def _compute_loss", "def _compute_loss_renamed")
    text = drift.format_report(_synthetic_results(head))
    assert "example/repo pkg/mod.py" in text
    assert "REMOVED" in text
    assert text.endswith(
        "DRIFT DETECTED: 0 changed, 1 removed, 1 unchanged across 2 tracked definitions"
    )

    clean = drift.format_report(_synthetic_results(_PINNED))
    assert clean.endswith(
        "no drift: 0 changed, 0 removed, 2 unchanged across 2 tracked definitions"
    )


def test_tracked_pins_match_vendor_provenance() -> None:
    """The pins in the tool must agree with the vendored/reimplementation provenance."""
    pins = {(t.repo, t.path): t.pinned_commit for t in drift.TRACKED_FILES}
    assert pins[("verl-project/verl", "verl/trainer/ppo/core_algos.py")] == (
        "74a718a492092312f1004fe25369975137388849"
    )
    assert pins[("OpenRLHF/OpenRLHF", "openrlhf/models/loss.py")] == (
        "bc71bb19464aca306b33080b2d2bb45d154e2f49"
    )
    assert pins[("huggingface/trl", "trl/trainer/grpo_trainer.py")] == (
        "95809b942eb5d11d0b06d749510d88be99230b73"
    )
    assert drift.EXIT_NO_DRIFT == 0
    assert drift.EXIT_ERROR == 1
    assert drift.EXIT_DRIFT == 3
