from __future__ import annotations

import json

import pytest
from rich.console import Console

from slop_code.common import SUMMARY_FILENAME
from slop_code.entrypoints.utils import count_expected_checkpoints
from slop_code.entrypoints.utils import display_and_save_summary
from slop_code.metrics.checkpoint import driver as checkpoint_driver


def _measurement() -> dict:
    return {
        "evaluator": "scb-check",
        "status": "measured",
        "requested_version": "0.1.3",
        "resolved_version": "0.1.3",
        "record_persisted": True,
        "snapshot_preserved": True,
        "snapshot_tree_sha256": "a" * 64,
        "snapshot_hash_algorithm": (
            checkpoint_driver.SCB_CHECK_SNAPSHOT_HASH_ALGORITHM
        ),
        "environment_lock_sha256": (
            checkpoint_driver.scb_check_lock_sha256()
        ),
        "environment_project_sha256": (
            checkpoint_driver._evaluator_identity()["project_sha256"]
        ),
    }


def test_display_and_save_summary_uses_new_composite_formulas(tmp_path):
    results_file = tmp_path / "checkpoint_results.jsonl"
    rows = [
        {
            "problem": "prob1",
            "idx": 1,
            "strict_pass_rate": 1.0,
            "isolated_pass_rate": 1.0,
            "verbosity": 0.6,
            "erosion": 0.6,
            "scb_check": _measurement(),
        },
        {
            "problem": "prob1",
            "idx": 2,
            "strict_pass_rate": 1.0,
            "isolated_pass_rate": 1.0,
            "verbosity": 0.3,
            "erosion": 0.4,
            "scb_check": _measurement(),
        },
    ]
    results_file.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    config = {
        "model": {"name": "test-model"},
        "thinking": "none",
        "prompt_path": "test_prompt.jinja",
        "agent": {"type": "test-agent", "version": "v1"},
        "problems": ["prob1"],
    }
    console = Console(record=True)

    summary = display_and_save_summary(
        results_file, tmp_path, config, console, expected_checkpoints=2
    )

    assert summary is not None
    expected_first = 0.6
    expected_second = 0.3
    assert summary.verbosity.mean == pytest.approx(
        (expected_first + expected_second) / 2
    )
    assert summary.erosion.mean == pytest.approx(0.5)
    assert summary.erosion.count == 2

    saved = json.loads((tmp_path / SUMMARY_FILENAME).read_text())
    assert saved["verbosity"]["mean"] == summary.verbosity.mean
    assert saved["erosion"]["mean"] == summary.erosion.mean
    assert saved["scb_check"]["requested_version"] == "0.1.3"
    assert saved["scb_check"]["resolved_versions"] == ["0.1.3"]
    assert saved["scb_check"]["measured_checkpoints"] == 2
    assert saved["scb_check"]["expected_checkpoints"] == 2

    rendered = console.export_text()
    assert "Mean verbosity score" in rendered
    assert "Mean erosion score" in rendered
    assert "scb-check evaluator" in rendered
    assert "scb-check coverage" in rendered
    assert "2/2 (100.0%)" in rendered


def test_expected_checkpoint_count_does_not_skip_missing_problem(tmp_path):
    """A missing configured problem cannot silently shrink the denominator."""
    with pytest.raises(ValueError, match="missing-problem"):
        count_expected_checkpoints(
            {"problems": ["missing-problem"]},
            tmp_path,
        )
