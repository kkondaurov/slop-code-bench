"""Tests for checkpoint resume functionality."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import pytest
import yaml

from slop_code.agent_runner.models import UsageTracker
from slop_code.agent_runner.reporting import CheckpointState
from slop_code.agent_runner.reporting import MetricsTracker
from slop_code.agent_runner.resume import InvalidationReason
from slop_code.agent_runner.resume import ResumeInfo
from slop_code.agent_runner.resume import _aggregate_prior_usage
from slop_code.agent_runner.resume import _detect_resume_from_artifacts
from slop_code.agent_runner.resume import _evaluation_status
from slop_code.agent_runner.resume import detect_resume_point
from slop_code.common import INFERENCE_RESULT_FILENAME
from slop_code.common import RUN_INFO_FILENAME
from slop_code.common import SNAPSHOT_DIR_NAME
from slop_code.common.llms import TokenUsage
from slop_code.evaluation.report import CorrectnessResults
from slop_code.evaluation.report import PassPolicy
from slop_code.execution.models import SetupConfig


def _inference_result(
    *,
    cost: float = 0.0,
    steps: int = 0,
    net_tokens: dict[str, int] | None = None,
    had_error: bool = False,
) -> dict[str, object]:
    now = datetime.now().isoformat()
    tokens = {
        "input": 0,
        "output": 0,
        "cache_read": 0,
        "cache_write": 0,
        "reasoning": 0,
    }
    tokens.update(net_tokens or {})
    return {
        "started": now,
        "completed": now,
        "elapsed": 0.0,
        "had_error": had_error,
        "usage": {
            "cost": cost,
            "steps": steps,
            "net_tokens": tokens,
            "current_tokens": dict(tokens),
        },
    }


class TestSetupConfigResumeCommands:
    """Tests for resume_commands in SetupConfig."""

    def test_resume_commands_default_empty(self) -> None:
        """Resume commands default to empty list."""
        config = SetupConfig()
        assert config.resume_commands == []

    def test_resume_commands_can_be_set(self) -> None:
        """Resume commands can be set explicitly."""
        config = SetupConfig(
            resume_commands=["pip install -r requirements.txt"]
        )
        assert config.resume_commands == ["pip install -r requirements.txt"]

    def test_resume_commands_multiple(self) -> None:
        """Multiple resume commands can be set."""
        config = SetupConfig(
            resume_commands=[
                "pip install -r requirements.txt",
                "python -m pip install --upgrade pip",
            ]
        )
        assert len(config.resume_commands) == 2


class TestResumeInfo:
    """Tests for ResumeInfo dataclass."""

    def test_resume_info_creation(self, tmp_path: Path) -> None:
        """Test basic ResumeInfo creation."""
        snapshot_dir = tmp_path / "checkpoint_1" / "snapshot"
        snapshot_dir.mkdir(parents=True)

        info = ResumeInfo(
            resume_from_checkpoint="checkpoint_2",
            completed_checkpoints=["checkpoint_1"],
            last_snapshot_dir=snapshot_dir,
            prior_usage=UsageTracker(cost=1.5, steps=10),
        )

        assert info.resume_from_checkpoint == "checkpoint_2"
        assert info.completed_checkpoints == ["checkpoint_1"]
        assert info.last_snapshot_dir == snapshot_dir
        assert info.prior_usage.cost == 1.5
        assert info.prior_usage.steps == 10


class TestDetectResumePoint:
    """Tests for detect_resume_point function."""

    def test_returns_resume_info_when_no_run_info_and_no_artifacts(
        self, tmp_path: Path
    ) -> None:
        """Returns ResumeInfo to start from first checkpoint when no prior state exists."""
        result = detect_resume_point(tmp_path, ["checkpoint_1", "checkpoint_2"])
        # When no run_info.yaml and no checkpoint directories exist,
        # returns ResumeInfo with all checkpoints invalidated (need to run from start)
        assert result is not None
        assert result.resume_from_checkpoint == "checkpoint_1"
        assert result.completed_checkpoints == []
        assert result.invalidated_checkpoints == [
            "checkpoint_1",
            "checkpoint_2",
        ]

    def test_returns_resume_info_when_all_checkpoints_completed(
        self, tmp_path: Path
    ) -> None:
        """Returns ResumeInfo with empty resume_from when all checkpoints completed."""
        # Create run_info.yaml with all checkpoints completed
        run_info = {
            "summary": {
                "checkpoints": {
                    "checkpoint_1": CheckpointState.RAN,
                    "checkpoint_2": CheckpointState.RAN,
                }
            }
        }
        with (tmp_path / RUN_INFO_FILENAME).open("w") as f:
            yaml.dump(run_info, f)

        # Create solve-complete artifacts, not only optimistic run_info state.
        for checkpoint_name in ("checkpoint_1", "checkpoint_2"):
            checkpoint_dir = tmp_path / checkpoint_name
            (checkpoint_dir / SNAPSHOT_DIR_NAME).mkdir(parents=True)
            (checkpoint_dir / INFERENCE_RESULT_FILENAME).write_text(
                json.dumps(_inference_result())
            )

        result = detect_resume_point(tmp_path, ["checkpoint_1", "checkpoint_2"])
        assert result is not None
        assert result.resume_from_checkpoint == ""  # Empty = nothing to resume
        assert result.completed_checkpoints == ["checkpoint_1", "checkpoint_2"]
        assert result.invalidated_checkpoints == []

    def test_returns_resume_info_when_no_checkpoints_completed(
        self, tmp_path: Path
    ) -> None:
        """Returns ResumeInfo to resume from first checkpoint when none completed."""
        run_info = {
            "summary": {
                "checkpoints": {
                    "checkpoint_1": CheckpointState.SKIPPED,
                    "checkpoint_2": CheckpointState.SKIPPED,
                }
            }
        }
        with (tmp_path / RUN_INFO_FILENAME).open("w") as f:
            yaml.dump(run_info, f)

        result = detect_resume_point(tmp_path, ["checkpoint_1", "checkpoint_2"])
        assert result is not None
        assert result.resume_from_checkpoint == "checkpoint_1"
        assert result.completed_checkpoints == []
        assert result.invalidated_checkpoints == [
            "checkpoint_1",
            "checkpoint_2",
        ]

    def test_detects_resume_point_after_first_checkpoint(
        self, tmp_path: Path
    ) -> None:
        """Detects correct resume point when first checkpoint completed."""
        # Create run_info.yaml
        run_info = {
            "summary": {
                "checkpoints": {
                    "checkpoint_1": CheckpointState.RAN,
                    "checkpoint_2": CheckpointState.ERROR,
                }
            }
        }
        with (tmp_path / RUN_INFO_FILENAME).open("w") as f:
            yaml.dump(run_info, f)

        # Create checkpoint_1 snapshot and inference result
        checkpoint_1_dir = tmp_path / "checkpoint_1"
        (checkpoint_1_dir / SNAPSHOT_DIR_NAME).mkdir(parents=True)
        inference_result = _inference_result(
            cost=0.5,
            steps=5,
            net_tokens={"input": 100},
        )
        with (checkpoint_1_dir / INFERENCE_RESULT_FILENAME).open("w") as f:
            json.dump(inference_result, f)

        result = detect_resume_point(
            tmp_path, ["checkpoint_1", "checkpoint_2", "checkpoint_3"]
        )

        assert result is not None
        assert result.resume_from_checkpoint == "checkpoint_2"
        assert result.completed_checkpoints == ["checkpoint_1"]
        assert result.last_snapshot_dir == checkpoint_1_dir / SNAPSHOT_DIR_NAME
        assert result.prior_usage.cost == 0.5
        assert result.prior_usage.steps == 5

    def test_detects_resume_point_after_multiple_checkpoints(
        self, tmp_path: Path
    ) -> None:
        """Detects correct resume point when multiple checkpoints completed."""
        run_info = {
            "summary": {
                "checkpoints": {
                    "checkpoint_1": CheckpointState.RAN,
                    "checkpoint_2": CheckpointState.RAN,
                    "checkpoint_3": CheckpointState.ERROR,
                }
            }
        }
        with (tmp_path / RUN_INFO_FILENAME).open("w") as f:
            yaml.dump(run_info, f)

        # Create snapshots and inference results
        for i, cost in [(1, 0.5), (2, 0.8)]:
            checkpoint_dir = tmp_path / f"checkpoint_{i}"
            (checkpoint_dir / SNAPSHOT_DIR_NAME).mkdir(parents=True)
            inference_result = _inference_result(cost=cost, steps=i * 5)
            with (checkpoint_dir / INFERENCE_RESULT_FILENAME).open("w") as f:
                json.dump(inference_result, f)

        result = detect_resume_point(
            tmp_path, ["checkpoint_1", "checkpoint_2", "checkpoint_3"]
        )

        assert result is not None
        assert result.resume_from_checkpoint == "checkpoint_3"
        assert result.completed_checkpoints == ["checkpoint_1", "checkpoint_2"]
        assert (
            result.last_snapshot_dir
            == tmp_path / "checkpoint_2" / SNAPSHOT_DIR_NAME
        )
        # Aggregated cost from both checkpoints
        assert result.prior_usage.cost == 1.3
        assert result.prior_usage.steps == 15

    def test_complete_artifact_supersedes_newer_stale_skipped_run_info(
        self,
        tmp_path: Path,
    ) -> None:
        """A hard kill after inference must not cause duplicate model spend."""
        now = datetime.now()
        run_info_path = tmp_path / RUN_INFO_FILENAME
        run_info_path.write_text(
            yaml.safe_dump(
                {
                    "summary": {
                        "started": now.isoformat(),
                        "ended": now.isoformat(),
                        "duration_seconds": 0.0,
                        "total_cost": 0.0,
                        "total_steps": 0,
                        "total_usage": UsageTracker().model_dump(),
                        "checkpoints": {
                            "checkpoint_1": CheckpointState.RAN,
                            "checkpoint_2": CheckpointState.SKIPPED,
                        },
                        "state": "error",
                        "passed_policy": False,
                    }
                }
            ),
            encoding="utf-8",
        )
        for index, cost in ((1, 1.0), (2, 2.0)):
            checkpoint_dir = tmp_path / f"checkpoint_{index}"
            (checkpoint_dir / SNAPSHOT_DIR_NAME).mkdir(parents=True)
            (checkpoint_dir / INFERENCE_RESULT_FILENAME).write_text(
                json.dumps(
                    _inference_result(
                        cost=cost,
                        steps=index,
                    )
                ),
                encoding="utf-8",
            )

        newest_artifact_mtime = max(
            (tmp_path / f"checkpoint_{index}" / INFERENCE_RESULT_FILENAME)
            .stat()
            .st_mtime_ns
            for index in (1, 2)
        )
        stale_metadata_mtime = newest_artifact_mtime + 1_000_000_000
        os.utime(
            run_info_path,
            ns=(stale_metadata_mtime, stale_metadata_mtime),
        )

        result = detect_resume_point(
            tmp_path,
            ["checkpoint_1", "checkpoint_2"],
        )

        assert result is not None
        assert result.resume_from_checkpoint == ""
        assert result.completed_checkpoints == [
            "checkpoint_1",
            "checkpoint_2",
        ]
        assert result.invalidated_checkpoints == []
        assert result.prior_usage.cost == 3.0
        assert result.run_info_reconciliation_required is True

    def test_handles_missing_snapshot_directory(self, tmp_path: Path) -> None:
        """Checkpoint with missing snapshot is not considered complete."""
        run_info = {
            "summary": {
                "checkpoints": {
                    "checkpoint_1": CheckpointState.RAN,
                    "checkpoint_2": CheckpointState.RAN,
                }
            }
        }
        with (tmp_path / RUN_INFO_FILENAME).open("w") as f:
            yaml.dump(run_info, f)

        # Only create snapshot for checkpoint_1, not checkpoint_2
        (tmp_path / "checkpoint_1" / SNAPSHOT_DIR_NAME).mkdir(parents=True)
        inference_result = _inference_result(cost=0.5, steps=5)
        with (tmp_path / "checkpoint_1" / INFERENCE_RESULT_FILENAME).open(
            "w"
        ) as f:
            json.dump(inference_result, f)

        result = detect_resume_point(tmp_path, ["checkpoint_1", "checkpoint_2"])

        assert result is not None
        # Should resume from checkpoint_2 since it has no snapshot
        assert result.resume_from_checkpoint == "checkpoint_2"
        assert result.completed_checkpoints == ["checkpoint_1"]

    def test_handles_invalid_yaml(self, tmp_path: Path) -> None:
        """Invalid metadata falls back to artifacts instead of starting fresh."""
        with (tmp_path / RUN_INFO_FILENAME).open("w") as f:
            f.write("invalid: yaml: content: {{{{")

        result = detect_resume_point(tmp_path, ["checkpoint_1"])
        assert result is not None
        assert result.resume_from_checkpoint == "checkpoint_1"
        assert result.invalidated_checkpoints == ["checkpoint_1"]

    def test_invalid_yaml_uses_completed_checkpoint_artifacts(
        self, tmp_path: Path
    ) -> None:
        (tmp_path / RUN_INFO_FILENAME).write_text(
            "summary: [truncated",
            encoding="utf-8",
        )
        checkpoint_dir = tmp_path / "checkpoint_1"
        (checkpoint_dir / SNAPSHOT_DIR_NAME).mkdir(parents=True)
        (checkpoint_dir / INFERENCE_RESULT_FILENAME).write_text(
            json.dumps(_inference_result()),
            encoding="utf-8",
        )

        result = detect_resume_point(
            tmp_path,
            ["checkpoint_1", "checkpoint_2"],
        )

        assert result is not None
        assert result.completed_checkpoints == ["checkpoint_1"]
        assert result.resume_from_checkpoint == "checkpoint_2"

    def test_non_object_inference_result_is_invalid_not_an_exception(
        self, tmp_path: Path
    ) -> None:
        (tmp_path / RUN_INFO_FILENAME).write_text(
            yaml.safe_dump(
                {"summary": {"checkpoints": {"checkpoint_1": "ran"}}}
            ),
            encoding="utf-8",
        )
        checkpoint_dir = tmp_path / "checkpoint_1"
        (checkpoint_dir / SNAPSHOT_DIR_NAME).mkdir(parents=True)
        (checkpoint_dir / INFERENCE_RESULT_FILENAME).write_text(
            "[]",
            encoding="utf-8",
        )

        result = detect_resume_point(tmp_path, ["checkpoint_1"])

        assert result is not None
        assert result.completed_checkpoints == []
        assert result.resume_from_checkpoint == "checkpoint_1"

    @pytest.mark.parametrize(
        "mutation",
        (
            lambda result: result.pop("elapsed"),
            lambda result: result["usage"].update({"cost": -1.0}),
            lambda result: result.update(
                {
                    "started": "2026-01-02T00:00:00",
                    "completed": "2026-01-01T00:00:00",
                }
            ),
            lambda result: result.update(
                {
                    "started": "2026-01-01T00:00:00",
                    "completed": "2026-01-01T00:00:00+00:00",
                }
            ),
        ),
    )
    def test_semantically_invalid_inference_result_forces_rerun(
        self,
        tmp_path: Path,
        mutation,
    ) -> None:
        checkpoint_dir = tmp_path / "checkpoint_1"
        (checkpoint_dir / SNAPSHOT_DIR_NAME).mkdir(parents=True)
        payload = _inference_result()
        mutation(payload)
        (checkpoint_dir / INFERENCE_RESULT_FILENAME).write_text(
            json.dumps(payload),
            encoding="utf-8",
        )

        result = detect_resume_point(tmp_path, ["checkpoint_1"])

        assert result is not None
        assert result.completed_checkpoints == []
        assert result.resume_from_checkpoint == "checkpoint_1"

    @pytest.mark.parametrize(
        "metadata_kind",
        ("missing", "unreadable", "schema-invalid"),
    )
    def test_complete_artifacts_require_metadata_only_reconciliation(
        self,
        tmp_path: Path,
        metadata_kind: str,
    ) -> None:
        checkpoint_dir = tmp_path / "checkpoint_1"
        (checkpoint_dir / SNAPSHOT_DIR_NAME).mkdir(parents=True)
        (checkpoint_dir / INFERENCE_RESULT_FILENAME).write_text(
            json.dumps(_inference_result(cost=1.0, steps=2)),
            encoding="utf-8",
        )
        run_info_path = tmp_path / RUN_INFO_FILENAME
        if metadata_kind == "unreadable":
            run_info_path.mkdir()
        elif metadata_kind == "schema-invalid":
            run_info_path.write_text(
                yaml.safe_dump({"summary": {"checkpoints": {}}}),
                encoding="utf-8",
            )

        result = detect_resume_point(tmp_path, ["checkpoint_1"])

        assert result is not None
        assert result.resume_from_checkpoint == ""
        assert result.completed_checkpoints == ["checkpoint_1"]
        assert result.run_info_reconciliation_required is True

    def test_complete_artifacts_supersede_semantically_stale_aggregate(
        self,
        tmp_path: Path,
    ) -> None:
        checkpoint_dir = tmp_path / "checkpoint_1"
        (checkpoint_dir / SNAPSHOT_DIR_NAME).mkdir(parents=True)
        actual_result = _inference_result(cost=1.0, steps=2)
        (checkpoint_dir / INFERENCE_RESULT_FILENAME).write_text(
            json.dumps(actual_result),
            encoding="utf-8",
        )
        stale_usage = UsageTracker(cost=999.0, steps=999)
        now = datetime.now().isoformat()
        (tmp_path / RUN_INFO_FILENAME).write_text(
            yaml.safe_dump(
                {
                    "pass_policy": PassPolicy.ANY_CASE.value,
                    "summary": {
                        "started": now,
                        "ended": now,
                        "duration_seconds": 0.0,
                        "total_cost": stale_usage.cost,
                        "total_steps": stale_usage.steps,
                        "total_usage": stale_usage.model_dump(),
                        "checkpoints": {"checkpoint_1": CheckpointState.RAN},
                        "state": "running",
                        "passed_policy": False,
                    },
                }
            ),
            encoding="utf-8",
        )

        result = detect_resume_point(tmp_path, ["checkpoint_1"])

        assert result is not None
        assert result.resume_from_checkpoint == ""
        assert result.prior_usage.cost == 1.0
        assert result.prior_usage.steps == 2
        assert result.run_info_reconciliation_required is True

    def test_truthful_completed_run_info_needs_no_reconciliation(
        self,
        tmp_path: Path,
    ) -> None:
        checkpoint_dir = tmp_path / "checkpoint_1"
        (checkpoint_dir / SNAPSHOT_DIR_NAME).mkdir(parents=True)
        inference = _inference_result(
            cost=1.0,
            steps=2,
            net_tokens={"input": 7, "output": 3},
        )
        (checkpoint_dir / INFERENCE_RESULT_FILENAME).write_text(
            json.dumps(inference),
            encoding="utf-8",
        )
        now = datetime.now().isoformat()
        usage = inference["usage"]
        (tmp_path / RUN_INFO_FILENAME).write_text(
            yaml.safe_dump(
                {
                    "pass_policy": PassPolicy.ANY_CASE.value,
                    "summary": {
                        "started": now,
                        "ended": now,
                        "duration_seconds": 0.0,
                        "total_cost": 1.0,
                        "total_steps": 2,
                        "total_usage": usage,
                        "checkpoints": {"checkpoint_1": CheckpointState.RAN},
                        "state": "completed",
                        "passed_policy": True,
                    },
                }
            ),
            encoding="utf-8",
        )

        result = detect_resume_point(tmp_path, ["checkpoint_1"])

        assert result is not None
        assert result.run_info_reconciliation_required is False
        assert result.prior_usage.current_tokens.input == 7
        assert result.prior_usage.current_tokens.output == 3

    def test_handles_malformed_run_info(self, tmp_path: Path) -> None:
        """Returns ResumeInfo with invalidated checkpoints when run_info structure is unexpected."""
        with (tmp_path / RUN_INFO_FILENAME).open("w") as f:
            yaml.dump({"not_summary": {}}, f)

        result = detect_resume_point(tmp_path, ["checkpoint_1"])
        # When run_info.yaml has unexpected structure but is valid YAML,
        # checkpoints are treated as missing/invalid
        assert result is not None
        assert result.resume_from_checkpoint == "checkpoint_1"
        assert result.completed_checkpoints == []

    def test_missing_summary_uses_completed_checkpoint_artifacts(
        self, tmp_path: Path
    ) -> None:
        (tmp_path / RUN_INFO_FILENAME).write_text(
            yaml.safe_dump({"not_summary": {}}),
            encoding="utf-8",
        )
        checkpoint_dir = tmp_path / "checkpoint_1"
        (checkpoint_dir / SNAPSHOT_DIR_NAME).mkdir(parents=True)
        (checkpoint_dir / INFERENCE_RESULT_FILENAME).write_text(
            json.dumps(_inference_result()),
            encoding="utf-8",
        )

        result = detect_resume_point(
            tmp_path,
            ["checkpoint_1", "checkpoint_2"],
        )

        assert result is not None
        assert result.completed_checkpoints == ["checkpoint_1"]
        assert result.resume_from_checkpoint == "checkpoint_2"


class TestDetectResumeFromArtifacts:
    """Tests for _detect_resume_from_artifacts fallback function."""

    def test_returns_resume_info_when_no_snapshots(
        self, tmp_path: Path
    ) -> None:
        """Returns ResumeInfo to resume from first checkpoint when no snapshots exist."""
        result = _detect_resume_from_artifacts(
            tmp_path, ["checkpoint_1", "checkpoint_2"]
        )
        # When no checkpoint directories exist, returns ResumeInfo with all invalidated
        assert result is not None
        assert result.resume_from_checkpoint == "checkpoint_1"
        assert result.completed_checkpoints == []
        assert result.invalidated_checkpoints == [
            "checkpoint_1",
            "checkpoint_2",
        ]

    def test_resumes_from_checkpoint_with_error(self, tmp_path: Path) -> None:
        """Resumes from checkpoint that has snapshot + error."""
        # checkpoint_1: snapshot + no error (completed)
        checkpoint_1_dir = tmp_path / "checkpoint_1"
        (checkpoint_1_dir / SNAPSHOT_DIR_NAME).mkdir(parents=True)
        with (checkpoint_1_dir / INFERENCE_RESULT_FILENAME).open("w") as f:
            json.dump(_inference_result(cost=0.5, steps=5), f)

        # checkpoint_2: snapshot + error (resume from here)
        checkpoint_2_dir = tmp_path / "checkpoint_2"
        (checkpoint_2_dir / SNAPSHOT_DIR_NAME).mkdir(parents=True)
        with (checkpoint_2_dir / INFERENCE_RESULT_FILENAME).open("w") as f:
            json.dump(
                _inference_result(cost=0.3, steps=3, had_error=True),
                f,
            )

        result = _detect_resume_from_artifacts(
            tmp_path, ["checkpoint_1", "checkpoint_2", "checkpoint_3"]
        )

        assert result is not None
        assert result.resume_from_checkpoint == "checkpoint_2"
        assert result.completed_checkpoints == ["checkpoint_1"]
        assert result.prior_usage.cost == 0.5
        assert result.prior_usage.steps == 5

    def test_resumes_from_first_checkpoint_with_error(
        self, tmp_path: Path
    ) -> None:
        """Resumes from first checkpoint if it has error."""
        # checkpoint_1: snapshot + error (resume from here)
        checkpoint_1_dir = tmp_path / "checkpoint_1"
        (checkpoint_1_dir / SNAPSHOT_DIR_NAME).mkdir(parents=True)
        with (checkpoint_1_dir / INFERENCE_RESULT_FILENAME).open("w") as f:
            json.dump(
                _inference_result(cost=0.5, steps=5, had_error=True),
                f,
            )

        result = _detect_resume_from_artifacts(
            tmp_path, ["checkpoint_1", "checkpoint_2"]
        )

        # No completed checkpoints, returns ResumeInfo to resume from first
        assert result is not None
        assert result.resume_from_checkpoint == "checkpoint_1"
        assert result.completed_checkpoints == []
        assert result.invalidated_checkpoints == [
            "checkpoint_1",
            "checkpoint_2",
        ]

    def test_resumes_from_next_checkpoint_after_completed(
        self, tmp_path: Path
    ) -> None:
        """Resumes from checkpoint after last completed one."""
        # checkpoint_1: snapshot + no error (completed)
        checkpoint_1_dir = tmp_path / "checkpoint_1"
        (checkpoint_1_dir / SNAPSHOT_DIR_NAME).mkdir(parents=True)
        with (checkpoint_1_dir / INFERENCE_RESULT_FILENAME).open("w") as f:
            json.dump(_inference_result(cost=1.0, steps=10), f)

        # checkpoint_2: no snapshot (resume from here)
        # (directory doesn't exist)

        result = _detect_resume_from_artifacts(
            tmp_path, ["checkpoint_1", "checkpoint_2", "checkpoint_3"]
        )

        assert result is not None
        assert result.resume_from_checkpoint == "checkpoint_2"
        assert result.completed_checkpoints == ["checkpoint_1"]
        assert result.last_snapshot_dir == checkpoint_1_dir / SNAPSHOT_DIR_NAME
        assert result.prior_usage.cost == 1.0

    def test_returns_resume_info_when_all_completed(
        self, tmp_path: Path
    ) -> None:
        """Returns ResumeInfo with empty resume_from when all checkpoints completed."""
        for i in range(1, 3):
            checkpoint_dir = tmp_path / f"checkpoint_{i}"
            (checkpoint_dir / SNAPSHOT_DIR_NAME).mkdir(parents=True)
            with (checkpoint_dir / INFERENCE_RESULT_FILENAME).open("w") as f:
                json.dump(_inference_result(cost=0.5), f)

        result = _detect_resume_from_artifacts(
            tmp_path, ["checkpoint_1", "checkpoint_2"]
        )
        assert result is not None
        assert result.resume_from_checkpoint == ""  # Empty = nothing to resume
        assert result.completed_checkpoints == ["checkpoint_1", "checkpoint_2"]
        assert result.invalidated_checkpoints == []

    def test_resumes_when_snapshot_but_no_inference_result(
        self, tmp_path: Path
    ) -> None:
        """Resumes from checkpoint with snapshot but missing inference result."""
        # checkpoint_1: snapshot + no error (completed)
        checkpoint_1_dir = tmp_path / "checkpoint_1"
        (checkpoint_1_dir / SNAPSHOT_DIR_NAME).mkdir(parents=True)
        with (checkpoint_1_dir / INFERENCE_RESULT_FILENAME).open("w") as f:
            json.dump(_inference_result(cost=0.5, steps=5), f)

        # checkpoint_2: snapshot but no inference result (resume from here)
        checkpoint_2_dir = tmp_path / "checkpoint_2"
        (checkpoint_2_dir / SNAPSHOT_DIR_NAME).mkdir(parents=True)

        result = _detect_resume_from_artifacts(
            tmp_path, ["checkpoint_1", "checkpoint_2", "checkpoint_3"]
        )

        assert result is not None
        assert result.resume_from_checkpoint == "checkpoint_2"
        assert result.completed_checkpoints == ["checkpoint_1"]

    def test_resumes_when_inference_result_is_invalid_json(
        self, tmp_path: Path
    ) -> None:
        """Resumes from checkpoint with invalid inference result JSON."""
        # checkpoint_1: snapshot + no error (completed)
        checkpoint_1_dir = tmp_path / "checkpoint_1"
        (checkpoint_1_dir / SNAPSHOT_DIR_NAME).mkdir(parents=True)
        with (checkpoint_1_dir / INFERENCE_RESULT_FILENAME).open("w") as f:
            json.dump(_inference_result(cost=0.5, steps=5), f)

        # checkpoint_2: snapshot + invalid JSON (resume from here)
        checkpoint_2_dir = tmp_path / "checkpoint_2"
        (checkpoint_2_dir / SNAPSHOT_DIR_NAME).mkdir(parents=True)
        with (checkpoint_2_dir / INFERENCE_RESULT_FILENAME).open("w") as f:
            f.write("not valid json")

        result = _detect_resume_from_artifacts(
            tmp_path, ["checkpoint_1", "checkpoint_2", "checkpoint_3"]
        )

        assert result is not None
        assert result.resume_from_checkpoint == "checkpoint_2"
        assert result.completed_checkpoints == ["checkpoint_1"]

    def test_aggregates_usage_from_completed_checkpoints(
        self, tmp_path: Path
    ) -> None:
        """Aggregates usage from all completed checkpoints."""
        # Create multiple completed checkpoints
        for i in range(1, 3):
            checkpoint_dir = tmp_path / f"checkpoint_{i}"
            (checkpoint_dir / SNAPSHOT_DIR_NAME).mkdir(parents=True)
            with (checkpoint_dir / INFERENCE_RESULT_FILENAME).open("w") as f:
                json.dump(
                    _inference_result(
                        cost=float(i),
                        steps=i * 10,
                        net_tokens={
                            "input": i * 100,
                            "output": i * 50,
                        },
                    ),
                    f,
                )

        # checkpoint_3: no snapshot (resume from here)

        result = _detect_resume_from_artifacts(
            tmp_path, ["checkpoint_1", "checkpoint_2", "checkpoint_3"]
        )

        assert result is not None
        assert result.resume_from_checkpoint == "checkpoint_3"
        assert result.completed_checkpoints == ["checkpoint_1", "checkpoint_2"]
        assert result.prior_usage.cost == 3.0  # 1 + 2
        assert result.prior_usage.steps == 30  # 10 + 20
        assert result.prior_usage.net_tokens.input == 300  # 100 + 200

    def test_detect_resume_point_uses_artifact_fallback(
        self, tmp_path: Path
    ) -> None:
        """detect_resume_point uses artifact fallback when no run_info.yaml."""
        # No run_info.yaml, but checkpoint_1 completed
        checkpoint_1_dir = tmp_path / "checkpoint_1"
        (checkpoint_1_dir / SNAPSHOT_DIR_NAME).mkdir(parents=True)
        with (checkpoint_1_dir / INFERENCE_RESULT_FILENAME).open("w") as f:
            json.dump(_inference_result(cost=1.0, steps=5), f)

        result = detect_resume_point(tmp_path, ["checkpoint_1", "checkpoint_2"])

        assert result is not None
        assert result.resume_from_checkpoint == "checkpoint_2"
        assert result.completed_checkpoints == ["checkpoint_1"]


class TestAggregatePriorUsage:
    """Tests for _aggregate_prior_usage function."""

    def test_aggregates_usage_from_multiple_checkpoints(
        self, tmp_path: Path
    ) -> None:
        """Correctly sums usage from multiple checkpoint results."""
        # Create inference results for two checkpoints
        for i in range(1, 3):
            checkpoint_dir = tmp_path / f"checkpoint_{i}"
            checkpoint_dir.mkdir()
            result = _inference_result(
                cost=float(i),
                steps=i * 10,
                net_tokens={
                    "input": i * 100,
                    "output": i * 50,
                },
            )
            with (checkpoint_dir / INFERENCE_RESULT_FILENAME).open("w") as f:
                json.dump(result, f)

        usage = _aggregate_prior_usage(
            tmp_path, ["checkpoint_1", "checkpoint_2"]
        )

        assert usage.cost == 3.0  # 1 + 2
        assert usage.steps == 30  # 10 + 20
        assert usage.net_tokens.input == 300  # 100 + 200
        assert usage.net_tokens.output == 150  # 50 + 100

    def test_rejects_missing_inference_result(self, tmp_path: Path) -> None:
        """Completed-checkpoint accounting must never silently go partial."""
        # Only create result for checkpoint_2
        checkpoint_dir = tmp_path / "checkpoint_2"
        checkpoint_dir.mkdir()
        result = _inference_result(cost=1.5, steps=15)
        with (checkpoint_dir / INFERENCE_RESULT_FILENAME).open("w") as f:
            json.dump(result, f)

        with pytest.raises(ValueError, match="no inference result"):
            _aggregate_prior_usage(tmp_path, ["checkpoint_1", "checkpoint_2"])

    def test_rejects_invalid_json(self, tmp_path: Path) -> None:
        checkpoint_dir = tmp_path / "checkpoint_1"
        checkpoint_dir.mkdir()
        with (checkpoint_dir / INFERENCE_RESULT_FILENAME).open("w") as f:
            f.write("not valid json")

        with pytest.raises(ValueError, match="invalid inference result"):
            _aggregate_prior_usage(tmp_path, ["checkpoint_1"])

    def test_rejects_missing_usage_fields(self, tmp_path: Path) -> None:
        checkpoint_dir = tmp_path / "checkpoint_1"
        checkpoint_dir.mkdir()
        result = {"usage": {"cost": 1.0}}  # missing steps and tokens
        with (checkpoint_dir / INFERENCE_RESULT_FILENAME).open("w") as f:
            json.dump(result, f)

        with pytest.raises(ValueError, match="invalid inference result"):
            _aggregate_prior_usage(tmp_path, ["checkpoint_1"])


def test_evaluation_status_rejects_infrastructure_failure(
    tmp_path: Path,
) -> None:
    report = CorrectnessResults(
        problem_name="prob",
        problem_version=1,
        checkpoint_name="checkpoint_1",
        checkpoint_version=1,
        duration=0.01,
        entrypoint="python main.py",
        pytest_exit_code=3,
        pytest_collected=0,
        infrastructure_failure=True,
    )
    report.save(tmp_path)

    valid, reason = _evaluation_status(tmp_path)

    assert valid is False
    assert reason == InvalidationReason.EVALUATION_ERROR


class TestMetricsTrackerOnResume:
    """Tests for MetricsTracker behavior during resume.

    When resuming a run, completed checkpoints are skipped but their usage
    should still be accumulated in the MetricsTracker via finish_checkpoint().
    """

    def test_finish_checkpoint_accumulates_usage(self) -> None:
        """Verify finish_checkpoint correctly accumulates usage from summaries.

        This tests the component used when loading existing checkpoint summaries
        during resume - the metrics tracker should accumulate usage from each
        skipped checkpoint's saved summary.
        """
        now = datetime.now()
        tracker = MetricsTracker(
            current_checkpoint="checkpoint_1",
            usage=UsageTracker(cost=0.0, steps=0),
            started=now,
            checkpoint_started=now,
        )

        # Simulate loading a completed checkpoint's usage
        checkpoint_1_usage = UsageTracker(
            cost=1.5,
            steps=10,
            net_tokens=TokenUsage(input=100, output=50),
        )
        tracker.finish_checkpoint(checkpoint_1_usage)

        assert tracker.usage.cost == 1.5
        assert tracker.usage.steps == 10
        assert tracker.usage.net_tokens.input == 100
        assert tracker.usage.net_tokens.output == 50

    def test_finish_checkpoint_accumulates_multiple_checkpoints(self) -> None:
        """Verify multiple finish_checkpoint calls accumulate correctly.

        When resuming after multiple completed checkpoints, each checkpoint's
        usage should be added to the running total.
        """
        now = datetime.now()
        tracker = MetricsTracker(
            current_checkpoint="resuming",
            usage=UsageTracker(cost=0.0, steps=0),
            started=now,
            checkpoint_started=now,
        )

        # Simulate loading multiple completed checkpoints
        tracker.finish_checkpoint(
            UsageTracker(
                cost=1.0,
                steps=5,
                net_tokens=TokenUsage(input=100, output=50),
            )
        )
        tracker.finish_checkpoint(
            UsageTracker(
                cost=2.0,
                steps=15,
                net_tokens=TokenUsage(input=200, output=100),
            )
        )

        assert tracker.usage.cost == 3.0
        assert tracker.usage.steps == 20
        assert tracker.usage.net_tokens.input == 300
        assert tracker.usage.net_tokens.output == 150

    def test_finish_checkpoint_with_prior_usage(self) -> None:
        """Verify finish_checkpoint works when tracker has prior usage.

        When resuming, the MetricsTracker may be initialized with prior_usage.
        Calling finish_checkpoint should add to existing values.
        """
        now = datetime.now()
        # Initialize with prior usage (as happens during resume)
        tracker = MetricsTracker(
            current_checkpoint="checkpoint_2",
            usage=UsageTracker(
                cost=5.0,
                steps=50,
                net_tokens=TokenUsage(input=500, output=250),
            ),
            started=now,
            checkpoint_started=now,
        )

        # Add usage from a newly completed checkpoint
        tracker.finish_checkpoint(
            UsageTracker(
                cost=1.0,
                steps=10,
                net_tokens=TokenUsage(input=100, output=50),
            )
        )

        assert tracker.usage.cost == 6.0
        assert tracker.usage.steps == 60
        assert tracker.usage.net_tokens.input == 600
        assert tracker.usage.net_tokens.output == 300
