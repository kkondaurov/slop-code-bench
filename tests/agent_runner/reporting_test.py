"""Tests for agent runner reporting module."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock
from unittest.mock import patch

import pytest
import yaml

from slop_code.agent_runner.agent import CheckpointInferenceResult
from slop_code.agent_runner.models import UsageTracker
from slop_code.agent_runner.reporting import AgentCheckpointSummary
from slop_code.agent_runner.reporting import CheckpointState
from slop_code.agent_runner.reporting import RunSummary
from slop_code.agent_runner.reporting import _validate_saved_run_info
from slop_code.agent_runner.reporting import _write_run_info_atomically
from slop_code.agent_runner.reporting import save_agent_checkpoint_info
from slop_code.agent_runner.reporting import save_results
from slop_code.agent_runner.state import AgentStateEnum
from slop_code.common import to_relative_path
from slop_code.common.atomic import UnsafeAtomicWriteError
from slop_code.common.llms import TokenUsage
from slop_code.evaluation import PassPolicy


def _make_usage_tracker(cost: float = 0.5, steps: int = 10) -> UsageTracker:
    """Create a UsageTracker for testing."""
    return UsageTracker(
        cost=cost,
        steps=steps,
        current_tokens=TokenUsage(input=100, output=50),
        net_tokens=TokenUsage(input=100, output=50),
    )


class TestRunSummary:
    """Tests for RunSummary model."""

    def test_serialization_contains_checkpoints_dict(self) -> None:
        """Test that RunSummary serializes checkpoints as dict."""
        now = datetime.now()
        summary = RunSummary(
            started=now,
            ended=now,
            duration_seconds=60.0,
            total_cost=1.5,
            total_steps=10,
            total_usage=_make_usage_tracker(),
            checkpoints={
                "checkpoint_1": CheckpointState.RAN,
                "checkpoint_2": CheckpointState.SKIPPED,
                "checkpoint_3": CheckpointState.ERROR,
            },
            state=AgentStateEnum.COMPLETED,
        )
        data = summary.model_dump(mode="json")

        # Checkpoints should be a dict with state values
        assert "checkpoints" in data
        assert data["checkpoints"]["checkpoint_1"] == "ran"
        assert data["checkpoints"]["checkpoint_2"] == "skipped"
        assert data["checkpoints"]["checkpoint_3"] == "error"

        # Old format fields should NOT exist
        assert "checkpoints_inferred" not in data
        assert "checkpoints_skipped" not in data

    def test_datetime_validators(self) -> None:
        """Test that datetime fields accept ISO format strings."""
        now = datetime.now()
        iso_now = now.isoformat()

        summary = RunSummary(
            started=iso_now,  # string input
            ended=iso_now,  # string input
            duration_seconds=60.0,
            total_cost=1.5,
            total_steps=10,
            total_usage=_make_usage_tracker(),
            checkpoints={},
            state=AgentStateEnum.COMPLETED,
        )

        assert isinstance(summary.started, datetime)
        assert isinstance(summary.ended, datetime)


class TestAgentCheckpointSummary:
    def test_no_evaluation_does_not_hide_inference_error(
        self, tmp_path: Path
    ) -> None:
        summary = AgentCheckpointSummary.from_results(
            checkpoint_name="checkpoint_1",
            path=tmp_path,
            snapshot_dir=tmp_path / "snapshot",
            artifacts=tmp_path / "agent",
            usage=_make_usage_tracker(),
            had_error=True,
            pass_policy=PassPolicy.ANY_CASE,
            evaluation_result=None,
        )

        assert summary.passed_policy is False

    def test_infrastructure_failure_never_passes_policy(
        self, tmp_path: Path
    ) -> None:
        evaluation = Mock()
        evaluation.infrastructure_failure = True
        evaluation.pass_counts = {"Core": 1}
        evaluation.total_counts = {"Core": 1}

        summary = AgentCheckpointSummary.from_results(
            checkpoint_name="checkpoint_1",
            path=tmp_path,
            snapshot_dir=tmp_path / "snapshot",
            artifacts=tmp_path / "agent",
            usage=_make_usage_tracker(),
            had_error=False,
            pass_policy=PassPolicy.ANY_CASE,
            evaluation_result=evaluation,
        )

        assert summary.passed_policy is False


def test_atomic_run_info_replacement_preserves_malformed_evidence(
    tmp_path: Path,
) -> None:
    target = tmp_path / "run_info.yaml"
    malformed = "summary: [truncated"
    target.write_text(malformed, encoding="utf-8")

    _write_run_info_atomically(target, {"summary": {"state": "completed"}})

    assert yaml.safe_load(target.read_text()) == {
        "summary": {"state": "completed"}
    }
    backups = list(tmp_path.glob("run_info.corrupt-*.yaml"))
    assert len(backups) == 1
    assert backups[0].read_text() == malformed
    assert not (tmp_path / ".run_info.yaml.tmp").exists()


def test_atomic_run_info_preserves_mapping_shaped_semantic_corruption(
    tmp_path: Path,
) -> None:
    target = tmp_path / "run_info.yaml"
    malformed = "summary:\n  state: completed\n"
    target.write_text(malformed, encoding="utf-8")

    _write_run_info_atomically(target, {"summary": {"state": "error"}})

    backups = list(tmp_path.glob("run_info.corrupt-*.yaml"))
    assert len(backups) == 1
    assert backups[0].read_text(encoding="utf-8") == malformed


def test_atomic_run_info_write_failure_preserves_existing_file(
    tmp_path: Path,
) -> None:
    target = tmp_path / "run_info.yaml"
    original = "summary:\n  state: running\n"
    target.write_text(original, encoding="utf-8")

    with (
        patch(
            "slop_code.agent_runner.reporting.yaml.dump",
            side_effect=OSError("disk full"),
        ),
        pytest.raises(OSError, match="disk full"),
    ):
        _write_run_info_atomically(
            target,
            {"summary": {"state": "completed"}},
        )

    assert target.read_text() == original
    assert not (tmp_path / ".run_info.yaml.tmp").exists()


def test_atomic_run_info_write_failure_restores_non_file_target(
    tmp_path: Path,
) -> None:
    target = tmp_path / "run_info.yaml"
    target.mkdir()
    (target / "evidence.txt").write_text("preserve me", encoding="utf-8")

    with (
        patch(
            "slop_code.agent_runner.reporting.atomic_write_text",
            side_effect=OSError("disk full"),
        ),
        pytest.raises(OSError, match="disk full"),
    ):
        _write_run_info_atomically(
            target,
            {"summary": {"state": "completed"}},
        )

    assert target.is_dir()
    assert (target / "evidence.txt").read_text(encoding="utf-8") == (
        "preserve me"
    )
    assert not list(tmp_path.glob("run_info.corrupt-*.yaml"))


def test_atomic_run_info_rejects_symlink_without_copying_target(
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside"
    outside.write_text("private evidence", encoding="utf-8")
    target = tmp_path / "run_info.yaml"
    target.symlink_to(outside)

    with pytest.raises(UnsafeAtomicWriteError, match="symlink target"):
        _write_run_info_atomically(
            target,
            {"summary": {"state": "completed"}},
        )

    assert outside.read_text(encoding="utf-8") == "private evidence"
    assert not list(tmp_path.glob("run_info.corrupt-*.yaml"))


def test_run_info_validation_rejects_mixed_timestamp_domains() -> None:
    usage = _make_usage_tracker()
    run_info = {
        "summary": {
            "started": "2026-01-01T00:00:00",
            "ended": "2026-01-01T00:00:00+00:00",
            "duration_seconds": 0.0,
            "total_cost": usage.cost,
            "total_steps": usage.steps,
            "total_usage": usage.model_dump(),
            "checkpoints": {},
            "state": "completed",
            "passed_policy": True,
        }
    }

    assert _validate_saved_run_info(run_info) is None


class TestCheckpointInferenceResultPaths:
    """Tests for CheckpointInferenceResult path fields."""

    def test_path_fields_serialize_correctly(self) -> None:
        """Test that path fields serialize to strings."""
        now = datetime.now()
        result = CheckpointInferenceResult(
            started=now,
            completed=now,
            elapsed=60.0,
            usage=_make_usage_tracker(),
            had_error=False,
            checkpoint_path=Path("/output/checkpoint_1"),
            snapshot_dir=Path("snapshot"),
            artifacts_dir=Path("agent"),
        )
        data = result.model_dump(mode="json")

        assert data["checkpoint_path"] == "/output/checkpoint_1"
        assert data["snapshot_dir"] == "snapshot"
        assert data["artifacts_dir"] == "agent"

    def test_path_fields_optional(self) -> None:
        """Test that path fields default to None."""
        now = datetime.now()
        result = CheckpointInferenceResult(
            started=now,
            completed=now,
            elapsed=60.0,
            usage=_make_usage_tracker(),
            had_error=False,
        )

        assert result.checkpoint_path is None
        assert result.snapshot_dir is None
        assert result.artifacts_dir is None


class TestSaveAgentCheckpointInfo:
    """Tests for save_agent_checkpoint_info function."""

    def test_populates_path_fields(self, tmp_path: Path) -> None:
        """Test that path fields are populated in inference_result.json."""
        now = datetime.now()
        checkpoint_result = CheckpointInferenceResult(
            started=now,
            completed=now,
            elapsed=60.0,
            usage=_make_usage_tracker(),
            had_error=False,
        )

        # Mock the diff and agent
        diff = Mock()
        diff.model_dump_json.return_value = "{}"

        agent = Mock()
        agent.save_artifacts = Mock()

        save_agent_checkpoint_info(
            tmp_path, diff, checkpoint_result, agent, compress_artifacts=False
        )

        # Read the saved inference_result.json
        inference_file = tmp_path / "inference_result.json"
        assert inference_file.exists()

        with inference_file.open("r") as f:
            saved_data = json.load(f)

        # Verify path fields are populated
        assert saved_data["checkpoint_path"] == to_relative_path(tmp_path)
        assert saved_data["snapshot_dir"] == "snapshot"
        assert saved_data["artifacts_dir"] == "agent"


class TestSaveResults:
    """Tests for save_results function."""

    def test_creates_run_info_yaml(self, tmp_path: Path) -> None:
        """Test that save_results creates run_info.yaml with summary section."""
        # Create mock checkpoint summary
        checkpoint_summary = Mock()
        checkpoint_summary.checkpoint_name = "checkpoint_1"
        checkpoint_summary.passed_policy = True
        checkpoint_summary.had_error = False

        # Create mock metrics tracker
        metrics_tracker = Mock()
        metrics_tracker.state = AgentStateEnum.COMPLETED
        metrics_tracker.usage = _make_usage_tracker(cost=1.5, steps=10)
        metrics_tracker.started = datetime.now()
        metrics_tracker.error_type = None
        metrics_tracker.error_message = None
        metrics_tracker.error_traceback = None

        # Create mock run spec
        run_spec = Mock()
        run_spec.problem.name = "test_problem"
        run_spec.problem.version = 1
        run_spec.problem.entry_file = "main.py"
        run_spec.problem.checkpoints = {
            "checkpoint_1": Mock(),
            "checkpoint_2": Mock(),
        }
        run_spec.problem.model_dump.return_value = {
            "name": "test_problem",
            "version": 1,
            "entry_file": "main.py",
        }
        run_spec.model_dump.return_value = {
            "seed": 42,
            "pass_policy": "any-case",
            "skip_evaluation": False,
            "template": "test.jinja",
            "environment": {},
            "problem": {},
        }
        run_spec.seed = 42
        run_spec.pass_policy = PassPolicy.ANY
        run_spec.skip_evaluation = False
        run_spec.environment.get_command.return_value = "python main.py"

        save_results(
            results=[checkpoint_summary],
            metrics_tracker=metrics_tracker,
            run_spec=run_spec,
            output_path=tmp_path,
        )

        # Verify run_info.yaml was written
        run_info_file = tmp_path / "run_info.yaml"
        assert run_info_file.exists()

        with run_info_file.open("r") as f:
            saved_data = yaml.safe_load(f)

        # Verify structure: top-level has run spec fields, summary is nested
        # Note: problem and environment are explicitly removed by save_results()
        assert "summary" in saved_data
        assert "seed" in saved_data

        # Verify summary contains checkpoint states
        summary = saved_data["summary"]
        assert "checkpoints" in summary
        assert summary["checkpoints"]["checkpoint_1"] == "ran"
        assert summary["checkpoints"]["checkpoint_2"] == "skipped"

        # Verify summary fields
        assert "duration_seconds" in summary
        assert "total_cost" in summary
        assert "total_steps" in summary
        assert "state" in summary

    def test_checkpoint_error_state(self, tmp_path: Path) -> None:
        """Test that checkpoint with error gets 'error' state."""
        # Create mock checkpoint summary with error
        checkpoint_summary = Mock()
        checkpoint_summary.checkpoint_name = "checkpoint_1"
        checkpoint_summary.passed_policy = False
        checkpoint_summary.had_error = True

        # Create mock metrics tracker
        metrics_tracker = Mock()
        metrics_tracker.state = AgentStateEnum.ERROR
        metrics_tracker.usage = _make_usage_tracker(cost=0.5, steps=5)
        metrics_tracker.started = datetime.now()
        metrics_tracker.error_type = "RuntimeError"
        metrics_tracker.error_message = "Something went wrong"
        metrics_tracker.error_traceback = "..."

        # Create mock run spec
        run_spec = Mock()
        run_spec.problem.name = "test_problem"
        run_spec.problem.version = 1
        run_spec.problem.entry_file = "main.py"
        run_spec.problem.checkpoints = {"checkpoint_1": Mock()}
        run_spec.problem.model_dump.return_value = {"name": "test_problem"}
        run_spec.model_dump.return_value = {
            "seed": 42,
            "pass_policy": "any-case",
            "skip_evaluation": False,
            "environment": {},
            "problem": {},
        }
        run_spec.seed = 42
        run_spec.pass_policy = PassPolicy.ANY
        run_spec.skip_evaluation = False
        run_spec.environment.get_command.return_value = "python main.py"

        result = save_results(
            results=[checkpoint_summary],
            metrics_tracker=metrics_tracker,
            run_spec=run_spec,
            output_path=tmp_path,
        )

        # Verify checkpoint has error state
        assert result["summary"]["checkpoints"]["checkpoint_1"] == "error"
        assert result["summary"]["error_type"] == "RuntimeError"

    def test_genuine_evaluation_error_has_distinct_state(
        self, tmp_path: Path
    ) -> None:
        checkpoint_summary = Mock()
        checkpoint_summary.checkpoint_name = "checkpoint_1"
        checkpoint_summary.passed_policy = False
        checkpoint_summary.had_error = False
        checkpoint_summary.evaluation_error_message = "evaluator crashed"

        metrics_tracker = Mock()
        metrics_tracker.state = AgentStateEnum.ERROR
        metrics_tracker.usage = _make_usage_tracker()
        metrics_tracker.started = datetime.now()
        metrics_tracker.error_type = "EvaluationError"
        metrics_tracker.error_message = "evaluator crashed"
        metrics_tracker.error_traceback = "..."

        run_spec = Mock()
        run_spec.problem.name = "test_problem"
        run_spec.problem.checkpoints = {"checkpoint_1": Mock()}
        run_spec.model_dump.return_value = {
            "seed": 42,
            "pass_policy": "any-case",
            "skip_evaluation": False,
            "environment": {},
            "problem": {},
        }
        run_spec.skip_evaluation = False

        result = save_results(
            results=[checkpoint_summary],
            metrics_tracker=metrics_tracker,
            run_spec=run_spec,
            output_path=tmp_path,
        )

        assert result["summary"]["checkpoints"] == {
            "checkpoint_1": "evaluation_error"
        }
        assert result["summary"]["passed_policy"] is False
