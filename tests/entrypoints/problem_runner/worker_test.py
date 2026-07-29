"""Resume-boundary tests for the per-problem worker."""

from __future__ import annotations

import json
import queue
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock
from unittest.mock import patch

import pytest
import yaml

from slop_code.agent_runner.models import AgentRunSpec
from slop_code.agent_runner.models import UsageTracker
from slop_code.agent_runner.resume import CheckpointStatus
from slop_code.agent_runner.resume import InvalidationReason
from slop_code.agent_runner.resume import ResumeInfo
from slop_code.common.atomic import UnsafeAtomicWriteError
from slop_code.entrypoints.problem_runner.models import RunTaskConfig
from slop_code.entrypoints.problem_runner.worker import (
    _persist_completed_run_info_reconciliation,
)
from slop_code.entrypoints.problem_runner.worker import (
    _quarantine_invalidated_checkpoints,
)
from slop_code.entrypoints.problem_runner.worker import (
    _write_quarantine_manifest,
)
from slop_code.entrypoints.problem_runner.worker import run_agent_on_problem
from slop_code.evaluation import CheckpointConfig
from slop_code.evaluation import CorrectnessResults
from slop_code.evaluation import GroupType
from slop_code.evaluation import PassPolicy
from slop_code.evaluation import ProblemConfig
from slop_code.execution.local_streaming import LocalEnvironmentSpec
from slop_code.execution.models import CommandConfig
from slop_code.provenance import artifact_checksums


def test_invalidated_checkpoint_is_quarantined_and_checksums_bind_it(
    tmp_path: Path,
) -> None:
    checkpoint_dir = tmp_path / "checkpoint_1"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "evidence.txt").write_text("old evidence")
    resume_info = ResumeInfo(
        resume_from_checkpoint="checkpoint_1",
        completed_checkpoints=[],
        last_snapshot_dir=None,
        prior_usage=UsageTracker(),
        checkpoint_statuses=[
            CheckpointStatus(
                name="checkpoint_1",
                is_valid=False,
                reason=InvalidationReason.SPEC_CHANGED,
            )
        ],
        invalidated_checkpoints=["checkpoint_1"],
    )

    batch_dir = _quarantine_invalidated_checkpoints(tmp_path, resume_info)

    assert batch_dir is not None
    assert not checkpoint_dir.exists()
    assert (batch_dir / "checkpoint_1" / "evidence.txt").read_text() == (
        "old evidence"
    )
    manifest = json.loads((batch_dir / "manifest.json").read_text())
    assert manifest["checkpoints"][0]["reason"] == "spec_changed"
    evidence_path = (batch_dir / "checkpoint_1" / "evidence.txt").relative_to(
        tmp_path
    )
    assert evidence_path.as_posix() in artifact_checksums(tmp_path)


def test_missing_invalidated_checkpoint_does_not_create_empty_quarantine(
    tmp_path: Path,
) -> None:
    resume_info = ResumeInfo(
        resume_from_checkpoint="checkpoint_1",
        completed_checkpoints=[],
        last_snapshot_dir=None,
        prior_usage=UsageTracker(),
        invalidated_checkpoints=["checkpoint_1"],
    )

    assert _quarantine_invalidated_checkpoints(tmp_path, resume_info) is None
    assert list(tmp_path.iterdir()) == []


def test_quarantine_manifest_rejects_symlink_target(tmp_path: Path) -> None:
    batch_dir = tmp_path / "batch"
    batch_dir.mkdir()
    outside = tmp_path / "outside"
    outside.write_text("untouched", encoding="utf-8")
    (batch_dir / "manifest.json").symlink_to(outside)

    with pytest.raises(UnsafeAtomicWriteError, match="symlink target"):
        _write_quarantine_manifest(batch_dir, {"schema_version": 1})

    assert outside.read_text(encoding="utf-8") == "untouched"


def test_fully_resumed_worker_reconstructs_failing_verdict(
    tmp_path: Path,
) -> None:
    checkpoint_dir = tmp_path / "checkpoint_1"
    checkpoint_dir.mkdir()
    CorrectnessResults(
        problem_name="problem",
        problem_version=1,
        checkpoint_name="checkpoint_1",
        checkpoint_version=1,
        duration=0.1,
        entrypoint="python main.py",
        pass_counts={GroupType.CORE: 0},
        total_counts={GroupType.CORE: 1},
        pytest_exit_code=1,
        pytest_collected=1,
    ).save(checkpoint_dir)
    resume_info = ResumeInfo(
        resume_from_checkpoint="",
        completed_checkpoints=["checkpoint_1"],
        last_snapshot_dir=checkpoint_dir / "snapshot",
        prior_usage=UsageTracker(cost=1.25, steps=7),
    )
    problem = Mock()
    problem.iterate_checkpoint_items.return_value = [("checkpoint_1", Mock())]
    config = cast(
        "RunTaskConfig",
        SimpleNamespace(
            one_shot=False,
            resume=True,
            prompt_template="prompt",
            env_spec=Mock(),
            disable_evaluation=False,
            pass_policy=PassPolicy.ANY_CASE,
        ),
    )

    with (
        patch(
            "slop_code.entrypoints.problem_runner.worker.apply_one_shot_mode",
            return_value=problem,
        ),
        patch(
            "slop_code.entrypoints.problem_runner.worker.detect_resume_point",
            return_value=resume_info,
        ),
    ):
        result = run_agent_on_problem(
            problem,
            "problem",
            config,
            queue.Queue(),
            tmp_path,
        )

    assert result["summary"]["state"] == "completed"
    assert result["summary"]["passed_policy"] is False
    assert result["summary"]["total_cost"] == 1.25


def test_completed_artifact_reconciliation_persists_truthful_run_info(
    tmp_path: Path,
) -> None:
    checkpoint = CheckpointConfig(
        name="checkpoint_1",
        version=1,
        order=1,
    )
    problem = ProblemConfig(
        name="problem",
        path=tmp_path,
        version=1,
        description="resume reconciliation probe",
        tags=["test"],
        checkpoints={checkpoint.name: checkpoint},
        entry_file="main.py",
    )
    run_spec = AgentRunSpec(
        seed=42,
        template="{{ spec }}",
        problem=problem,
        environment=LocalEnvironmentSpec(
            type="local",
            name="test",
            commands=CommandConfig(command="python"),
        ),
        image="unused",
        pass_policy=PassPolicy.ANY_CASE,
        skip_evaluation=True,
    )
    checkpoint_dir = tmp_path / checkpoint.name
    (checkpoint_dir / "snapshot").mkdir(parents=True)
    now = datetime.now()
    usage = UsageTracker(cost=2.75, steps=9)
    (checkpoint_dir / "inference_result.json").write_text(
        json.dumps(
            {
                "started": now.isoformat(),
                "completed": now.isoformat(),
                "elapsed": 0.0,
                "usage": usage.model_dump(),
                "had_error": False,
                "error_message": None,
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "run_info.yaml").write_text(
        yaml.safe_dump(
            {"summary": {"checkpoints": {checkpoint.name: "skipped"}}}
        ),
        encoding="utf-8",
    )
    resume_info = ResumeInfo(
        resume_from_checkpoint="",
        completed_checkpoints=[checkpoint.name],
        last_snapshot_dir=checkpoint_dir / "snapshot",
        prior_usage=usage,
        run_info_reconciliation_required=True,
    )

    result = _persist_completed_run_info_reconciliation(
        run_spec,
        resume_info,
        tmp_path,
    )

    persisted = yaml.safe_load((tmp_path / "run_info.yaml").read_text())
    assert result["summary"]["checkpoints"] == {checkpoint.name: "ran"}
    assert result["summary"]["state"] == "completed"
    assert result["summary"]["total_cost"] == 2.75
    assert result["summary"]["passed_policy"] is True
    assert persisted == result


@pytest.mark.parametrize(
    "metadata_kind",
    ("missing", "malformed-file", "unreadable-directory"),
)
def test_fully_resumed_worker_atomically_repairs_metadata_without_agent(
    tmp_path: Path,
    metadata_kind: str,
) -> None:
    checkpoint = CheckpointConfig(
        name="checkpoint_1",
        version=1,
        order=1,
    )
    problem = ProblemConfig(
        name="problem",
        path=tmp_path,
        version=1,
        description="metadata-only resume probe",
        tags=["test"],
        checkpoints={checkpoint.name: checkpoint},
        entry_file="main.py",
    )
    environment = LocalEnvironmentSpec(
        type="local",
        name="test",
        commands=CommandConfig(command="python"),
    )
    checkpoint_dir = tmp_path / checkpoint.name
    (checkpoint_dir / "snapshot").mkdir(parents=True)
    now = datetime.now()
    usage = UsageTracker(cost=3.5, steps=11)
    (checkpoint_dir / "inference_result.json").write_text(
        json.dumps(
            {
                "started": now.isoformat(),
                "completed": now.isoformat(),
                "elapsed": 0.0,
                "usage": usage.model_dump(),
                "had_error": False,
                "error_message": None,
            }
        ),
        encoding="utf-8",
    )
    malformed_run_info = "summary: [truncated"
    run_info_path = tmp_path / "run_info.yaml"
    if metadata_kind == "malformed-file":
        run_info_path.write_text(
            malformed_run_info,
            encoding="utf-8",
        )
    elif metadata_kind == "unreadable-directory":
        run_info_path.mkdir()
        (run_info_path / "preserved.txt").write_text(
            "non-file metadata evidence",
            encoding="utf-8",
        )
    resume_info = ResumeInfo(
        resume_from_checkpoint="",
        completed_checkpoints=[checkpoint.name],
        last_snapshot_dir=checkpoint_dir / "snapshot",
        prior_usage=usage,
        run_info_reconciliation_required=True,
    )
    config = cast(
        "RunTaskConfig",
        SimpleNamespace(
            one_shot=False,
            resume=True,
            prompt_template="{{ spec }}",
            env_spec=environment,
            disable_evaluation=True,
            pass_policy=PassPolicy.ANY_CASE,
            seed=42,
            concurrent_evaluation=False,
            verbosity=0,
            image="unused",
            agent_config=SimpleNamespace(
                type="codex",
                version="0.124.0",
            ),
            model_def=SimpleNamespace(name="gpt-5.5"),
        ),
    )

    with (
        patch(
            "slop_code.entrypoints.problem_runner.worker.apply_one_shot_mode",
            return_value=problem,
        ),
        patch(
            "slop_code.entrypoints.problem_runner.worker.detect_resume_point",
            return_value=resume_info,
        ),
        patch(
            "slop_code.entrypoints.problem_runner.worker.Agent.from_config"
        ) as agent_factory,
        patch(
            "slop_code.entrypoints.problem_runner.worker.runner.run_agent"
        ) as run_agent,
    ):
        result = run_agent_on_problem(
            problem,
            problem.name,
            config,
            queue.Queue(),
            tmp_path,
        )

    agent_factory.assert_not_called()
    run_agent.assert_not_called()
    assert result["summary"]["state"] == "completed"
    assert result["summary"]["total_cost"] == 3.5

    persisted = yaml.safe_load((tmp_path / "run_info.yaml").read_text())
    assert persisted["summary"]["state"] == "completed"
    assert persisted["summary"]["passed_policy"] is True
    assert persisted["summary"]["total_cost"] == 3.5
    assert persisted["summary"]["checkpoints"] == {checkpoint.name: "ran"}
    backups = list(tmp_path.glob("run_info.corrupt-*.yaml"))
    if metadata_kind == "missing":
        assert backups == []
    elif metadata_kind == "malformed-file":
        assert len(backups) == 1
        assert backups[0].read_text(encoding="utf-8") == malformed_run_info
    else:
        assert len(backups) == 1
        assert backups[0].is_dir()
        assert (backups[0] / "preserved.txt").read_text(encoding="utf-8") == (
            "non-file metadata evidence"
        )
