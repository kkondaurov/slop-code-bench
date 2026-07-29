from __future__ import annotations

import json
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock
from unittest.mock import Mock
from unittest.mock import call
from unittest.mock import patch
from unittest.mock import sentinel

import pytest
import yaml

from slop_code.agent_runner import runner
from slop_code.agent_runner.agent import RETRY_PROMPT
from slop_code.agent_runner.agent import Agent
from slop_code.agent_runner.models import AgentCostLimits
from slop_code.agent_runner.models import AgentError
from slop_code.agent_runner.models import UsageTracker
from slop_code.agent_runner.resume import ResumeInfo
from slop_code.agent_runner.resume import detect_resume_point
from slop_code.common import INFERENCE_RESULT_FILENAME
from slop_code.common import PROMPT_FILENAME
from slop_code.common.atomic import UnsafeAtomicWriteError
from slop_code.common.llms import TokenUsage
from slop_code.evaluation.report import CorrectnessResults
from slop_code.evaluation.report import GroupType
from slop_code.evaluation.report import PassPolicy
from slop_code.execution import DockerEnvironmentSpec


class StubCheckpoint:
    def __init__(self, name: str, spec_text: str) -> None:
        self.name = name
        self._spec_text = spec_text

    def get_spec_text(self) -> str:
        return self._spec_text


@dataclass(frozen=True)
class ReplayCase:
    path_exists: bool
    is_file: bool
    supports: bool
    expected: bool


def _usage(cost: float = 0.0, steps: int = 0) -> UsageTracker:
    return UsageTracker(
        cost=cost,
        steps=steps,
        net_tokens=TokenUsage(),
        current_tokens=TokenUsage(),
    )


def _inference_result(cost: float, steps: int) -> dict[str, object]:
    return {
        "started": "2026-01-01T00:00:00",
        "completed": "2026-01-01T00:00:00",
        "elapsed": 0.0,
        "usage": _usage(cost=cost, steps=steps).model_dump(),
        "had_error": False,
    }


def _passing_report(checkpoint_name: str) -> CorrectnessResults:
    return CorrectnessResults(
        problem_name="prob",
        problem_version=1,
        checkpoint_name=checkpoint_name,
        checkpoint_version=1,
        duration=0.01,
        entrypoint="python main.py",
        pass_counts={GroupType.CORE: 1},
        total_counts={GroupType.CORE: 1},
        pytest_exit_code=0,
        pytest_collected=1,
    )


def test_evaluate_snapshot_saves_infrastructure_report_then_raises(
    tmp_path: Path,
) -> None:
    report = MagicMock()
    report.infrastructure_failure = True
    report.pytest_exit_code = 3
    report.pytest_collected = 0

    with (
        patch(
            "slop_code.agent_runner.runner.evaluate_checkpoint",
            return_value=report,
        ),
        patch(
            "slop_code.agent_runner.runner.measure_snapshot_quality"
        ) as measure_quality,
        pytest.raises(runner.EvaluationError, match="infrastructure failure"),
    ):
        runner.evaluate_agent_snapshot(
            checkpoint=SimpleNamespace(name="checkpoint_1"),
            save_dir=tmp_path / "checkpoint_1",
            snapshot_dir=tmp_path / "snapshot",
            problem=MagicMock(),
            environment=MagicMock(),
        )

    report.save.assert_called_once_with(tmp_path / "checkpoint_1")
    measure_quality.assert_not_called()


def test_terminal_progress_does_not_double_count_final_checkpoint() -> None:
    agent = Mock(spec=Agent)
    agent.usage = _usage(cost=2.5, steps=7)
    metrics = Mock()
    metrics.state = runner.AgentStateEnum.COMPLETED
    metrics.model_copy.return_value = sentinel.metrics_snapshot
    progress_queue = queue.Queue()

    runner.agent_progress_watcher(
        agent,
        metrics,
        progress_queue,
        "prob",
    )

    problem_name, active_usage, metrics_snapshot = progress_queue.get_nowait()
    assert problem_name == "prob"
    assert active_usage.cost == 0.0
    assert active_usage.steps == 0
    assert metrics_snapshot is sentinel.metrics_snapshot


class RetryProbeAgent(Agent):
    def __init__(
        self,
        *,
        max_retries: int,
        errors: list[Exception],
    ) -> None:
        super().__init__(
            agent_name="retry_probe",
            problem_name="prob",
            cost_limits=AgentCostLimits(
                step_limit=0,
                cost_limit=0.0,
                net_cost_limit=0.0,
                max_retries=max_retries,
            ),
            pricing=None,
            verbose=False,
        )
        self.errors = errors
        self.run_tasks: list[str] = []
        self.retry_count = 0

    @classmethod
    def _from_config(cls, *args: object, **kwargs: object) -> Agent:
        raise NotImplementedError

    def setup(self, session: object) -> None:
        _ = session

    def run(self, task: str) -> None:
        self.run_tasks.append(task)
        if self.errors:
            raise self.errors.pop(0)

    def retry(self) -> None:
        self.retry_count += 1
        self.run(RETRY_PROMPT)

    def reset(self) -> None:
        pass

    def save_artifacts(self, path: Path) -> None:
        _ = path

    def cleanup(self) -> None:
        pass


class CapturingLogger:
    def __init__(self) -> None:
        self.errors: list[tuple[str, dict[str, object]]] = []

    def error(self, event: str, **kwargs: object) -> None:
        self.errors.append((event, kwargs))


def test_run_checkpoint_retries_agent_errors_with_continue_prompt() -> None:
    agent = RetryProbeAgent(
        max_retries=1,
        errors=[AgentError("transient")],
    )

    result = agent.run_checkpoint("full checkpoint prompt")

    assert result.had_error is False
    assert agent.retry_count == 1
    assert agent.run_tasks == ["full checkpoint prompt", RETRY_PROMPT]


def test_run_checkpoint_stops_after_retry_budget() -> None:
    agent = RetryProbeAgent(
        max_retries=1,
        errors=[AgentError("first"), AgentError("second")],
    )

    result = agent.run_checkpoint("full checkpoint prompt")

    assert result.had_error is True
    assert "second" in (result.error_message or "")
    assert agent.retry_count == 1
    assert agent.run_tasks == ["full checkpoint prompt", RETRY_PROMPT]


def test_run_checkpoint_does_not_retry_non_agent_errors() -> None:
    agent = RetryProbeAgent(
        max_retries=3,
        errors=[RuntimeError("programming error")],
    )

    result = agent.run_checkpoint("full checkpoint prompt")

    assert result.had_error is True
    assert "programming error" in (result.error_message or "")
    assert agent.retry_count == 0
    assert agent.run_tasks == ["full checkpoint prompt"]


def test_run_checkpoint_logs_exact_agent_error_message() -> None:
    agent = RetryProbeAgent(
        max_retries=0,
        errors=[AgentError("provider said: rate limit exceeded")],
    )
    logger = CapturingLogger()
    agent.log = logger  # type: ignore[assignment]

    agent.run_checkpoint("full checkpoint prompt")

    assert logger.errors
    _, kwargs = logger.errors[0]
    assert kwargs["error_message"] == "provider said: rate limit exceeded"


def test_agent_cost_limits_default_to_two_retries() -> None:
    limits = AgentCostLimits(
        step_limit=0,
        cost_limit=0.0,
        net_cost_limit=0.0,
    )

    assert limits.max_retries == 2


@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt])
def test_inference_error_survives_checkpoint_finalization_failure(
    tmp_path: Path,
    error_type: type[BaseException],
) -> None:
    primary = error_type("inference exploded")
    finalization_error = RuntimeError("snapshot cleanup exploded")
    agent = MagicMock(spec=Agent)
    agent.usage = _usage(cost=1.0, steps=2)
    session = MagicMock()
    session.finish_checkpoint.side_effect = finalization_error

    with (
        patch(
            "slop_code.agent_runner.runner._run_checkpoint_task",
            side_effect=primary,
        ),
        pytest.raises(error_type) as caught,
    ):
        runner._run_inference(
            checkpoint_name="checkpoint_1",
            session=session,
            agent=agent,
            task="solve it",
            save_dir=tmp_path,
        )

    assert caught.value is primary
    assert any(
        "snapshot cleanup exploded" in note
        for note in getattr(primary, "__notes__", [])
    )


@pytest.mark.parametrize(
    ("filename", "writer"),
    [
        (
            runner.common.EVALUATION_ERROR_FILENAME,
            lambda checkpoint_dir: runner._save_evaluation_error(
                "checkpoint_1",
                checkpoint_dir,
                RuntimeError("eval failed"),
                "traceback",
            ),
        ),
        (
            runner.common.RESUME_ERROR_FILENAME,
            lambda checkpoint_dir: runner._save_resume_error(
                "checkpoint_1",
                checkpoint_dir,
                "restore",
                SimpleNamespace(
                    exit_code=17,
                    stdout="",
                    stderr="failed",
                ),
            ),
        ),
    ],
)
def test_checkpoint_error_writer_rejects_symlink_target(
    tmp_path: Path,
    filename: str,
    writer: Callable[[Path], object],
) -> None:
    checkpoint_dir = tmp_path / "checkpoint_1"
    checkpoint_dir.mkdir()
    outside = tmp_path / "outside"
    outside.write_text("untouched", encoding="utf-8")
    (checkpoint_dir / filename).symlink_to(outside)

    with pytest.raises(UnsafeAtomicWriteError, match="symlink target"):
        writer(checkpoint_dir)

    assert outside.read_text(encoding="utf-8") == "untouched"


@pytest.mark.parametrize(
    "case",
    [
        ReplayCase(
            path_exists=False, is_file=False, supports=True, expected=False
        ),
        ReplayCase(
            path_exists=True, is_file=False, supports=True, expected=False
        ),
        ReplayCase(
            path_exists=True, is_file=True, supports=False, expected=False
        ),
        ReplayCase(
            path_exists=True, is_file=True, supports=True, expected=True
        ),
    ],
)
def test_should_run_replay(
    tmp_path: Path,
    case: ReplayCase,
) -> None:
    agent = Mock(spec=Agent)
    agent.supports_replay.return_value = case.supports

    replay_path = tmp_path / "replay.json"
    if case.path_exists:
        replay_path.write_text("{}")
        if not case.is_file:
            replay_path.unlink()
            replay_path.mkdir()

    assert (
        runner._should_run_replay(
            replay_path if case.path_exists else None, agent
        )
        is case.expected
    )

    if case.path_exists:
        assert agent.supports_replay.call_count >= 1
    else:
        assert agent.supports_replay.call_count == 0


def test_run_checkpoint_task_uses_replay_when_available(
    tmp_path: Path,
) -> None:
    agent = Mock(spec=Agent)
    replay_result = sentinel.replay_result
    agent.run_replay.return_value = replay_result
    replay_path = tmp_path / "replay.json"

    with patch(
        "slop_code.agent_runner.runner._should_run_replay",
        return_value=True,
    ):
        result = runner._run_checkpoint_task(
            agent=agent,
            task="ignored",
            checkpoint_name="ckpt",
            replay_path=replay_path,
        )

    assert result is replay_result
    agent.run_replay.assert_called_once()
    agent.run_checkpoint.assert_not_called()


def test_run_checkpoint_task_runs_inference_when_no_replay() -> None:
    agent = Mock(spec=Agent)
    inference_result = sentinel.inference_result
    agent.run_checkpoint.return_value = inference_result

    with patch(
        "slop_code.agent_runner.runner._should_run_replay",
        return_value=False,
    ):
        result = runner._run_checkpoint_task(
            agent=agent,
            task="solve this",
            checkpoint_name="ckpt",
            replay_path=None,
        )

    assert result is inference_result
    agent.run_checkpoint.assert_called_once_with("solve this")
    agent.run_replay.assert_not_called()


def test_get_task_for_checkpoint_renders_prompt_and_writes_file(
    tmp_path: Path,
) -> None:
    spec_text = "Start with %%%ENTRYPOINT:entry_file%%% and run %%%ENTRYPOINT:entry_command%%%"
    environment = Mock()
    environment.format_entry_file.return_value = "formatted/main.py"
    environment.get_command.return_value = "uv run formatted/main.py"

    prompt = runner.get_task_for_checkpoint(
        checkpoint_name="checkpoint_1",
        spec_text=spec_text,
        template="{{ 'CONT' if is_continuation else 'START' }} :: {{ spec }}",
        entry_file="main.py",
        environment=environment,
        is_first_checkpoint=True,
        output_path=tmp_path,
    )

    expected_text = (
        "START :: Start with formatted/main.py and run uv run formatted/main.py"
    )
    assert prompt == expected_text

    written = (tmp_path / PROMPT_FILENAME).read_text()
    assert written == expected_text

    environment.format_entry_file.assert_called_once_with("main.py")
    environment.get_command.assert_called_once_with(
        "main.py", is_agent_run=True
    )


def test_get_task_for_checkpoint_includes_agent_info_in_context(
    tmp_path: Path,
) -> None:
    """Test that agent_type, agent_version, and model_name are available in templates."""
    spec_text = "Test spec"
    environment = Mock()
    environment.format_entry_file.return_value = "main.py"
    environment.get_command.return_value = "python main.py"

    template = (
        "Agent: {{ agent_type }} v{{ agent_version }} | "
        "Model: {{ model_name }} | "
        "{{ spec }}"
    )

    prompt = runner.get_task_for_checkpoint(
        checkpoint_name="checkpoint_1",
        spec_text=spec_text,
        template=template,
        entry_file="main.py",
        environment=environment,
        is_first_checkpoint=True,
        output_path=tmp_path,
        agent_type="claude_code",
        agent_version="2.0.51",
        model_name="opus-4.5",
    )

    expected = "Agent: claude_code v2.0.51 | Model: opus-4.5 | Test spec"
    assert prompt == expected


def test_get_task_for_checkpoint_handles_none_agent_version(
    tmp_path: Path,
) -> None:
    """Test that None agent_version renders as empty string."""
    environment = Mock()
    environment.format_entry_file.return_value = "main.py"
    environment.get_command.return_value = "python main.py"

    template = "Agent: {{ agent_type }}{% if agent_version %}-{{ agent_version }}{% endif %}"

    prompt = runner.get_task_for_checkpoint(
        checkpoint_name="checkpoint_1",
        spec_text="spec",
        template=template,
        entry_file="main.py",
        environment=environment,
        is_first_checkpoint=True,
        output_path=tmp_path,
        agent_type="gemini",
        agent_version=None,
        model_name="gemini-2.0",
    )

    # agent_version is converted to "" when None, so conditional is false
    assert prompt == "Agent: gemini"


@pytest.mark.parametrize("base_dir", [None, Path("/prior/snapshot")])
def test_create_agent_session_uses_static_assets_and_environment(
    base_dir: Path | None,
) -> None:
    problem_config = Mock()
    problem_config.path = Path("/problem")
    problem_config.static_assets = {"foo": "bar"}
    environment_spec = Mock()

    resolved_assets = {"foo": sentinel.asset}
    with (
        patch(
            "slop_code.agent_runner.runner.resolve_static_assets",
            return_value=resolved_assets,
        ) as resolve_assets,
        patch(
            "slop_code.agent_runner.runner.Session.from_environment_spec",
            return_value=sentinel.session,
        ) as from_env_spec,
    ):
        session = runner.create_agent_session(
            problem_config,
            environment_spec,
            base_dir=base_dir,
        )

    assert session is sentinel.session
    resolve_assets.assert_called_once_with(
        base_path=problem_config.path,
        assets=problem_config.static_assets,
    )
    from_env_spec.assert_called_once_with(
        spec=environment_spec,
        base_dir=base_dir,
        static_assets=resolved_assets,
        is_agent_infer=True,
    )


def test_run_problem_uses_fresh_session_from_prior_snapshot(
    tmp_path: Path,
) -> None:
    """Each checkpoint gets a new session before hidden-test evaluation."""
    agent = MagicMock(spec=Agent)
    agent.usage = UsageTracker()
    agent.hit_net_rate_limit.return_value = False

    environment = Mock()
    environment.get_resume_commands.return_value = []
    problem = Mock()
    problem.name = "prob"
    problem.checkpoints = {"checkpoint_1": Mock(), "checkpoint_2": Mock()}

    run_spec = Mock()
    run_spec.problem = problem
    run_spec.environment = environment
    run_spec.template = "{{ spec }}"
    run_spec.compress_artifacts = False
    run_spec.skip_evaluation = False
    run_spec.concurrent_evaluation = False
    run_spec.pass_policy = PassPolicy.ANY_CASE
    run_spec.agent_type = "codex"
    run_spec.agent_version = "0.124.0"
    run_spec.model_name = "gpt-5.5"

    checkpoints = [
        (StubCheckpoint("checkpoint_1", "one"), tmp_path / "checkpoint_1"),
        (StubCheckpoint("checkpoint_2", "two"), tmp_path / "checkpoint_2"),
    ]
    sessions = [MagicMock(name="session_1"), MagicMock(name="session_2")]
    lifecycle: list[str] = []
    for index, session in enumerate(sessions, start=1):
        session.__enter__.return_value = session
        session.__exit__.side_effect = (
            lambda *_args, session_index=index: lifecycle.append(
                f"exit:{session_index}"
            )
        )

    session_indices = {
        id(session): index for index, session in enumerate(sessions, start=1)
    }
    agent.setup.side_effect = lambda *, session: lifecycle.append(
        f"setup:{session_indices[id(session)]}"
    )
    agent.cleanup.side_effect = lambda: lifecycle.append("cleanup")
    agent.finish_checkpoint.side_effect = (
        lambda *, reset_context: lifecycle.append(f"reset:{reset_context}")
    )

    inference_result = Mock(had_error=False)
    report = SimpleNamespace(pass_counts={}, total_counts={})
    evaluated: list[str] = []

    def run_in_session(**kwargs):
        save_dir = kwargs["save_dir"]
        return save_dir / "snapshot", inference_result, sentinel.diff

    def evaluate_after_cleanup(**kwargs):
        index = len(evaluated)
        assert sessions[index].__exit__.called
        assert agent.cleanup.call_count == index + 1
        checkpoint_name = kwargs["checkpoint"].name
        evaluated.append(checkpoint_name)
        lifecycle.append(f"eval:{checkpoint_name}")
        return report, sentinel.quality

    agent_runner = runner.AgentRunner(
        run_spec=run_spec,
        agent=agent,
        output_path=tmp_path,
        progress_queue=queue.Queue(),
    )

    with (
        patch(
            "slop_code.agent_runner.runner.get_checkpoints",
            return_value=iter(checkpoints),
        ),
        patch(
            "slop_code.agent_runner.runner.create_agent_session",
            side_effect=sessions,
        ) as create_session,
        patch(
            "slop_code.agent_runner.runner.run_checkpoint",
            side_effect=run_in_session,
        ),
        patch(
            "slop_code.agent_runner.runner.reporting.save_agent_checkpoint_info"
        ),
        patch(
            "slop_code.agent_runner.runner.evaluate_agent_snapshot",
            side_effect=evaluate_after_cleanup,
        ),
        patch.object(
            agent_runner,
            "_run_resume_commands",
            wraps=agent_runner._run_resume_commands,
        ) as resume_commands,
    ):
        results = agent_runner._run_problem()

    first_snapshot = tmp_path / "checkpoint_1" / "snapshot"
    assert [result.snapshot_dir for result in results] == [
        first_snapshot,
        tmp_path / "checkpoint_2" / "snapshot",
    ]
    assert create_session.call_args_list == [
        call(
            problem_config=problem,
            environment_spec=environment,
            base_dir=None,
        ),
        call(
            problem_config=problem,
            environment_spec=environment,
            base_dir=first_snapshot,
        ),
    ]
    assert agent.setup.call_args_list == [
        call(session=sessions[0]),
        call(session=sessions[1]),
    ]
    agent.finish_checkpoint.assert_called_once_with(reset_context=True)
    resume_commands.assert_called_once_with(
        "checkpoint_2",
        tmp_path / "checkpoint_2",
    )
    assert evaluated == ["checkpoint_1", "checkpoint_2"]
    assert lifecycle == [
        "setup:1",
        "cleanup",
        "exit:1",
        "eval:checkpoint_1",
        "reset:True",
        "setup:2",
        "cleanup",
        "exit:2",
        "eval:checkpoint_2",
    ]


def test_resume_commands_run_as_non_root_docker_user(tmp_path: Path) -> None:
    """Restored dependencies remain writable by the inference agent."""
    environment = DockerEnvironmentSpec.model_validate(
        {
            "type": "docker",
            "name": "test",
            "docker": {"image": "python:3.12"},
            "setup": {"resume_commands": ["python -m venv .venv"]},
        }
    )
    agent_runner = runner.AgentRunner(
        run_spec=Mock(environment=environment),
        agent=MagicMock(spec=Agent),
        output_path=tmp_path,
        progress_queue=queue.Queue(),
    )
    session = MagicMock()
    runtime = session.exec.return_value
    runtime.execute.return_value = SimpleNamespace(exit_code=0, stderr="")
    agent_runner._session = session

    agent_runner._run_resume_commands("checkpoint_2", tmp_path / "checkpoint_2")

    session.exec.assert_called_once_with(
        command="python -m venv .venv",
        disable_setup=True,
        user="1000:1000",
    )
    runtime.cleanup.assert_called_once_with()


def test_resume_command_failure_is_durable_and_fails_closed(
    tmp_path: Path,
) -> None:
    environment = DockerEnvironmentSpec.model_validate(
        {
            "type": "docker",
            "name": "test",
            "docker": {"image": "python:3.12"},
            "setup": {"resume_commands": ["install-dependencies"]},
        }
    )
    agent_runner = runner.AgentRunner(
        run_spec=Mock(environment=environment),
        agent=MagicMock(spec=Agent),
        output_path=tmp_path,
        progress_queue=queue.Queue(),
    )
    session = MagicMock()
    runtime = session.exec.return_value
    runtime.execute.return_value = SimpleNamespace(
        exit_code=17,
        stdout="install output",
        stderr="dependency unavailable",
    )
    agent_runner._session = session
    checkpoint_dir = tmp_path / "checkpoint_2"

    with pytest.raises(runner.ResumeCommandError, match="exit code 17"):
        agent_runner._run_resume_commands("checkpoint_2", checkpoint_dir)

    payload = json.loads(
        (checkpoint_dir / runner.common.RESUME_ERROR_FILENAME).read_text()
    )
    assert payload["checkpoint"] == "checkpoint_2"
    assert payload["command"] == "install-dependencies"
    assert payload["exit_code"] == 17
    assert payload["stdout"] == "install output"
    assert payload["stderr"] == "dependency unavailable"
    runtime.cleanup.assert_called_once_with()


def test_resume_cleanup_failure_does_not_mask_command_failure(
    tmp_path: Path,
) -> None:
    environment = DockerEnvironmentSpec.model_validate(
        {
            "type": "docker",
            "name": "test",
            "docker": {"image": "python:3.12"},
            "setup": {"resume_commands": ["install-dependencies"]},
        }
    )
    agent_runner = runner.AgentRunner(
        run_spec=Mock(environment=environment),
        agent=MagicMock(spec=Agent),
        output_path=tmp_path,
        progress_queue=queue.Queue(),
    )
    runtime = MagicMock()
    runtime.execute.return_value = SimpleNamespace(
        exit_code=17,
        stdout="",
        stderr="dependency unavailable",
    )
    runtime.cleanup.side_effect = RuntimeError("cleanup goblin")
    session = MagicMock()
    session.exec.return_value = runtime
    agent_runner._session = session

    with pytest.raises(runner.ResumeCommandError) as caught:
        agent_runner._run_resume_commands(
            "checkpoint_2",
            tmp_path / "checkpoint_2",
        )

    assert "exit code 17" in str(caught.value)
    assert any(
        "cleanup goblin" in note
        for note in getattr(caught.value, "__notes__", [])
    )


@pytest.mark.parametrize("control_flow_type", [KeyboardInterrupt, SystemExit])
def test_resume_cleanup_never_swallows_control_flow(
    tmp_path: Path,
    control_flow_type: type[BaseException],
) -> None:
    environment = DockerEnvironmentSpec.model_validate(
        {
            "type": "docker",
            "name": "test",
            "docker": {"image": "python:3.12"},
            "setup": {"resume_commands": ["install-dependencies"]},
        }
    )
    agent_runner = runner.AgentRunner(
        run_spec=Mock(environment=environment),
        agent=MagicMock(spec=Agent),
        output_path=tmp_path,
        progress_queue=queue.Queue(),
    )
    runtime = MagicMock()
    runtime.execute.return_value = SimpleNamespace(
        exit_code=17,
        stdout="",
        stderr="dependency unavailable",
    )
    cancellation = control_flow_type("stop cleanup")
    runtime.cleanup.side_effect = cancellation
    session = MagicMock()
    session.exec.return_value = runtime
    agent_runner._session = session

    with pytest.raises(control_flow_type) as caught:
        agent_runner._run_resume_commands(
            "checkpoint_2",
            tmp_path / "checkpoint_2",
        )

    assert caught.value is cancellation
    assert any(
        "Earlier resume command failure" in note
        for note in getattr(cancellation, "__notes__", [])
    )


def test_concurrent_eval_artifact_failure_preserves_primary_error(
    tmp_path: Path,
) -> None:
    run_spec = Mock(problem=Mock(), environment=Mock())
    agent_runner = runner.AgentRunner(
        run_spec=run_spec,
        agent=MagicMock(spec=Agent),
        output_path=tmp_path,
        progress_queue=queue.Queue(),
    )
    summary = runner.AgentCheckpointSummary(
        checkpoint_name="checkpoint_1",
        path=tmp_path / "checkpoint_1",
        snapshot_dir=tmp_path / "checkpoint_1" / "snapshot",
        artifacts=tmp_path / "checkpoint_1" / "agent",
        usage=_usage(),
        passed_policy=None,
        had_error=False,
    )
    primary = RuntimeError("evaluator exploded")

    with (
        patch(
            "slop_code.agent_runner.runner.evaluate_agent_snapshot",
            side_effect=primary,
        ),
        patch(
            "slop_code.agent_runner.runner._save_evaluation_error",
            side_effect=OSError("artifact disk full"),
        ),
    ):
        agent_runner._eval_one(
            SimpleNamespace(name="checkpoint_1"),
            summary,
        )

    assert agent_runner._failed_evals["checkpoint_1"] == (
        "RuntimeError: evaluator exploded"
    )
    assert any(
        "artifact disk full" in note
        for note in getattr(primary, "__notes__", [])
    )


@pytest.mark.parametrize("control_flow_type", [KeyboardInterrupt, SystemExit])
def test_evaluation_error_persistence_never_swallows_control_flow(
    tmp_path: Path,
    control_flow_type: type[BaseException],
) -> None:
    checkpoint_dir = tmp_path / "checkpoint_1"
    checkpoint_dir.mkdir()
    agent_runner = runner.AgentRunner(
        run_spec=Mock(problem=Mock(), environment=Mock()),
        agent=MagicMock(spec=Agent),
        output_path=tmp_path,
        progress_queue=queue.Queue(),
    )
    summary = runner.AgentCheckpointSummary(
        checkpoint_name="checkpoint_1",
        path=checkpoint_dir,
        snapshot_dir=checkpoint_dir / "snapshot",
        artifacts=checkpoint_dir / "agent",
        usage=_usage(),
        had_error=False,
    )
    cancellation = control_flow_type("stop evidence write")

    with (
        patch(
            "slop_code.agent_runner.runner.evaluate_agent_snapshot",
            side_effect=RuntimeError("evaluator exploded"),
        ),
        patch(
            "slop_code.agent_runner.runner._save_evaluation_error",
            side_effect=cancellation,
        ),
        pytest.raises(control_flow_type) as caught,
    ):
        agent_runner._eval_one(
            SimpleNamespace(name="checkpoint_1"),
            summary,
            preserve_cancellation=True,
        )

    assert caught.value is cancellation
    assert agent_runner._failed_evals_snapshot() == {}
    assert any(
        "Earlier evaluation error artifact failure" in note
        for note in getattr(cancellation, "__notes__", [])
    )


def test_synchronous_evaluation_repair_preserves_cancellation(
    tmp_path: Path,
) -> None:
    checkpoint_dir = tmp_path / "checkpoint_1"
    checkpoint_dir.mkdir()
    agent_runner = runner.AgentRunner(
        run_spec=Mock(
            problem=Mock(),
            environment=Mock(),
            pass_policy=PassPolicy.ANY_CASE,
        ),
        agent=MagicMock(spec=Agent),
        output_path=tmp_path,
        progress_queue=queue.Queue(),
    )
    summary = runner.AgentCheckpointSummary(
        checkpoint_name="checkpoint_1",
        path=checkpoint_dir,
        snapshot_dir=checkpoint_dir / "snapshot",
        artifacts=checkpoint_dir / "agent",
        usage=_usage(),
        had_error=False,
    )
    cancellation = KeyboardInterrupt("stop repair")

    with (
        patch(
            "slop_code.agent_runner.runner.evaluate_agent_snapshot",
            side_effect=cancellation,
        ),
        pytest.raises(KeyboardInterrupt, match="stop repair") as caught,
    ):
        agent_runner._repair_checkpoint_evaluation(
            SimpleNamespace(name="checkpoint_1"),
            summary,
        )

    assert caught.value is cancellation
    assert agent_runner._failed_evals_snapshot() == {}
    assert not (
        checkpoint_dir / runner.common.EVALUATION_ERROR_FILENAME
    ).exists()


def test_failed_eval_reads_wait_for_metrics_lock(tmp_path: Path) -> None:
    agent_runner = runner.AgentRunner(
        run_spec=Mock(),
        agent=MagicMock(spec=Agent),
        output_path=tmp_path,
        progress_queue=queue.Queue(),
    )
    agent_runner._failed_evals["checkpoint_1"] = "failed"  # noqa: SLF001
    entered = threading.Event()
    finished = threading.Event()

    def read_failures() -> None:
        entered.set()
        agent_runner._failed_evals_snapshot()
        finished.set()

    agent_runner._metrics_lock.acquire()  # noqa: SLF001
    reader = threading.Thread(target=read_failures)
    try:
        reader.start()
        assert entered.wait(timeout=1)
        assert not finished.wait(timeout=0.05)
    finally:
        agent_runner._metrics_lock.release()  # noqa: SLF001
        reader.join(timeout=1)

    assert finished.is_set()


@pytest.mark.parametrize("failure_phase", ("clear", "metrics"))
def test_concurrent_eval_guards_post_evaluation_failures(
    tmp_path: Path,
    failure_phase: str,
) -> None:
    checkpoint_dir = tmp_path / "checkpoint_1"
    checkpoint_dir.mkdir()
    agent_runner = runner.AgentRunner(
        run_spec=Mock(problem=Mock(), environment=Mock()),
        agent=MagicMock(spec=Agent),
        output_path=tmp_path,
        progress_queue=queue.Queue(),
    )
    summary = runner.AgentCheckpointSummary(
        checkpoint_name="checkpoint_1",
        path=checkpoint_dir,
        snapshot_dir=checkpoint_dir / "snapshot",
        artifacts=checkpoint_dir / "agent",
        usage=_usage(),
        had_error=False,
    )
    report = Mock()
    patches = [
        patch(
            "slop_code.agent_runner.runner.evaluate_agent_snapshot",
            return_value=(report, None),
        )
    ]
    if failure_phase == "clear":
        patches.append(
            patch(
                "slop_code.agent_runner.runner._clear_evaluation_error",
                side_effect=OSError("cannot clear stale marker"),
            )
        )
    else:
        patches.append(
            patch.object(
                runner.MetricsTracker,
                "record_checkpoint_result",
                side_effect=RuntimeError("metrics mutation failed"),
            )
        )

    with patches[0], patches[1]:
        agent_runner._eval_one(
            SimpleNamespace(name="checkpoint_1"),
            summary,
        )

    assert "checkpoint_1" in agent_runner._failed_evals
    assert "checkpoint_1" not in agent_runner._eval_reports
    error_record = json.loads(
        (checkpoint_dir / runner.common.EVALUATION_ERROR_FILENAME).read_text()
    )
    assert error_record["error_type"] in {"OSError", "RuntimeError"}


def test_cleanup_checkpoint_session_retains_session_when_exit_raises(
    tmp_path: Path,
) -> None:
    agent = MagicMock(spec=Agent)
    agent_runner = runner.AgentRunner(
        run_spec=Mock(),
        agent=agent,
        output_path=tmp_path,
        progress_queue=queue.Queue(),
    )
    session = MagicMock()
    session.__exit__.side_effect = RuntimeError("session cleanup failed")
    agent_runner._session = session

    with pytest.raises(RuntimeError, match="session cleanup failed"):
        agent_runner._cleanup_checkpoint_session()

    agent.cleanup.assert_called_once_with()
    session.__exit__.assert_called_once_with(None, None, None)
    assert agent_runner._session is session


def test_cleanup_checkpoint_session_exits_session_when_agent_cleanup_raises(
    tmp_path: Path,
) -> None:
    agent = MagicMock(spec=Agent)
    agent.cleanup.side_effect = RuntimeError("agent cleanup failed")
    agent_runner = runner.AgentRunner(
        run_spec=Mock(),
        agent=agent,
        output_path=tmp_path,
        progress_queue=queue.Queue(),
    )
    session = MagicMock()
    agent_runner._session = session

    with pytest.raises(RuntimeError, match="agent cleanup failed"):
        agent_runner._cleanup_checkpoint_session()

    session.__exit__.assert_called_once_with(None, None, None)
    assert agent_runner._session is None


def test_managed_session_runtime_is_cleaned_exactly_once(
    tmp_path: Path,
) -> None:
    """Agent and Session must not both close the same spawned runtime."""
    runtime = Mock()
    agent = MagicMock(spec=Agent)
    agent._runtime = runtime  # noqa: SLF001

    def conventional_agent_cleanup() -> None:
        if agent._runtime is not None:  # noqa: SLF001
            agent._runtime.cleanup()  # noqa: SLF001

    agent.cleanup.side_effect = conventional_agent_cleanup
    agent_runner = runner.AgentRunner(
        run_spec=Mock(),
        agent=agent,
        output_path=tmp_path,
        progress_queue=queue.Queue(),
    )
    session = MagicMock()
    session.__exit__.side_effect = lambda *_args: runtime.cleanup()
    agent_runner._session = session

    agent_runner._cleanup_checkpoint_session()

    agent.cleanup.assert_called_once_with()
    runtime.cleanup.assert_called_once_with()
    assert agent._runtime is None  # noqa: SLF001


def test_checkpoint_primary_error_survives_cleanup_and_commits_usage_once(
    tmp_path: Path,
) -> None:
    agent = MagicMock(spec=Agent)
    agent.usage = _usage(cost=2.5, steps=7)
    agent.cleanup.side_effect = RuntimeError("agent cleanup failed")
    run_spec = Mock(
        problem=Mock(
            name="prob",
            checkpoints={"checkpoint_1": Mock()},
        ),
        environment=Mock(),
        template="{{ spec }}",
        compress_artifacts=False,
        skip_evaluation=True,
        concurrent_evaluation=False,
        pass_policy=PassPolicy.ANY_CASE,
        agent_type="codex",
        agent_version="0.124.0",
        model_name="gpt-5.5",
    )
    agent_runner = runner.AgentRunner(
        run_spec=run_spec,
        agent=agent,
        output_path=tmp_path,
        progress_queue=queue.Queue(),
    )
    checkpoint = StubCheckpoint("checkpoint_1", "")
    session = MagicMock()

    def attach_session(*_args: object, **_kwargs: object) -> None:
        agent_runner._session = session

    with (
        patch(
            "slop_code.agent_runner.runner.get_checkpoints",
            return_value=iter([(checkpoint, tmp_path / checkpoint.name)]),
        ),
        patch.object(
            agent_runner,
            "_setup_for_checkpoint",
            side_effect=attach_session,
        ),
        patch(
            "slop_code.agent_runner.runner.run_checkpoint",
            side_effect=ValueError("solve exploded"),
        ),
        patch(
            "slop_code.agent_runner.runner._save_agent_artifacts_after_checkpoint_error"
        ),
        patch.object(agent_runner, "setup"),
        patch(
            "slop_code.agent_runner.runner.reporting.save_results",
            return_value={"summary": {"passed_policy": False}},
        ) as save_results,
        pytest.raises(ValueError, match="solve exploded") as exc_info,
    ):
        agent_runner.run()

    assert any(
        "Secondary checkpoint cleanup failure" in note
        and "agent cleanup failed" in note
        for note in (exc_info.value.__notes__ or [])
    )
    session.__exit__.assert_called_once_with(None, None, None)
    assert agent_runner.metrics_tracker.usage.cost == 2.5
    assert agent_runner.metrics_tracker.usage.steps == 7
    saved_metrics = save_results.call_args.args[1]
    assert saved_metrics.usage.cost == 2.5
    assert saved_metrics.usage.steps == 7

    # A later safety-net call cannot double-count the failed checkpoint.
    agent_runner._commit_checkpoint_usage()
    assert agent_runner.metrics_tracker.usage.cost == 2.5
    assert agent_runner.metrics_tracker.usage.steps == 7


def test_cleanup_only_failure_commits_usage_once(tmp_path: Path) -> None:
    agent = MagicMock(spec=Agent)
    agent.usage = _usage(cost=1.25, steps=3)
    agent.cleanup.side_effect = RuntimeError("cleanup-only failure")
    run_spec = Mock(
        problem=Mock(
            name="prob",
            checkpoints={"checkpoint_1": Mock()},
        ),
        environment=Mock(),
        template="{{ spec }}",
        compress_artifacts=False,
        skip_evaluation=True,
        concurrent_evaluation=False,
        pass_policy=PassPolicy.ANY_CASE,
        agent_type="codex",
        agent_version="0.124.0",
        model_name="gpt-5.5",
    )
    agent_runner = runner.AgentRunner(
        run_spec=run_spec,
        agent=agent,
        output_path=tmp_path,
        progress_queue=queue.Queue(),
    )
    checkpoint = StubCheckpoint("checkpoint_1", "")
    session = MagicMock()

    def attach_session(*_args: object, **_kwargs: object) -> None:
        agent_runner._session = session

    with (
        patch(
            "slop_code.agent_runner.runner.get_checkpoints",
            return_value=iter([(checkpoint, tmp_path / checkpoint.name)]),
        ),
        patch.object(
            agent_runner,
            "_setup_for_checkpoint",
            side_effect=attach_session,
        ),
        patch(
            "slop_code.agent_runner.runner.run_checkpoint",
            return_value=(
                tmp_path / checkpoint.name / "snapshot",
                Mock(had_error=False),
                sentinel.diff,
            ),
        ),
        patch(
            "slop_code.agent_runner.runner.reporting.save_agent_checkpoint_info"
        ),
        pytest.raises(RuntimeError, match="cleanup-only failure"),
    ):
        agent_runner._run_problem()

    session.__exit__.assert_called_once_with(None, None, None)
    assert agent_runner.metrics_tracker.usage.cost == 1.25
    assert agent_runner.metrics_tracker.usage.steps == 3
    agent_runner._commit_checkpoint_usage()
    assert agent_runner.metrics_tracker.usage.cost == 1.25
    assert agent_runner.metrics_tracker.usage.steps == 3


def test_finish_persists_cleanup_failure_as_secondary_to_primary(
    tmp_path: Path,
) -> None:
    problem = Mock(name="prob", checkpoints={})
    run_spec = Mock(problem=problem, skip_evaluation=False)
    run_spec.model_dump.return_value = {
        "problem": {},
        "environment": {},
        "skip_evaluation": False,
    }
    agent_runner = runner.AgentRunner(
        run_spec=run_spec,
        agent=MagicMock(spec=Agent),
        output_path=tmp_path,
        progress_queue=queue.Queue(),
    )
    primary = ValueError("solve exploded")
    agent_runner.metrics_tracker.record_error(
        primary,
        traceback_text="primary traceback",
    )
    agent_runner.metrics_tracker.state = runner.AgentStateEnum.ERROR
    cleanup_error = RuntimeError("session cleanup still failed")

    with (
        patch.object(
            agent_runner,
            "_cleanup_checkpoint_session",
            side_effect=cleanup_error,
        ),
        pytest.raises(RuntimeError, match="session cleanup still failed"),
    ):
        agent_runner.finish()

    persisted = yaml.safe_load(
        (tmp_path / runner.common.RUN_INFO_FILENAME).read_text()
    )["summary"]
    assert persisted["error_type"] == "ValueError"
    assert persisted["error_message"] == "solve exploded"
    assert persisted["error_traceback"] == "primary traceback"
    assert persisted["secondary_errors"] == [
        "run finalization cleanup: RuntimeError: session cleanup still failed"
    ]


def test_finish_preserves_cleanup_failure_when_result_save_also_fails(
    tmp_path: Path,
) -> None:
    run_spec = Mock(
        problem=Mock(name="prob", checkpoints={}),
        concurrent_evaluation=False,
    )
    agent_runner = runner.AgentRunner(
        run_spec=run_spec,
        agent=MagicMock(spec=Agent),
        output_path=tmp_path,
        progress_queue=queue.Queue(),
    )
    cleanup_error = RuntimeError("cleanup exploded")
    save_error = OSError("disk full")

    with (
        patch.object(
            agent_runner,
            "_cleanup_checkpoint_session",
            side_effect=cleanup_error,
        ),
        patch(
            "slop_code.agent_runner.runner.reporting.save_results",
            side_effect=save_error,
        ),
        pytest.raises(RuntimeError, match="cleanup exploded") as exc_info,
    ):
        agent_runner.finish()

    assert exc_info.value is cleanup_error
    assert any(
        "run result persistence" in note and "OSError: disk full" in note
        for note in (cleanup_error.__notes__ or [])
    )


def test_solve_cleanup_and_save_failures_retain_original_primary(
    tmp_path: Path,
) -> None:
    agent = MagicMock(spec=Agent)
    agent.usage = _usage()
    run_spec = Mock(
        problem=Mock(name="prob", checkpoints={}),
        concurrent_evaluation=False,
    )
    agent_runner = runner.AgentRunner(
        run_spec=run_spec,
        agent=agent,
        output_path=tmp_path,
        progress_queue=queue.Queue(),
    )
    solve_error = ValueError("solve exploded")
    cleanup_error = RuntimeError("cleanup exploded")
    save_error = OSError("disk full")

    with (
        patch.object(agent_runner, "setup"),
        patch.object(
            agent_runner,
            "_run_problem",
            side_effect=solve_error,
        ),
        patch.object(
            agent_runner,
            "_cleanup_checkpoint_session",
            side_effect=cleanup_error,
        ),
        patch(
            "slop_code.agent_runner.runner.reporting.save_results",
            side_effect=save_error,
        ),
        pytest.raises(ValueError, match="solve exploded") as exc_info,
    ):
        agent_runner.run()

    assert exc_info.value is solve_error
    notes = solve_error.__notes__ or []
    assert any(
        "run finalization cleanup failure" in note
        and "RuntimeError: cleanup exploded" in note
        for note in notes
    )
    assert any(
        "run finalization detail" in note and "OSError: disk full" in note
        for note in notes
    )
    assert agent_runner.metrics_tracker.error_type == "ValueError"
    assert agent_runner.metrics_tracker.secondary_errors == [
        "run finalization cleanup: RuntimeError: cleanup exploded"
    ]


def test_finish_waits_for_pending_concurrent_evaluation_cleanup(
    tmp_path: Path,
) -> None:
    """Finalization cannot return while an evaluator cleanup is in flight."""
    run_spec = Mock()
    run_spec.problem.name = "prob"
    agent_runner = runner.AgentRunner(
        run_spec=run_spec,
        agent=Mock(spec=Agent),
        output_path=tmp_path,
        progress_queue=queue.Queue(),
    )
    agent_runner.metrics_tracker.state = runner.AgentStateEnum.COMPLETED
    cleanup_started = threading.Event()
    cleanup_finished = threading.Event()

    def delayed_evaluator_cleanup() -> None:
        cleanup_started.set()
        time.sleep(0.05)
        cleanup_finished.set()

    evaluator = threading.Thread(target=delayed_evaluator_cleanup)
    evaluator.start()
    assert cleanup_started.wait(timeout=1)
    agent_runner._pending_eval_thread = evaluator

    with patch(
        "slop_code.agent_runner.runner.reporting.save_results",
        return_value={"summary": {"passed_policy": True}},
    ):
        agent_runner.finish()

    assert cleanup_finished.is_set()
    assert not evaluator.is_alive()
    assert agent_runner._pending_eval_thread is None


def test_finish_joins_progress_watcher_after_terminal_event(
    tmp_path: Path,
) -> None:
    agent = Mock(spec=Agent)
    agent.usage = _usage(cost=4.0, steps=9)
    run_spec = Mock()
    run_spec.problem.name = "prob"
    progress_queue = queue.Queue()
    agent_runner = runner.AgentRunner(
        run_spec=run_spec,
        agent=agent,
        output_path=tmp_path,
        progress_queue=progress_queue,
    )
    agent_runner.metrics_tracker.state = runner.AgentStateEnum.COMPLETED
    watcher = threading.Thread(
        target=runner.agent_progress_watcher,
        args=(
            agent,
            agent_runner.metrics_tracker,
            progress_queue,
            "prob",
        ),
    )
    watcher.start()
    agent_runner.progress_thread = watcher

    with patch(
        "slop_code.agent_runner.runner.reporting.save_results",
        return_value={"summary": {"passed_policy": True}},
    ):
        agent_runner.finish()

    assert not watcher.is_alive()
    problem_name, active_usage, terminal_metrics = progress_queue.get_nowait()
    assert problem_name == "prob"
    assert active_usage.cost == 0.0
    assert active_usage.steps == 0
    assert terminal_metrics.state == runner.AgentStateEnum.COMPLETED


def test_run_problem_resume_does_not_treat_first_executed_as_checkpoint_1(
    tmp_path: Path,
) -> None:
    """When resuming and skipping early checkpoints, continuation must be true.

    This ensures `_run_problem()` only treats the real `checkpoint_1` as the
    first checkpoint for prompt rendering purposes.
    """

    agent = Mock(spec=Agent)
    agent.usage = _usage()

    run_spec = Mock()
    run_spec.problem = Mock()
    run_spec.problem.name = "prob"
    run_spec.problem.checkpoints = {
        "checkpoint_1": Mock(),
        "checkpoint_2": Mock(),
        "checkpoint_3": Mock(),
    }
    run_spec.skip_evaluation = True
    run_spec.concurrent_evaluation = False
    run_spec.pass_policy = Mock()

    resume_info = ResumeInfo(
        resume_from_checkpoint="checkpoint_3",
        completed_checkpoints=["checkpoint_1", "checkpoint_2"],
        last_snapshot_dir=None,
        prior_usage=_usage(),
    )

    ar = runner.AgentRunner(
        run_spec=run_spec,
        agent=agent,
        output_path=tmp_path,
        progress_queue=queue.Queue(),
        resume_info=resume_info,
    )

    checkpoints = [
        (StubCheckpoint("checkpoint_1", ""), tmp_path / "checkpoint_1"),
        (StubCheckpoint("checkpoint_2", ""), tmp_path / "checkpoint_2"),
        (StubCheckpoint("checkpoint_3", ""), tmp_path / "checkpoint_3"),
    ]

    existing_summary = Mock()
    existing_summary.usage = _usage()
    existing_summary.snapshot_dir = tmp_path / "checkpoint_2" / "snapshot"

    summary = Mock()
    summary.passed = True
    summary.passed_policy = True
    summary.had_error = False
    summary.checkpoint_name = "checkpoint_3"
    summary.usage = _usage()

    with (
        patch(
            "slop_code.agent_runner.runner.get_checkpoints",
            return_value=iter(checkpoints),
        ),
        patch.object(
            runner.AgentRunner,
            "_load_checkpoint_summary",
            return_value=existing_summary,
        ),
        patch.object(
            runner.AgentRunner,
            "_run_checkpoint",
            return_value=summary,
        ) as run_ckpt,
    ):
        ar._run_problem()

    # Only checkpoint_3 should be executed, and it must not be treated as
    # "first" just because it's the first executed after resume.
    run_ckpt.assert_called_once()
    _, _, is_first_checkpoint, prior_snapshot_dir = run_ckpt.call_args.args
    assert is_first_checkpoint is False
    assert prior_snapshot_dir == existing_summary.snapshot_dir


def test_run_problem_resume_uses_last_saved_snapshot_when_summary_is_missing(
    tmp_path: Path,
) -> None:
    """The first resumed solve starts from ResumeInfo's saved snapshot."""
    agent = Mock(spec=Agent)
    agent.usage = _usage()

    run_spec = Mock()
    run_spec.problem = Mock()
    run_spec.problem.name = "prob"
    run_spec.problem.checkpoints = {
        "checkpoint_1": Mock(),
        "checkpoint_2": Mock(),
    }
    run_spec.skip_evaluation = True
    run_spec.concurrent_evaluation = False
    run_spec.pass_policy = PassPolicy.ANY_CASE

    last_snapshot_dir = tmp_path / "checkpoint_1" / "snapshot"
    resume_info = ResumeInfo(
        resume_from_checkpoint="checkpoint_2",
        completed_checkpoints=["checkpoint_1"],
        last_snapshot_dir=last_snapshot_dir,
        prior_usage=_usage(),
    )
    agent_runner = runner.AgentRunner(
        run_spec=run_spec,
        agent=agent,
        output_path=tmp_path,
        progress_queue=queue.Queue(),
        resume_info=resume_info,
    )

    checkpoints = [
        (StubCheckpoint("checkpoint_1", ""), tmp_path / "checkpoint_1"),
        (StubCheckpoint("checkpoint_2", ""), tmp_path / "checkpoint_2"),
    ]
    resumed_summary = Mock(
        checkpoint_name="checkpoint_2",
        had_error=False,
        passed_policy=True,
        snapshot_dir=tmp_path / "checkpoint_2" / "snapshot",
        usage=_usage(),
    )

    with (
        patch(
            "slop_code.agent_runner.runner.get_checkpoints",
            return_value=iter(checkpoints),
        ),
        patch.object(
            agent_runner,
            "_load_checkpoint_summary",
            return_value=None,
        ),
        patch.object(
            agent_runner,
            "_run_checkpoint",
            return_value=resumed_summary,
        ) as run_checkpoint_mock,
    ):
        agent_runner._run_problem()

    run_checkpoint_mock.assert_called_once()
    checkpoint, save_dir, is_first, prior_snapshot = (
        run_checkpoint_mock.call_args.args
    )
    assert checkpoint is checkpoints[1][0]
    assert save_dir == checkpoints[1][1]
    assert is_first is False
    assert prior_snapshot == last_snapshot_dir


def test_run_problem_resume_does_not_double_count_prior_usage(
    tmp_path: Path,
) -> None:
    """Skipped checkpoints are already included in resume_info.prior_usage."""

    agent = Mock(spec=Agent)
    agent.usage = _usage(cost=4.0, steps=40)

    run_spec = Mock()
    run_spec.problem = Mock()
    run_spec.problem.name = "prob"
    run_spec.problem.checkpoints = {
        "checkpoint_1": Mock(),
        "checkpoint_2": Mock(),
        "checkpoint_3": Mock(),
    }
    run_spec.skip_evaluation = True
    run_spec.concurrent_evaluation = False
    run_spec.pass_policy = Mock()

    resume_info = ResumeInfo(
        resume_from_checkpoint="checkpoint_3",
        completed_checkpoints=["checkpoint_1", "checkpoint_2"],
        last_snapshot_dir=None,
        prior_usage=_usage(cost=3.0, steps=30),
    )

    ar = runner.AgentRunner(
        run_spec=run_spec,
        agent=agent,
        output_path=tmp_path,
        progress_queue=queue.Queue(),
        resume_info=resume_info,
    )
    ar.metrics_tracker.usage = resume_info.prior_usage.model_copy(deep=True)

    checkpoints = [
        (StubCheckpoint("checkpoint_1", ""), tmp_path / "checkpoint_1"),
        (StubCheckpoint("checkpoint_2", ""), tmp_path / "checkpoint_2"),
        (StubCheckpoint("checkpoint_3", ""), tmp_path / "checkpoint_3"),
    ]

    skipped_summaries = [
        Mock(usage=_usage(cost=1.0, steps=10)),
        Mock(usage=_usage(cost=2.0, steps=20)),
    ]

    summary = Mock()
    summary.passed_policy = True
    summary.had_error = False
    summary.checkpoint_name = "checkpoint_3"
    summary.usage = agent.usage

    with (
        patch(
            "slop_code.agent_runner.runner.get_checkpoints",
            return_value=iter(checkpoints),
        ),
        patch.object(
            runner.AgentRunner,
            "_load_checkpoint_summary",
            side_effect=skipped_summaries,
        ),
        patch.object(
            runner.AgentRunner,
            "_run_checkpoint",
            return_value=summary,
        ),
    ):
        ar._run_problem()

    assert ar.metrics_tracker.usage.cost == 7.0
    assert ar.metrics_tracker.usage.steps == 70


def test_run_problem_resume_does_not_duplicate_preloaded_checkpoint_results(
    tmp_path: Path,
) -> None:
    """Resume should keep one checkpoint result entry per checkpoint."""

    agent = Mock(spec=Agent)
    agent.usage = _usage(cost=4.0, steps=40)

    run_spec = Mock()
    run_spec.problem = Mock()
    run_spec.problem.name = "prob"
    run_spec.problem.checkpoints = {
        "checkpoint_1": Mock(),
        "checkpoint_2": Mock(),
        "checkpoint_3": Mock(),
    }
    run_spec.compress_artifacts = False
    run_spec.skip_evaluation = True
    run_spec.concurrent_evaluation = False
    run_spec.pass_policy = Mock()

    resume_info = ResumeInfo(
        resume_from_checkpoint="checkpoint_3",
        completed_checkpoints=["checkpoint_1", "checkpoint_2"],
        last_snapshot_dir=None,
        prior_usage=_usage(cost=3.0, steps=30),
    )

    ar = runner.AgentRunner(
        run_spec=run_spec,
        agent=agent,
        output_path=tmp_path,
        progress_queue=queue.Queue(),
        resume_info=resume_info,
    )
    ar.metrics_tracker.usage = resume_info.prior_usage.model_copy(deep=True)
    # Mimic setup() preloading completed checkpoint results.
    ar.metrics_tracker.record_checkpoint_result("checkpoint_1", None)
    ar.metrics_tracker.record_checkpoint_result("checkpoint_2", None)

    for checkpoint_name, usage in (
        ("checkpoint_1", {"cost": 1.0, "steps": 10}),
        ("checkpoint_2", {"cost": 2.0, "steps": 20}),
    ):
        checkpoint_dir = tmp_path / checkpoint_name
        checkpoint_dir.mkdir()
        with (checkpoint_dir / INFERENCE_RESULT_FILENAME).open("w") as f:
            json.dump(_inference_result(**usage), f)

    checkpoints = [
        (StubCheckpoint("checkpoint_1", ""), tmp_path / "checkpoint_1"),
        (StubCheckpoint("checkpoint_2", ""), tmp_path / "checkpoint_2"),
        (StubCheckpoint("checkpoint_3", ""), tmp_path / "checkpoint_3"),
    ]

    summary = Mock()
    summary.passed_policy = True
    summary.had_error = False
    summary.checkpoint_name = "checkpoint_3"
    summary.usage = agent.usage

    def _run_checkpoint_side_effect(*args: object, **kwargs: object) -> Mock:
        ar.metrics_tracker.record_checkpoint_result("checkpoint_3", None)
        return summary

    with (
        patch(
            "slop_code.agent_runner.runner.get_checkpoints",
            return_value=iter(checkpoints),
        ),
        patch.object(
            runner.AgentRunner,
            "_run_checkpoint",
            side_effect=_run_checkpoint_side_effect,
        ),
    ):
        ar._run_problem()

    assert [r.name for r in ar.metrics_tracker.checkpoint_results] == [
        "checkpoint_1",
        "checkpoint_2",
        "checkpoint_3",
    ]


def test_evaluation_only_resume_retries_without_inference_and_preserves_later_solve(
    tmp_path: Path,
) -> None:
    checkpoint_names = ["checkpoint_1", "checkpoint_2"]
    for index, checkpoint_name in enumerate(checkpoint_names, start=1):
        checkpoint_dir = tmp_path / checkpoint_name
        (checkpoint_dir / runner.common.SNAPSHOT_DIR_NAME).mkdir(parents=True)
        (checkpoint_dir / INFERENCE_RESULT_FILENAME).write_text(
            json.dumps(_inference_result(float(index), index)),
            encoding="utf-8",
        )
    _passing_report("checkpoint_2").save(tmp_path / "checkpoint_2")
    (
        tmp_path / "checkpoint_1" / runner.common.EVALUATION_ERROR_FILENAME
    ).write_text(
        json.dumps(
            {
                "checkpoint": "checkpoint_1",
                "error_type": "RuntimeError",
                "error_message": "initial evaluator outage",
                "traceback": "",
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / runner.common.RUN_INFO_FILENAME).write_text(
        yaml.safe_dump(
            {
                "summary": {
                    "checkpoints": {
                        "checkpoint_1": "evaluation_error",
                        "checkpoint_2": "ran",
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    resume_info = detect_resume_point(
        tmp_path,
        checkpoint_names,
        require_evaluation=True,
    )
    assert resume_info is not None
    assert resume_info.completed_checkpoints == checkpoint_names
    assert resume_info.evaluation_only_checkpoints == ["checkpoint_1"]
    assert resume_info.invalidated_checkpoints == []
    assert resume_info.resume_from_checkpoint == ""

    problem = Mock(
        name="prob",
        checkpoints={name: Mock() for name in checkpoint_names},
    )
    run_spec = Mock(
        problem=problem,
        environment=Mock(),
        compress_artifacts=False,
        skip_evaluation=False,
        concurrent_evaluation=False,
        pass_policy=PassPolicy.ANY_CASE,
    )
    checkpoints = [
        (StubCheckpoint(name, ""), tmp_path / name) for name in checkpoint_names
    ]

    def save_public_results(
        results: list[runner.AgentCheckpointSummary],
        agent_runner: runner.AgentRunner,
    ) -> dict:
        save_spec = Mock()
        save_spec.problem = problem
        save_spec.skip_evaluation = False
        save_spec.model_dump.return_value = {
            "problem": {},
            "environment": {},
            "skip_evaluation": False,
        }
        return runner.reporting.save_results(
            results,
            agent_runner.metrics_tracker,
            save_spec,
            tmp_path,
        )

    # A failed repair remains a harness error, but checkpoint_2's already-paid
    # solve must remain RAN rather than being invalidated or re-inferred.
    first_agent = Mock(spec=Agent)
    first_agent.usage = _usage()
    first_runner = runner.AgentRunner(
        run_spec=run_spec,
        agent=first_agent,
        output_path=tmp_path,
        progress_queue=queue.Queue(),
        resume_info=resume_info,
    )
    first_runner.metrics_tracker.usage = resume_info.prior_usage.model_copy(
        deep=True
    )
    with (
        patch(
            "slop_code.agent_runner.runner.get_checkpoints",
            return_value=iter(checkpoints),
        ),
        patch.object(
            first_runner,
            "_run_checkpoint",
            side_effect=AssertionError("inference must not run"),
        ) as first_inference,
        patch(
            "slop_code.agent_runner.runner.evaluate_agent_snapshot",
            side_effect=RuntimeError("evaluator still unavailable"),
        ),
    ):
        first_results = first_runner._run_problem()

    first_inference.assert_not_called()
    failed_run_info = save_public_results(first_results, first_runner)
    assert failed_run_info["summary"]["state"] == "error"
    assert failed_run_info["summary"]["passed_policy"] is False
    assert failed_run_info["summary"]["checkpoints"] == {
        "checkpoint_1": "evaluation_error",
        "checkpoint_2": "ran",
    }

    retry_info = detect_resume_point(
        tmp_path,
        checkpoint_names,
        require_evaluation=True,
    )
    assert retry_info is not None
    assert retry_info.completed_checkpoints == checkpoint_names
    assert retry_info.evaluation_only_checkpoints == ["checkpoint_1"]

    repaired_report = _passing_report("checkpoint_1")

    def successful_evaluation(*, save_dir: Path, **_kwargs):
        repaired_report.save(save_dir)
        return repaired_report, Mock()

    retry_agent = Mock(spec=Agent)
    retry_agent.usage = _usage()
    retry_runner = runner.AgentRunner(
        run_spec=run_spec,
        agent=retry_agent,
        output_path=tmp_path,
        progress_queue=queue.Queue(),
        resume_info=retry_info,
    )
    retry_runner.metrics_tracker.usage = retry_info.prior_usage.model_copy(
        deep=True
    )
    with (
        patch(
            "slop_code.agent_runner.runner.get_checkpoints",
            return_value=iter(checkpoints),
        ),
        patch.object(
            retry_runner,
            "_run_checkpoint",
            side_effect=AssertionError("inference must not run"),
        ) as retry_inference,
        patch(
            "slop_code.agent_runner.runner.evaluate_agent_snapshot",
            side_effect=successful_evaluation,
        ),
    ):
        repaired_results = retry_runner._run_problem()

    retry_inference.assert_not_called()
    assert not (
        tmp_path / "checkpoint_1" / runner.common.EVALUATION_ERROR_FILENAME
    ).exists()
    assert (
        tmp_path / "checkpoint_1" / runner.common.EVALUATION_FILENAME
    ).exists()
    repaired_run_info = save_public_results(repaired_results, retry_runner)
    assert repaired_run_info["summary"]["state"] == "completed"
    assert repaired_run_info["summary"]["passed_policy"] is True
    assert repaired_run_info["summary"]["checkpoints"] == {
        "checkpoint_1": "ran",
        "checkpoint_2": "ran",
    }


def test_run_problem_concurrent_eval_bounds_inflight(tmp_path):
    """Concurrent eval runs in a rolling background thread.

    Verifies: every checkpoint is evaluated, AT MOST ONE eval is in flight at
    a time (so in-flight containers stay bounded to 1 solve + 1 eval), reports
    are merged back into the summaries, and the run does not deadlock.
    """
    agent = Mock(spec=Agent)
    agent.usage = _usage()

    run_spec = Mock()
    run_spec.problem = Mock()
    run_spec.problem.name = "prob"
    run_spec.problem.checkpoints = {
        f"checkpoint_{i}": Mock() for i in range(1, 5)
    }
    run_spec.skip_evaluation = False
    run_spec.concurrent_evaluation = True
    run_spec.environment = Mock()
    run_spec.pass_policy = Mock()
    run_spec.pass_policy.check.return_value = True

    ar = runner.AgentRunner(
        run_spec=run_spec,
        agent=agent,
        output_path=tmp_path,
        progress_queue=queue.Queue(),
    )

    names = [f"checkpoint_{i}" for i in range(1, 5)]
    checkpoints = [(StubCheckpoint(n, ""), tmp_path / n) for n in names]

    def make_summary(name: str) -> Mock:
        s = Mock()
        s.checkpoint_name = name
        s.had_error = False
        s.passed_policy = None
        s.snapshot_dir = tmp_path / name / "snapshot"
        s.path = tmp_path / name
        s.artifacts = tmp_path / name / "artifacts"
        s.usage = _usage()
        return s

    solved: list[str] = []

    def fake_run_checkpoint(  # noqa: ANN001
        checkpoint, save_dir, is_first, prior_snapshot_dir
    ):
        del save_dir, is_first, prior_snapshot_dir
        solved.append(checkpoint.name)
        time.sleep(0.02)  # let an eval overlap the next solve
        return make_summary(checkpoint.name)

    inflight = {"cur": 0, "max": 0}
    lock = threading.Lock()
    evaluated: list[str] = []

    def fake_eval(*, checkpoint, save_dir, snapshot_dir, problem, environment):  # noqa: ANN001
        with lock:
            inflight["cur"] += 1
            inflight["max"] = max(inflight["max"], inflight["cur"])
        time.sleep(0.05)
        with lock:
            inflight["cur"] -= 1
            evaluated.append(checkpoint.name)
        return (Mock(), None)

    with (
        patch(
            "slop_code.agent_runner.runner.get_checkpoints",
            return_value=iter(checkpoints),
        ),
        patch.object(
            runner.AgentRunner,
            "_run_checkpoint",
            side_effect=fake_run_checkpoint,
        ),
        patch(
            "slop_code.agent_runner.runner.evaluate_agent_snapshot",
            side_effect=fake_eval,
        ),
        patch.object(runner.MetricsTracker, "record_checkpoint_result"),
        patch.object(runner.MetricsTracker, "finish_checkpoint"),
    ):
        results = ar._run_problem()

    assert solved == names
    assert sorted(evaluated) == sorted(names)
    # The core guarantee: never more than one eval running at once.
    assert inflight["max"] == 1
    # Reports folded back; one summary per checkpoint.
    assert len(results) == 4
    assert all(n in ar._eval_reports for n in names)


def test_run_problem_concurrent_eval_failure_is_durable_and_unsuccessful(
    tmp_path: Path,
) -> None:
    """Evaluator infrastructure errors must never look like model results."""
    agent = Mock(spec=Agent)
    agent.usage = _usage()

    run_spec = Mock()
    run_spec.problem = Mock()
    run_spec.problem.name = "prob"
    run_spec.problem.checkpoints = {
        "checkpoint_1": Mock(),
        "checkpoint_2": Mock(),
        "checkpoint_3": Mock(),
    }
    run_spec.skip_evaluation = False
    run_spec.concurrent_evaluation = True
    # Use a real PassPolicy so _merge_eval_reports can rebuild summaries via
    # AgentCheckpointSummary.from_results (which validates with pydantic).
    run_spec.pass_policy = PassPolicy.ANY_CASE

    ar = runner.AgentRunner(
        run_spec=run_spec,
        agent=agent,
        output_path=tmp_path,
        progress_queue=queue.Queue(),
        resume_info=None,
    )

    checkpoints = [
        (StubCheckpoint(f"checkpoint_{i}", ""), tmp_path / f"checkpoint_{i}")
        for i in (1, 2, 3)
    ]

    def make_summary(cp_name: str) -> runner.AgentCheckpointSummary:
        return runner.AgentCheckpointSummary(
            checkpoint_name=cp_name,
            passed_policy=True,
            had_error=False,
            snapshot_dir=tmp_path / f"{cp_name}_snap",
            path=tmp_path / cp_name,
            artifacts=tmp_path / cp_name / "agent",
            usage=_usage(),
        )

    summaries = {
        f"checkpoint_{i}": make_summary(f"checkpoint_{i}") for i in (1, 2, 3)
    }
    eval_calls: list[str] = []

    def make_report() -> Mock:
        """Minimal report shape consumed by metrics and pass policy."""
        report = Mock()
        report.pass_counts = {"Core": 1}
        report.total_counts = {"Core": 1}
        return report

    def eval_snapshot_side_effect(*, checkpoint, **_kwargs):
        eval_calls.append(checkpoint.name)
        if checkpoint.name == "checkpoint_1":
            raise RuntimeError("boom: simulated eval failure")
        return (make_report(), Mock())

    with (
        patch(
            "slop_code.agent_runner.runner.get_checkpoints",
            return_value=iter(checkpoints),
        ),
        patch.object(
            runner.AgentRunner,
            "_run_checkpoint",
            side_effect=lambda ckpt, *_a, **_kw: summaries[ckpt.name],
        ),
        patch(
            "slop_code.agent_runner.runner.evaluate_agent_snapshot",
            side_effect=eval_snapshot_side_effect,
        ),
    ):
        results = ar._run_problem()

    # 1. cp1's failure is recorded with the exception text; cp2/cp3 are not.
    assert "checkpoint_1" in ar._failed_evals
    assert "boom" in ar._failed_evals["checkpoint_1"]
    assert "checkpoint_2" not in ar._failed_evals
    assert "checkpoint_3" not in ar._failed_evals

    # cp2 can already be in flight when cp1's background eval fails, but the
    # harness must not begin cp3 after discovering an infrastructure failure.
    assert eval_calls == ["checkpoint_1", "checkpoint_2"]
    assert [result.checkpoint_name for result in results] == [
        "checkpoint_1",
        "checkpoint_2",
    ]
    assert results[0].passed_policy is False
    assert "boom" in (results[0].evaluation_error_message or "")
    assert ar.metrics_tracker.state == runner.AgentStateEnum.ERROR

    error_path = (
        tmp_path / "checkpoint_1" / runner.common.EVALUATION_ERROR_FILENAME
    )
    error_payload = json.loads(error_path.read_text())
    assert error_payload["checkpoint"] == "checkpoint_1"
    assert error_payload["error_type"] == "RuntimeError"
    assert error_payload["error_message"] == "boom: simulated eval failure"

    # The saved public result must distinguish a harness evaluator error from
    # a failed model solution and must never claim the run passed.
    save_spec = Mock()
    save_spec.problem = run_spec.problem
    save_spec.skip_evaluation = False
    save_spec.model_dump.return_value = {
        "problem": {},
        "environment": {},
        "skip_evaluation": False,
    }
    run_info = runner.reporting.save_results(
        results,
        ar.metrics_tracker,
        save_spec,
        tmp_path,
    )
    assert run_info["summary"]["state"] == "error"
    assert run_info["summary"]["passed_policy"] is False
    assert run_info["summary"]["checkpoints"] == {
        "checkpoint_1": "evaluation_error",
        "checkpoint_2": "ran",
        "checkpoint_3": "skipped",
    }


@pytest.mark.parametrize("primary_type", [RuntimeError, KeyboardInterrupt])
def test_concurrent_eval_failure_does_not_replace_simultaneous_solve_failure(
    tmp_path: Path,
    primary_type: type[BaseException],
) -> None:
    agent = MagicMock(spec=Agent)
    agent.usage = _usage()
    problem = Mock(
        name="prob",
        checkpoints={
            "checkpoint_1": Mock(),
            "checkpoint_2": Mock(),
        },
    )
    run_spec = Mock(
        problem=problem,
        environment=Mock(),
        skip_evaluation=False,
        concurrent_evaluation=True,
        pass_policy=PassPolicy.ANY_CASE,
    )
    run_spec.model_dump.return_value = {
        "problem": {},
        "environment": {},
        "skip_evaluation": False,
        "concurrent_evaluation": True,
    }
    agent_runner = runner.AgentRunner(
        run_spec=run_spec,
        agent=agent,
        output_path=tmp_path,
        progress_queue=queue.Queue(),
    )
    checkpoints = [
        (StubCheckpoint(name, ""), tmp_path / name)
        for name in ("checkpoint_1", "checkpoint_2")
    ]
    for _checkpoint, checkpoint_dir in checkpoints:
        (checkpoint_dir / "snapshot").mkdir(parents=True)

    checkpoint_1_summary = runner.AgentCheckpointSummary(
        checkpoint_name="checkpoint_1",
        path=tmp_path / "checkpoint_1",
        snapshot_dir=tmp_path / "checkpoint_1" / "snapshot",
        artifacts=tmp_path / "checkpoint_1" / "agent",
        usage=_usage(),
        passed_policy=True,
        had_error=False,
    )
    allow_eval_failure = threading.Event()
    evaluator_raised = threading.Event()
    primary_error = primary_type("solve exploded")

    def run_checkpoint_side_effect(checkpoint, *_args, **_kwargs):  # noqa: ANN001
        if checkpoint.name == "checkpoint_1":
            return checkpoint_1_summary
        allow_eval_failure.set()
        assert evaluator_raised.wait(timeout=1)
        raise primary_error

    def evaluate_side_effect(**_kwargs):
        assert allow_eval_failure.wait(timeout=1)
        evaluator_raised.set()
        raise RuntimeError("evaluator exploded")

    with (
        patch.object(agent_runner, "setup"),
        patch(
            "slop_code.agent_runner.runner.get_checkpoints",
            return_value=iter(checkpoints),
        ),
        patch.object(
            agent_runner,
            "_run_checkpoint",
            side_effect=run_checkpoint_side_effect,
        ),
        patch(
            "slop_code.agent_runner.runner.evaluate_agent_snapshot",
            side_effect=evaluate_side_effect,
        ),
        pytest.raises(primary_type) as exc_info,
    ):
        agent_runner.run()

    assert exc_info.value is primary_error
    persisted = yaml.safe_load(
        (tmp_path / runner.common.RUN_INFO_FILENAME).read_text()
    )["summary"]
    assert persisted["error_type"] == primary_type.__qualname__
    assert persisted["error_message"] == "solve exploded"
    assert persisted["state"] == "error"
    assert persisted["passed_policy"] is False
    assert persisted["checkpoints"] == {
        "checkpoint_1": "evaluation_error",
        "checkpoint_2": "skipped",
    }
    assert persisted["secondary_errors"] == [
        "concurrent evaluation: EvaluationError: "
        "Checkpoint evaluation failed: checkpoint_1"
    ]
    assert (
        tmp_path / "checkpoint_1" / runner.common.EVALUATION_ERROR_FILENAME
    ).exists()
