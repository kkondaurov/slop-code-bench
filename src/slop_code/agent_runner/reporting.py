from __future__ import annotations

import json
import math
import shutil
import tarfile
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import field_validator

from slop_code import common
from slop_code.agent_runner.agent import Agent
from slop_code.agent_runner.agent import CheckpointInferenceResult
from slop_code.agent_runner.models import AgentRunSpec
from slop_code.agent_runner.models import UsageTracker
from slop_code.agent_runner.state import AgentStateEnum
from slop_code.common.atomic import atomic_write_text
from slop_code.evaluation import CheckpointConfig
from slop_code.evaluation import CorrectnessResults
from slop_code.evaluation import GroupType
from slop_code.evaluation import PassPolicy
from slop_code.execution import SnapshotDiff
from slop_code.logging import get_logger

logger = get_logger(__name__)


class AgentCheckpointSummary(BaseModel):
    """Result of running an agent on a single checkpoint.

    Attributes:
        checkpoint_name: Name of the checkpoint.
        path: Path to the checkpoint directory.
        snapshot_dir: Relative path to the snapshot directory.
        artifacts: Relative path to the agent artifacts directory.
        usage: Usage tracker with cost and token counts.
        passed_policy: Whether checkpoint passed the configured pass policy.
        had_error: Whether an error occurred during execution.
        error_message: Error message if one occurred.
    """

    checkpoint_name: str
    path: Path
    snapshot_dir: Path
    artifacts: Path
    usage: UsageTracker
    passed_policy: bool | None = None
    error_message: str | None = None
    evaluation_error_message: str | None = None
    had_error: bool

    @classmethod
    def from_results(
        cls,
        checkpoint_name: str,
        path: Path,
        snapshot_dir: Path,
        artifacts: Path,
        usage: UsageTracker,
        *,
        had_error: bool,
        pass_policy: PassPolicy,
        evaluation_result: CorrectnessResults | None = None,
    ) -> AgentCheckpointSummary:
        if isinstance(pass_policy, str):
            pass_policy = PassPolicy(pass_policy)

        if evaluation_result is not None:
            # A structured report may still represent an evaluator/container
            # failure. Counts from such a report are evidence, not a valid
            # model score.
            if (
                getattr(evaluation_result, "infrastructure_failure", False)
                is True
            ):
                passed_policy = False
            else:
                passed_policy = pass_policy.check(
                    evaluation_result.pass_counts,
                    evaluation_result.total_counts,
                )
        else:
            # No evaluation - ANY_CASE permits continuation only when inference
            # itself succeeded.
            passed_policy = not had_error and pass_policy == PassPolicy.ANY_CASE

        return cls(
            checkpoint_name=checkpoint_name,
            path=path,
            snapshot_dir=snapshot_dir,
            artifacts=artifacts,
            usage=usage.model_copy(deep=True),
            had_error=had_error,
            passed_policy=passed_policy,
        )


def _preserve_malformed_run_info(target: Path) -> Path | None:
    """Copy malformed run metadata before an atomic replacement."""
    # The atomic writer below rejects symlinks. Avoid following one while
    # trying to preserve malformed evidence first.
    if target.is_symlink():
        return None
    if not target.exists():
        return None
    malformed = False
    try:
        with target.open(encoding="utf-8") as handle:
            malformed = _validate_saved_run_info(yaml.safe_load(handle)) is None
    except (OSError, yaml.YAMLError):
        malformed = True
    if not malformed:
        return None

    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S%f")
    backup = target.with_name(f"run_info.corrupt-{timestamp}.yaml")
    moved = False
    if not target.is_file():
        # A directory, FIFO, or other non-regular artifact cannot be replaced
        # by os.replace(temp_file, target). Move it intact so the normal atomic
        # writer can recreate run_info.yaml at the expected path.
        target.replace(backup)
        moved = True
    else:
        try:
            shutil.copy2(target, backup)
        except OSError:
            # A regular file can be stat-able but unreadable. Renaming only
            # needs write access to the parent and preserves the exact bytes.
            target.replace(backup)
            moved = True
    logger.warning(
        "Preserved malformed run info before replacement",
        source=str(target),
        backup=str(backup),
        moved=moved,
    )
    return backup


def _write_run_info_atomically(target: Path, run_info: dict[str, Any]) -> None:
    """Write run metadata without exposing readers to a partial YAML file."""
    serialized = yaml.dump(run_info, indent=2, sort_keys=True)
    backup = _preserve_malformed_run_info(target)
    try:
        atomic_write_text(target, serialized)
    except BaseException:  # noqa: BLE001
        # A non-regular or unreadable target may have needed to be moved out of
        # the way. If the atomic writer failed before publishing a replacement,
        # restore the exact original path; the caller still receives the write
        # failure and can retry safely.
        if backup is not None and not target.exists():
            backup.replace(target)
        raise


class CheckpointState:
    """Valid states for checkpoint execution."""

    RAN = "ran"
    SKIPPED = "skipped"
    ERROR = "error"
    EVALUATION_ERROR = "evaluation_error"


_VALID_CHECKPOINT_STATES = {
    CheckpointState.RAN,
    CheckpointState.SKIPPED,
    CheckpointState.ERROR,
    CheckpointState.EVALUATION_ERROR,
}


def _has_evaluation_error(result: AgentCheckpointSummary) -> bool:
    """Recognize a persisted evaluator failure, including legacy summaries.

    Older callers and lightweight tests may supply summary-shaped objects that
    predate ``evaluation_error_message``. An unconstrained ``Mock`` also
    synthesizes a truthy attribute on access, so only a real non-empty string
    is evidence of an evaluator failure.
    """
    message = getattr(result, "evaluation_error_message", None)
    return isinstance(message, str) and bool(message.strip())


class RunSummary(BaseModel):
    """Execution summary for an agent problem run.

    This contains only aggregated/summary information. Detailed per-checkpoint
    data should be read from individual inference_result.json and
    evaluation.json files in each checkpoint directory.

    Attributes:
        started: Timestamp when execution started
        ended: Timestamp when execution ended
        duration_seconds: Total duration in seconds
        total_cost: Total cost across all checkpoints
        total_steps: Total steps across all checkpoints
        total_usage: Aggregated usage tracker
        checkpoints: Dict mapping checkpoint name to state (ran/skipped/error)
        state: Final execution state
        error_type: Type of error if one occurred
        error_message: Error message if one occurred
        error_traceback: Error traceback if one occurred
        passed_policy: Whether all checkpoints passed the policy
    """

    model_config = ConfigDict(use_enum_values=True)

    # Timing
    started: datetime
    ended: datetime
    duration_seconds: float

    # Aggregated usage
    total_cost: float
    total_steps: int
    total_usage: UsageTracker

    # Checkpoint tracking: checkpoint_name -> state (ran/skipped/error)
    checkpoints: dict[str, str]

    # State/error
    state: AgentStateEnum
    error_type: str | None = None
    error_message: str | None = None
    error_traceback: str | None = None
    secondary_errors: list[str] = Field(default_factory=list)

    # Aggregated evaluation
    passed_policy: bool | None = None

    @field_validator("ended")
    @classmethod
    def validate_ended(cls, v: str | datetime) -> datetime:
        if isinstance(v, str):
            return datetime.fromisoformat(v)
        return v

    @field_validator("started")
    @classmethod
    def validate_started(cls, v: str | datetime) -> datetime:
        if isinstance(v, str):
            return datetime.fromisoformat(v)
        return v


def _is_nonnegative_number(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, int | float)
        and math.isfinite(value)
        and value >= 0
    )


def _is_nonnegative_integer(value: object) -> bool:
    return type(value) is int and value >= 0


def _usage_has_valid_domains(value: object) -> bool:
    """Validate persisted usage without accepting Pydantic coercions."""
    if not isinstance(value, dict):
        return False
    if not _is_nonnegative_number(value.get("cost")):
        return False
    if not _is_nonnegative_integer(value.get("steps")):
        return False
    for token_key in ("net_tokens", "current_tokens"):
        tokens = value.get(token_key)
        if not isinstance(tokens, dict):
            return False
        for field_name in common.TokenUsage.model_fields:
            if not _is_nonnegative_integer(tokens.get(field_name)):
                return False
    return True


def _validate_saved_run_info(value: object) -> RunSummary | None:
    """Return a fully validated saved summary, or ``None`` when untrusted."""
    if not isinstance(value, dict):
        return None
    raw_summary = value.get("summary")
    if not isinstance(raw_summary, dict):
        return None
    if not _is_nonnegative_number(raw_summary.get("duration_seconds")):
        return None
    if not _is_nonnegative_number(raw_summary.get("total_cost")):
        return None
    if not _is_nonnegative_integer(raw_summary.get("total_steps")):
        return None
    if not _usage_has_valid_domains(raw_summary.get("total_usage")):
        return None
    passed_policy = raw_summary.get("passed_policy")
    if passed_policy is not None and type(passed_policy) is not bool:
        return None
    secondary_errors = raw_summary.get("secondary_errors", [])
    if not isinstance(secondary_errors, list) or any(
        not isinstance(error, str) or not error.strip()
        for error in secondary_errors
    ):
        return None
    checkpoint_states = raw_summary.get("checkpoints")
    if not isinstance(checkpoint_states, dict) or any(
        not isinstance(name, str)
        or not isinstance(state, str)
        or state not in _VALID_CHECKPOINT_STATES
        for name, state in checkpoint_states.items()
    ):
        return None
    try:
        summary = RunSummary.model_validate(raw_summary)
    except (TypeError, ValueError):
        return None
    try:
        ended_before_started = summary.ended < summary.started
        timestamp_duration = (summary.ended - summary.started).total_seconds()
    except TypeError:
        # Offset-aware and offset-naive datetimes are individually valid to
        # Pydantic but do not inhabit a comparable timestamp domain.
        return None
    if ended_before_started:
        return None
    if not math.isclose(
        summary.duration_seconds,
        timestamp_duration,
        rel_tol=1e-9,
        abs_tol=1e-6,
    ):
        return None
    if not math.isclose(
        summary.total_cost,
        summary.total_usage.cost,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        return None
    if summary.total_steps != summary.total_usage.steps:
        return None
    return summary


class CheckpointEvalResult(BaseModel):
    """Result of a single checkpoint evaluation for progress tracking.

    Attributes:
        name: Checkpoint name
        passed: Whether pass_rate == 1.0 (all tests including regression passed)
        iso_passed: Whether checkpoint_pass_rate == 1.0 (all non-regression tests passed)
        pass_rate: Overall pass rate (0.0 to 1.0)
        checkpoint_pass_rate: Isolated pass rate excluding regression tests
    """

    name: str
    passed: bool
    iso_passed: bool
    core_passed: bool = False
    pass_rate: float = 0.0
    checkpoint_pass_rate: float = 0.0


class MetricsTracker(BaseModel):
    """Tracks metrics and state during agent execution.

    Attributes:
        state: Current execution state of the agent
        current_checkpoint: Name of the checkpoint being processed
        usage: Usage tracker for cost and tokens
        started: Timestamp when execution started
        checkpoint_started: Timestamp when current checkpoint started
        checkpoint_results: List of evaluation results for completed checkpoints
    """

    model_config = ConfigDict(extra="forbid", use_enum_values=True)
    state: AgentStateEnum = AgentStateEnum.INITIALIZED
    current_checkpoint: str
    usage: UsageTracker
    started: datetime
    checkpoint_started: datetime
    error_type: str | None = None
    error_message: str | None = None
    error_traceback: str | None = None
    secondary_errors: list[str] = Field(default_factory=list)
    checkpoint_results: list[CheckpointEvalResult] = Field(default_factory=list)

    def finish_checkpoint(self, usage: UsageTracker) -> None:
        """Update usage metrics after checkpoint completion.

        Args:
            usage: UsageTracker with usage data to incorporate
        """
        self.usage.cost += usage.cost
        self.usage.steps += usage.steps
        self.usage.net_tokens += usage.net_tokens
        self.usage.current_tokens += usage.current_tokens

    def had_error(self) -> bool:
        """Check if the agent is in an error state.

        Returns:
            True if agent state is ERROR, False otherwise
        """
        return self.state in {
            AgentStateEnum.ERROR,
        }

    def record_error(
        self,
        error: BaseException,
        *,
        traceback_text: str | None = None,
    ) -> None:
        """Capture error details for downstream reporting."""
        self.error_type = type(error).__name__
        self.error_message = str(error)
        self.error_traceback = traceback_text

    def record_secondary_error(
        self,
        error: BaseException,
        *,
        phase: str,
    ) -> None:
        """Persist a cleanup/finalization failure without replacing primary."""
        self.secondary_errors.append(
            f"{phase}: {type(error).__qualname__}: {error}"
        )

    def record_checkpoint_result(
        self,
        name: str,
        evaluation_result: CorrectnessResults | None,
    ) -> None:
        """Record evaluation result for a checkpoint.

        Calculates pass rates and stores the result for progress tracking.

        Args:
            name: Checkpoint name
            evaluation_result: Evaluation results, or None if evaluation was skipped
        """
        checkpoint_result: CheckpointEvalResult
        if evaluation_result is None:
            checkpoint_result = CheckpointEvalResult(
                name=name,
                passed=False,
                iso_passed=False,
            )
        else:
            # Calculate pass rates
            pass_counts = evaluation_result.pass_counts
            total_counts = evaluation_result.total_counts

            total_passed = sum(pass_counts.values())
            total_total = sum(total_counts.values())
            pass_rate = total_passed / total_total if total_total > 0 else 0.0

            # Checkpoint pass rate excludes regression tests
            regression_passed = pass_counts.get(GroupType.REGRESSION, 0)
            regression_total = total_counts.get(GroupType.REGRESSION, 0)
            checkpoint_passed = total_passed - regression_passed
            checkpoint_total = total_total - regression_total
            checkpoint_pass_rate = (
                checkpoint_passed / checkpoint_total
                if checkpoint_total > 0
                else 0.0
            )

            core_total = total_counts.get(GroupType.CORE, 0)
            core_passed_count = pass_counts.get(GroupType.CORE, 0)
            core_pass_rate = (
                core_passed_count / core_total if core_total > 0 else 0.0
            )

            checkpoint_result = CheckpointEvalResult(
                name=name,
                passed=math.isclose(pass_rate, 1.0),
                iso_passed=math.isclose(checkpoint_pass_rate, 1.0),
                core_passed=math.isclose(core_pass_rate, 1.0),
                pass_rate=pass_rate,
                checkpoint_pass_rate=checkpoint_pass_rate,
            )

        for index, existing in enumerate(self.checkpoint_results):
            if existing.name == name:
                self.checkpoint_results[index] = checkpoint_result
                return

        self.checkpoint_results.append(checkpoint_result)


def setup_run_output_directory(
    run_spec: AgentRunSpec, output_path: Path
) -> None:
    """Set up the output directory for an agent run.

    Creates the output directory and saves the problem configuration as YAML.
    The run specification and execution summary are saved together in
    run_info.yaml by save_results() at the end of the run.

    Args:
        run_spec: Specification for the agent run
        output_path: Directory path where output should be saved
    """
    logger.debug(
        "Setting up output directory",
        output_path=output_path,
        problem=run_spec.problem.name,
    )

    problem = common.serialize_path_dict(
        run_spec.problem.model_dump(mode="json")
    )
    with (output_path / common.PROBLEM_CONFIG_NAME).open("w") as f:
        yaml.dump(
            problem,
            f,
            indent=2,
            sort_keys=True,
        )


def setup_checkpoint_output_directory(
    checkpoint: CheckpointConfig, output_path: Path
) -> Path:
    """Set up output directory for a specific checkpoint.

    Creates a subdirectory for the checkpoint and saves its configuration.

    Args:
        checkpoint: Checkpoint configuration
        output_path: Base output directory path

    Returns:
        Path to the created checkpoint output directory
    """
    logger.debug(
        "Setting up checkpoint output directory",
        output_path=output_path,
        checkpoint=checkpoint.name,
    )

    checkpoint_save_dir = output_path / checkpoint.name
    checkpoint_save_dir.mkdir(parents=True, exist_ok=True)
    with (checkpoint_save_dir / common.CHECKPOINT_CONFIG_NAME).open("w") as f:
        yaml.dump(
            common.serialize_path_dict(checkpoint.model_dump(mode="json")),
            f,
            indent=2,
            sort_keys=True,
        )
    return checkpoint_save_dir


def save_results(
    results: list[AgentCheckpointSummary],
    metrics_tracker: MetricsTracker,
    run_spec: AgentRunSpec,
    output_path: Path,
) -> dict[str, Any]:
    """Save agent run info to YAML file.

    Saves a combined run_info.yaml with run specification and execution summary.
    Per-checkpoint details are stored in individual inference_result.json
    and evaluation.json files.

    Args:
        results: List of checkpoint results
        metrics_tracker: Metrics tracker with aggregated usage
        run_spec: Original run specification
        output_path: Directory to save results

    Returns:
        Dictionary containing the complete run info data
    """
    logger.info(
        "Saving run info",
        problem=run_spec.problem.name,
        checkpoints=len(results),
        output_path=output_path,
    )

    # Build checkpoint state dict: name -> state (ran/skipped/error)
    all_checkpoint_names = list(run_spec.problem.checkpoints.keys())
    results_by_name = {r.checkpoint_name: r for r in results}
    checkpoints_state: dict[str, str] = {}
    for name in all_checkpoint_names:
        if name in results_by_name:
            result = results_by_name[name]
            if _has_evaluation_error(result):
                checkpoints_state[name] = CheckpointState.EVALUATION_ERROR
            elif result.had_error:
                checkpoints_state[name] = CheckpointState.ERROR
            else:
                checkpoints_state[name] = CheckpointState.RAN
        else:
            checkpoints_state[name] = CheckpointState.SKIPPED

    ended = datetime.now()
    secondary_errors = getattr(metrics_tracker, "secondary_errors", [])
    if not isinstance(secondary_errors, list) or any(
        not isinstance(error, str) for error in secondary_errors
    ):
        secondary_errors = []
    summary = RunSummary(
        started=metrics_tracker.started,
        ended=ended,
        duration_seconds=(ended - metrics_tracker.started).total_seconds(),
        total_cost=metrics_tracker.usage.cost,
        total_steps=metrics_tracker.usage.steps,
        total_usage=metrics_tracker.usage,
        checkpoints=checkpoints_state,
        state=metrics_tracker.state,
        error_type=metrics_tracker.error_type,
        error_message=metrics_tracker.error_message,
        error_traceback=metrics_tracker.error_traceback,
        secondary_errors=secondary_errors,
        # Errored/skipped checkpoints are complete failures
        passed_policy=all(
            not r.had_error
            and not _has_evaluation_error(r)
            and (run_spec.skip_evaluation or bool(r.passed_policy))
            for r in results
        )
        and len(results) == len(all_checkpoint_names),
    )

    # Build combined run_info structure
    spec_dump = common.serialize_path_dict(run_spec.model_dump(mode="json"))
    spec_dump.pop("environment")
    spec_dump.pop("problem")
    spec_dump.update(common.get_save_spec_dump())

    run_info: dict[str, Any] = {
        **spec_dump,
        "summary": common.serialize_path_dict(summary.model_dump(mode="json")),
    }

    _write_run_info_atomically(
        output_path / common.RUN_INFO_FILENAME,
        run_info,
    )

    logger.info(
        "Run info saved successfully",
        problem=run_spec.problem.name,
        passed_policy=summary.passed_policy,
        total_cost=summary.total_cost,
    )
    return run_info


def save_agent_artifacts(
    output_path: Path,
    agent: Agent,
    *,
    compress_artifacts: bool = False,
) -> str:
    """Save agent-native artifacts and return the artifact path name."""
    output_path.mkdir(parents=True, exist_ok=True)
    if compress_artifacts:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            agent.save_artifacts(temp_path)
            tar_path = output_path / common.AGENT_TAR_FILENAME
            with tarfile.open(tar_path, "w:gz") as tar:
                for item in temp_path.iterdir():
                    tar.add(item, arcname=item.name)
        return common.AGENT_TAR_FILENAME

    agent_dir = output_path / common.AGENT_DIR_NAME
    agent_dir.mkdir(parents=True, exist_ok=True)
    agent.save_artifacts(agent_dir)
    return common.AGENT_DIR_NAME


def save_agent_checkpoint_info(
    output_path: Path,
    diff: SnapshotDiff,
    checkpoint_result: CheckpointInferenceResult,
    agent: Agent,
    *,
    compress_artifacts: bool = False,
) -> None:
    logger.info(
        "Saving agent checkpoint info",
        output_path=output_path,
        compress_artifacts=compress_artifacts,
    )
    with (output_path / common.DIFF_FILENAME).open("w") as f:
        f.write(diff.model_dump_json())

    artifacts_name = save_agent_artifacts(
        output_path,
        agent,
        compress_artifacts=compress_artifacts,
    )

    # Add path fields to checkpoint result before saving
    result_with_paths = checkpoint_result.model_copy(
        update={
            "checkpoint_path": common.to_relative_path(output_path),
            "snapshot_dir": common.SNAPSHOT_DIR_NAME,
            "artifacts_dir": artifacts_name,
        }
    )

    with (output_path / common.INFERENCE_RESULT_FILENAME).open("w") as f:
        f.write(
            json.dumps(
                common.serialize_path_dict(
                    result_with_paths.model_dump(mode="json")
                ),
                indent=2,
                sort_keys=True,
            )
        )
