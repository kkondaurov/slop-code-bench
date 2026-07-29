"""Resume functionality for checkpoint-based execution.

This module provides utilities for detecting and resuming from the last
successful checkpoint when a run is interrupted or fails mid-execution.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from dataclasses import field
from enum import Enum
from pathlib import Path

import yaml

from slop_code.agent_runner.agent import CheckpointInferenceResult
from slop_code.agent_runner.models import UsageTracker
from slop_code.agent_runner.reporting import CheckpointState
from slop_code.agent_runner.reporting import RunSummary
from slop_code.agent_runner.reporting import _validate_saved_run_info
from slop_code.agent_runner.state import AgentStateEnum
from slop_code.common import EVALUATION_ERROR_FILENAME
from slop_code.common import EVALUATION_FILENAME
from slop_code.common import INFERENCE_RESULT_FILENAME
from slop_code.common import PROMPT_FILENAME
from slop_code.common import RUN_INFO_FILENAME
from slop_code.common import SNAPSHOT_DIR_NAME
from slop_code.common import render_prompt
from slop_code.common.llms import TokenUsage
from slop_code.evaluation.config import CheckpointConfig
from slop_code.evaluation.config import ProblemConfig
from slop_code.evaluation.report import CorrectnessResults
from slop_code.evaluation.report import PassPolicy
from slop_code.execution.models import EnvironmentSpec
from slop_code.logging import get_logger

logger = get_logger(__name__)


class InvalidationReason(Enum):
    """Reasons why a checkpoint needs to be re-run."""

    SPEC_CHANGED = "spec_changed"
    HAD_ERROR = "had_error"
    MISSING_SNAPSHOT = "missing_snapshot"
    MISSING_RESULT = "missing_result"
    MISSING_DIR = "missing_directory"
    UNREADABLE_RESULT = "unreadable_result"
    DEPENDS_ON_INVALID = "depends_on_invalid"
    MISSING_EVALUATION = "missing_evaluation"
    EVALUATION_ERROR = "evaluation_error"


@dataclass
class CheckpointStatus:
    """Status of a single checkpoint for resume detection."""

    name: str
    is_valid: bool
    reason: InvalidationReason | None = None


@dataclass
class ResumeInfo:
    """Information needed to resume a run from a checkpoint.

    Attributes:
        resume_from_checkpoint: Name of the checkpoint to resume from
        completed_checkpoints: Names of checkpoints that completed successfully
        last_snapshot_dir: Path to the last successful snapshot directory
        prior_usage: Aggregated usage from completed checkpoints
        checkpoint_statuses: Detailed status for each checkpoint examined
        invalidated_checkpoints: List of checkpoint names that will be re-run
    """

    resume_from_checkpoint: str
    completed_checkpoints: list[str]
    last_snapshot_dir: Path | None
    prior_usage: UsageTracker
    checkpoint_statuses: list[CheckpointStatus] = field(default_factory=list)
    invalidated_checkpoints: list[str] = field(default_factory=list)
    evaluation_only_checkpoints: list[str] = field(default_factory=list)
    run_info_reconciliation_required: bool = False


def _evaluation_status(
    checkpoint_dir: Path,
) -> tuple[bool, InvalidationReason | None]:
    if (checkpoint_dir / EVALUATION_ERROR_FILENAME).exists():
        return False, InvalidationReason.EVALUATION_ERROR
    if not (checkpoint_dir / EVALUATION_FILENAME).exists():
        return False, InvalidationReason.MISSING_EVALUATION
    try:
        result = CorrectnessResults.from_dir(checkpoint_dir)
    except (
        AttributeError,
        OSError,
        TypeError,
        ValueError,
        KeyError,
        json.JSONDecodeError,
    ):
        return False, InvalidationReason.MISSING_EVALUATION
    if result.infrastructure_failure:
        return False, InvalidationReason.EVALUATION_ERROR
    return True, None


def _valid_usage_mapping(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    cost = value.get("cost")
    if (
        isinstance(cost, bool)
        or not isinstance(cost, int | float)
        or not math.isfinite(cost)
        or cost < 0
    ):
        return False
    steps = value.get("steps")
    if type(steps) is not int or steps < 0:
        return False
    for token_key in ("net_tokens", "current_tokens"):
        tokens = value.get(token_key)
        if not isinstance(tokens, dict):
            return False
        for field_name in TokenUsage.model_fields:
            token_count = tokens.get(field_name)
            if type(token_count) is not int or token_count < 0:
                return False
    return True


def _load_inference_result(
    result_path: Path,
) -> CheckpointInferenceResult | None:
    """Load a schema- and domain-validated inference result."""
    try:
        with result_path.open(encoding="utf-8") as handle:
            result = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        logger.warning(
            "Failed to read inference_result.json",
            path=str(result_path),
            error=str(error),
        )
        return None
    if not isinstance(result, dict):
        logger.warning(
            "Inference result is not a JSON object",
            path=str(result_path),
            result_type=type(result).__name__,
        )
        return None
    required_fields = {"started", "completed", "elapsed", "usage", "had_error"}
    missing_fields = required_fields.difference(result)
    if missing_fields:
        logger.warning(
            "Inference result is missing required fields",
            path=str(result_path),
            missing_fields=sorted(missing_fields),
        )
        return None
    if type(result.get("had_error")) is not bool:
        logger.warning(
            "Inference result had_error is not a boolean",
            path=str(result_path),
            had_error_type=type(result.get("had_error")).__name__,
        )
        return None
    elapsed = result.get("elapsed")
    if (
        isinstance(elapsed, bool)
        or not isinstance(elapsed, int | float)
        or not math.isfinite(elapsed)
        or elapsed < 0
    ):
        logger.warning(
            "Inference result elapsed is outside its valid domain",
            path=str(result_path),
            elapsed=elapsed,
        )
        return None
    if not _valid_usage_mapping(result.get("usage")):
        logger.warning(
            "Inference result usage is outside its valid domain",
            path=str(result_path),
        )
        return None
    try:
        parsed = CheckpointInferenceResult.model_validate(result)
    except (TypeError, ValueError) as error:
        logger.warning(
            "Inference result failed schema validation",
            path=str(result_path),
            error=str(error),
        )
        return None
    try:
        completed_before_started = parsed.completed < parsed.started
        actual_elapsed = (parsed.completed - parsed.started).total_seconds()
    except TypeError as error:
        # Pydantic accepts both offset-aware and offset-naive ISO timestamps,
        # but Python deliberately refuses to order or subtract a mixed pair.
        # Persisted evidence with incompatible timestamp domains is invalid; it
        # must trigger checkpoint recovery rather than abort resume discovery.
        logger.warning(
            "Inference result timestamps are not comparable",
            path=str(result_path),
            error=str(error),
        )
        return None
    if completed_before_started:
        logger.warning(
            "Inference result completed before it started",
            path=str(result_path),
        )
        return None
    if not math.isclose(
        parsed.elapsed,
        actual_elapsed,
        rel_tol=1e-9,
        abs_tol=0.01,
    ):
        logger.warning(
            "Inference result elapsed disagrees with its timestamps",
            path=str(result_path),
            elapsed=parsed.elapsed,
            timestamp_elapsed=actual_elapsed,
        )
        return None
    if not parsed.had_error and parsed.error_message:
        logger.warning(
            "Successful inference result contains an error message",
            path=str(result_path),
        )
        return None
    return parsed


def _generate_expected_prompt(
    problem_config: ProblemConfig,
    checkpoint: CheckpointConfig,
    prompt_template: str,
    environment: EnvironmentSpec,
    entry_file: str,
    *,
    is_first_checkpoint: bool,
) -> str:
    """Generate what the prompt SHOULD be for a checkpoint.

    Args:
        checkpoint: Checkpoint configuration
        prompt_template: Jinja2 template string for prompts
        environment: Environment specification
        entry_file: Entry file path
        is_first_checkpoint: Whether this is the first checkpoint

    Returns:
        Rendered prompt string
    """
    return render_prompt(
        spec_text=problem_config.get_checkpoint_spec(checkpoint.name),
        context={"is_continuation": not is_first_checkpoint},
        prompt_template=prompt_template,
        entry_file=environment.format_entry_file(entry_file),
        entry_command=environment.get_command(entry_file, is_agent_run=True),
    )


def _prompts_match(saved_prompt: str, expected_prompt: str) -> bool:
    """Compare prompts with whitespace normalization.

    Args:
        saved_prompt: The prompt that was saved to disk
        expected_prompt: The prompt generated from current config

    Returns:
        True if prompts match, False otherwise
    """
    return saved_prompt.strip() == expected_prompt.strip()


def _check_prompt_mismatch(
    problem_config: ProblemConfig,
    checkpoint_dir: Path,
    checkpoint_name: str,
    checkpoint_names: list[str],
    prompt_template: str,
    environment: EnvironmentSpec,
    entry_file: str,
    checkpoints: list[CheckpointConfig],
) -> bool:
    """Check if saved prompt matches expected prompt for a checkpoint.

    Args:
        checkpoint_dir: Directory containing checkpoint files
        checkpoint_name: Name of the checkpoint
        checkpoint_names: Ordered list of all checkpoint names
        prompt_template: Jinja2 template string
        environment: Environment specification
        entry_file: Entry file path
        checkpoints: List of checkpoint configurations

    Returns:
        True if there's a mismatch (should invalidate), False if prompts match
    """
    prompt_path = checkpoint_dir / PROMPT_FILENAME
    if not prompt_path.exists():
        # No saved prompt - checkpoint incomplete anyway
        return False

    try:
        saved_prompt = prompt_path.read_text()
    except OSError:
        # Can't read prompt - treat as incomplete
        return False

    checkpoint_config = next(
        (c for c in checkpoints if c.name == checkpoint_name), None
    )
    if checkpoint_config is None:
        # Can't find config - shouldn't happen, but don't invalidate
        return False

    is_first = checkpoint_name == checkpoint_names[0]
    expected = _generate_expected_prompt(
        problem_config,
        checkpoint_config,
        prompt_template,
        environment,
        entry_file,
        is_first_checkpoint=is_first,
    )

    if not _prompts_match(saved_prompt, expected):
        logger.info(
            "Prompt mismatch detected",
            checkpoint=checkpoint_name,
        )
        return True

    return False


def _detect_resume_from_artifacts(
    output_path: Path,
    checkpoint_names: list[str],
    problem_config: ProblemConfig | None = None,
    prompt_template: str | None = None,
    environment: EnvironmentSpec | None = None,
    entry_file: str | None = None,
    checkpoints: list[CheckpointConfig] | None = None,
    *,
    require_evaluation: bool = False,
) -> ResumeInfo | None:
    """Fallback resume detection when run_info.yaml is missing.

    Checks checkpoint directories for snapshots and inference results
    to determine resume point. Optionally validates prompts if parameters
    are provided.

    Args:
        output_path: Path to the problem's output directory
        checkpoint_names: Ordered list of checkpoint names from problem config
        prompt_template: Optional Jinja2 template for prompt validation
        environment: Optional environment spec for prompt validation
        entry_file: Optional entry file for prompt validation
        checkpoints: Optional checkpoint configs for prompt validation

    Returns:
        ResumeInfo if resumable state found, None if should start fresh
    """
    completed: list[str] = []
    evaluation_only: list[str] = []
    statuses: list[CheckpointStatus] = []
    can_validate_prompts = all(
        [problem_config, prompt_template, environment, entry_file, checkpoints]
    )
    first_invalid_reason: InvalidationReason | None = None

    for name in checkpoint_names:
        # If we've already found an invalid checkpoint, mark rest as dependent
        if first_invalid_reason is not None:
            statuses.append(
                CheckpointStatus(
                    name=name,
                    is_valid=False,
                    reason=InvalidationReason.DEPENDS_ON_INVALID,
                )
            )
            continue

        checkpoint_dir = output_path / name
        snapshot_dir = checkpoint_dir / SNAPSHOT_DIR_NAME

        # Force re-run if checkpoint directory is missing
        if not checkpoint_dir.exists():
            first_invalid_reason = InvalidationReason.MISSING_DIR
            statuses.append(
                CheckpointStatus(
                    name=name,
                    is_valid=False,
                    reason=InvalidationReason.MISSING_DIR,
                )
            )
            continue

        if not snapshot_dir.exists():
            first_invalid_reason = InvalidationReason.MISSING_SNAPSHOT
            statuses.append(
                CheckpointStatus(
                    name=name,
                    is_valid=False,
                    reason=InvalidationReason.MISSING_SNAPSHOT,
                )
            )
            continue

        # Snapshot exists - check inference result for errors
        result_path = checkpoint_dir / INFERENCE_RESULT_FILENAME
        if not result_path.exists():
            first_invalid_reason = InvalidationReason.MISSING_RESULT
            statuses.append(
                CheckpointStatus(
                    name=name,
                    is_valid=False,
                    reason=InvalidationReason.MISSING_RESULT,
                )
            )
            continue

        result = _load_inference_result(result_path)
        if result is None:
            first_invalid_reason = InvalidationReason.UNREADABLE_RESULT
            statuses.append(
                CheckpointStatus(
                    name=name,
                    is_valid=False,
                    reason=InvalidationReason.UNREADABLE_RESULT,
                )
            )
            continue

        if result.had_error:
            first_invalid_reason = InvalidationReason.HAD_ERROR
            statuses.append(
                CheckpointStatus(
                    name=name,
                    is_valid=False,
                    reason=InvalidationReason.HAD_ERROR,
                )
            )
            continue

        # Check prompt matches if validation is enabled
        if can_validate_prompts and _check_prompt_mismatch(
            problem_config,
            checkpoint_dir,
            name,
            checkpoint_names,
            prompt_template,  # type: ignore[arg-type]
            environment,  # type: ignore[arg-type]
            entry_file,  # type: ignore[arg-type]
            checkpoints,  # type: ignore[arg-type]
        ):
            first_invalid_reason = InvalidationReason.SPEC_CHANGED
            statuses.append(
                CheckpointStatus(
                    name=name,
                    is_valid=False,
                    reason=InvalidationReason.SPEC_CHANGED,
                )
            )
            continue

        # The solve is complete even if evaluation must be repaired. Evaluation
        # never feeds the agent, so later immutable snapshots remain valid.
        completed.append(name)
        if require_evaluation:
            evaluation_valid, evaluation_reason = _evaluation_status(
                checkpoint_dir
            )
            if not evaluation_valid:
                evaluation_only.append(name)
                statuses.append(
                    CheckpointStatus(
                        name=name,
                        is_valid=False,
                        reason=evaluation_reason,
                    )
                )
                continue
        statuses.append(CheckpointStatus(name=name, is_valid=True))

    # Find checkpoints whose solve (rather than only evaluation) is invalid.
    completed_set = set(completed)
    resume_from = None
    invalidated: list[str] = []
    for name in checkpoint_names:
        if name not in completed_set:
            if resume_from is None:
                resume_from = name
            invalidated.append(name)

    if not completed and not invalidated:
        # No checkpoint directories exist at all, start fresh
        return None

    if not resume_from:
        # All checkpoints completed - return ResumeInfo with empty resume_from
        prior_usage = _aggregate_prior_usage(output_path, completed)
        last_snapshot_dir = output_path / completed[-1] / SNAPSHOT_DIR_NAME
        return ResumeInfo(
            resume_from_checkpoint="",  # Empty string = nothing to resume
            completed_checkpoints=completed,
            last_snapshot_dir=last_snapshot_dir,
            prior_usage=prior_usage,
            checkpoint_statuses=statuses,
            invalidated_checkpoints=[],
            evaluation_only_checkpoints=evaluation_only,
        )

    # Aggregate usage and build ResumeInfo
    prior_usage = _aggregate_prior_usage(output_path, completed)
    last_snapshot_dir = (
        output_path / completed[-1] / SNAPSHOT_DIR_NAME if completed else None
    )

    logger.info(
        "Detected resume point from artifacts (no run_info.yaml)",
        resume_from=resume_from,
        completed_count=len(completed),
        invalidated_count=len(invalidated),
    )

    return ResumeInfo(
        resume_from_checkpoint=resume_from,
        completed_checkpoints=completed,
        last_snapshot_dir=last_snapshot_dir,
        prior_usage=prior_usage,
        checkpoint_statuses=statuses,
        invalidated_checkpoints=invalidated,
        evaluation_only_checkpoints=evaluation_only,
    )


def _usage_matches(
    saved: UsageTracker,
    artifact_usage: UsageTracker,
) -> bool:
    """Compare persisted aggregate usage with checkpoint-source evidence."""
    if not math.isclose(
        saved.cost,
        artifact_usage.cost,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        return False
    return (
        saved.steps == artifact_usage.steps
        and saved.net_tokens == artifact_usage.net_tokens
        and saved.current_tokens == artifact_usage.current_tokens
    )


def _completed_artifact_verdict(
    run_info: object,
    output_path: Path,
    checkpoint_names: list[str],
    *,
    require_evaluation: bool,
) -> bool | None:
    """Recompute a completed run verdict from checkpoint evidence.

    ``None`` means the saved policy is unavailable or invalid, which itself is
    a reason to rebuild metadata from the current immutable run specification.
    """
    if not require_evaluation:
        return True
    if not isinstance(run_info, dict):
        return None
    try:
        pass_policy = PassPolicy(run_info.get("pass_policy"))
    except (TypeError, ValueError):
        return None
    for checkpoint_name in checkpoint_names:
        try:
            evaluation = CorrectnessResults.from_dir(
                output_path / checkpoint_name
            )
        except (
            AttributeError,
            OSError,
            TypeError,
            ValueError,
            KeyError,
            json.JSONDecodeError,
        ):
            return None
        if evaluation.infrastructure_failure or not pass_policy.check(
            evaluation.pass_counts,
            evaluation.total_counts,
        ):
            return False
    return True


def _completed_state_is_consistent(
    summary: RunSummary,
    *,
    expected_verdict: bool,
) -> bool:
    """Reject impossible terminal-state combinations without erasing errors."""
    state = AgentStateEnum(summary.state)
    if state == AgentStateEnum.COMPLETED:
        return summary.error_type is None and summary.error_message is None
    if state == AgentStateEnum.FAILED:
        return not expected_verdict
    if state == AgentStateEnum.ERROR:
        # A cleanup/finalization error can coexist with otherwise complete,
        # valid checkpoint artifacts. Preserve it when metadata records the
        # actual error instead of treating every ERROR state as stale.
        return bool(
            summary.error_type
            or summary.error_message
            or summary.secondary_errors
        )
    return state == AgentStateEnum.HIT_RATE_LIMITED


def _completed_run_info_mismatches(
    summary: RunSummary,
    run_info: object,
    artifact_view: ResumeInfo,
    output_path: Path,
    checkpoint_names: list[str],
    *,
    require_evaluation: bool,
) -> list[str]:
    """Describe contradictions between a final summary and durable artifacts."""
    reasons: list[str] = []
    expected_states = dict.fromkeys(checkpoint_names, CheckpointState.RAN)
    if summary.checkpoints != expected_states:
        reasons.append("checkpoint states")
    if not _usage_matches(summary.total_usage, artifact_view.prior_usage):
        reasons.append("aggregate usage")

    expected_verdict = _completed_artifact_verdict(
        run_info,
        output_path,
        checkpoint_names,
        require_evaluation=require_evaluation,
    )
    if expected_verdict is None:
        reasons.append("pass policy")
        return reasons
    if summary.passed_policy is not expected_verdict:
        reasons.append("verdict")
    if not _completed_state_is_consistent(
        summary,
        expected_verdict=expected_verdict,
    ):
        reasons.append("terminal state")
    return reasons


def detect_resume_point(
    output_path: Path,
    checkpoint_names: list[str],
    problem_config: ProblemConfig | None = None,
    prompt_template: str | None = None,
    environment: EnvironmentSpec | None = None,
    entry_file: str | None = None,
    checkpoints: list[CheckpointConfig] | None = None,
    *,
    require_evaluation: bool = False,
) -> ResumeInfo | None:
    """Detect where to resume from based on existing output.

    Analyzes the output directory to find completed checkpoints and determine
    where to resume execution. Optionally validates prompts if parameters
    are provided.

    Args:
        output_path: Path to the problem's output directory
        checkpoint_names: Ordered list of checkpoint names from problem config
        prompt_template: Optional Jinja2 template for prompt validation
        environment: Optional environment spec for prompt validation
        entry_file: Optional entry file for prompt validation
        checkpoints: Optional checkpoint configs for prompt validation

    Returns:
        ResumeInfo if resumable state found, None if should start fresh
    """

    def fallback_to_artifacts(
        reason: str, **details: object
    ) -> ResumeInfo | None:
        logger.warning(
            "Falling back to checkpoint artifacts for resume detection",
            reason=reason,
            output_path=str(output_path),
            **details,
        )
        artifact_view = _detect_resume_from_artifacts(
            output_path,
            checkpoint_names,
            problem_config=problem_config,
            prompt_template=prompt_template,
            environment=environment,
            entry_file=entry_file,
            checkpoints=checkpoints,
            require_evaluation=require_evaluation,
        )
        if (
            artifact_view is not None
            and not artifact_view.resume_from_checkpoint
            and not artifact_view.evaluation_only_checkpoints
        ):
            # Complete checkpoint artifacts prevent duplicate model spend, but
            # absent or untrusted run metadata still needs a metadata-only
            # worker pass. Otherwise the CLI filters the problem as complete
            # and leaves downstream reporting with no trustworthy run_info.
            artifact_view.run_info_reconciliation_required = True
        return artifact_view

    run_info_path = output_path / RUN_INFO_FILENAME
    if not run_info_path.exists():
        logger.debug(
            "No run_info.yaml found, checking for artifacts",
            output_path=str(output_path),
        )
        return fallback_to_artifacts("run_info.yaml missing")

    try:
        with run_info_path.open() as f:
            run_info = yaml.safe_load(f)
    except (OSError, yaml.YAMLError) as e:
        return fallback_to_artifacts(
            "run_info.yaml unreadable",
            error=str(e),
        )

    validated_summary = _validate_saved_run_info(run_info)
    if validated_summary is None:
        return fallback_to_artifacts(
            "run_info.yaml failed schema or semantic validation"
        )
    checkpoint_states = validated_summary.checkpoints

    # ``run_info.yaml`` is written only at run finalization. A hard kill after
    # a resumed checkpoint durably writes its inference marker can therefore
    # leave a perfectly valid summary that still says "skipped" or "error".
    # Valid, prompt/config-bound checkpoint artifacts are the durable completion
    # record. Filesystem mtimes are deliberately irrelevant here: copying,
    # archiving, or timestamp normalization must never turn an already-paid
    # solve back into work.
    artifact_view = _detect_resume_from_artifacts(
        output_path,
        checkpoint_names,
        problem_config=problem_config,
        prompt_template=prompt_template,
        environment=environment,
        entry_file=entry_file,
        checkpoints=checkpoints,
        require_evaluation=require_evaluation,
    )
    artifact_completions_missing_from_run_info: list[str] = []
    if artifact_view is not None:
        for name in artifact_view.completed_checkpoints:
            checkpoint_dir = output_path / name
            state = checkpoint_states.get(name)
            evaluation_failure_recorded = (
                checkpoint_dir / EVALUATION_ERROR_FILENAME
            ).exists()
            metadata_solve_complete = state in {
                CheckpointState.RAN,
                CheckpointState.EVALUATION_ERROR,
            } or (
                state == CheckpointState.ERROR and evaluation_failure_recorded
            )
            if metadata_solve_complete:
                continue
            artifact_completions_missing_from_run_info.append(name)
    if artifact_completions_missing_from_run_info:
        logger.warning(
            "Complete checkpoint artifacts supersede stale run_info states",
            checkpoints=artifact_completions_missing_from_run_info,
            output_path=str(output_path),
        )
        if artifact_view is None:  # pragma: no cover - guarded above.
            raise AssertionError("artifact resume view unexpectedly missing")
        artifact_view.run_info_reconciliation_required = True
        return artifact_view

    if (
        artifact_view is not None
        and not artifact_view.resume_from_checkpoint
        and not artifact_view.evaluation_only_checkpoints
    ):
        mismatch_reasons = _completed_run_info_mismatches(
            validated_summary,
            run_info,
            artifact_view,
            output_path,
            checkpoint_names,
            require_evaluation=require_evaluation,
        )
        if mismatch_reasons:
            logger.warning(
                "Complete checkpoint artifacts contradict run_info summary",
                reasons=mismatch_reasons,
                output_path=str(output_path),
            )
            artifact_view.run_info_reconciliation_required = True
            return artifact_view

    can_validate_prompts = all(
        [problem_config, prompt_template, environment, entry_file, checkpoints]
    )

    # Find completed checkpoints and first incomplete one
    completed: list[str] = []
    evaluation_only: list[str] = []
    statuses: list[CheckpointStatus] = []
    first_invalid_reason: InvalidationReason | None = None

    for name in checkpoint_names:
        # If we've already found an invalid checkpoint, mark rest as dependent
        if first_invalid_reason is not None:
            statuses.append(
                CheckpointStatus(
                    name=name,
                    is_valid=False,
                    reason=InvalidationReason.DEPENDS_ON_INVALID,
                )
            )
            continue

        state = checkpoint_states.get(name)
        checkpoint_dir = output_path / name
        snapshot_dir = checkpoint_dir / SNAPSHOT_DIR_NAME

        # Force re-run if checkpoint directory is missing
        if not checkpoint_dir.exists():
            logger.debug(
                "Checkpoint directory missing, forcing re-run",
                checkpoint=name,
            )
            first_invalid_reason = InvalidationReason.MISSING_DIR
            statuses.append(
                CheckpointStatus(
                    name=name,
                    is_valid=False,
                    reason=InvalidationReason.MISSING_DIR,
                )
            )
            continue

        result_path = checkpoint_dir / INFERENCE_RESULT_FILENAME
        inference_valid = False
        if result_path.exists():
            inference_result = _load_inference_result(result_path)
            inference_valid = (
                inference_result is not None and not inference_result.had_error
            )
        evaluation_failure_recorded = (
            checkpoint_dir / EVALUATION_ERROR_FILENAME
        ).exists()
        solve_state_valid = state in {
            CheckpointState.RAN,
            CheckpointState.EVALUATION_ERROR,
        } or (state == CheckpointState.ERROR and evaluation_failure_recorded)

        if solve_state_valid and snapshot_dir.exists() and inference_valid:
            # Check prompt matches if validation is enabled
            if can_validate_prompts and _check_prompt_mismatch(
                problem_config,
                checkpoint_dir,
                name,
                checkpoint_names,
                prompt_template,  # type: ignore[arg-type]
                environment,  # type: ignore[arg-type]
                entry_file,  # type: ignore[arg-type]
                checkpoints,  # type: ignore[arg-type]
            ):
                # Prompt mismatch - invalidate this and all subsequent
                logger.debug(
                    "Checkpoint has prompt mismatch",
                    checkpoint=name,
                )
                first_invalid_reason = InvalidationReason.SPEC_CHANGED
                statuses.append(
                    CheckpointStatus(
                        name=name,
                        is_valid=False,
                        reason=InvalidationReason.SPEC_CHANGED,
                    )
                )
                continue

            logger.debug(
                "Checkpoint completed",
                checkpoint=name,
            )
            completed.append(name)
            if require_evaluation:
                evaluation_valid, evaluation_reason = _evaluation_status(
                    checkpoint_dir
                )
                if not evaluation_valid:
                    evaluation_only.append(name)
                    statuses.append(
                        CheckpointStatus(
                            name=name,
                            is_valid=False,
                            reason=evaluation_reason,
                        )
                    )
                    continue
            statuses.append(CheckpointStatus(name=name, is_valid=True))
        else:
            logger.debug(
                "Checkpoint not completed",
                checkpoint=name,
                state=state,
                snapshot_exists=snapshot_dir.exists(),
            )
            # Determine the specific reason
            if not snapshot_dir.exists():
                reason = InvalidationReason.MISSING_SNAPSHOT
            elif state == CheckpointState.ERROR or (
                result_path.exists() and not inference_valid
            ):
                reason = InvalidationReason.HAD_ERROR
            else:
                reason = InvalidationReason.MISSING_RESULT
            first_invalid_reason = reason
            statuses.append(
                CheckpointStatus(
                    name=name,
                    is_valid=False,
                    reason=reason,
                )
            )

    # Evaluation-only repairs preserve solve snapshots and do not invalidate
    # later checkpoints. Only solve failures participate in the dependency
    # chain and inference resume point.
    evaluation_only_set = set(evaluation_only)
    invalidated = [
        status.name
        for status in statuses
        if not status.is_valid and status.name not in evaluation_only_set
    ]

    if not invalidated:
        # All checkpoints completed - return ResumeInfo with empty resume_from
        logger.debug(
            "All checkpoints completed, no resume needed",
            completed_count=len(completed),
        )
        prior_usage = _aggregate_prior_usage(output_path, completed)
        last_snapshot_dir = output_path / completed[-1] / SNAPSHOT_DIR_NAME
        return ResumeInfo(
            resume_from_checkpoint="",  # Empty string = nothing to resume
            completed_checkpoints=completed,
            last_snapshot_dir=last_snapshot_dir,
            prior_usage=prior_usage,
            checkpoint_statuses=statuses,
            invalidated_checkpoints=[],
            evaluation_only_checkpoints=evaluation_only,
        )

    if not completed and not invalidated:
        # No checkpoint directories exist at all, start fresh
        logger.debug("No checkpoint directories found, starting fresh")
        return None

    resume_from = invalidated[0] if invalidated else ""

    # Calculate prior usage from completed checkpoints
    prior_usage = _aggregate_prior_usage(output_path, completed)

    # Get the snapshot from the last completed checkpoint (if any)
    last_snapshot_dir = (
        output_path / completed[-1] / SNAPSHOT_DIR_NAME if completed else None
    )

    logger.info(
        "Detected resume point",
        resume_from=resume_from,
        completed_count=len(completed),
        invalidated_count=len(invalidated),
        last_completed=completed[-1] if completed else None,
        prior_cost=prior_usage.cost,
        prior_steps=prior_usage.steps,
    )

    return ResumeInfo(
        resume_from_checkpoint=resume_from,
        completed_checkpoints=completed,
        last_snapshot_dir=last_snapshot_dir,
        prior_usage=prior_usage,
        checkpoint_statuses=statuses,
        invalidated_checkpoints=invalidated,
        evaluation_only_checkpoints=evaluation_only,
    )


def _aggregate_prior_usage(
    output_path: Path,
    completed: list[str],
) -> UsageTracker:
    """Aggregate usage from completed checkpoint inference results.

    Args:
        output_path: Path to the problem's output directory
        completed: List of completed checkpoint names

    Returns:
        UsageTracker with aggregated usage from all completed checkpoints
    """
    total_cost = 0.0
    total_steps = 0
    total_net_tokens = TokenUsage()
    total_current_tokens = TokenUsage()

    for checkpoint_name in completed:
        result_path = output_path / checkpoint_name / INFERENCE_RESULT_FILENAME
        if not result_path.exists():
            raise ValueError(
                "Completed checkpoint has no inference result: "
                f"{checkpoint_name}"
            )

        result = _load_inference_result(result_path)
        if result is None:
            raise ValueError(
                "Completed checkpoint has an invalid inference result: "
                f"{checkpoint_name}"
            )
        parsed_usage = result.usage
        total_cost += parsed_usage.cost
        total_steps += parsed_usage.steps

        total_net_tokens += parsed_usage.net_tokens
        total_current_tokens += parsed_usage.current_tokens

    return UsageTracker(
        cost=total_cost,
        steps=total_steps,
        net_tokens=total_net_tokens,
        current_tokens=total_current_tokens,
    )


def format_resume_summary(
    info: ResumeInfo, problem_name: str | None = None
) -> str:
    """Format a human-readable summary of resume detection results.

    Args:
        info: ResumeInfo from detect_resume_point
        problem_name: Optional problem name for the header

    Returns:
        Formatted string summarizing the resume state
    """
    reason_descriptions = {
        InvalidationReason.SPEC_CHANGED: "specification changed",
        InvalidationReason.HAD_ERROR: "previous run had error",
        InvalidationReason.MISSING_SNAPSHOT: "missing snapshot",
        InvalidationReason.MISSING_RESULT: "missing results",
        InvalidationReason.MISSING_DIR: "directory missing",
        InvalidationReason.UNREADABLE_RESULT: "unreadable results",
        InvalidationReason.DEPENDS_ON_INVALID: "depends on invalid checkpoint",
        InvalidationReason.MISSING_EVALUATION: "missing evaluation",
        InvalidationReason.EVALUATION_ERROR: "evaluation failed",
    }

    lines = []
    if problem_name:
        lines.append(f"{problem_name}:")
    lines.append(f"  Resume from: {info.resume_from_checkpoint}")
    lines.append(f"  Completed: {', '.join(info.completed_checkpoints)}")

    if info.invalidated_checkpoints:
        lines.append("  Will re-run:")
        for status in info.checkpoint_statuses:
            if status.name in info.invalidated_checkpoints and status.reason:
                desc = reason_descriptions.get(status.reason, "unknown reason")
                lines.append(f"    - {status.name} ({desc})")

    if info.evaluation_only_checkpoints:
        lines.append("  Will re-evaluate without inference:")
        for status in info.checkpoint_statuses:
            if (
                status.name in info.evaluation_only_checkpoints
                and status.reason
            ):
                desc = reason_descriptions.get(status.reason, "unknown reason")
                lines.append(f"    - {status.name} ({desc})")

    return "\n".join(lines)
