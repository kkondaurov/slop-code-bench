"""Agent execution wrapper for problem runner.

This module provides the function that executes an agent on a single problem.
"""

from __future__ import annotations

import json
import queue
from datetime import datetime
from pathlib import Path
from typing import Any

from slop_code import evaluation
from slop_code.agent_runner import AgentRunSpec
from slop_code.agent_runner import runner
from slop_code.agent_runner.agent import Agent
from slop_code.agent_runner.models import UsageTracker
from slop_code.agent_runner.reporting import AgentCheckpointSummary
from slop_code.agent_runner.reporting import MetricsTracker
from slop_code.agent_runner.resume import ResumeInfo
from slop_code.agent_runner.resume import detect_resume_point
from slop_code.agent_runner.resume import format_resume_summary
from slop_code.agent_runner.state import AgentStateEnum
from slop_code.common import QUARANTINED_CHECKPOINTS_DIR_NAME
from slop_code.common.atomic import atomic_write_text
from slop_code.common.llms import TokenUsage
from slop_code.entrypoints.problem_runner.models import RunTaskConfig
from slop_code.entrypoints.problem_runner.one_shot import apply_one_shot_mode
from slop_code.logging import get_logger

logger = get_logger(__name__)


def _send_completed_progress(
    problem_name: str,
    resume_info: ResumeInfo,
    output_path: Path,
    progress_queue: queue.Queue,
) -> None:
    """Send a progress update for a fully-completed problem so live stats are accurate."""
    now = datetime.now()
    metrics = MetricsTracker(
        state=AgentStateEnum.COMPLETED,
        current_checkpoint=resume_info.completed_checkpoints[-1]
        if resume_info.completed_checkpoints
        else "",
        usage=resume_info.prior_usage.model_copy(deep=True),
        started=now,
        checkpoint_started=now,
    )
    for checkpoint_name in resume_info.completed_checkpoints:
        checkpoint_dir = output_path / checkpoint_name
        eval_result = runner._load_eval_result(checkpoint_dir)
        metrics.record_checkpoint_result(checkpoint_name, eval_result)
    dummy_usage = UsageTracker(
        cost=0.0,
        steps=0,
        current_tokens=TokenUsage(),
        net_tokens=TokenUsage(),
    )
    progress_queue.put((problem_name, dummy_usage, metrics))


def _completed_run_passed_policy(
    resume_info: ResumeInfo,
    output_path: Path,
    config: RunTaskConfig,
) -> bool:
    """Reconstruct the saved run verdict instead of inventing a pass."""
    if config.disable_evaluation:
        return True
    for checkpoint_name in resume_info.completed_checkpoints:
        eval_result = runner._load_eval_result(output_path / checkpoint_name)
        if eval_result is None or eval_result.infrastructure_failure:
            return False
        if not config.pass_policy.check(
            eval_result.pass_counts,
            eval_result.total_counts,
        ):
            return False
    return True


def _build_run_spec(
    problem_config: evaluation.ProblemConfig,
    config: RunTaskConfig,
) -> AgentRunSpec:
    """Build the immutable per-problem run specification."""
    return AgentRunSpec(
        seed=config.seed,
        template=config.prompt_template,
        problem=problem_config,
        environment=config.env_spec,
        pass_policy=config.pass_policy,
        skip_evaluation=config.disable_evaluation,
        concurrent_evaluation=config.concurrent_evaluation,
        verbose=config.verbosity > 0,
        image=config.image,
        agent_type=config.agent_config.type,
        agent_version=config.agent_config.version,
        model_name=config.model_def.name,
    )


def _persist_completed_run_info_reconciliation(
    run_spec: AgentRunSpec,
    resume_info: ResumeInfo,
    output_path: Path,
) -> dict[str, Any]:
    """Rewrite stale metadata from schema-valid completed artifacts.

    Resume detection itself is intentionally read-only because it is also used
    by dry-run previews. A no-inference worker reaches this helper only when the
    artifact view proved that every configured solve is already complete.
    """
    now = datetime.now()
    metrics = MetricsTracker(
        state=AgentStateEnum.COMPLETED,
        current_checkpoint=(
            resume_info.completed_checkpoints[-1]
            if resume_info.completed_checkpoints
            else ""
        ),
        usage=resume_info.prior_usage.model_copy(deep=True),
        started=now,
        checkpoint_started=now,
    )
    completed = set(resume_info.completed_checkpoints)
    summaries: list[AgentCheckpointSummary] = []
    for (
        checkpoint_name,
        _checkpoint,
    ) in run_spec.problem.iterate_checkpoint_items():
        if checkpoint_name not in completed:
            raise ValueError(
                "Cannot reconcile run_info with an incomplete checkpoint: "
                f"{checkpoint_name}"
            )
        checkpoint_dir = output_path / checkpoint_name
        inference_result = runner._load_inference_result(  # noqa: SLF001
            checkpoint_dir / runner.common.INFERENCE_RESULT_FILENAME
        )
        if inference_result is None or inference_result.had_error:
            raise ValueError(
                "Cannot reconcile run_info from an invalid inference result: "
                f"{checkpoint_name}"
            )
        evaluation_result = None
        if not run_spec.skip_evaluation:
            evaluation_result = runner._load_eval_result(checkpoint_dir)
            if (
                evaluation_result is None
                or evaluation_result.infrastructure_failure
            ):
                raise ValueError(
                    "Cannot reconcile run_info from an invalid evaluation: "
                    f"{checkpoint_name}"
                )
        metrics.record_checkpoint_result(checkpoint_name, evaluation_result)
        summaries.append(
            AgentCheckpointSummary.from_results(
                checkpoint_name=checkpoint_name,
                path=checkpoint_dir,
                snapshot_dir=checkpoint_dir / runner.common.SNAPSHOT_DIR_NAME,
                artifacts=runner.get_artifacts_path(
                    checkpoint_dir,
                    compress=run_spec.compress_artifacts,
                ),
                usage=inference_result.usage,
                had_error=False,
                pass_policy=run_spec.pass_policy,
                evaluation_result=evaluation_result,
            )
        )
    return runner.reporting.save_results(
        summaries,
        metrics,
        run_spec,
        output_path,
    )


def _write_quarantine_manifest(
    batch_dir: Path,
    manifest: dict[str, object],
) -> None:
    target = batch_dir / "manifest.json"
    atomic_write_text(
        target,
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    )


def _quarantine_invalidated_checkpoints(
    output_path: Path,
    resume_info: ResumeInfo,
) -> Path | None:
    """Move invalid checkpoint evidence aside before a clean re-run.

    The hidden quarantine directory cannot be mistaken for a configured
    checkpoint, but remains under the problem output tree so provenance
    checksums bind the preserved evidence.

    Args:
        output_path: Base output directory for the problem
        resume_info: Resume information with invalidated checkpoints
    """
    if not resume_info.invalidated_checkpoints:
        return None
    existing_checkpoints = [
        checkpoint_name
        for checkpoint_name in resume_info.invalidated_checkpoints
        if (output_path / checkpoint_name).exists()
        or (output_path / checkpoint_name).is_symlink()
    ]
    if not existing_checkpoints:
        return None

    quarantine_root = output_path / QUARANTINED_CHECKPOINTS_DIR_NAME
    timestamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%f%z")
    batch_dir = quarantine_root / timestamp
    suffix = 1
    while batch_dir.exists():
        batch_dir = quarantine_root / f"{timestamp}-{suffix}"
        suffix += 1
    batch_dir.mkdir(parents=True)
    reason_by_checkpoint = {
        status.name: status.reason.value if status.reason is not None else None
        for status in resume_info.checkpoint_statuses
    }
    records: list[dict[str, object]] = []
    manifest: dict[str, object] = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "checkpoints": records,
    }
    _write_quarantine_manifest(batch_dir, manifest)

    logger.info(
        "Quarantining invalidated checkpoint directories",
        checkpoints=existing_checkpoints,
        destination=str(batch_dir),
    )

    for checkpoint_name in existing_checkpoints:
        checkpoint_dir = output_path / checkpoint_name
        destination = batch_dir / checkpoint_name
        checkpoint_dir.replace(destination)
        records.append(
            {
                "checkpoint": checkpoint_name,
                "reason": reason_by_checkpoint.get(checkpoint_name),
                "original_path": checkpoint_name,
                "quarantined_path": destination.relative_to(
                    output_path
                ).as_posix(),
            }
        )
        _write_quarantine_manifest(batch_dir, manifest)
        logger.debug(
            "Quarantined checkpoint directory",
            checkpoint=checkpoint_name,
            source=str(checkpoint_dir),
            destination=str(destination),
        )
    return batch_dir


def run_agent_on_problem(
    problem_config: evaluation.ProblemConfig,
    problem_name: str,
    config: RunTaskConfig,
    progress_queue: queue.Queue,
    output_path: Path,
) -> dict[str, Any]:
    """Execute an agent on a problem with progress reporting.

    Creates an AgentRunSpec from the config and runs the agent with
    progress updates sent to the queue.

    Args:
        problem_config: Problem configuration
        problem_name: Name of the problem
        config: Shared execution configuration
        progress_queue: Queue for progress updates
        output_path: Directory for output files

    Returns:
        Dictionary containing the run results including summary with state,
        passed_policy, and any error information.
    """
    problem_config = apply_one_shot_mode(
        problem_config=problem_config, one_shot=config.one_shot
    )

    # Detect resume point if resume mode is enabled
    resume_info: ResumeInfo | None = None
    if config.resume:
        checkpoint_items = list(problem_config.iterate_checkpoint_items())
        checkpoint_names = [name for name, _ in checkpoint_items]
        checkpoints = [cp for _, cp in checkpoint_items]
        resume_info = detect_resume_point(
            output_path,
            checkpoint_names,
            problem_config=problem_config,
            prompt_template=config.prompt_template,
            environment=config.env_spec,
            entry_file=problem_config.entry_file,
            checkpoints=checkpoints,
            require_evaluation=not config.disable_evaluation,
        )
        if resume_info:
            # Check if all checkpoints are already completed
            if (
                not resume_info.resume_from_checkpoint
                and not resume_info.evaluation_only_checkpoints
            ):
                logger.info(
                    "All checkpoints completed, skipping",
                    problem=problem_name,
                    completed=len(resume_info.completed_checkpoints),
                )
                _send_completed_progress(
                    problem_name, resume_info, output_path, progress_queue
                )
                reconciled_run_info: dict[str, Any] | None = None
                if resume_info.run_info_reconciliation_required:
                    reconciled_run_info = (
                        _persist_completed_run_info_reconciliation(
                            _build_run_spec(problem_config, config),
                            resume_info,
                            output_path,
                        )
                    )
                passed_policy = (
                    bool(reconciled_run_info["summary"]["passed_policy"])
                    if reconciled_run_info is not None
                    else _completed_run_passed_policy(
                        resume_info,
                        output_path,
                        config,
                    )
                )
                return {
                    "summary": {
                        "state": AgentStateEnum.COMPLETED.value,
                        "passed_policy": passed_policy,
                        "total_cost": resume_info.prior_usage.cost,
                        "total_steps": resume_info.prior_usage.steps,
                        "total_usage": resume_info.prior_usage.model_dump(),
                        "checkpoints": dict.fromkeys(
                            resume_info.completed_checkpoints,
                            "ran",
                        ),
                    }
                }

            # Log detailed resume summary
            summary = format_resume_summary(resume_info, problem_name)
            logger.info(
                "Resuming from checkpoint",
                problem=problem_name,
                checkpoint=resume_info.resume_from_checkpoint,
                completed=len(resume_info.completed_checkpoints),
                invalidated=resume_info.invalidated_checkpoints,
                summary=summary,
            )

            # Preserve invalidated evidence outside checkpoint discovery.
            _quarantine_invalidated_checkpoints(output_path, resume_info)

    run_spec = _build_run_spec(problem_config, config)

    return runner.run_agent(
        run_spec=run_spec,
        agent=Agent.from_config(
            config.agent_config,
            model=config.model_def,
            credential=config.credential,
            problem_name=problem_name,
            verbose=run_spec.verbose,
            image=config.image,
            thinking_preset=config.thinking_preset,
            thinking_max_tokens=config.thinking_max_tokens,
        ),
        output_path=output_path,
        progress_queue=progress_queue,
        resume_info=resume_info,
    )
