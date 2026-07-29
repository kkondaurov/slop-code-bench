"""Main entry point for run summary computation.

This module provides the orchestrating function that combines all aggregators
to produce a complete RunSummary.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from slop_code.metrics.models import RunSummary
from slop_code.metrics.summary import aggregators
from slop_code.metrics.summary.stats import group_by_problem


def _configured_problem_names(
    config: dict,
    expected_problem_names: Sequence[str] | None,
) -> list[str]:
    """Resolve a publishable problem denominator from configuration only."""
    raw_names: object = (
        expected_problem_names
        if expected_problem_names is not None
        else config.get("problems")
    )
    if (
        not isinstance(raw_names, Sequence)
        or isinstance(raw_names, str | bytes)
        or not raw_names
        or not all(isinstance(name, str) and name for name in raw_names)
    ):
        raise ValueError(
            "expected_problem_names or config['problems'] must provide a "
            "non-empty configured problem denominator"
        )
    return list(dict.fromkeys(raw_names))


def _reject_duplicate_checkpoint_identities(
    checkpoints: Sequence[dict[str, Any]],
) -> None:
    """Reject duplicate rows before any counts or metric totals are computed."""
    seen: dict[tuple[str, int], int] = {}
    for position, checkpoint in enumerate(checkpoints, start=1):
        problem = checkpoint.get("problem")
        index = checkpoint.get("idx")
        if not isinstance(problem, str) or not problem:
            raise ValueError(
                f"checkpoint row {position} has no valid problem identity"
            )
        if type(index) is not int or index < 1:
            raise ValueError(
                f"checkpoint row {position} has no valid checkpoint index"
            )
        identity = (problem, index)
        first_position = seen.get(identity)
        if first_position is not None:
            raise ValueError(
                "duplicate checkpoint identity "
                f"{problem}/checkpoint_{index} in rows "
                f"{first_position} and {position}"
            )
        seen[identity] = position


def compute_run_summary(
    config: dict,
    checkpoints: list[dict[str, Any]],
    expected_checkpoints: int,
    expected_problem_names: Sequence[str] | None = None,
) -> RunSummary:
    """Compute complete summary statistics from checkpoint data.

    Args:
        config: Run configuration dictionary.
        checkpoints: List of checkpoint data dictionaries.
        expected_checkpoints: Total checkpoints the run was configured
            to attempt (sum across the problem list). Used as the
            pct_checkpoints_* denominator; if the agent crashed, the
            produced count is less than this and missing checkpoints
            count as unsolved.
        expected_problem_names: Configured problem identities. Missing whole
            problems count as unsolved. Defaults to ``config['problems']``.
            Produced checkpoint rows are never used as the denominator.

    Returns:
        RunSummary with all computed statistics.
    """
    configured_names = _configured_problem_names(
        config,
        expected_problem_names,
    )

    configured_set = set(configured_names)
    included_checkpoints = [
        checkpoint
        for checkpoint in checkpoints
        if checkpoint.get("problem") in configured_set
    ]
    _reject_duplicate_checkpoint_identities(included_checkpoints)
    problems = group_by_problem(included_checkpoints)

    return RunSummary(
        model=config["model"]["name"],
        thinking=config["thinking"],
        prompt=Path(config["prompt_path"]).stem,
        agent_type=config["agent"]["type"],
        agent_version=config["agent"].get("version"),
        num_problems=len(problems),
        expected_problems=len(configured_names),
        num_checkpoints=len(included_checkpoints),
        expected_checkpoints=expected_checkpoints,
        costs=aggregators.compute_costs_stats(included_checkpoints, problems),
        time=aggregators.compute_time_stats(included_checkpoints, problems),
        tokens=aggregators.compute_tokens_stats(included_checkpoints, problems),
        steps=aggregators.compute_steps_stats(included_checkpoints, problems),
        **aggregators.compute_solve_rates(
            included_checkpoints,
            problems,
            expected_checkpoints,
            configured_names,
        ),
        pass_rates=aggregators.compute_pass_rates_stats(
            included_checkpoints, problems
        ),
        cc=aggregators.compute_cc_stats(included_checkpoints),
        ratios=aggregators.compute_ratios_stats(included_checkpoints),
        **aggregators.compute_composite_scores(included_checkpoints),
        scb_check=aggregators.compute_scb_check_coverage(
            included_checkpoints, expected_checkpoints
        ),
    )
