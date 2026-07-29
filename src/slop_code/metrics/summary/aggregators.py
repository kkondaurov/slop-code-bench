"""Per-category aggregation functions for run summary computation.

This module provides functions that compute specific categories of
aggregate statistics from checkpoint data.
"""

from __future__ import annotations

import math
import statistics
from typing import Any

from slop_code.logging import get_logger
from slop_code.metrics.checkpoint.driver import SCB_CHECK_NAME
from slop_code.metrics.checkpoint.driver import SCB_CHECK_VERSION
from slop_code.metrics.checkpoint.driver import is_valid_scb_check_metadata
from slop_code.metrics.models import CostsStats
from slop_code.metrics.models import CyclomaticComplexityStats
from slop_code.metrics.models import MetricStats
from slop_code.metrics.models import PassRatesByType
from slop_code.metrics.models import PassRatesStats
from slop_code.metrics.models import RatiosStats
from slop_code.metrics.models import ScbCheckCoverage
from slop_code.metrics.models import StepsStats
from slop_code.metrics.models import TimeStats
from slop_code.metrics.models import TokenMeans
from slop_code.metrics.models import TokenStats
from slop_code.metrics.summary.stats import compute_metric_stats
from slop_code.metrics.summary.stats import compute_pass_rate
from slop_code.metrics.summary.stats import compute_ratio_values
from slop_code.metrics.summary.stats import extract_metric_values

logger = get_logger(__name__)


def safe_mean(values: list[float]) -> float:
    """Compute mean of values, returning 0.0 for empty lists."""
    return statistics.mean(values) if values else 0.0


def aggregate_per_problem(
    problems: dict[str, list[dict[str, Any]]],
    key: str,
    default: float = 0,
) -> list[float]:
    """Sum values for a given key across all checkpoints within each problem."""
    return [
        sum(c.get(key, default) for c in chkpts) for chkpts in problems.values()
    ]


def compute_costs_stats(
    checkpoints: list[dict[str, Any]],
    problems: dict[str, list[dict[str, Any]]],
) -> CostsStats:
    """Compute cost statistics at checkpoint and problem levels."""
    checkpoint_costs = extract_metric_values(checkpoints, "cost")
    problem_costs = aggregate_per_problem(problems, "cost")

    return CostsStats(
        checkpoint=compute_metric_stats(checkpoint_costs),
        problem=compute_metric_stats(problem_costs),
        total=sum(checkpoint_costs) if checkpoint_costs else 0.0,
    )


def compute_time_stats(
    checkpoints: list[dict[str, Any]],
    problems: dict[str, list[dict[str, Any]]],
) -> TimeStats:
    """Compute time statistics at checkpoint and problem levels."""
    checkpoint_times = extract_metric_values(checkpoints, "duration")
    problem_times = [
        sum(float(c["duration"]) for c in chkpts if "duration" in c)
        for chkpts in problems.values()
        if any("duration" in c for c in chkpts)
    ]

    return TimeStats(
        checkpoint=compute_metric_stats(checkpoint_times),
        problem=compute_metric_stats(problem_times),
    )


def compute_tokens_stats(
    checkpoints: list[dict[str, Any]],
    problems: dict[str, list[dict[str, Any]]],
) -> TokenStats:
    """Compute token statistics: totals and per-level means."""
    token_types = ["input", "output", "cache_read", "cache_write", "reasoning"]

    # Extract checkpoint-level values and compute means
    checkpoint_values = {
        t: extract_metric_values(checkpoints, t) for t in token_types
    }
    checkpoint_means = TokenMeans(
        **{t: safe_mean(checkpoint_values[t]) for t in token_types}
    )

    # Aggregate per-problem totals and compute means
    problem_totals = {
        t: aggregate_per_problem(problems, t) for t in token_types
    }
    problem_means = TokenMeans(
        **{t: safe_mean(problem_totals[t]) for t in token_types}
    )

    return TokenStats(
        **{
            t: int(sum(checkpoint_values[t])) if checkpoint_values[t] else 0
            for t in token_types
        },
        checkpoint=checkpoint_means,
        problem=problem_means,
    )


def compute_steps_stats(
    checkpoints: list[dict[str, Any]],
    problems: dict[str, list[dict[str, Any]]],
) -> StepsStats:
    """Compute step statistics at checkpoint and problem levels."""
    checkpoint_steps = extract_metric_values(checkpoints, "steps")
    problem_steps = aggregate_per_problem(problems, "steps")

    return StepsStats(
        checkpoint=compute_metric_stats(checkpoint_steps),
        problem=compute_metric_stats(problem_steps),
    )


def _is_problem_fully_solved(chkpts: list[dict[str, Any]]) -> bool:
    rates = [c.get("strict_pass_rate", 0.0) for c in chkpts]
    if not rates or not all(math.isclose(pr, 1.0) for pr in rates):
        return False

    indices = [checkpoint.get("idx") for checkpoint in chkpts]
    if any(
        type(index) is not int or index < 1  # type: ignore[operator]
        for index in indices
    ):
        return False
    typed_indices = [int(index) for index in indices]
    if len(set(typed_indices)) != len(typed_indices):
        return False
    last_index = max(typed_indices)
    if set(typed_indices) != set(range(1, last_index + 1)):
        return False

    if any("is_last" in c for c in chkpts):
        terminal_indices = [
            int(checkpoint["idx"])
            for checkpoint in chkpts
            if checkpoint.get("is_last") is True
        ]
        return terminal_indices == [last_index]

    return True


def compute_solve_rates(
    checkpoints: list[dict[str, Any]],
    problems: dict[str, list[dict[str, Any]]],
    expected_checkpoints: int,
    expected_problem_names: list[str],
) -> dict[str, float | None]:
    """Compute solve rate percentages normalized to the benchmark total.

    Checkpoint and problem percentages use configured denominators. Using only
    produced records would let crashes shrink either denominator and inflate
    the result.

    Args:
        checkpoints: Checkpoint records actually produced by the run.
        problems: Checkpoints grouped by problem (for problem-level stats).
        expected_checkpoints: Total checkpoints configured for the run.
        expected_problem_names: Ordered, unique configured problem names.

    Returns:
        Dict with raw counts and percentages.
    """
    if expected_checkpoints <= 0:
        raise ValueError(
            f"expected_checkpoints must be positive, got {expected_checkpoints}"
        )
    pass_rates_list = extract_metric_values(checkpoints, "strict_pass_rate")
    iso_pass_rates_list = extract_metric_values(
        checkpoints, "isolated_pass_rate"
    )
    core_pass_rates_list = extract_metric_values(checkpoints, "core_pass_rate")

    if not pass_rates_list or not iso_pass_rates_list:
        return {}
    if not expected_problem_names:
        raise ValueError("expected_problem_names must not be empty")

    # Count checkpoints that fully pass (rate = 1.0)
    checkpoints_solved = sum(
        1 for pr in pass_rates_list if math.isclose(pr, 1.0)
    )
    iso_solved = sum(1 for pr in iso_pass_rates_list if math.isclose(pr, 1.0))
    core_solved = sum(1 for pr in core_pass_rates_list if math.isclose(pr, 1.0))

    # Count problems fully vs partially solved
    fully_solved = sum(
        1
        for name in expected_problem_names
        if _is_problem_fully_solved(problems.get(name, []))
    )
    partially_solved = sum(
        1
        for name in expected_problem_names
        if (chkpts := problems.get(name, []))
        if (rates := [c.get("strict_pass_rate", 0.0) for c in chkpts])
        and any(pr == 1.0 for pr in rates)
    )

    num_problems = len(expected_problem_names)

    return {
        "pct_checkpoints_solved": (checkpoints_solved / expected_checkpoints)
        * 100,
        "pct_checkpoints_iso_solved": (iso_solved / expected_checkpoints) * 100,
        "pct_checkpoints_core_solved": (core_solved / expected_checkpoints)
        * 100,
        "pct_problems_solved": (fully_solved / num_problems) * 100,
        "pct_problems_partial": (partially_solved / num_problems) * 100,
        "problem_solved": fully_solved,
        "problem_partial": partially_solved,
        "checkpoints_solved": checkpoints_solved,
        "checkpoints_iso_solved": iso_solved,
        "checkpoints_core_solved": core_solved,
    }


def compute_pass_rates_stats(
    checkpoints: list[dict[str, Any]],
    problems: dict[str, list[dict[str, Any]]],
) -> PassRatesStats:
    """Compute pass rate statistics by test type at checkpoint and problem levels.

    Note: Checkpoints with no tests for a given type (e.g., no error tests) are
    excluded from the average for that type.
    """
    test_types = ["core", "total", "error", "functionality", "regression"]

    def compute_pass_rates_by_type(
        chkpt: dict[str, Any],
    ) -> dict[str, float | None]:
        """Extract pass rates for all test types from a single checkpoint."""
        return {
            "total": compute_pass_rate(
                chkpt.get("passed_tests"), chkpt.get("total_tests")
            ),
            **{
                t: compute_pass_rate(
                    chkpt.get(f"{t}_passed"), chkpt.get(f"{t}_total")
                )
                for t in ["core", "error", "functionality", "regression"]
            },
        }

    def mean_excluding_none(values: list[float | None]) -> float:
        """Compute mean of non-None values, returning 0.0 if all are None."""
        valid = [v for v in values if v is not None]
        return statistics.mean(valid) if valid else 0.0

    # Collect checkpoint-level pass rates (excluding None for checkpoints with 0 tests)
    checkpoint_pass_rates: dict[str, list[float | None]] = {
        t: [] for t in test_types
    }
    for chkpt in checkpoints:
        checkpoint_rates = compute_pass_rates_by_type(chkpt)
        for test_type in test_types:
            checkpoint_pass_rates[test_type].append(checkpoint_rates[test_type])

    # Collect problem-level pass rates (mean across checkpoints per problem, excluding None)
    problem_pass_rates: dict[str, list[float]] = {t: [] for t in test_types}
    for problem_chkpts in problems.values():
        for test_type in test_types:
            rates_for_problem = [
                compute_pass_rates_by_type(c)[test_type] for c in problem_chkpts
            ]
            problem_pass_rates[test_type].append(
                mean_excluding_none(rates_for_problem)
            )

    return PassRatesStats(
        checkpoint=PassRatesByType(
            **{
                t: mean_excluding_none(checkpoint_pass_rates[t])
                for t in test_types
            }
        ),
        problem=PassRatesByType(
            **{t: safe_mean(problem_pass_rates[t]) for t in test_types}
        ),
    )


def compute_cc_stats(
    checkpoints: list[dict[str, Any]],
) -> CyclomaticComplexityStats:
    """Compute CC (cyclomatic complexity) statistics."""
    high_counts = extract_metric_values(checkpoints, "cc_high_count")
    high_means = extract_metric_values(checkpoints, "high_cc_mean")
    max_values = extract_metric_values(checkpoints, "cc_max")

    return CyclomaticComplexityStats(
        high_count=compute_metric_stats(high_counts),
        high_mean=compute_metric_stats(high_means),
        max=compute_metric_stats(max_values),
    )


def compute_ratios_stats(checkpoints: list[dict[str, Any]]) -> RatiosStats:
    """Compute quality ratio statistics (per LOC)."""
    rubric_ratios = compute_ratio_values(
        checkpoints, "rubric_total_flags", "loc"
    )
    lint_ratios = compute_ratio_values(checkpoints, "lint_errors", "loc")
    violation_pct_values = extract_metric_values(checkpoints, "violation_pct")

    return RatiosStats(
        rubric=compute_metric_stats(rubric_ratios),
        lint=compute_metric_stats(lint_ratios),
        violation_pct=compute_metric_stats(violation_pct_values),
    )


def compute_composite_scores(
    checkpoints: list[dict[str, Any]],
) -> dict[str, MetricStats]:
    """Aggregate only evaluator-owned, fully evidenced composite scores."""
    measured = [
        checkpoint
        for checkpoint in checkpoints
        if is_valid_scb_check_metadata(checkpoint.get("scb_check"))
    ]
    verbosity_values = [
        float(value)
        for checkpoint in measured
        if isinstance((value := checkpoint.get("verbosity")), int | float)
        and not isinstance(value, bool)
    ]
    erosion_values = [
        float(value)
        for checkpoint in measured
        if isinstance((value := checkpoint.get("erosion")), int | float)
        and not isinstance(value, bool)
    ]

    return {
        "verbosity": compute_metric_stats(verbosity_values),
        "erosion": compute_metric_stats(erosion_values),
    }


def compute_scb_check_coverage(
    checkpoints: list[dict[str, Any]],
    expected_checkpoints: int,
) -> ScbCheckCoverage:
    """Count explicit quality measurements against the configured total."""
    measured = 0
    failed = 0
    missing_snapshot = 0
    missing_metadata = 0
    resolved_versions: set[str] = set()

    for checkpoint in checkpoints:
        metadata = checkpoint.get("scb_check")
        if not isinstance(metadata, dict):
            missing_metadata += 1
            continue

        resolved = metadata.get("resolved_version")
        if isinstance(resolved, str) and resolved:
            resolved_versions.add(resolved)

        status = metadata.get("status")
        if is_valid_scb_check_metadata(metadata):
            measured += 1
        elif status in {"measured", "failed"}:
            failed += 1
        elif status == "missing_snapshot":
            missing_snapshot += 1
        else:
            missing_metadata += 1

    missing_records = max(expected_checkpoints - len(checkpoints), 0)
    unmeasured = max(expected_checkpoints - measured, 0)
    coverage_pct = (
        measured / expected_checkpoints * 100
        if expected_checkpoints > 0
        else 0.0
    )
    return ScbCheckCoverage(
        evaluator=SCB_CHECK_NAME,
        requested_version=SCB_CHECK_VERSION,
        resolved_versions=sorted(resolved_versions),
        measured_checkpoints=measured,
        expected_checkpoints=expected_checkpoints,
        failed_checkpoints=failed,
        missing_snapshot_checkpoints=missing_snapshot,
        missing_metadata_checkpoints=missing_metadata,
        missing_checkpoint_records=missing_records,
        unmeasured_checkpoints=unmeasured,
        coverage_pct=coverage_pct,
    )
