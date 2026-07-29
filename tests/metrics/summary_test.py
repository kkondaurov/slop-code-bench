"""Tests for run summary computation."""

from __future__ import annotations

import pytest

from slop_code.metrics.checkpoint import driver as checkpoint_driver
from slop_code.metrics.summary import (
    compute_run_summary as _compute_run_summary,
)
from slop_code.metrics.summary import save_summary_json


@pytest.fixture
def mock_config() -> dict:
    """Return a minimal config dict for compute_run_summary."""
    return {
        "model": {"name": "test-model"},
        "thinking": "none",
        "prompt_path": "test.jinja",
        "agent": {"type": "test-agent", "version": "1.0"},
    }


def _measured_scb_check() -> dict:
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


def compute_run_summary(
    config: dict,
    checkpoints: list[dict],
    expected_checkpoints: int,
    expected_problem_names: list[str] | None = None,
):
    """Supply explicit denominators in tests that focus on other metrics."""
    if expected_problem_names is None and not config.get("problems"):
        expected_problem_names = list(
            dict.fromkeys(
                checkpoint["problem"]
                for checkpoint in checkpoints
                if checkpoint.get("problem")
            )
        )
    return _compute_run_summary(
        config,
        checkpoints,
        expected_checkpoints,
        expected_problem_names=expected_problem_names,
    )


class TestRunSummaryDeltaRemoval:
    """Tests for summary contract after delta removal."""

    def test_does_not_serialize_delta_stats(self, mock_config):
        checkpoints = [
            {
                "problem": "test",
                "idx": 1,
                "strict_pass_rate": 0.8,
                "delta.loc": 50.0,
                "delta.ast_grep_violations": -20.0,
                "delta.churn_ratio": 0.1,
            }
        ]

        summary = compute_run_summary(
            mock_config,
            checkpoints,
            expected_checkpoints=max(len(checkpoints), 1),
        )

        assert "delta" not in summary.model_dump()


class TestRunSummaryCounts:
    """Tests for count aggregation in compute_run_summary."""

    def test_counts_problems_and_checkpoints(self, mock_config):
        """Test that summary counts problems and checkpoints correctly."""
        checkpoints = [
            {"problem": "prob1", "idx": 1, "strict_pass_rate": 0.8},
            {"problem": "prob1", "idx": 2, "strict_pass_rate": 1.0},
            {"problem": "prob2", "idx": 1, "strict_pass_rate": 0.5},
        ]
        summary = compute_run_summary(
            mock_config,
            checkpoints,
            expected_checkpoints=max(len(checkpoints), 1),
        )

        assert summary.num_problems == 2
        assert summary.expected_problems == 2
        assert summary.num_checkpoints == 3

    def test_missing_problem_denominator_is_rejected(self, mock_config):
        """Produced rows cannot silently define a publishable denominator."""
        with pytest.raises(ValueError, match="configured problem denominator"):
            _compute_run_summary(
                mock_config,
                [{"problem": "produced", "idx": 1}],
                expected_checkpoints=1,
            )

    def test_configured_absent_problem_is_unsolved_and_stale_rows_are_ignored(
        self, mock_config
    ):
        """The configured suite, not produced rows, defines the denominator."""
        mock_config["problems"] = ["produced", "absent"]
        checkpoints = [
            {
                "problem": "produced",
                "idx": 1,
                "is_last": True,
                "strict_pass_rate": 1.0,
                "isolated_pass_rate": 1.0,
            },
            {
                "problem": "stale-from-another-selection",
                "idx": 1,
                "is_last": True,
                "strict_pass_rate": 1.0,
                "isolated_pass_rate": 1.0,
                "cost": 99.0,
            },
        ]

        summary = compute_run_summary(
            mock_config,
            checkpoints,
            expected_checkpoints=2,
        )

        assert summary.expected_problems == 2
        assert summary.num_problems == 1
        assert summary.num_checkpoints == 1
        assert summary.problem_solved == 1
        assert summary.pct_problems_solved == 50.0
        assert summary.pct_checkpoints_solved == 50.0
        assert summary.costs.total == 0.0


class TestRunSummaryCosts:
    """Tests for cost aggregation in compute_run_summary."""

    def test_aggregates_costs(self, mock_config):
        """Test that summary aggregates costs correctly."""
        checkpoints = [
            {
                "problem": "prob1",
                "idx": 1,
                "cost": 0.10,
                "strict_pass_rate": 1.0,
            },
            {
                "problem": "prob1",
                "idx": 2,
                "cost": 0.20,
                "strict_pass_rate": 1.0,
            },
            {
                "problem": "prob2",
                "idx": 1,
                "cost": 0.15,
                "strict_pass_rate": 1.0,
            },
        ]
        summary = compute_run_summary(
            mock_config,
            checkpoints,
            expected_checkpoints=max(len(checkpoints), 1),
        )

        assert abs(summary.costs.total - 0.45) < 0.001
        assert abs(summary.costs.checkpoint.mean - 0.15) < 0.001
        # Problem costs: prob1=0.30, prob2=0.15
        assert abs(summary.costs.problem.mean - 0.225) < 0.001

    def test_summary_json_rejects_non_finite_values(
        self, mock_config, tmp_path
    ):
        summary = compute_run_summary(
            mock_config,
            [
                {
                    "problem": "prob1",
                    "idx": 1,
                    "cost": float("nan"),
                    "strict_pass_rate": 1.0,
                    "isolated_pass_rate": 1.0,
                }
            ],
            expected_checkpoints=1,
        )

        with pytest.raises(ValueError, match="JSON compliant"):
            save_summary_json(summary, tmp_path)


class TestRunSummarySolveRates:
    """Tests for solve rate computation in compute_run_summary."""

    def test_computes_checkpoint_solve_rate(self, mock_config):
        """Test that summary computes checkpoint solve rate correctly."""
        checkpoints = [
            {
                "problem": "prob1",
                "idx": 1,
                "strict_pass_rate": 1.0,
                "isolated_pass_rate": 1.0,
            },
            {
                "problem": "prob1",
                "idx": 2,
                "strict_pass_rate": 0.8,
                "isolated_pass_rate": 0.8,
            },
            {
                "problem": "prob2",
                "idx": 1,
                "strict_pass_rate": 1.0,
                "isolated_pass_rate": 1.0,
            },
            {
                "problem": "prob2",
                "idx": 2,
                "strict_pass_rate": 1.0,
                "isolated_pass_rate": 1.0,
            },
        ]
        summary = compute_run_summary(
            mock_config,
            checkpoints,
            expected_checkpoints=max(len(checkpoints), 1),
        )

        # 3 out of 4 checkpoints have pass_rate == 1.0
        assert summary.pct_checkpoints_solved == 75.0

    def test_computes_problem_solve_rate(self, mock_config):
        """Test that summary computes problem solve rate correctly."""
        checkpoints = [
            {
                "problem": "prob1",
                "idx": 1,
                "strict_pass_rate": 1.0,
                "isolated_pass_rate": 1.0,
            },
            {
                "problem": "prob1",
                "idx": 2,
                "strict_pass_rate": 1.0,
                "isolated_pass_rate": 1.0,
            },  # fully solved
            {
                "problem": "prob2",
                "idx": 1,
                "strict_pass_rate": 0.8,
                "isolated_pass_rate": 0.8,
            },
            {
                "problem": "prob2",
                "idx": 2,
                "strict_pass_rate": 0.9,
                "isolated_pass_rate": 0.9,
            },  # not fully solved
        ]
        summary = compute_run_summary(
            mock_config,
            checkpoints,
            expected_checkpoints=max(len(checkpoints), 1),
        )

        # Only 1 problem fully solved
        assert summary.pct_problems_solved == 50.0

    def test_incomplete_problem_is_not_fully_solved(self, mock_config):
        """Missing later checkpoints prevent a problem from being fully solved."""
        checkpoints = [
            {
                "problem": "crashed_after_first",
                "idx": 1,
                "is_last": False,
                "strict_pass_rate": 1.0,
                "isolated_pass_rate": 1.0,
            },
            {
                "problem": "finished",
                "idx": 1,
                "is_last": False,
                "strict_pass_rate": 1.0,
                "isolated_pass_rate": 1.0,
            },
            {
                "problem": "finished",
                "idx": 2,
                "is_last": True,
                "strict_pass_rate": 1.0,
                "isolated_pass_rate": 1.0,
            },
        ]
        summary = compute_run_summary(
            mock_config,
            checkpoints,
            expected_checkpoints=4,
        )

        assert summary.problem_solved == 1
        assert summary.pct_problems_solved == 50.0

    @pytest.mark.parametrize("indices", [[1, 3]])
    def test_missing_or_duplicate_checkpoint_identity_is_not_fully_solved(
        self, mock_config, indices
    ):
        """Passing rows cannot hide a gap or duplicate in checkpoint identity."""
        checkpoints = [
            {
                "problem": "incomplete",
                "idx": index,
                "is_last": position == len(indices) - 1,
                "strict_pass_rate": 1.0,
                "isolated_pass_rate": 1.0,
            }
            for position, index in enumerate(indices)
        ]
        mock_config["problems"] = ["incomplete"]

        summary = compute_run_summary(
            mock_config,
            checkpoints,
            expected_checkpoints=3,
        )

        assert summary.problem_solved == 0
        assert summary.pct_problems_solved == 0.0
        assert summary.problem_partial == 1
        assert summary.pct_problems_partial == 100.0

    def test_duplicate_checkpoint_identity_is_rejected_before_aggregation(
        self, mock_config
    ):
        checkpoints = [
            {
                "problem": "duplicated",
                "idx": 1,
                "strict_pass_rate": 1.0,
                "isolated_pass_rate": 1.0,
                "cost": cost,
            }
            for cost in (1.0, 2.0)
        ]

        with pytest.raises(ValueError, match="duplicate checkpoint identity"):
            compute_run_summary(
                mock_config,
                checkpoints,
                expected_checkpoints=2,
            )

    def test_computes_partial_solve_rate(self, mock_config):
        """Test that summary computes partial solve rate correctly."""
        checkpoints = [
            {
                "problem": "prob1",
                "idx": 1,
                "strict_pass_rate": 1.0,
                "isolated_pass_rate": 1.0,
            },
            {
                "problem": "prob1",
                "idx": 2,
                "strict_pass_rate": 0.5,
                "isolated_pass_rate": 0.5,
            },  # has at least one 1.0
            {
                "problem": "prob2",
                "idx": 1,
                "strict_pass_rate": 0.8,
                "isolated_pass_rate": 0.8,
            },
            {
                "problem": "prob2",
                "idx": 2,
                "strict_pass_rate": 0.9,
                "isolated_pass_rate": 0.9,
            },  # no 1.0
        ]
        summary = compute_run_summary(
            mock_config,
            checkpoints,
            expected_checkpoints=max(len(checkpoints), 1),
        )

        # Only prob1 has at least one pass_rate == 1.0
        assert summary.pct_problems_partial == 50.0


class TestRunSummaryPassRates:
    """Tests for pass rate aggregation in compute_run_summary."""

    def test_computes_pass_rates_by_type(self, mock_config):
        """Test that summary computes pass rates by test type."""
        checkpoints = [
            {
                "problem": "prob1",
                "idx": 1,
                "strict_pass_rate": 0.8,
                "total_tests": 10,
                "passed_tests": 8,
                "core_total": 5,
                "core_passed": 5,
                "functionality_total": 3,
                "functionality_passed": 2,
                "error_total": 2,
                "error_passed": 1,
                "regression_total": 0,
                "regression_passed": 0,
            },
        ]
        summary = compute_run_summary(
            mock_config,
            checkpoints,
            expected_checkpoints=max(len(checkpoints), 1),
        )

        assert summary.pass_rates.checkpoint.total == 0.8
        assert summary.pass_rates.checkpoint.core == 1.0
        assert abs(summary.pass_rates.checkpoint.functionality - 2 / 3) < 0.01
        assert summary.pass_rates.checkpoint.error == 0.5

    def test_pass_rates_exclude_zero_total_checkpoints(self, mock_config):
        """Test that checkpoints with 0 tests for a type are excluded from averages.

        Regression test: Previously, checkpoints with regression_total=0 would
        contribute 0.0 to the mean, dragging down the average incorrectly.
        """
        checkpoints = [
            {
                "problem": "prob1",
                "idx": 1,
                "strict_pass_rate": 1.0,
                "total_tests": 10,
                "passed_tests": 10,
                "core_total": 5,
                "core_passed": 5,
                "functionality_total": 5,
                "functionality_passed": 5,
                "error_total": 0,  # No error tests
                "error_passed": 0,
                "regression_total": 0,  # No regression tests (first checkpoint)
                "regression_passed": 0,
            },
            {
                "problem": "prob1",
                "idx": 2,
                "strict_pass_rate": 0.9,
                "total_tests": 20,
                "passed_tests": 18,
                "core_total": 5,
                "core_passed": 4,
                "functionality_total": 5,
                "functionality_passed": 4,
                "error_total": 5,
                "error_passed": 5,  # 100% error rate
                "regression_total": 5,
                "regression_passed": 5,  # 100% regression rate
            },
        ]
        summary = compute_run_summary(
            mock_config,
            checkpoints,
            expected_checkpoints=max(len(checkpoints), 1),
        )

        # Core: mean of (5/5, 4/5) = (1.0 + 0.8) / 2 = 0.9
        assert abs(summary.pass_rates.checkpoint.core - 0.9) < 0.01

        # Error: only checkpoint 2 has error tests (5/5 = 1.0)
        # checkpoint 1 should be EXCLUDED, not counted as 0.0
        assert summary.pass_rates.checkpoint.error == 1.0

        # Regression: only checkpoint 2 has regression tests (5/5 = 1.0)
        # checkpoint 1 should be EXCLUDED, not counted as 0.0
        assert summary.pass_rates.checkpoint.regression == 1.0

    def test_pass_rates_multiple_checkpoints_with_zero_tests(self, mock_config):
        """Test pass rate averaging with multiple checkpoints having no tests."""
        checkpoints = [
            {
                "problem": "prob1",
                "idx": 1,
                "strict_pass_rate": 1.0,
                "total_tests": 10,
                "passed_tests": 10,
                "core_total": 10,
                "core_passed": 10,
                "error_total": 0,
                "error_passed": 0,
                "regression_total": 0,
                "regression_passed": 0,
            },
            {
                "problem": "prob2",
                "idx": 1,
                "strict_pass_rate": 1.0,
                "total_tests": 10,
                "passed_tests": 10,
                "core_total": 10,
                "core_passed": 10,
                "error_total": 0,
                "error_passed": 0,
                "regression_total": 0,
                "regression_passed": 0,
            },
            {
                "problem": "prob1",
                "idx": 2,
                "strict_pass_rate": 0.8,
                "total_tests": 20,
                "passed_tests": 16,
                "core_total": 10,
                "core_passed": 8,
                "error_total": 5,
                "error_passed": 4,  # 80% error rate
                "regression_total": 5,
                "regression_passed": 4,  # 80% regression rate
            },
        ]
        summary = compute_run_summary(
            mock_config,
            checkpoints,
            expected_checkpoints=max(len(checkpoints), 1),
        )

        # Error: only 1 checkpoint has error tests (4/5 = 0.8)
        # The 2 checkpoints with error_total=0 should be EXCLUDED
        assert summary.pass_rates.checkpoint.error == 0.8

        # Regression: only 1 checkpoint has regression tests (4/5 = 0.8)
        assert summary.pass_rates.checkpoint.regression == 0.8

    def test_pass_rates_all_checkpoints_have_zero_tests_returns_zero(
        self, mock_config
    ):
        """Test that pass rate is 0.0 when ALL checkpoints have 0 tests for a type."""
        checkpoints = [
            {
                "problem": "prob1",
                "idx": 1,
                "strict_pass_rate": 1.0,
                "total_tests": 10,
                "passed_tests": 10,
                "core_total": 10,
                "core_passed": 10,
                "error_total": 0,  # No error tests
                "error_passed": 0,
            },
            {
                "problem": "prob1",
                "idx": 2,
                "strict_pass_rate": 1.0,
                "total_tests": 10,
                "passed_tests": 10,
                "core_total": 10,
                "core_passed": 10,
                "error_total": 0,  # No error tests
                "error_passed": 0,
            },
        ]
        summary = compute_run_summary(
            mock_config,
            checkpoints,
            expected_checkpoints=max(len(checkpoints), 1),
        )

        # When no checkpoints have error tests, the rate should be 0.0
        # (not NaN or an error)
        assert summary.pass_rates.checkpoint.error == 0.0


class TestRunSummaryCcMetrics:
    """Tests for cyclomatic complexity aggregation in compute_run_summary."""

    def test_computes_cc_stats(self, mock_config):
        """Test that summary aggregates CC metrics across checkpoints."""
        checkpoints = [
            {
                "problem": "prob1",
                "idx": 1,
                "strict_pass_rate": 1.0,
                "cc_high_count": 5,
                "high_cc_mean": 12.0,
                "cc_max": 30,
            },
            {
                "problem": "prob2",
                "idx": 1,
                "strict_pass_rate": 0.8,
                "cc_high_count": 3,
                "high_cc_mean": 18.0,
                "cc_max": 24,
            },
        ]
        summary = compute_run_summary(
            mock_config,
            checkpoints,
            expected_checkpoints=max(len(checkpoints), 1),
        )

        assert summary.cc.high_count.mean == 4.0
        assert summary.cc.high_count.count == 2
        assert summary.cc.high_mean.mean == 15.0
        assert summary.cc.max.max == 30
        assert summary.cc.max.min == 24


class TestRunSummaryCompositeScores:
    """Tests for composite verbosity/erosion score aggregation."""

    def test_uses_saved_checkpoint_verbosity_and_erosion(self, mock_config):
        checkpoints = [
            {
                "problem": "prob1",
                "idx": 1,
                "strict_pass_rate": 1.0,
                "isolated_pass_rate": 1.0,
                "verbosity": 0.95,
                "erosion": 0.6,
                "scb_check": _measured_scb_check(),
                "ast_grep_violations": 999,
                "rubric_total_flags": 999,
                "mass.high_cc_pct": 0.01,
            }
        ]
        summary = compute_run_summary(
            mock_config,
            checkpoints,
            expected_checkpoints=max(len(checkpoints), 1),
        )

        assert summary.verbosity.mean == pytest.approx(0.95)
        assert summary.verbosity.count == 1
        assert summary.erosion.mean == pytest.approx(0.6)
        assert summary.erosion.count == 1

    def test_composites_ignore_raw_checkpoint_fields(self, mock_config):
        checkpoints = [
            {
                "problem": "prob1",
                "idx": 1,
                "strict_pass_rate": 1.0,
                "isolated_pass_rate": 1.0,
                "verbosity": 0.95,
                "erosion": 0.2,
                "scb_check": _measured_scb_check(),
                "loc": 1,
                "clone_lines": 999,
                "functions": 1,
                "methods": 0,
                "trivial_wrappers": 999,
                "single_use_functions": 999,
                "cc_concentration": 0.2,
                "mass.high_cc_pct": 0.2,
            },
            {
                "problem": "prob1",
                "idx": 2,
                "strict_pass_rate": 1.0,
                "isolated_pass_rate": 1.0,
                "verbosity": 0.35,
                "erosion": 0.4,
                "scb_check": _measured_scb_check(),
                "loc": 1,
                "clone_lines": 999,
                "functions": 1,
                "methods": 0,
                "trivial_wrappers": 999,
                "single_use_functions": 999,
                "cc_concentration": 0.4,
                "mass.high_cc_pct": 0.4,
            },
        ]
        summary = compute_run_summary(
            mock_config,
            checkpoints,
            expected_checkpoints=max(len(checkpoints), 1),
        )

        assert summary.erosion.count == 2
        assert summary.erosion.mean == pytest.approx(0.3)
        assert summary.verbosity.mean == pytest.approx((0.95 + 0.35) / 2)

    def test_skips_missing_saved_composites(self, mock_config):
        checkpoints = [
            {
                "problem": "prob1",
                "idx": 1,
                "strict_pass_rate": 1.0,
                "isolated_pass_rate": 1.0,
            },
            {
                "problem": "prob1",
                "idx": 2,
                "strict_pass_rate": 1.0,
                "isolated_pass_rate": 1.0,
            },
        ]
        summary = compute_run_summary(
            mock_config,
            checkpoints,
            expected_checkpoints=max(len(checkpoints), 1),
        )

        assert summary.erosion.count == 0
        assert summary.erosion.mean is None
        assert summary.verbosity.mean is None

    def test_rejects_composites_from_failed_evaluator(self, mock_config):
        checkpoint = {
            "problem": "prob1",
            "idx": 1,
            "strict_pass_rate": 1.0,
            "isolated_pass_rate": 1.0,
            "verbosity": 0.99,
            "erosion": 0.99,
            "scb_check": {
                **_measured_scb_check(),
                "status": "failed",
            },
        }

        summary = compute_run_summary(
            mock_config,
            [checkpoint],
            expected_checkpoints=1,
        )

        assert summary.scb_check.measured_checkpoints == 0
        assert summary.verbosity.count == 0
        assert summary.erosion.count == 0

    def test_time_is_empty_without_duration(self, mock_config):
        summary = compute_run_summary(
            mock_config,
            [{"problem": "prob1", "idx": 1, "strict_pass_rate": 1.0}],
            expected_checkpoints=1,
        )

        assert summary.time.checkpoint.mean is None
        assert summary.time.problem.mean is None


class TestRunSummaryScbCheckCoverage:
    """Tests for explicit evaluator-version and coverage accounting."""

    def test_counts_measured_failed_and_missing_checkpoints(self, mock_config):
        checkpoints = [
            {
                "problem": "prob1",
                "idx": 1,
                "scb_check": {
                    **_measured_scb_check(),
                },
            },
            {
                "problem": "prob1",
                "idx": 2,
                "scb_check": {
                    "status": "failed",
                    "requested_version": "0.1.3",
                    "resolved_version": "0.1.3",
                },
            },
            {
                "problem": "prob2",
                "idx": 1,
                "scb_check": {
                    "status": "missing_snapshot",
                    "requested_version": "0.1.3",
                    "resolved_version": None,
                },
            },
            {"problem": "prob2", "idx": 2},
        ]

        summary = compute_run_summary(
            mock_config,
            checkpoints,
            expected_checkpoints=5,
        )

        assert summary.scb_check.requested_version == "0.1.3"
        assert summary.scb_check.resolved_versions == ["0.1.3"]
        assert summary.scb_check.measured_checkpoints == 1
        assert summary.scb_check.expected_checkpoints == 5
        assert summary.scb_check.failed_checkpoints == 1
        assert summary.scb_check.missing_snapshot_checkpoints == 1
        assert summary.scb_check.missing_metadata_checkpoints == 1
        assert summary.scb_check.missing_checkpoint_records == 1
        assert summary.scb_check.unmeasured_checkpoints == 4
        assert summary.scb_check.coverage_pct == pytest.approx(20.0)
