"""Exhaustive tests for AST-grep metrics.

Tests all functions in slop_code.metrics.languages.python.ast_grep including:
- Public API: calculate_ast_grep_metrics, build_ast_grep_rules_lookup
- Helper functions: _is_sg_available, _get_ast_grep_rules_path, _count_rules_in_file
"""

from __future__ import annotations

import shutil
from pathlib import Path
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from slop_code.metrics.languages.python import AST_GREP_RULES_DIR
from slop_code.metrics.languages.python import AST_GREP_RULES_PATH
from slop_code.metrics.languages.python import _get_ast_grep_rules_dir
from slop_code.metrics.languages.python import _get_ast_grep_rules_path
from slop_code.metrics.languages.python import _get_sg_executable
from slop_code.metrics.languages.python import _is_sg_available
from slop_code.metrics.languages.python import calculate_ast_grep_metrics
from slop_code.metrics.languages.python.ast_grep import (
    build_ast_grep_rules_lookup,
)
from slop_code.metrics.models import AstGrepMetrics
from slop_code.metrics.models import AstGrepViolation


def _write(tmp_path: Path, name: str, content: str) -> Path:
    """Helper to write a file and return its path."""
    path = tmp_path / name
    path.write_text(content)
    return path


# =============================================================================
# sg Availability Tests
# =============================================================================


class TestSgAvailability:
    """Tests for _is_sg_available."""

    def test_sg_available_when_installed(self) -> None:
        with (
            patch.dict("os.environ", {}, clear=True),
            patch("shutil.which", return_value="/usr/bin/sg"),
        ):
            assert _is_sg_available() is True
            assert _get_sg_executable() == "/usr/bin/sg"

    def test_sg_unavailable_when_not_installed(self) -> None:
        with (
            patch.dict("os.environ", {}, clear=True),
            patch("shutil.which", return_value=None),
        ):
            assert _is_sg_available() is False

    def test_explicit_binary_override(self, tmp_path: Path) -> None:
        executable = tmp_path / "sg"
        executable.write_text("", encoding="utf-8")
        with patch.dict(
            "os.environ", {"AST_GREP_BIN": str(executable)}, clear=True
        ):
            assert _get_sg_executable() == str(executable)


# =============================================================================
# Rules Directory Tests
# =============================================================================


class TestGetAstGrepRulesPath:
    """Tests for _get_ast_grep_rules_path."""

    def test_default_rules_path(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            rules_path = _get_ast_grep_rules_path()
            assert rules_path.name == "slop_rules.yaml"
            assert rules_path.parent.name == "configs"
            assert "configs" in str(rules_path)
            assert _get_ast_grep_rules_dir() == rules_path

    def test_env_override(self, tmp_path: Path) -> None:
        custom_file = tmp_path / "custom-rules.yaml"
        custom_file.write_text("id: test-rule\n")
        with patch.dict(
            "os.environ", {"AST_GREP_RULES_PATH": str(custom_file)}
        ):
            assert _get_ast_grep_rules_path() == custom_file

    def test_default_rules_path_exists(self) -> None:
        """Verify the default rules file actually exists."""
        assert AST_GREP_RULES_PATH.exists(), (
            f"Expected {AST_GREP_RULES_PATH} to exist"
        )
        assert AST_GREP_RULES_PATH.is_file()
        assert AST_GREP_RULES_DIR == AST_GREP_RULES_PATH


class TestBuildAstGrepRulesLookup:
    """Tests for build_ast_grep_rules_lookup."""

    def test_uses_curated_slop_ruleset(self) -> None:
        lookup = build_ast_grep_rules_lookup()

        included_rules = {
            "manual-sum-loop",
            "guard-return-none",
            "nested-if-no-else",
            "json-roundtrip-dumps-loads",
            "dict-str-any",
            "object-type-annotation",
        }
        excluded_rules = {
            "bare-except-pass",
            "mutable-default-arg",
            "pandas-iterrows",
            "untyped-list-annotation",
        }

        for rule_id in included_rules:
            assert lookup[rule_id]["category"] == "slop"
            assert lookup[rule_id]["subcategory"] == "slop"
        for rule_id in excluded_rules:
            assert rule_id not in lookup


# =============================================================================
# Calculate AST-grep Metrics Tests
# =============================================================================


class TestCalculateAstGrepMetrics:
    """Tests for calculate_ast_grep_metrics."""

    def test_fails_when_sg_unavailable(self, tmp_path: Path) -> None:
        source = _write(tmp_path, "test.py", "x = 1")
        with (
            patch.dict("os.environ", {}, clear=True),
            patch(
                "slop_code.metrics.languages.python.ast_grep.shutil.which",
                return_value=None,
            ),
            pytest.raises(RuntimeError, match="ast-grep.*required"),
        ):
            calculate_ast_grep_metrics(source)

    def test_fails_when_rules_file_missing(self, tmp_path: Path) -> None:
        source = _write(tmp_path, "test.py", "x = 1")
        with (
            patch(
                "slop_code.metrics.languages.python.ast_grep.shutil.which",
                return_value="/usr/bin/sg",
            ),
            patch.dict(
                "os.environ", {"AST_GREP_RULES_PATH": "/nonexistent.yaml"}
            ),
            pytest.raises(
                FileNotFoundError, match="rules file not found"
            ),
        ):
            calculate_ast_grep_metrics(source)

    def test_fails_when_no_rules(self, tmp_path: Path) -> None:
        source = _write(tmp_path, "test.py", "x = 1")
        rules_path = tmp_path / "empty-rules.yaml"
        rules_path.write_text("")

        with (
            patch(
                "slop_code.metrics.languages.python.ast_grep.shutil.which",
                return_value="/usr/bin/sg",
            ),
            patch.dict("os.environ", {"AST_GREP_RULES_PATH": str(rules_path)}),
            pytest.raises(RuntimeError, match="contains no valid rules"),
        ):
            calculate_ast_grep_metrics(source)

    def test_parses_violations_from_sg_output(self, tmp_path: Path) -> None:
        source = _write(
            tmp_path,
            "test.py",
            """
def bad():
    try:
        pass
    except:
        pass
""",
        )
        rules_path = tmp_path / "rules.yaml"
        # Rule YAML includes metadata
        rules_path.write_text(
            "id: bare-except-pass\n"
            "language: python\n"
            "metadata:\n"
            "  category: safety\n"
            "  weight: 2\n"
            "rule:\n"
            "  pattern: 'pass'"
        )

        mock_output = (
            '{"ruleId": "bare-except-pass", '
            '"severity": "warning", '
            '"range": {"start": {"line": 5, "column": 8}, '
            '"end": {"line": 5, "column": 12}}}'
        )

        with (
            patch(
                "slop_code.metrics.languages.python.ast_grep.shutil.which",
                return_value="/usr/bin/sg",
            ),
            patch.dict("os.environ", {"AST_GREP_RULES_PATH": str(rules_path)}),
            patch(
                "slop_code.metrics.languages.python.ast_grep.subprocess.run"
            ) as mock_run,
        ):
            mock_run.return_value = MagicMock(
                stdout=mock_output,
                stderr="",
                returncode=0,
            )
            result = calculate_ast_grep_metrics(source)

        assert result.total_violations == 1
        assert result.rules_checked == 1
        assert len(result.violations) == 1
        assert result.violations[0].rule_id == "bare-except-pass"
        assert result.violations[0].severity == "warning"
        assert result.violations[0].line == 5
        assert result.violations[0].column == 8
        assert result.counts == {"bare-except-pass": 1}

    def test_aggregates_multiple_violations(self, tmp_path: Path) -> None:
        source = _write(tmp_path, "test.py", "x = 1")
        rules_path = tmp_path / "rules.yaml"
        rules_path.write_text("id: test-rule")

        mock_output = "\n".join(
            [
                '{"ruleId": "test-rule", "severity": "warning", '
                '"range": {"start": {"line": 1, "column": 0}, '
                '"end": {"line": 1, "column": 5}}}',
                '{"ruleId": "test-rule", "severity": "warning", '
                '"range": {"start": {"line": 2, "column": 0}, '
                '"end": {"line": 2, "column": 5}}}',
            ]
        )

        with (
            patch(
                "slop_code.metrics.languages.python.ast_grep.shutil.which",
                return_value="/usr/bin/sg",
            ),
            patch.dict("os.environ", {"AST_GREP_RULES_PATH": str(rules_path)}),
            patch(
                "slop_code.metrics.languages.python.ast_grep.subprocess.run"
            ) as mock_run,
        ):
            mock_run.return_value = MagicMock(
                stdout=mock_output,
                stderr="",
                returncode=0,
            )
            result = calculate_ast_grep_metrics(source)

        assert result.total_violations == 2
        assert result.counts == {"test-rule": 2}

    def test_counts_multiple_rules_in_one_file(self, tmp_path: Path) -> None:
        source = _write(tmp_path, "test.py", "x = 1")
        rules_path = tmp_path / "rules.yaml"
        rules_path.write_text(
            "---\n"
            "id: first-rule\n"
            "language: python\n"
            "rule:\n"
            "  pattern: x\n"
            "---\n"
            "id: second-rule\n"
            "language: python\n"
            "rule:\n"
            "  pattern: y\n"
        )

        mock_output = "\n".join(
            [
                '{"ruleId": "first-rule", "severity": "warning", '
                '"range": {"start": {"line": 1, "column": 0}, '
                '"end": {"line": 1, "column": 5}}}',
                '{"ruleId": "second-rule", "severity": "warning", '
                '"range": {"start": {"line": 2, "column": 0}, '
                '"end": {"line": 2, "column": 5}}}',
            ]
        )

        with (
            patch(
                "slop_code.metrics.languages.python.ast_grep.shutil.which",
                return_value="/usr/bin/sg",
            ),
            patch.dict("os.environ", {"AST_GREP_RULES_PATH": str(rules_path)}),
            patch(
                "slop_code.metrics.languages.python.ast_grep.subprocess.run"
            ) as mock_run,
        ):
            mock_run.return_value = MagicMock(
                stdout=mock_output,
                stderr="",
                returncode=0,
            )
            result = calculate_ast_grep_metrics(source)

        assert mock_run.call_count == 1
        assert result.rules_checked == 2
        assert result.counts == {"first-rule": 1, "second-rule": 1}

    def test_fails_on_subprocess_error(self, tmp_path: Path) -> None:
        source = _write(tmp_path, "test.py", "x = 1")
        rules_path = tmp_path / "rules.yaml"
        rules_path.write_text(
            "id: test\nlanguage: python\nrule:\n  pattern: 'x'"
        )

        with (
            patch(
                "slop_code.metrics.languages.python.ast_grep.shutil.which",
                return_value="/usr/bin/sg",
            ),
            patch.dict("os.environ", {"AST_GREP_RULES_PATH": str(rules_path)}),
            patch(
                "slop_code.metrics.languages.python.ast_grep.subprocess.run"
            ) as mock_run,
        ):
            mock_run.side_effect = OSError("sg not found")
            with pytest.raises(RuntimeError, match="failed to run AST-grep"):
                calculate_ast_grep_metrics(source)

    def test_fails_on_nonzero_scan(self, tmp_path: Path) -> None:
        source = _write(tmp_path, "test.py", "x = 1")
        rules_path = tmp_path / "rules.yaml"
        rules_path.write_text(
            "id: test\nlanguage: python\nrule:\n  pattern: 'x'"
        )

        with (
            patch(
                "slop_code.metrics.languages.python.ast_grep.shutil.which",
                return_value="/usr/bin/sg",
            ),
            patch.dict("os.environ", {"AST_GREP_RULES_PATH": str(rules_path)}),
            patch(
                "slop_code.metrics.languages.python.ast_grep.subprocess.run"
            ) as mock_run,
        ):
            mock_run.return_value = MagicMock(
                stdout="",
                stderr="invalid rule",
                returncode=2,
            )
            with pytest.raises(RuntimeError, match="scan failed"):
                calculate_ast_grep_metrics(source)

    def test_fails_on_malformed_json(self, tmp_path: Path) -> None:
        source = _write(tmp_path, "test.py", "x = 1")
        rules_path = tmp_path / "rules.yaml"
        rules_path.write_text("id: test")

        with (
            patch(
                "slop_code.metrics.languages.python.ast_grep.shutil.which",
                return_value="/usr/bin/sg",
            ),
            patch.dict("os.environ", {"AST_GREP_RULES_PATH": str(rules_path)}),
            patch(
                "slop_code.metrics.languages.python.ast_grep.subprocess.run"
            ) as mock_run,
        ):
            mock_run.return_value = MagicMock(
                stdout="not valid json",
                stderr="",
                returncode=0,
            )
            with pytest.raises(RuntimeError, match="failed to parse"):
                calculate_ast_grep_metrics(source)

    def test_fails_on_missing_json_fields(
        self, tmp_path: Path
    ) -> None:
        source = _write(tmp_path, "test.py", "x = 1")
        rules_path = tmp_path / "rules.yaml"
        rules_path.write_text("id: test")

        # Missing 'range' field
        mock_output = '{"ruleId": "test", "severity": "warning"}'

        with (
            patch(
                "slop_code.metrics.languages.python.ast_grep.shutil.which",
                return_value="/usr/bin/sg",
            ),
            patch.dict("os.environ", {"AST_GREP_RULES_PATH": str(rules_path)}),
            patch(
                "slop_code.metrics.languages.python.ast_grep.subprocess.run"
            ) as mock_run,
        ):
            mock_run.return_value = MagicMock(
                stdout=mock_output,
                stderr="",
                returncode=0,
            )
            with pytest.raises(RuntimeError, match="failed to parse"):
                calculate_ast_grep_metrics(source)

    def test_handles_empty_output(self, tmp_path: Path) -> None:
        source = _write(tmp_path, "test.py", "x = 1")
        rules_path = tmp_path / "rules.yaml"
        rules_path.write_text("id: test")

        with (
            patch(
                "slop_code.metrics.languages.python.ast_grep.shutil.which",
                return_value="/usr/bin/sg",
            ),
            patch.dict("os.environ", {"AST_GREP_RULES_PATH": str(rules_path)}),
            patch(
                "slop_code.metrics.languages.python.ast_grep.subprocess.run"
            ) as mock_run,
        ):
            mock_run.return_value = MagicMock(
                stdout="",
                stderr="",
                returncode=0,
            )
            result = calculate_ast_grep_metrics(source)

        assert result.total_violations == 0
        assert result.rules_checked == 1


# =============================================================================
# AST-grep Metrics Model Tests
# =============================================================================


class TestAstGrepMetricsModel:
    """Tests for AstGrepMetrics and AstGrepViolation models."""

    def test_ast_grep_violation_serialization(self) -> None:
        violation = AstGrepViolation(
            rule_id="test-rule",
            severity="warning",
            category="verbosity",
            subcategory="verbose-code",
            weight=3,
            line=10,
            column=4,
            end_line=10,
            end_column=8,
        )

        data = violation.model_dump()
        assert data["rule_id"] == "test-rule"
        assert data["severity"] == "warning"
        assert data["category"] == "verbosity"
        assert data["subcategory"] == "verbose-code"
        assert data["weight"] == 3
        assert data["line"] == 10
        assert data["column"] == 4
        assert data["end_line"] == 10
        assert data["end_column"] == 8

    def test_ast_grep_violation_defaults(self) -> None:
        """Test that category and subcategory have sensible defaults."""
        violation = AstGrepViolation(
            rule_id="test-rule",
            severity="warning",
            line=1,
            column=0,
            end_line=1,
            end_column=5,
        )

        assert violation.category == ""
        assert violation.subcategory == "unknown"
        assert violation.weight == 1

    def test_ast_grep_metrics_serialization(self) -> None:
        violation = AstGrepViolation(
            rule_id="test-rule",
            severity="warning",
            line=10,
            column=4,
            end_line=10,
            end_column=8,
        )
        metrics = AstGrepMetrics(
            violations=[violation],
            total_violations=1,
            counts={"test-rule": 1},
            rules_checked=5,
        )

        data = metrics.model_dump()
        assert data["total_violations"] == 1
        assert data["rules_checked"] == 5
        assert len(data["violations"]) == 1
        assert data["violations"][0]["rule_id"] == "test-rule"
        assert data["counts"] == {"test-rule": 1}

    def test_ast_grep_metrics_empty(self) -> None:
        metrics = AstGrepMetrics(
            violations=[],
            total_violations=0,
            counts={},
            rules_checked=0,
        )

        data = metrics.model_dump()
        assert data["total_violations"] == 0
        assert data["rules_checked"] == 0
        assert data["violations"] == []
        assert data["counts"] == {}


# =============================================================================
# Integration Tests
# =============================================================================


class TestAstGrepMetricsIntegration:
    """Integration tests that use actual ast-grep if available."""

    @pytest.mark.skipif(
        not shutil.which("sg"), reason="ast-grep (sg) not installed"
    )
    def test_real_sg_scan_with_clean_code(self, tmp_path: Path) -> None:
        """Test with actual ast-grep scanning on clean code."""
        source = _write(
            tmp_path,
            "clean.py",
            """
def greet(name: str) -> str:
    return f"Hello, {name}!"
""",
        )

        result = calculate_ast_grep_metrics(source)

        # Should have checked rules but found few/no violations in clean code
        assert result.rules_checked > 0

    @pytest.mark.skipif(
        not shutil.which("sg"), reason="ast-grep (sg) not installed"
    )
    def test_real_sg_scan_with_bad_code(self, tmp_path: Path) -> None:
        """Test with actual ast-grep scanning on code with bad patterns."""
        source = _write(
            tmp_path,
            "bad.py",
            """
def example():
    try:
        do_something()
    except:
        pass

def check_value(x):
    if x == True:
        return True
    else:
        return False
""",
        )

        result = calculate_ast_grep_metrics(source)

        # Should have checked rules
        assert result.rules_checked > 0
        # This code has patterns that should be caught:
        # - bare except with pass
        # - comparing to True
        # - if/else returning True/False
