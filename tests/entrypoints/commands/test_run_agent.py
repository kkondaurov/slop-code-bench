"""Tests for run_agent command helper functions."""

import json
from pathlib import Path
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
import typer
import yaml

from slop_code import provenance
from slop_code.agent_runner.models import UsageTracker
from slop_code.agent_runner.resume import ResumeInfo
from slop_code.common import CHECKPOINT_RESULTS_FILENAME
from slop_code.common import EVALUATION_FILENAME
from slop_code.common import INFERENCE_RESULT_FILENAME
from slop_code.common import POSTPROCESSING_FILENAME
from slop_code.common import SUMMARY_FILENAME
from slop_code.entrypoints.commands.run_agent import PostprocessingResult
from slop_code.entrypoints.commands.run_agent import _build_cli_flags
from slop_code.entrypoints.commands.run_agent import _check_problem_needs_rerun
from slop_code.entrypoints.commands.run_agent import (
    _configure_named_profile_temp_root,
)
from slop_code.entrypoints.commands.run_agent import (
    _create_checkpoint_results_and_summary,
)
from slop_code.entrypoints.commands.run_agent import _create_task_config
from slop_code.entrypoints.commands.run_agent import _discover_problems
from slop_code.entrypoints.commands.run_agent import _exit_on_incomplete_run
from slop_code.entrypoints.commands.run_agent import (
    _filter_problems_for_execution,
)
from slop_code.entrypoints.commands.run_agent import _final_run_status
from slop_code.entrypoints.commands.run_agent import (
    _finalize_after_primary_error,
)
from slop_code.entrypoints.commands.run_agent import (
    _finalize_initialization_error,
)
from slop_code.entrypoints.commands.run_agent import (
    _finalize_run_provenance_or_raise,
)
from slop_code.entrypoints.commands.run_agent import _get_nested
from slop_code.entrypoints.commands.run_agent import _handle_early_completion
from slop_code.entrypoints.commands.run_agent import _handle_resume_validation
from slop_code.entrypoints.commands.run_agent import (
    _load_and_validate_run_config,
)
from slop_code.entrypoints.commands.run_agent import (
    _named_checkpoint_domain_errors,
)
from slop_code.entrypoints.commands.run_agent import _prepare_run_artifacts
from slop_code.entrypoints.commands.run_agent import _preview_dry_run
from slop_code.entrypoints.commands.run_agent import (
    _publish_new_owned_run_directory,
)
from slop_code.entrypoints.commands.run_agent import _resolve_output_directory
from slop_code.entrypoints.commands.run_agent import _resolve_problem_names
from slop_code.entrypoints.commands.run_agent import (
    _validate_preexisting_output_provenance,
)
from slop_code.entrypoints.commands.run_agent import _validate_problem_paths
from slop_code.entrypoints.commands.run_agent import _validate_resume_config
from slop_code.entrypoints.commands.run_agent import (
    _write_postprocessing_result,
)
from slop_code.problem_catalog import CatalogManifest


def _initialize_test_provenance(
    root: Path,
    storage_dir: Path,
    identity_dir: Path,
) -> None:
    with (
        patch.object(provenance, "_git_repository_metadata", return_value={}),
        patch.object(provenance, "_host_metadata", return_value={}),
        patch.object(
            provenance,
            "_image_metadata",
            return_value={"available": False},
        ),
        patch.object(
            provenance,
            "_container_tool_versions",
            return_value={"error": "no_image"},
        ),
    ):
        provenance.start_run_provenance(
            repository_root=root,
            run_dir=storage_dir,
            identity_run_dir=identity_dir,
            profile="test-profile",
            model_provider="test",
            model_name="model",
            agent_type="codex",
            agent_version="1",
            thinking="high",
            seed=42,
            problem_names=["problem"],
            catalog_version="v1",
            catalog_commit="a" * 40,
            num_workers=1,
            evaluate=True,
            environment_name="test",
            source_image_name="",
            base_image_name="",
            agent_image_name="",
            invocation=["slop-code", "run"],
        )


class TestGetNested:
    """Tests for _get_nested helper function."""

    def test_simple_key(self):
        """Test getting a simple top-level key."""
        data = {"name": "value"}
        assert _get_nested(data, "name") == "value"

    def test_nested_key(self):
        """Test getting a nested key with dot notation."""
        data = {"model": {"provider": "anthropic", "name": "opus-4"}}
        assert _get_nested(data, "model.provider") == "anthropic"
        assert _get_nested(data, "model.name") == "opus-4"

    def test_deeply_nested_key(self):
        """Test getting a deeply nested key."""
        data = {"level1": {"level2": {"level3": "deep_value"}}}
        assert _get_nested(data, "level1.level2.level3") == "deep_value"

    def test_missing_key(self):
        """Test that missing keys return None."""
        data = {"name": "value"}
        assert _get_nested(data, "missing") is None
        assert _get_nested(data, "missing.nested") is None

    def test_missing_nested_key(self):
        """Test that missing nested keys return None."""
        data = {"model": {"provider": "anthropic"}}
        assert _get_nested(data, "model.name") is None

    def test_non_dict_intermediate(self):
        """Test that non-dict intermediate values return None."""
        data = {"model": "string_value"}
        assert _get_nested(data, "model.provider") is None

    def test_empty_dict(self):
        """Test with empty dictionary."""
        assert _get_nested({}, "any.key") is None


def test_named_checkpoint_rates_must_match_counts() -> None:
    report = {
        "cost": 0.0,
        "duration": 1.0,
        "steps": 1,
        "input": 1,
        "output": 1,
        "cache_read": 0,
        "cache_write": 0,
        "reasoning": 0,
        "passed_tests": 7,
        "total_tests": 10,
        "core_passed": 2,
        "core_total": 2,
        "functionality_passed": 2,
        "functionality_total": 3,
        "error_passed": 2,
        "error_total": 3,
        "regression_passed": 1,
        "regression_total": 2,
        "strict_pass_rate": 0.8,
        "core_pass_rate": 1.0,
        "isolated_pass_rate": 0.75,
    }

    errors = _named_checkpoint_domain_errors(report)

    assert errors == [
        "strict_pass_rate must equal passed_tests/total_tests (0.7)"
    ]


class TestBuildCliFlags:
    """Tests for _build_cli_flags helper function."""

    def test_no_overrides(self):
        """Test with no overrides provided."""
        result = _build_cli_flags(None, None, None, None)
        assert result == {}

    def test_agent_override(self):
        """Test agent path override."""
        result = _build_cli_flags("my_agent", None, None, None)
        assert result == {"agent": "my_agent"}

    def test_environment_override(self):
        """Test environment path override."""
        result = _build_cli_flags(None, "my_env.yaml", None, None)
        assert result == {"environment": "my_env.yaml"}

    def test_prompt_override(self):
        """Test prompt path override."""
        result = _build_cli_flags(None, None, "my_prompt.jinja", None)
        assert result == {"prompt": "my_prompt.jinja"}

    def test_model_override(self):
        """Test model override parsing."""
        # Use a model that exists in the catalog
        result = _build_cli_flags(None, None, None, "anthropic/sonnet-4.5")
        assert "model" in result
        model = result["model"]
        assert model["provider"] == "anthropic"
        assert model["name"] == "sonnet-4.5"

    def test_all_overrides(self):
        """Test all overrides provided."""
        result = _build_cli_flags(
            "agent.yaml", "env.yaml", "prompt.jinja", "anthropic/sonnet-4.5"
        )
        assert result["agent"] == "agent.yaml"
        assert result["environment"] == "env.yaml"
        assert result["prompt"] == "prompt.jinja"
        assert "model" in result
        model = result["model"]
        assert model["provider"] == "anthropic"
        assert model["name"] == "sonnet-4.5"

    def test_invalid_model_format_raises(self):
        """Test error on invalid model format."""
        with pytest.raises(ValueError):
            _build_cli_flags(None, None, None, "invalid-model-format")


class TestLoadAndValidateRunConfig:
    """Tests for _load_and_validate_run_config helper function."""

    def test_defaults_only(self):
        """Test loading with all defaults."""
        result = _load_and_validate_run_config(None, {}, None)
        assert result.model.provider == "anthropic"
        assert result.model.name == "sonnet-4.5"

    def test_config_file_not_found(self, tmp_path):
        """Test error when config file doesn't exist."""
        with pytest.raises(typer.Exit) as excinfo:
            _load_and_validate_run_config(tmp_path / "missing.yaml", {}, None)
        assert excinfo.value.exit_code == 1

    def test_cli_flags_override(self):
        """Test CLI flags override defaults."""
        cli_flags = {
            "model": {"provider": "openai", "name": "gpt-4"},
        }
        result = _load_and_validate_run_config(None, cli_flags, None)
        assert result.model.provider == "openai"
        assert result.model.name == "gpt-4"

    def test_config_file_loading(self, tmp_path):
        """Test loading from config file."""
        config_file = tmp_path / "run_config.yaml"
        config_file.write_text(
            yaml.dump(
                {
                    "model": {"provider": "google", "name": "gemini-pro"},
                }
            )
        )
        result = _load_and_validate_run_config(config_file, {}, None)
        assert result.model.provider == "google"
        assert result.model.name == "gemini-pro"


class TestNamedProfileTempRoot:
    def test_unnamed_run_does_not_configure_profile_root(
        self, tmp_path: Path
    ) -> None:
        with patch(
            "slop_code.entrypoints.commands.run_agent.configure_named_profile_temp_root"
        ) as configure:
            result = _configure_named_profile_temp_root(tmp_path, None)

        assert result is None
        configure.assert_not_called()

    def test_named_run_configures_profile_root_before_execution(
        self, tmp_path: Path
    ) -> None:
        expected = tmp_path / "tmp" / "scbench-v2"
        with patch(
            "slop_code.entrypoints.commands.run_agent.configure_named_profile_temp_root",
            return_value=expected,
        ) as configure:
            result = _configure_named_profile_temp_root(
                tmp_path,
                "gpt-5.5-current-xhigh",
            )

        assert result == expected
        configure.assert_called_once_with(tmp_path)

    def test_invalid_explicit_profile_root_exits_nonzero(
        self, tmp_path: Path
    ) -> None:
        with (
            patch(
                "slop_code.entrypoints.commands.run_agent.configure_named_profile_temp_root",
                side_effect=ValueError("SLOP_CODE_TMPDIR must be absolute"),
            ),
            pytest.raises(typer.Exit) as exc_info,
        ):
            _configure_named_profile_temp_root(
                tmp_path,
                "paper-v2-reference",
            )

        assert exc_info.value.exit_code == 1


class TestResolveProblemNames:
    """Tests for _resolve_problem_names helper function."""

    def test_cli_takes_precedence(self):
        """Test CLI problems override config."""
        result = _resolve_problem_names(["cli_prob"], ["config_prob"])
        assert result == ["cli_prob"]

    def test_config_used_when_no_cli(self):
        """Test config problems used when no CLI."""
        result = _resolve_problem_names([], ["config_prob"])
        assert result == ["config_prob"]

    def test_empty_returns_empty(self):
        """Test empty inputs return empty list."""
        result = _resolve_problem_names([], [])
        assert result == []

    def test_multiple_cli_problems(self):
        """Test multiple CLI problems."""
        result = _resolve_problem_names(["prob1", "prob2"], ["config_prob"])
        assert result == ["prob1", "prob2"]

    def test_multiple_config_problems(self):
        """Test multiple config problems."""
        result = _resolve_problem_names([], ["prob1", "prob2", "prob3"])
        assert result == ["prob1", "prob2", "prob3"]


class TestResolveOutputDirectory:
    """Tests for _resolve_output_directory helper function."""

    def test_config_path_used(self, tmp_path):
        """Test config output path is used."""
        config_path = str(tmp_path / "config_out")
        result, existed = _resolve_output_directory(config_path, debug=False)
        assert result == Path(config_path)

    def test_debug_prefix_added(self, tmp_path):
        """Test DEBUG_ prefix in debug mode."""
        result, existed = _resolve_output_directory(
            str(tmp_path / "run_123"), debug=True
        )
        assert "DEBUG_" in str(result)

    def test_debug_prefix_with_nested_path(self):
        """Test DEBUG_ prefix with nested path."""
        result, existed = _resolve_output_directory(
            "outputs/run_123", debug=True
        )
        assert "outputs/DEBUG_run_123" in str(result)

    def test_debug_prefix_with_simple_path(self):
        """Test DEBUG_ prefix with simple path."""
        result, existed = _resolve_output_directory("run_123", debug=True)
        assert str(result) == "DEBUG_run_123"

    def test_preexisted_flag_true(self, tmp_path):
        """Test preexisted flag correctly set when directory exists."""
        existing_dir = tmp_path / "exists"
        existing_dir.mkdir()
        result, existed = _resolve_output_directory(
            str(existing_dir), debug=False
        )
        assert existed is True

    def test_preexisted_flag_false(self, tmp_path):
        """Test preexisted flag correctly set when directory doesn't exist."""
        result, existed = _resolve_output_directory(
            str(tmp_path / "new"), debug=False
        )
        assert existed is False

    def test_directory_created(self, tmp_path):
        """Test that directory is created."""
        new_path = tmp_path / "new_dir"
        result, existed = _resolve_output_directory(str(new_path), debug=False)
        assert result.exists()

    def test_read_only_resolution_does_not_create_directory(self, tmp_path):
        """Dry-run path resolution leaves a missing output path untouched."""
        new_path = tmp_path / "dry-run"

        result, existed = _resolve_output_directory(
            str(new_path),
            debug=False,
            create=False,
        )

        assert result == new_path
        assert existed is False
        assert not new_path.exists()


def test_dry_run_missing_problem_output_is_fresh_and_does_not_mutate(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run_dir = tmp_path / "missing-run"
    problem_path = tmp_path / "problems"
    problem_path.mkdir()

    with (
        patch(
            "slop_code.entrypoints.commands.run_agent.ProblemConfig.from_yaml",
            return_value=MagicMock(),
        ),
        patch(
            "slop_code.entrypoints.commands.run_agent.detect_resume_point"
        ) as detect,
    ):
        _preview_dry_run(
            ["prob1"],
            run_dir,
            problem_path,
            "prompt",
            MagicMock(),
        )

    output = capsys.readouterr().out
    assert "Would start fresh (no existing run)" in output
    assert "Resume from:" not in output
    assert "Directories to quarantine:" not in output
    assert not run_dir.exists()
    detect.assert_not_called()


def test_preexisting_named_or_provenanced_output_is_always_validated(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "provenance.json").write_text("{}\n", encoding="utf-8")

    with patch(
        "slop_code.entrypoints.commands.run_agent.validate_resumable_provenance"
    ) as validate:
        assert _validate_preexisting_output_provenance(
            run_dir,
            run_dir_preexisted=True,
            is_resuming=False,
            profile="gpt-5.5-current-xhigh",
        )

    validate.assert_called_once_with(run_dir)


def test_new_named_output_does_not_require_existing_provenance(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "new-run"

    with patch(
        "slop_code.entrypoints.commands.run_agent.validate_resumable_provenance"
    ) as validate:
        assert not _validate_preexisting_output_provenance(
            run_dir,
            run_dir_preexisted=False,
            is_resuming=False,
            profile="gpt-5.5-current-xhigh",
        )

    validate.assert_not_called()
    assert not run_dir.exists()


def test_new_named_output_is_published_with_valid_provenance(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "named-run"

    _publish_new_owned_run_directory(
        run_dir,
        lambda storage, identity: _initialize_test_provenance(
            tmp_path,
            storage,
            identity,
        ),
    )

    assert run_dir.is_dir()
    saved = json.loads((run_dir / "provenance.json").read_text())
    assert saved["final_status"] == "running"
    assert saved["run"]["directory"] == str(run_dir.resolve())
    provenance.validate_resumable_provenance(run_dir)


def test_new_named_output_crash_after_staging_remains_resumable(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "named-run"
    _publish_new_owned_run_directory(
        run_dir,
        lambda storage, identity: _initialize_test_provenance(
            tmp_path,
            storage,
            identity,
        ),
    )

    # Simulate evaluator/catalog staging and the persistent preflight append,
    # followed by a hard crash before config preparation.
    (run_dir / "inputs/scbench-v2-evaluator").mkdir(parents=True)
    (run_dir / "inputs/scbench-v2-catalog").mkdir()
    (run_dir / "scbench_v2_preflight.json").write_text(
        '{"attempts": [{"status": "verified"}]}\n',
        encoding="utf-8",
    )

    provenance.validate_resumable_provenance(run_dir)
    with (
        patch.object(provenance, "_git_repository_metadata", return_value={}),
        patch.object(provenance, "_host_metadata", return_value={}),
        patch.object(
            provenance,
            "_image_metadata",
            return_value={"available": False},
        ),
        patch.object(
            provenance,
            "_container_tool_versions",
            return_value={"error": "no_image"},
        ),
    ):
        provenance.start_run_provenance(
            repository_root=tmp_path,
            run_dir=run_dir,
            profile="test-profile",
            model_provider="test",
            model_name="model",
            agent_type="codex",
            agent_version="1",
            thinking="high",
            seed=42,
            problem_names=["problem"],
            catalog_version="v1",
            catalog_commit="a" * 40,
            num_workers=1,
            evaluate=True,
            environment_name="test",
            source_image_name="",
            base_image_name="",
            agent_image_name="",
            invocation=["slop-code", "run"],
            require_existing=True,
        )

    saved = json.loads((run_dir / "provenance.json").read_text())
    assert saved["invocations"][0]["status"] == (
        "interrupted_before_next_invocation"
    )


def test_failed_atomic_publication_never_exposes_unowned_target(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "named-run"

    def fail_after_initialization(storage: Path, identity: Path) -> None:
        _initialize_test_provenance(tmp_path, storage, identity)
        raise RuntimeError("crash before publication")

    with pytest.raises(RuntimeError, match="crash before publication"):
        _publish_new_owned_run_directory(run_dir, fail_after_initialization)

    assert not run_dir.exists()


class TestValidateResumeConfig:
    """Tests for _validate_resume_config function."""

    @pytest.fixture
    def mock_run_cfg(self):
        """Create a mock ResolvedRunConfig."""
        cfg = MagicMock()
        cfg.model_dump.return_value = {
            "model": {"provider": "anthropic", "name": "opus-4"},
            "agent": {"type": "claude_code"},
            "thinking": "low",
            "prompt_path": "/path/to/prompt.jinja",
        }
        return cfg

    @pytest.fixture
    def mock_env_spec(self):
        """Create a mock environment spec."""
        spec = MagicMock()
        spec.model_dump.return_value = {
            "type": "docker",
            "name": "python3.12",
            "docker": {"image": "python:3.12"},
        }
        return spec

    def test_no_saved_config(self, tmp_path, mock_run_cfg, mock_env_spec):
        """Test that missing config.yaml allows resume."""
        result = _validate_resume_config(tmp_path, mock_run_cfg, mock_env_spec)
        assert result == []

    def test_matching_config(self, tmp_path, mock_run_cfg, mock_env_spec):
        """Test that matching config returns no mismatches."""
        (tmp_path / "config.yaml").write_text(
            yaml.dump(
                {
                    "model": {"provider": "anthropic", "name": "opus-4"},
                    "agent": {"type": "claude_code"},
                    "thinking": "low",
                    "prompt_path": "/path/to/prompt.jinja",
                }
            )
        )
        (tmp_path / "environment.yaml").write_text(
            yaml.dump(
                {
                    "type": "docker",
                    "name": "python3.12",
                    "docker": {"image": "python:3.12"},
                }
            )
        )
        result = _validate_resume_config(tmp_path, mock_run_cfg, mock_env_spec)
        assert result == []

    def test_model_provider_mismatch(
        self, tmp_path, mock_run_cfg, mock_env_spec
    ):
        """Test detection of model provider mismatch."""
        (tmp_path / "config.yaml").write_text(
            yaml.dump(
                {
                    "model": {"provider": "openai", "name": "opus-4"},
                    "agent": {"type": "claude_code"},
                    "thinking": "low",
                    "prompt_path": "/path/to/prompt.jinja",
                }
            )
        )
        result = _validate_resume_config(tmp_path, mock_run_cfg, mock_env_spec)
        assert len(result) == 1
        assert result[0][0] == "model.provider"
        assert result[0][1] == "openai"
        assert result[0][2] == "anthropic"


class TestHandleResumeValidation:
    """Tests for _handle_resume_validation helper function."""

    @pytest.fixture
    def mock_run_cfg(self):
        """Create a mock ResolvedRunConfig."""
        cfg = MagicMock()
        cfg.model_dump.return_value = {
            "model": {"provider": "anthropic", "name": "opus-4"},
            "agent": {"type": "claude_code"},
            "thinking": "low",
            "prompt_path": "/path/to/prompt.jinja",
        }
        return cfg

    @pytest.fixture
    def mock_env_spec(self):
        """Create a mock environment spec."""
        spec = MagicMock()
        spec.model_dump.return_value = {
            "type": "docker",
            "name": "python3.12",
            "docker": {"image": "python:3.12"},
        }
        return spec

    def test_no_exit_when_not_resuming(
        self, tmp_path, mock_run_cfg, mock_env_spec
    ):
        """Test no exit when resume=False."""
        # Should not raise
        _handle_resume_validation(
            tmp_path, mock_run_cfg, mock_env_spec, resume=False, overwrite=False
        )

    def test_no_exit_when_overwrite(
        self, tmp_path, mock_run_cfg, mock_env_spec
    ):
        """Test no exit when overwrite=True."""
        # Write mismatched config
        (tmp_path / "config.yaml").write_text(
            yaml.dump({"model": {"provider": "openai", "name": "gpt-4"}})
        )
        # Should not raise because overwrite=True
        _handle_resume_validation(
            tmp_path, mock_run_cfg, mock_env_spec, resume=True, overwrite=True
        )

    def test_exits_on_mismatch(self, tmp_path, mock_run_cfg, mock_env_spec):
        """Test exits when config mismatches detected."""
        (tmp_path / "config.yaml").write_text(
            yaml.dump(
                {
                    "model": {"provider": "openai", "name": "gpt-4"},
                    "agent": {"type": "claude_code"},
                    "thinking": "low",
                    "prompt_path": "/path/to/prompt.jinja",
                }
            )
        )
        with pytest.raises(typer.Exit) as excinfo:
            _handle_resume_validation(
                tmp_path,
                mock_run_cfg,
                mock_env_spec,
                resume=True,
                overwrite=False,
            )
        assert excinfo.value.exit_code == 1

    def test_no_exit_when_config_matches(
        self, tmp_path, mock_run_cfg, mock_env_spec
    ):
        """Test no exit when config matches."""
        (tmp_path / "config.yaml").write_text(
            yaml.dump(
                {
                    "model": {"provider": "anthropic", "name": "opus-4"},
                    "agent": {"type": "claude_code"},
                    "thinking": "low",
                    "prompt_path": "/path/to/prompt.jinja",
                }
            )
        )
        (tmp_path / "environment.yaml").write_text(
            yaml.dump(
                {
                    "type": "docker",
                    "name": "python3.12",
                    "docker": {"image": "python:3.12"},
                }
            )
        )
        # Should not raise
        _handle_resume_validation(
            tmp_path, mock_run_cfg, mock_env_spec, resume=True, overwrite=False
        )


class TestHandleEarlyCompletion:
    """Tests for _handle_early_completion helper function."""

    def test_returns_true_when_nothing_to_do(self, tmp_path):
        """Test returns True when no problems to run."""
        console = MagicMock()
        with patch(
            "slop_code.entrypoints.commands.run_agent._finalize_run_provenance_or_raise"
        ):
            result = _handle_early_completion(
                problem_names=[],  # Empty - nothing to do
                run_dir=tmp_path,
                problems_base_path=tmp_path,
                console=console,
                evaluate=False,
                requested=["prob1"],
            )
        assert result is True

    def test_returns_false_when_work_remains(self, tmp_path):
        """Test returns False when problems need running."""
        console = MagicMock()
        result = _handle_early_completion(
            problem_names=["prob1"],
            run_dir=tmp_path,
            problems_base_path=tmp_path,
            console=console,
            evaluate=False,
            requested=["prob1"],
        )
        assert result is False

    def test_no_work_completion_rechecks_staged_catalog(self, tmp_path):
        check = MagicMock()
        with patch(
            "slop_code.entrypoints.commands.run_agent._finalize_run_provenance_or_raise"
        ):
            result = _handle_early_completion(
                problem_names=[],
                run_dir=tmp_path,
                problems_base_path=tmp_path,
                console=MagicMock(),
                evaluate=False,
                requested=["prob1"],
                catalog_integrity_check=check,
            )

        assert result is True
        check.assert_called_once_with()

    @patch(
        "slop_code.entrypoints.commands.run_agent._create_checkpoint_results_and_summary"
    )
    def test_calls_summary_when_evaluate_true(self, mock_summary, tmp_path):
        """Test summary is generated when evaluate=True and nothing to do."""
        console = MagicMock()
        postprocessing = MagicMock(successful=True)
        postprocessing.model_dump.return_value = {"status": "completed"}
        mock_summary.return_value = postprocessing
        with patch(
            "slop_code.entrypoints.commands.run_agent._finalize_run_provenance_or_raise"
        ) as finalize:
            result = _handle_early_completion(
                problem_names=[],
                run_dir=tmp_path,
                problems_base_path=tmp_path,
                console=console,
                evaluate=True,
                requested=["prob1"],
            )
        assert result is True
        mock_summary.assert_called_once()
        finalize.assert_called_once_with(
            tmp_path,
            status="completed",
            details={
                "no_work": True,
                "postprocessing": {"status": "completed"},
            },
        )

    @patch(
        "slop_code.entrypoints.commands.run_agent._create_checkpoint_results_and_summary"
    )
    def test_no_summary_when_evaluate_false(self, mock_summary, tmp_path):
        """Test no summary when evaluate=False."""
        console = MagicMock()
        with patch(
            "slop_code.entrypoints.commands.run_agent._finalize_run_provenance_or_raise"
        ) as finalize:
            result = _handle_early_completion(
                problem_names=[],
                run_dir=tmp_path,
                problems_base_path=tmp_path,
                console=console,
                evaluate=False,
                requested=["prob1"],
            )
        assert result is True
        mock_summary.assert_not_called()
        finalize.assert_called_once_with(
            tmp_path,
            status="completed",
            details={"no_work": True, "postprocessing": None},
        )

    @patch(
        "slop_code.entrypoints.commands.run_agent._create_checkpoint_results_and_summary"
    )
    def test_incomplete_postprocessing_exits_nonzero(
        self, mock_summary, tmp_path
    ):
        """A no-work resume cannot silently bless an incomplete report."""
        mock_summary.return_value = MagicMock(successful=False)

        mock_summary.return_value.model_dump.return_value = {
            "status": "incomplete"
        }
        with (
            patch(
                "slop_code.entrypoints.commands.run_agent._finalize_run_provenance_or_raise"
            ) as finalize,
            pytest.raises(typer.Exit) as exc_info,
        ):
            _handle_early_completion(
                problem_names=[],
                run_dir=tmp_path,
                problems_base_path=tmp_path,
                console=MagicMock(),
                evaluate=True,
                requested=["prob1"],
            )

        assert exc_info.value.exit_code == 1
        finalize.assert_called_once_with(
            tmp_path,
            status="incomplete_postprocessing",
            details={
                "no_work": True,
                "postprocessing": {"status": "incomplete"},
            },
        )


class TestReadOnlyProblemFiltering:
    def test_completed_stale_run_info_is_scheduled_for_metadata_repair(
        self,
        tmp_path: Path,
    ) -> None:
        output_path = tmp_path / "run" / "prob1"
        output_path.mkdir(parents=True)
        problem = MagicMock(entry_file="main.py")
        problem.iterate_checkpoint_items.return_value = [
            ("checkpoint_1", MagicMock(name="checkpoint_1"))
        ]
        resume_info = ResumeInfo(
            resume_from_checkpoint="",
            completed_checkpoints=["checkpoint_1"],
            last_snapshot_dir=output_path / "checkpoint_1" / "snapshot",
            prior_usage=UsageTracker(cost=1.0, steps=2),
            run_info_reconciliation_required=True,
        )

        with (
            patch(
                "slop_code.entrypoints.commands.run_agent.ProblemConfig.from_yaml",
                return_value=problem,
            ),
            patch(
                "slop_code.entrypoints.commands.run_agent.detect_resume_point",
                return_value=resume_info,
            ),
        ):
            needs_rerun, reason = _check_problem_needs_rerun(
                tmp_path / "run",
                "prob1",
                tmp_path / "problems" / "prob1",
                "prompt",
                MagicMock(),
            )

        assert needs_rerun is True
        assert reason == "stale run_info requires artifact reconciliation"

    @pytest.mark.parametrize("overwrite", [False, True])
    def test_read_only_filter_never_clears_outputs(
        self, tmp_path, overwrite
    ):
        with (
            patch(
                "slop_code.entrypoints.commands.run_agent._check_problem_needs_rerun",
                return_value=(True, "incomplete"),
            ),
            patch(
                "slop_code.entrypoints.commands.run_agent._clear_problem_outputs"
            ) as clear,
        ):
            to_run, _, _ = _filter_problems_for_execution(
                tmp_path,
                ["prob1"],
                tmp_path,
                "prompt",
                MagicMock(),
                overwrite=overwrite,
                resume=True,
                read_only=True,
            )

        assert to_run == ["prob1"]
        clear.assert_not_called()


class TestStrictPostprocessing:
    """Completeness failures must survive as durable run evidence."""

    def test_postprocessing_target_symlink_is_not_followed(
        self,
        tmp_path: Path,
    ) -> None:
        outside = tmp_path.parent / f"{tmp_path.name}-outside-postprocessing"
        outside.write_text("preserve me\n", encoding="utf-8")
        (tmp_path / POSTPROCESSING_FILENAME).symlink_to(outside)
        result = PostprocessingResult(
            status="incomplete",
            expected_problem_names=[],
            observed_problem_names=[],
            executed_problem_names=[],
            expected_checkpoints=0,
            report_count=0,
            summary_created=False,
            scb_check=None,
            errors=[],
        )

        with pytest.raises(OSError, match="symlink"):
            _write_postprocessing_result(tmp_path, result)

        assert outside.read_text(encoding="utf-8") == "preserve me\n"

    @staticmethod
    def _summary(
        tmp_path: Path,
        *,
        expected_problems: int,
        expected_checkpoints: int,
        measured_checkpoints: int,
    ) -> MagicMock:
        summary = MagicMock()
        summary.expected_problems = expected_problems
        summary.scb_check.expected_checkpoints = expected_checkpoints
        summary.scb_check.measured_checkpoints = measured_checkpoints
        summary.scb_check.failed_checkpoints = 0
        summary.scb_check.missing_snapshot_checkpoints = 0
        summary.scb_check.missing_metadata_checkpoints = 0
        summary.scb_check.missing_checkpoint_records = (
            expected_checkpoints - measured_checkpoints
        )
        summary.scb_check.unmeasured_checkpoints = (
            expected_checkpoints - measured_checkpoints
        )
        summary.scb_check.coverage_pct = (
            measured_checkpoints / expected_checkpoints * 100
        )
        summary.scb_check.requested_version = "0.1.3"
        summary.scb_check.resolved_versions = ["0.1.3"]
        summary.scb_check.model_dump.return_value = {
            "requested_version": "0.1.3",
            "resolved_versions": ["0.1.3"],
            "expected_checkpoints": expected_checkpoints,
            "measured_checkpoints": measured_checkpoints,
            "failed_checkpoints": 0,
            "missing_snapshot_checkpoints": 0,
            "missing_metadata_checkpoints": 0,
            "missing_checkpoint_records": (
                expected_checkpoints - measured_checkpoints
            ),
            "unmeasured_checkpoints": (
                expected_checkpoints - measured_checkpoints
            ),
            "coverage_pct": (
                measured_checkpoints / expected_checkpoints * 100
            ),
        }
        return summary

    def test_complete_reports_receive_completed_status(self, tmp_path):
        (tmp_path / "prob1").mkdir()
        (tmp_path / "config.yaml").write_text(
            yaml.safe_dump({"problems": ["prob1"]})
        )
        reports = [
            {"problem": "prob1", "checkpoint": "checkpoint_1"},
            {"problem": "prob1", "checkpoint": "checkpoint_2"},
        ]
        summary = self._summary(
            tmp_path,
            expected_problems=1,
            expected_checkpoints=2,
            measured_checkpoints=2,
        )
        problem = MagicMock(
            checkpoints={"checkpoint_1": MagicMock(), "checkpoint_2": MagicMock()}
        )

        def save_summary(*args, **kwargs):
            (tmp_path / SUMMARY_FILENAME).write_text("{}\n")
            return summary

        with (
            patch(
                "slop_code.entrypoints.commands.run_agent.count_expected_checkpoints",
                return_value=2,
            ),
            patch(
                "slop_code.entrypoints.commands.run_agent.ProblemConfig.from_yaml",
                return_value=problem,
            ),
            patch(
                "slop_code.entrypoints.commands.run_agent.evaluation_entry.create_problem_reports",
                return_value=(reports, []),
            ),
            patch(
                "slop_code.entrypoints.commands.run_agent.display_and_save_summary",
                side_effect=save_summary,
            ),
        ):
            result = _create_checkpoint_results_and_summary(
                tmp_path,
                tmp_path,
                ["prob1"],
                MagicMock(),
            )

        assert result.successful is True
        assert result.report_count == 2
        durable = json.loads((tmp_path / POSTPROCESSING_FILENAME).read_text())
        assert durable["status"] == "completed"

    def test_named_profile_requires_evaluation_and_inference_evidence(
        self,
        tmp_path,
    ):
        problem_dir = tmp_path / "prob1"
        for checkpoint_name in ("checkpoint_1", "checkpoint_2"):
            (problem_dir / checkpoint_name).mkdir(parents=True)
        (tmp_path / "config.yaml").write_text(
            yaml.safe_dump(
                {
                    "profile": "paper-v2-reference",
                    "problems": ["prob1"],
                }
            )
        )
        reports = [
            {"problem": "prob1", "checkpoint": "checkpoint_1"},
            {"problem": "prob1", "checkpoint": "checkpoint_2"},
        ]
        summary = self._summary(
            tmp_path,
            expected_problems=1,
            expected_checkpoints=2,
            measured_checkpoints=2,
        )
        problem = MagicMock(
            checkpoints={"checkpoint_1": MagicMock(), "checkpoint_2": MagicMock()}
        )

        def save_summary(*args, **kwargs):
            (tmp_path / SUMMARY_FILENAME).write_text("{}\n")
            return summary

        with (
            patch(
                "slop_code.entrypoints.commands.run_agent.count_expected_checkpoints",
                return_value=2,
            ),
            patch(
                "slop_code.entrypoints.commands.run_agent.ProblemConfig.from_yaml",
                return_value=problem,
            ),
            patch(
                "slop_code.entrypoints.commands.run_agent.evaluation_entry.create_problem_reports",
                return_value=(reports, []),
            ),
            patch(
                "slop_code.entrypoints.commands.run_agent.display_and_save_summary",
                side_effect=save_summary,
            ),
        ):
            result = _create_checkpoint_results_and_summary(
                tmp_path,
                tmp_path,
                ["prob1"],
                MagicMock(),
            )

        assert result.successful is False
        kinds = [error["kind"] for error in result.errors]
        assert kinds.count("missing_evaluation_evidence") == 2
        assert kinds.count("missing_inference_evidence") == 2

    @pytest.mark.parametrize(
        ("field", "impossible_value", "message"),
        [
            ("cost", -0.01, "cost must be"),
            ("duration", float("nan"), "duration must be"),
            ("steps", -1, "steps must be"),
            ("input", -1, "input must be"),
            ("passed_tests", 5, "passed_tests cannot exceed total_tests"),
        ],
    )
    def test_named_profile_rejects_impossible_metrics_before_aggregation(
        self,
        tmp_path: Path,
        field: str,
        impossible_value: int | float,
        message: str,
    ) -> None:
        checkpoint_dir = tmp_path / "prob1" / "checkpoint_1"
        checkpoint_dir.mkdir(parents=True)
        for filename in (EVALUATION_FILENAME, INFERENCE_RESULT_FILENAME):
            (checkpoint_dir / filename).write_text("{}\n", encoding="utf-8")
        (tmp_path / "config.yaml").write_text(
            yaml.safe_dump(
                {"profile": "paper-v2-reference", "problems": ["prob1"]}
            ),
            encoding="utf-8",
        )
        report: dict[str, object] = {
            "problem": "prob1",
            "checkpoint": "checkpoint_1",
            "cost": 0.0,
            "duration": 1.0,
            "steps": 1,
            "input": 0,
            "output": 0,
            "cache_read": 0,
            "cache_write": 0,
            "reasoning": 0,
            "passed_tests": 4,
            "total_tests": 4,
            "core_passed": 1,
            "core_total": 1,
            "functionality_passed": 1,
            "functionality_total": 1,
            "error_passed": 1,
            "error_total": 1,
            "regression_passed": 1,
            "regression_total": 1,
            "strict_pass_rate": 1.0,
            "core_pass_rate": 1.0,
            "isolated_pass_rate": 1.0,
        }
        report[field] = impossible_value
        summary = self._summary(
            tmp_path,
            expected_problems=1,
            expected_checkpoints=1,
            measured_checkpoints=1,
        )
        problem = MagicMock(checkpoints={"checkpoint_1": MagicMock()})

        def save_summary(*args, **kwargs):
            (tmp_path / SUMMARY_FILENAME).write_text("{}\n")
            return summary

        with (
            patch(
                "slop_code.entrypoints.commands.run_agent.count_expected_checkpoints",
                return_value=1,
            ),
            patch(
                "slop_code.entrypoints.commands.run_agent.ProblemConfig.from_yaml",
                return_value=problem,
            ),
            patch(
                "slop_code.entrypoints.commands.run_agent.evaluation_entry.create_problem_reports",
                return_value=([report], []),
            ),
            patch(
                "slop_code.entrypoints.commands.run_agent.display_and_save_summary",
                side_effect=save_summary,
            ),
        ):
            result = _create_checkpoint_results_and_summary(
                tmp_path,
                tmp_path,
                ["prob1"],
                MagicMock(),
            )

        domain_errors = [
            error
            for error in result.errors
            if error["kind"] == "invalid_checkpoint_metric_domain"
        ]
        assert any(message in error["message"] for error in domain_errors)
        assert result.report_count == 0
        assert not (tmp_path / CHECKPOINT_RESULTS_FILENAME).exists()

    def test_missing_problem_and_coverage_are_durable_non_success(
        self, tmp_path
    ):
        (tmp_path / "prob1").mkdir()
        (tmp_path / "config.yaml").write_text(
            yaml.safe_dump({"problems": ["prob1", "prob2"]})
        )
        # A stale row from a previous selection must not survive regeneration.
        (tmp_path / CHECKPOINT_RESULTS_FILENAME).write_text(
            '{"problem":"stale","checkpoint":"checkpoint_9"}\n'
        )
        reports = [
            {"problem": "prob1", "checkpoint": "checkpoint_1"},
        ]
        summary = self._summary(
            tmp_path,
            expected_problems=2,
            expected_checkpoints=3,
            measured_checkpoints=1,
        )

        def problem_config(path):
            checkpoint_count = 1 if path.name == "prob1" else 2
            return MagicMock(
                checkpoints={
                    f"checkpoint_{index}": MagicMock()
                    for index in range(1, checkpoint_count + 1)
                }
            )

        def save_summary(*args, **kwargs):
            (tmp_path / SUMMARY_FILENAME).write_text("{}\n")
            return summary

        with (
            patch(
                "slop_code.entrypoints.commands.run_agent.count_expected_checkpoints",
                return_value=3,
            ),
            patch(
                "slop_code.entrypoints.commands.run_agent.ProblemConfig.from_yaml",
                side_effect=problem_config,
            ),
            patch(
                "slop_code.entrypoints.commands.run_agent.evaluation_entry.create_problem_reports",
                return_value=(reports, []),
            ),
            patch(
                "slop_code.entrypoints.commands.run_agent.display_and_save_summary",
                side_effect=save_summary,
            ),
        ):
            result = _create_checkpoint_results_and_summary(
                tmp_path,
                tmp_path,
                ["prob1", "prob2"],
                MagicMock(),
            )

        assert result.successful is False
        assert {error["kind"] for error in result.errors} >= {
            "missing_problem_output",
            "missing_checkpoint_reports",
            "checkpoint_report_count_mismatch",
            "incomplete_scb_check_coverage",
        }
        rows = [
            json.loads(line)
            for line in (tmp_path / CHECKPOINT_RESULTS_FILENAME)
            .read_text()
            .splitlines()
        ]
        assert rows == reports
        durable = json.loads((tmp_path / POSTPROCESSING_FILENAME).read_text())
        assert durable["status"] == "incomplete"

    @pytest.mark.parametrize(
        ("problem_success", "postprocessing_success", "expected"),
        [
            (True, True, "completed"),
            (False, True, "incomplete_problem_execution"),
            (True, False, "incomplete_postprocessing"),
            (
                False,
                False,
                "incomplete_problem_execution_and_postprocessing",
            ),
        ],
    )
    def test_final_status_cannot_hide_postprocessing_failure(
        self,
        problem_success,
        postprocessing_success,
        expected,
    ):
        task = MagicMock(success=problem_success)
        postprocessing = PostprocessingResult(
            status="completed" if postprocessing_success else "incomplete",
            expected_problem_names=["prob1"],
            observed_problem_names=["prob1"],
            executed_problem_names=["prob1"],
            expected_checkpoints=1,
            report_count=1,
            summary_created=True,
            scb_check={},
            errors=[] if postprocessing_success else [{"kind": "error"}],
        )

        assert _final_run_status([task], postprocessing) == expected


class TestRunFinalization:
    def test_preflight_failure_is_finalized_and_resumable(
        self,
        tmp_path: Path,
    ) -> None:
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        _initialize_test_provenance(tmp_path, run_dir, run_dir)
        evidence = {"status": "failed", "errors": ["catalog drift"]}
        (run_dir / "scbench_v2_preflight.json").write_text(
            json.dumps({"attempts": [evidence]}),
            encoding="utf-8",
        )
        primary = RuntimeError("preflight failed")

        _finalize_initialization_error(
            repository_root=tmp_path,
            run_dir=run_dir,
            primary_error=primary,
            source_image_name="",
            base_image_name="",
            agent_image_name="",
            preflight=evidence,
            executed_problem_names=["problem"],
        )

        saved = json.loads((run_dir / "provenance.json").read_text())
        assert saved["final_status"] == "failed"
        assert saved["preflight"] == evidence
        assert saved["artifacts"]["complete"] is False
        provenance.validate_resumable_provenance(run_dir)

    def test_docker_build_failure_binds_written_configs_and_is_resumable(
        self,
        tmp_path: Path,
    ) -> None:
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        _initialize_test_provenance(tmp_path, run_dir, run_dir)
        (run_dir / "config.yaml").write_text("profile: test\n")
        (run_dir / "environment.yaml").write_text("type: docker\n")
        primary = RuntimeError("docker build failed")

        _finalize_initialization_error(
            repository_root=tmp_path,
            run_dir=run_dir,
            primary_error=primary,
            source_image_name="source",
            base_image_name="base",
            agent_image_name="agent",
            preflight={"status": "verified"},
            executed_problem_names=["problem"],
        )

        saved = json.loads((run_dir / "provenance.json").read_text())
        assert saved["final_status"] == "failed"
        assert saved["inputs"]["resolved_config"]["sha256"] is not None
        assert saved["inputs"]["resolved_environment"]["sha256"] is not None
        provenance.validate_resumable_provenance(run_dir)

    def test_primary_error_survives_provenance_failure(
        self,
        tmp_path: Path,
    ) -> None:
        primary = ValueError("agent solve failed")
        with patch(
            "slop_code.entrypoints.commands.run_agent.finalize_run_provenance",
            side_effect=OSError("artifact disappeared"),
        ) as finalize:
            _finalize_after_primary_error(tmp_path, primary)

        finalize.assert_called_once_with(
            tmp_path,
            status="failed",
            error_type="ValueError",
            checksum_artifacts=False,
        )
        assert any(
            "Secondary provenance finalization failure" in note
            and "artifact disappeared" in note
            for note in (primary.__notes__ or [])
        )

    def test_hashing_failure_leaves_incomplete_provenance_marker(
        self,
        tmp_path: Path,
    ) -> None:
        with (
            patch(
                "slop_code.entrypoints.commands.run_agent.finalize_run_provenance",
                side_effect=[OSError("hash failed"), None],
            ) as finalize,
            pytest.raises(OSError, match="hash failed"),
        ):
            _finalize_run_provenance_or_raise(
                tmp_path,
                status="completed",
                details={"postprocessing": {"status": "completed"}},
            )

        assert finalize.call_count == 2
        fallback = finalize.call_args_list[1]
        assert fallback.args == (tmp_path,)
        assert fallback.kwargs["status"] == "incomplete_provenance"
        assert fallback.kwargs["error_type"] == "OSError"
        assert fallback.kwargs["checksum_artifacts"] is False
        assert fallback.kwargs["details"]["requested_status"] == "completed"

    @pytest.mark.parametrize(
        "status",
        [
            "incomplete_problem_execution",
            "incomplete_postprocessing",
            "incomplete_problem_execution_and_postprocessing",
            "incomplete_provenance",
        ],
    )
    def test_incomplete_status_exits_nonzero(self, status: str) -> None:
        with pytest.raises(typer.Exit) as exc_info:
            _exit_on_incomplete_run(status)

        assert exc_info.value.exit_code == 1

    def test_completed_status_returns_normally(self) -> None:
        _exit_on_incomplete_run("completed")


class TestCreateTaskConfig:
    """Tests for _create_task_config helper function."""

    def test_creates_valid_config(self):
        """Test task config creation with all parameters."""
        from slop_code.entrypoints import problem_runner

        mock_run_cfg = MagicMock()
        mock_run_cfg.thinking = "low"
        mock_run_cfg.thinking_max_tokens = None
        mock_run_cfg.prompt_content = "template content"
        mock_run_cfg.pass_policy = MagicMock()
        mock_run_cfg.one_shot = MagicMock()

        result = _create_task_config(
            problem_base_path=Path("/problems"),
            run_dir=Path("/output"),
            env_spec=MagicMock(),
            agent_config=MagicMock(),
            model_def=MagicMock(),
            credential=MagicMock(),
            run_cfg=mock_run_cfg,
            seed=42,
            verbosity=1,
            debug=False,
            evaluate=True,
            live_progress=True,
            image_name="test:image",
            resume=False,
        )

        assert isinstance(result, problem_runner.RunTaskConfig)
        assert result.run_dir == Path("/output")
        assert result.image == "test:image"
        assert result.seed == 42
        assert result.debug is False
        assert result.disable_evaluation is False
        assert result.resume is False

    def test_disable_evaluation_flag(self):
        """Test disable_evaluation is inverse of evaluate."""
        mock_run_cfg = MagicMock()
        mock_run_cfg.thinking = "low"
        mock_run_cfg.thinking_max_tokens = None
        mock_run_cfg.prompt_content = "template"
        mock_run_cfg.pass_policy = MagicMock()
        mock_run_cfg.one_shot = MagicMock()

        result = _create_task_config(
            problem_base_path=Path("/problems"),
            run_dir=Path("/output"),
            env_spec=MagicMock(),
            agent_config=MagicMock(),
            model_def=MagicMock(),
            credential=MagicMock(),
            run_cfg=mock_run_cfg,
            seed=None,
            verbosity=0,
            debug=True,
            evaluate=False,  # evaluate=False
            live_progress=False,
            image_name="",
            resume=True,
        )

        assert result.disable_evaluation is True  # Should be inverse
        assert result.debug is True
        assert result.resume is True


class TestPrepareRunArtifacts:
    """Tests for _prepare_run_artifacts helper function."""

    def test_writes_problem_catalog_manifest(self, tmp_path):
        """Run start persists problem_catalog.json metadata."""
        run_cfg = MagicMock()
        run_cfg.model_dump.return_value = {
            "model": {"provider": "anthropic", "name": "sonnet-4.5"}
        }
        env_spec = MagicMock()
        env_spec.model_dump.return_value = {"type": "local", "name": "local"}
        agent_config = MagicMock()
        agent_config.docker_template = None
        manifest = CatalogManifest(version="v1.0.0", commit="abc123")

        image_name = _prepare_run_artifacts(
            run_dir=tmp_path,
            env_spec=env_spec,
            agent_config=agent_config,
            run_cfg=run_cfg,
            catalog_manifest=manifest,
        )

        assert image_name == ""
        saved_manifest = yaml.safe_load(
            (tmp_path / "problem_catalog.json").read_text()
        )
        assert saved_manifest == {"version": "v1.0.0", "commit": "abc123"}

    @pytest.mark.parametrize("target_name", ["config.yaml", "environment.yaml"])
    def test_configuration_symlink_is_never_followed(
        self,
        tmp_path: Path,
        target_name: str,
    ) -> None:
        outside = tmp_path / "outside.yaml"
        outside.write_text("untouched\n", encoding="utf-8")
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        (run_dir / target_name).symlink_to(outside)
        run_cfg = MagicMock()
        run_cfg.model_dump.return_value = {"profile": "test"}
        env_spec = MagicMock()
        env_spec.model_dump.return_value = {
            "type": "local",
            "name": "local",
        }
        agent_config = MagicMock()
        agent_config.docker_template = None
        manifest = CatalogManifest(version="v1.0.0", commit="abc123")

        with pytest.raises(OSError, match="symlink"):
            _prepare_run_artifacts(
                run_dir=run_dir,
                env_spec=env_spec,
                agent_config=agent_config,
                run_cfg=run_cfg,
                catalog_manifest=manifest,
            )

        assert outside.read_text(encoding="utf-8") == "untouched\n"


class TestDiscoverProblems:
    """Tests for _discover_problems helper function."""

    def test_discovers_valid_problems(self, tmp_path):
        """Test discovery of valid problems."""
        prob_dir = tmp_path / "test_problem"
        prob_dir.mkdir()
        (prob_dir / "config.yaml").write_text(
            yaml.dump(
                {
                    "name": "test_problem",
                    "version": 1,
                    "description": "Test problem",
                    "category": "test",
                    "entry_file": "main.py",
                    "checkpoints": {"checkpoint_1": {"version": 1, "order": 1}},
                }
            )
        )
        result = _discover_problems(tmp_path)
        assert "test_problem" in result

    def test_skips_not_set_category(self, tmp_path):
        """Test problems with NOT_SET category are skipped."""
        prob_dir = tmp_path / "not_set_prob"
        prob_dir.mkdir()
        (prob_dir / "config.yaml").write_text(
            yaml.dump(
                {
                    "name": "not_set_prob",
                    "category": "NOT_SET",
                    "checkpoints": {"checkpoint_1": {"order": 1}},
                }
            )
        )
        result = _discover_problems(tmp_path)
        assert "not_set_prob" not in result

    def test_skips_invalid_config(self, tmp_path):
        """Test problems with invalid config are skipped."""
        prob_dir = tmp_path / "invalid_prob"
        prob_dir.mkdir()
        (prob_dir / "config.yaml").write_text("invalid: yaml: [")
        result = _discover_problems(tmp_path)
        assert "invalid_prob" not in result

    def test_skips_no_checkpoints(self, tmp_path):
        """Test problems with no checkpoints are skipped."""
        prob_dir = tmp_path / "no_checkpoints"
        prob_dir.mkdir()
        (prob_dir / "config.yaml").write_text(
            yaml.dump(
                {
                    "name": "no_checkpoints",
                    "category": "test",
                    "checkpoints": {},
                }
            )
        )
        result = _discover_problems(tmp_path)
        assert "no_checkpoints" not in result

    def test_multiple_problems(self, tmp_path):
        """Test discovering multiple valid problems."""
        for name in ["prob_a", "prob_b", "prob_c"]:
            prob_dir = tmp_path / name
            prob_dir.mkdir()
            (prob_dir / "config.yaml").write_text(
                yaml.dump(
                    {
                        "name": name,
                        "version": 1,
                        "description": f"Test problem {name}",
                        "category": "test",
                        "entry_file": "main.py",
                        "checkpoints": {
                            "checkpoint_1": {"version": 1, "order": 1}
                        },
                    }
                )
            )
        result = _discover_problems(tmp_path)
        assert len(result) == 3
        assert set(result) == {"prob_a", "prob_b", "prob_c"}

    def test_returns_sorted(self, tmp_path):
        """Test that problems are returned sorted."""
        for name in ["zebra", "apple", "mango"]:
            prob_dir = tmp_path / name
            prob_dir.mkdir()
            (prob_dir / "config.yaml").write_text(
                yaml.dump(
                    {
                        "name": name,
                        "version": 1,
                        "description": f"Test problem {name}",
                        "category": "test",
                        "entry_file": "main.py",
                        "checkpoints": {
                            "checkpoint_1": {"version": 1, "order": 1}
                        },
                    }
                )
            )
        result = _discover_problems(tmp_path)
        assert result == ["apple", "mango", "zebra"]


class TestValidateProblemPaths:
    """Tests for _validate_problem_paths helper function."""

    def test_passes_for_existing_paths(self, tmp_path):
        """Test validation passes when paths exist."""
        prob_dir = tmp_path / "exists"
        prob_dir.mkdir()
        (prob_dir / "config.yaml").write_text("name: exists\n")
        # Should not raise
        _validate_problem_paths(["exists"], tmp_path)

    def test_exits_for_missing_path(self, tmp_path):
        """Test exits when path doesn't exist."""
        with pytest.raises(typer.Exit) as excinfo:
            _validate_problem_paths(["missing"], tmp_path)
        assert excinfo.value.exit_code == 1

    def test_multiple_existing_paths(self, tmp_path):
        """Test validation with multiple existing paths."""
        for name in ["prob1", "prob2", "prob3"]:
            prob_dir = tmp_path / name
            prob_dir.mkdir()
            (prob_dir / "config.yaml").write_text(f"name: {name}\n")
        # Should not raise
        _validate_problem_paths(["prob1", "prob2", "prob3"], tmp_path)

    def test_fails_on_first_missing(self, tmp_path):
        """Test validation fails on first missing path."""
        (tmp_path / "exists1").mkdir()
        ((tmp_path / "exists1") / "config.yaml").write_text("name: exists1\n")
        (tmp_path / "exists2").mkdir()
        ((tmp_path / "exists2") / "config.yaml").write_text("name: exists2\n")
        with pytest.raises(typer.Exit):
            _validate_problem_paths(["exists1", "missing", "exists2"], tmp_path)

    def test_empty_list_passes(self, tmp_path):
        """Test validation passes with empty list."""
        # Should not raise
        _validate_problem_paths([], tmp_path)
