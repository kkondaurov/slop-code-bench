from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
from collections.abc import Callable
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import typer
import yaml
from rich.console import Console

from slop_code import problem_catalog
from slop_code.agent_runner.agent import AgentConfigBase
from slop_code.agent_runner.credentials import API_KEY_STORE
from slop_code.agent_runner.credentials import CredentialNotFoundError
from slop_code.agent_runner.credentials import ProviderCredential
from slop_code.agent_runner.registry import build_agent_config
from slop_code.agent_runner.resume import detect_resume_point
from slop_code.common import CHECKPOINT_RESULTS_FILENAME
from slop_code.common import CONFIG_FILENAME
from slop_code.common import ENV_CONFIG_NAME
from slop_code.common import EVALUATION_FILENAME
from slop_code.common import INFERENCE_RESULT_FILENAME
from slop_code.common import POSTPROCESSING_FILENAME
from slop_code.common import SUMMARY_FILENAME
from slop_code.common import serialize_path_dict
from slop_code.common.atomic import atomic_write_text
from slop_code.common.llms import ModelCatalog
from slop_code.common.llms import ModelDefinition
from slop_code.common.temp import configure_named_profile_temp_root
from slop_code.entrypoints import evaluation as evaluation_entry
from slop_code.entrypoints import problem_runner
from slop_code.entrypoints import utils
from slop_code.entrypoints.commands import common
from slop_code.entrypoints.config import ResolvedRunConfig
from slop_code.entrypoints.config import load_run_config
from slop_code.entrypoints.config import loader as config_loader
from slop_code.entrypoints.config.loader import load_config_from_run_dir
from slop_code.entrypoints.evaluation.metrics import update_results_jsonl
from slop_code.entrypoints.utils import count_expected_checkpoints
from slop_code.entrypoints.utils import display_and_save_summary
from slop_code.evaluation import ProblemConfig
from slop_code.execution import EnvironmentSpecType
from slop_code.execution import docker_runtime
from slop_code.logging import get_logger
from slop_code.provenance import PROVENANCE_FILENAME
from slop_code.provenance import ProvenanceIntegrityError
from slop_code.provenance import finalize_run_provenance
from slop_code.provenance import refresh_run_provenance_context
from slop_code.provenance import start_run_provenance
from slop_code.provenance import validate_resumable_provenance
from slop_code.scbench_v2 import SCB_CHECK_PROJECT_ENV
from slop_code.scbench_v2 import SCB_CHECK_VENV_ENV
from slop_code.scbench_v2 import STAGED_CATALOG_PATH
from slop_code.scbench_v2 import STAGED_EVALUATOR_PATH
from slop_code.scbench_v2 import NamedProfileContext
from slop_code.scbench_v2 import ScbenchV2PreflightError
from slop_code.scbench_v2 import run_named_profile_preflight
from slop_code.scbench_v2 import verify_staged_catalog
from slop_code.scbench_v2 import verify_staged_evaluator

logger = get_logger(__name__)


@dataclass(frozen=True)
class PostprocessingResult:
    """Durable completeness evidence for benchmark report generation."""

    status: str
    expected_problem_names: list[str]
    observed_problem_names: list[str]
    executed_problem_names: list[str]
    expected_checkpoints: int
    report_count: int
    summary_created: bool
    scb_check: dict[str, Any] | None
    errors: list[dict[str, str]]

    @property
    def successful(self) -> bool:
        return self.status == "completed"

    def model_dump(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return asdict(self)


def _named_checkpoint_domain_errors(
    report: dict[str, object],
) -> list[str]:
    """Reject impossible correctness and telemetry before aggregation."""
    errors: list[str] = []

    for key in ("cost", "duration"):
        value = report.get(key)
        if (
            not isinstance(value, int | float)
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or value < 0
        ):
            errors.append(f"{key} must be a finite non-negative number")

    for key in (
        "steps",
        "input",
        "output",
        "cache_read",
        "cache_write",
        "reasoning",
    ):
        value = report.get(key)
        if type(value) is not int or value < 0:
            errors.append(f"{key} must be a non-negative integer")

    count_pairs = (
        ("passed_tests", "total_tests"),
        ("core_passed", "core_total"),
        ("functionality_passed", "functionality_total"),
        ("error_passed", "error_total"),
        ("regression_passed", "regression_total"),
    )
    for passed_key, total_key in count_pairs:
        passed = report.get(passed_key)
        total = report.get(total_key)
        if type(passed) is not int or passed < 0:
            errors.append(f"{passed_key} must be a non-negative integer")
            continue
        if type(total) is not int or total < 0:
            errors.append(f"{total_key} must be a non-negative integer")
            continue
        if passed > total:
            errors.append(f"{passed_key} cannot exceed {total_key}")

    for key in ("strict_pass_rate", "core_pass_rate", "isolated_pass_rate"):
        value = report.get(key)
        if (
            not isinstance(value, int | float)
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or not 0 <= value <= 1
        ):
            errors.append(f"{key} must be a finite number between 0 and 1")

    totals = [report.get(key) for _, key in count_pairs[1:]]
    passed_counts = [report.get(key) for key, _ in count_pairs[1:]]
    if (
        all(type(value) is int and value >= 0 for value in totals)
        and report.get("total_tests") != sum(totals)
    ):
        errors.append("total_tests must equal the sum of group totals")
    if (
        all(type(value) is int and value >= 0 for value in passed_counts)
        and report.get("passed_tests") != sum(passed_counts)
    ):
        errors.append("passed_tests must equal the sum of group pass counts")

    rate_inputs = (
        ("strict_pass_rate", "passed_tests", "total_tests", None, None),
        ("core_pass_rate", "core_passed", "core_total", None, None),
        (
            "isolated_pass_rate",
            "passed_tests",
            "total_tests",
            "regression_passed",
            "regression_total",
        ),
    )
    for rate_key, passed_key, total_key, excluded_passed, excluded_total in (
        rate_inputs
    ):
        rate = report.get(rate_key)
        passed = report.get(passed_key)
        total = report.get(total_key)
        if (
            not isinstance(rate, int | float)
            or isinstance(rate, bool)
            or type(passed) is not int
            or type(total) is not int
        ):
            continue
        if excluded_passed is not None and excluded_total is not None:
            excluded_passed_value = report.get(excluded_passed)
            excluded_total_value = report.get(excluded_total)
            if (
                type(excluded_passed_value) is not int
                or type(excluded_total_value) is not int
            ):
                continue
            passed -= excluded_passed_value
            total -= excluded_total_value
        if passed < 0 or total < 0 or passed > total:
            continue
        expected_rate = passed / total if total else 0.0
        if not math.isclose(
            float(rate),
            expected_rate,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            errors.append(
                f"{rate_key} must equal {passed_key}/{total_key} "
                f"({expected_rate})"
            )
    return errors


def _write_postprocessing_result(
    run_dir: Path,
    result: PostprocessingResult,
) -> None:
    target = run_dir / POSTPROCESSING_FILENAME
    atomic_write_text(
        target,
        json.dumps(result.model_dump(), indent=2, sort_keys=True) + "\n",
    )


def _get_nested(data: dict[str, object], path: str) -> object | None:
    """Get nested value using dot notation (e.g., 'model.name').

    Args:
        data: Dictionary to search
        path: Dot-separated path to value

    Returns:
        Value at path, or None if not found
    """
    keys = path.split(".")
    current: object = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _validate_resume_config(
    run_dir: Path,
    run_cfg: ResolvedRunConfig,
    env_spec: EnvironmentSpecType,
) -> list[tuple[str, object, object]]:
    """Validate current config matches saved config for resume.

    Compares critical configuration fields that should not change between resume:
    - Model (provider, name)
    - Agent type
    - Thinking preset
    - Prompt template path
    - Environment (type, name, docker image)

    Args:
        run_dir: Path to existing run directory
        run_cfg: Current resolved run configuration
        env_spec: Current environment specification

    Returns:
        List of (field_name, saved_value, current_value) for mismatches.
        Empty list if configuration is compatible.
    """
    mismatches: list[tuple[str, object, object]] = []

    config_path = run_dir / CONFIG_FILENAME
    env_path = run_dir / ENV_CONFIG_NAME

    # No saved config - allow resume (old runs before this feature)
    if not config_path.exists():
        return []

    # Load saved config
    try:
        with config_path.open() as f:
            saved_config = yaml.safe_load(f)
    except (yaml.YAMLError, OSError) as e:
        logger.warning("Failed to load saved config.yaml", error=str(e))
        return []

    if saved_config is None:
        return []

    # Build current config dict for comparison
    current_config = run_cfg.model_dump(mode="json")

    # Fields to validate from config.yaml
    config_fields = [
        "model.provider",
        "model.name",
        "agent.type",
        "thinking",
        "prompt_path",
    ]

    for field in config_fields:
        saved_val = _get_nested(saved_config, field)
        current_val = _get_nested(current_config, field)

        # Skip if saved config doesn't have this field (schema evolution)
        if saved_val is None:
            continue

        # Normalize paths for comparison
        if field == "prompt_path":
            saved_val = str(saved_val) if saved_val else None
            current_val = str(current_val) if current_val else None

        if saved_val != current_val:
            mismatches.append((field, saved_val, current_val))

    # Load and validate environment config
    if env_path.exists():
        try:
            with env_path.open() as f:
                saved_env = yaml.safe_load(f)
        except (yaml.YAMLError, OSError) as e:
            logger.warning(
                "Failed to load saved environment.yaml", error=str(e)
            )
            saved_env = None

        if saved_env:
            current_env = env_spec.model_dump(mode="json")

            env_fields = ["type", "name"]
            for field in env_fields:
                saved_val = _get_nested(saved_env, field)
                current_val = _get_nested(current_env, field)

                if saved_val is None:
                    continue

                if saved_val != current_val:
                    mismatches.append(
                        (f"environment.{field}", saved_val, current_val)
                    )

            # Check Docker image if Docker environment
            if saved_env.get("type") == "docker":
                saved_image = _get_nested(saved_env, "docker.image")
                current_image = _get_nested(current_env, "docker.image")

                if saved_image and saved_image != current_image:
                    mismatches.append(
                        ("environment.docker.image", saved_image, current_image)
                    )

    return mismatches


def _clear_problem_outputs(run_dir: Path, problem_name: str) -> None:
    """Remove any prior outputs for a problem (but keep the run dir)."""
    shutil.rmtree(run_dir / problem_name, ignore_errors=True)


def _check_problem_needs_rerun(
    run_dir: Path,
    problem_name: str,
    problem_path: Path,
    prompt_template: str,
    environment: EnvironmentSpecType,
    *,
    require_evaluation: bool = True,
) -> tuple[bool, str | None]:
    """Check if a problem needs to be run based on checkpoint state.

    Uses detect_resume_point() which handles both:
    1. Structured detection from run_info.yaml if present
    2. Artifact-based fallback (inference_result.json + snapshot) if missing
    3. Prompt validation to detect spec changes

    Args:
        run_dir: The run output directory
        problem_name: Name of the problem
        problem_path: Path to the problem definition
        prompt_template: Current prompt template content
        environment: Current environment spec

    Returns:
        (needs_rerun, reason) - reason is None if doesn't need rerun,
        otherwise a human-readable explanation
    """
    output_path = run_dir / problem_name

    # If output directory doesn't exist, need to run
    if not output_path.exists():
        return True, "no output directory"

    # Load problem config for checkpoint validation
    try:
        problem_config = ProblemConfig.from_yaml(problem_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Failed to load problem config",
            problem=problem_name,
            error=str(exc),
        )
        return True, "invalid problem config"

    checkpoint_items = list(problem_config.iterate_checkpoint_items())
    checkpoint_names = [name for name, _ in checkpoint_items]
    checkpoints = [cp for _, cp in checkpoint_items]

    # Use detect_resume_point for both run_info.yaml and artifact-based detection
    resume_info = detect_resume_point(
        output_path,
        checkpoint_names,
        problem_config=problem_config,
        prompt_template=prompt_template,
        environment=environment,
        entry_file=problem_config.entry_file,
        checkpoints=checkpoints,
        require_evaluation=require_evaluation,
    )

    # No valid state found at all
    if resume_info is None:
        return True, "no valid checkpoint state"

    # All checkpoints completed and valid
    if (
        not resume_info.resume_from_checkpoint
        and not resume_info.evaluation_only_checkpoints
    ):
        if resume_info.run_info_reconciliation_required:
            return True, "stale run_info requires artifact reconciliation"
        return False, None

    # Some checkpoints need to be re-run
    reasons = []
    for status in resume_info.checkpoint_statuses:
        if not status.is_valid and status.reason:
            reasons.append(f"{status.name}: {status.reason.value}")
    reason_str = "; ".join(reasons) if reasons else "incomplete checkpoints"
    return True, reason_str


def _filter_problems_for_execution(
    run_dir: Path,
    problem_names: list[str],
    problem_path: Path,
    prompt_template: str,
    environment: EnvironmentSpecType,
    *,
    overwrite: bool,
    resume: bool,
    require_evaluation: bool = True,
    read_only: bool = False,
) -> tuple[list[str], list[str], dict[str, str]]:
    """Filter problems based on completion status and prompt changes.

    Determines which problems need to be run based on their completion state
    and whether prompts have changed since the last run.

    Args:
        run_dir: The run output directory
        problem_names: List of requested problem names
        problem_path: Base path to problem definitions
        prompt_template: Current prompt template content
        environment: Current environment spec
        overwrite: If True, rerun all problems regardless of state
        resume: If True, preserve partial checkpoint outputs
        read_only: If True, never clear outputs while selecting work

    Returns:
        Tuple of (problems_to_run, skipped_problems, rerun_reasons)
        where rerun_reasons maps problem name to why it needs rerun
    """
    if overwrite:
        # Clear all outputs and rerun everything
        if not read_only:
            for p in problem_names:
                _clear_problem_outputs(run_dir, p)
        return list(problem_names), [], {}

    to_run: list[str] = []
    skipped: list[str] = []
    rerun_reasons: dict[str, str] = {}

    for p in problem_names:
        needs_rerun, reason = _check_problem_needs_rerun(
            run_dir,
            p,
            problem_path / p,
            prompt_template,
            environment,
            require_evaluation=require_evaluation,
        )
        if needs_rerun:
            to_run.append(p)
            if reason:
                rerun_reasons[p] = reason
            if not resume and not read_only:
                _clear_problem_outputs(run_dir, p)
        else:
            skipped.append(p)

    return to_run, skipped, rerun_reasons


def _build_cli_flags(
    agent_config_path: str | None,
    environment_config_path: str | None,
    prompt_template_path: str | None,
    model_override: str | None,
) -> dict[str, object]:
    """Build CLI flags dict for config loading.

    Args:
        agent_config_path: Agent config override
        environment_config_path: Environment config override
        prompt_template_path: Prompt template override
        model_override: Model override in "provider/name" format

    Returns:
        Dictionary of CLI flags for config loading

    Raises:
        ValueError: If model_override format is invalid
    """
    cli_flags: dict[str, object] = {}

    if agent_config_path is not None:
        cli_flags["agent"] = agent_config_path

    if environment_config_path is not None:
        cli_flags["environment"] = environment_config_path

    if prompt_template_path is not None:
        cli_flags["prompt"] = prompt_template_path

    if model_override is not None:
        parsed = utils.parse_model_override(model_override)
        cli_flags["model"] = {
            "provider": parsed.provider,
            "name": parsed.name,
        }

    return cli_flags


def _resolve_environment_and_credentials(
    run_cfg: ResolvedRunConfig,
    provider_api_key_env: str | None,
) -> tuple[EnvironmentSpecType, ModelDefinition, ProviderCredential]:
    """Resolve environment spec, model definition, and credentials.

    Args:
        run_cfg: The resolved run configuration
        provider_api_key_env: Optional override for API key environment variable

    Returns:
        Tuple of (environment_spec, model_definition, credential)

    Raises:
        typer.Exit: If model not found in catalog or credentials unavailable
    """
    # Resolve environment spec from loaded config
    env_spec = config_loader.resolve_environment(
        run_cfg.environment_config_path or run_cfg.environment
    )
    env_spec_typed = cast("EnvironmentSpecType", env_spec)

    # Look up ModelDefinition from catalog
    model_def = ModelCatalog.get(run_cfg.model.name)
    if model_def is None:
        typer.echo(
            typer.style(
                f"Model '{run_cfg.model.name}' not found in catalog",
                fg=typer.colors.RED,
                bold=True,
            )
        )
        raise typer.Exit(1)

    # Resolve credential for the provider
    try:
        credential = API_KEY_STORE.resolve(
            run_cfg.model.provider,
            env_var_override=provider_api_key_env,
        )
    except CredentialNotFoundError as e:
        typer.echo(
            typer.style(
                f"Credential error: {e}", fg=typer.colors.RED, bold=True
            )
        )
        raise typer.Exit(1) from e

    return env_spec_typed, model_def, credential


def _build_agent_config(run_cfg: ResolvedRunConfig) -> AgentConfigBase:
    """Build agent configuration from resolved run config.

    Args:
        run_cfg: The resolved run configuration

    Returns:
        Built agent configuration

    Note:
        Always uses run_cfg.agent which already has CLI overrides applied,
        rather than re-loading from the file path.
    """
    return build_agent_config(run_cfg.agent)


def _discover_problems(problem_path: Path) -> list[str]:
    """Auto-discover problems from the problem directory.

    Args:
        problem_path: Base path containing problem directories

    Returns:
        List of valid problem names found
    """
    problem_names: list[str] = []
    for problem in problem_catalog.discover_problem_dirs(problem_path):
        try:
            cfg = ProblemConfig.from_yaml(problem)
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Problem config not valid",
                problem=problem.name,
                error=str(exc),
            )
            continue

        if cfg.category == "NOT_SET":
            continue

        checkpoints = [cp for _, cp in cfg.iterate_checkpoint_items()]
        if not checkpoints:
            typer.echo(
                typer.style(
                    f"Problem '{problem}' has no checkpoints",
                    fg=typer.colors.RED,
                    bold=True,
                )
            )
            continue
        problem_names.append(problem.name)

    return problem_names


def _validate_problem_paths(
    problem_names: list[str], problem_path: Path
) -> None:
    """Validate that all problem paths exist.

    Args:
        problem_names: List of problem names to validate
        problem_path: Base path containing problem directories

    Raises:
        typer.Exit: If any problem path does not exist
    """
    for problem in problem_names:
        full_path = problem_path / problem
        if not full_path.exists() or not (full_path / "config.yaml").exists():
            typer.echo(
                typer.style(
                    f"Problem path '{full_path}' does not exist.",
                    fg=typer.colors.RED,
                    bold=True,
                )
            )
            raise typer.Exit(1)


def _preview_dry_run(
    problem_names: list[str],
    run_dir: Path,
    problem_path: Path,
    prompt_template: str,
    env_spec: EnvironmentSpecType,
    *,
    require_evaluation: bool = True,
) -> None:
    """Preview what would be executed without making changes.

    Args:
        problem_names: List of problem names to preview
        run_dir: The run output directory
        problem_path: Base path to problem definitions
        prompt_template: Current prompt template content
        env_spec: Current environment spec
    """
    typer.echo(
        typer.style(
            "\nDRY RUN - Execution Preview:",
            fg=typer.colors.CYAN,
            bold=True,
        )
    )
    for problem_name in problem_names:
        full_problem_path = problem_path / problem_name
        try:
            problem_config = ProblemConfig.from_yaml(full_problem_path)
        except Exception as exc:  # noqa: BLE001
            typer.echo(
                typer.style(
                    f"\n{problem_name}: Failed to load config - {exc}",
                    fg=typer.colors.RED,
                )
            )
            continue

        output_path = run_dir / problem_name
        if not output_path.exists():
            typer.echo(
                typer.style(
                    f"\n{problem_name}:",
                    fg=typer.colors.CYAN,
                    bold=True,
                )
            )
            typer.echo("  Would start fresh (no existing run)")
            continue
        checkpoint_items = list(problem_config.iterate_checkpoint_items())
        checkpoint_names = [name for name, _ in checkpoint_items]
        checkpoints = [cp for _, cp in checkpoint_items]

        resume_info = detect_resume_point(
            output_path,
            checkpoint_names,
            problem_config=problem_config,
            prompt_template=prompt_template,
            environment=env_spec,
            entry_file=problem_config.entry_file,
            checkpoints=checkpoints,
            require_evaluation=require_evaluation,
        )

        # Skip fully completed problems (silently)
        if (
            resume_info
            and not resume_info.resume_from_checkpoint
            and not resume_info.evaluation_only_checkpoints
        ):
            continue

        typer.echo(
            typer.style(
                f"\n{problem_name}:",
                fg=typer.colors.CYAN,
                bold=True,
            )
        )

        if resume_info:
            typer.echo(f"  Resume from: {resume_info.resume_from_checkpoint}")
            typer.echo(
                f"  Completed: {', '.join(resume_info.completed_checkpoints)}"
            )
            if resume_info.invalidated_checkpoints:
                typer.echo("  Would quarantine and re-run:")
                for status in resume_info.checkpoint_statuses:
                    if not status.is_valid and status.reason:
                        typer.echo(
                            f"    - {status.name} ({status.reason.value})"
                        )
                existing_directories = [
                    output_path / cp_name
                    for cp_name in resume_info.invalidated_checkpoints
                    if (output_path / cp_name).exists()
                ]
                if existing_directories:
                    typer.echo("  Directories to quarantine:")
                    for cp_dir in existing_directories:
                        typer.echo(f"    - {cp_dir}")
            if resume_info.evaluation_only_checkpoints:
                typer.echo("  Would re-evaluate without inference:")
                for cp_name in resume_info.evaluation_only_checkpoints:
                    typer.echo(f"    - {cp_name}")
        else:
            if output_path.exists():
                typer.echo(
                    "  Would start fresh (no valid completed checkpoints)"
                )
            else:
                typer.echo("  Would start fresh (no existing run)")

    typer.echo(
        typer.style(
            "\nNo changes made (dry run).",
            fg=typer.colors.CYAN,
            bold=True,
        )
    )


def _prepare_run_artifacts(
    run_dir: Path,
    env_spec: EnvironmentSpecType,
    agent_config: AgentConfigBase,
    run_cfg: ResolvedRunConfig,
    catalog_manifest: problem_catalog.CatalogManifest,
) -> str:
    """Save configuration artifacts and build Docker image if needed.

    Args:
        run_dir: The run output directory
        env_spec: Environment specification
        agent_config: Agent configuration
        run_cfg: Resolved run configuration

    Returns:
        Docker image name (empty string if not using Docker)
    """
    # Replace configuration evidence atomically without following a target or
    # any parent symlink supplied by an incomplete pre-existing output.
    atomic_write_text(
        run_dir / ENV_CONFIG_NAME,
        yaml.dump(serialize_path_dict(env_spec.model_dump(mode="json"))),
    )
    atomic_write_text(
        run_dir / CONFIG_FILENAME,
        yaml.dump(serialize_path_dict(run_cfg.model_dump(mode="json"))),
    )
    problem_catalog.save_run_catalog_manifest(run_dir, catalog_manifest)

    # Build docker image if needed
    if isinstance(env_spec, docker_runtime.DockerEnvironmentSpec):
        if agent_config.docker_template is not None:
            return common.build_agent_docker(
                agent_config=agent_config,
                environment=env_spec,
                force_build=False,
                force_build_base=False,
            )
        # Agent doesn't have custom Dockerfile, use environment base image
        return env_spec.get_base_image()
    return ""


def _planned_agent_image_name(
    env_spec: EnvironmentSpecType,
    agent_config: AgentConfigBase,
) -> str:
    """Return the deterministic image identity before any image build."""
    if not isinstance(env_spec, docker_runtime.DockerEnvironmentSpec):
        return ""
    if agent_config.docker_template is None:
        return env_spec.get_base_image()
    return (
        f"{docker_runtime.IMAGE_NAME_PREFIX}:"
        f"{agent_config.get_image(env_spec.name)}"
    )


def _publish_new_owned_run_directory(
    run_dir: Path,
    initialize: Callable[[Path, Path], None],
) -> None:
    """Atomically publish a new named output that already has provenance.

    The final path is never created as an empty or partially initialized run.
    A hard crash before ``os.rename`` leaves only a hidden sibling staging
    directory; a crash after it leaves a valid running provenance document at
    the requested path.
    """
    absolute = run_dir.absolute()
    parent = absolute.parent
    parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{absolute.name}.initializing-",
            dir=parent,
        )
    )
    try:
        initialize(staging, absolute)
        if absolute.exists() or absolute.is_symlink():
            raise FileExistsError(
                f"output directory appeared during initialization: {absolute}"
            )
        staging.rename(absolute)
        parent_fd = os.open(
            parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    except BaseException:
        if staging.is_dir() and not staging.is_symlink():
            shutil.rmtree(staging)
        raise


def _load_and_validate_run_config(
    config: Path | None,
    cli_flags: dict[str, object],
    overrides: list[str] | None,
) -> ResolvedRunConfig:
    """Load and validate run configuration from file, flags, and overrides.

    Args:
        config: Path to config file, or None for defaults
        cli_flags: CLI flag overrides
        overrides: Key=value override strings

    Returns:
        Validated and resolved run configuration

    Raises:
        typer.Exit: If config file not found or validation fails
    """
    try:
        return load_run_config(
            config_path=config,
            cli_flags=cli_flags,
            cli_overrides=overrides or [],
        )
    except FileNotFoundError as exc:
        typer.echo(typer.style(str(exc), fg=typer.colors.RED, bold=True))
        raise typer.Exit(1) from exc
    except ValueError as exc:
        typer.echo(
            typer.style(f"Config error: {exc}", fg=typer.colors.RED, bold=True)
        )
        raise typer.Exit(1) from exc


def _resolve_problem_names(
    cli_problem_names: list[str],
    config_problems: list[str],
    *,
    is_resuming: bool = False,
) -> list[str]:
    """Resolve problem names from CLI and config sources.

    On fresh runs, CLI problem names take precedence (replace config).
    On resume, CLI problem names are ADDED to config problems (merge).
    If both are empty, returns empty list for auto-discovery.

    Args:
        cli_problem_names: Problems specified on command line
        config_problems: Problems specified in config file
        is_resuming: If True, merge CLI with config instead of replacing

    Returns:
        Final list of problem names (may be empty for discovery)
    """
    if cli_problem_names:
        if is_resuming:
            # When resuming, merge CLI problems with saved config
            merged = list(config_problems)
            for p in cli_problem_names:
                if p not in merged:
                    merged.append(p)
            if merged != list(config_problems):
                typer.echo(
                    typer.style(
                        f"Adding problems to resume: {', '.join(cli_problem_names)}",
                        fg=typer.colors.CYAN,
                    )
                )
            return merged
        # Fresh run: CLI replaces config
        return cli_problem_names
    if config_problems:
        typer.echo(
            typer.style(
                f"Using problems from config: {', '.join(config_problems)}",
                fg=typer.colors.CYAN,
            )
        )
        return list(config_problems)
    return []


def _resolve_output_directory(
    config_output_path: str,
    debug: bool,
    *,
    create: bool = True,
) -> tuple[Path, bool]:
    """Resolve and prepare output directory.

    Args:
        config_output_path: Output path from config
        debug: If True, prepend DEBUG_ prefix.
        create: Whether to create a missing directory.

    Returns:
        Tuple of (run_dir, preexisted) where preexisted indicates
        if the directory existed before creation.
    """
    output_path_str = config_output_path
    if debug:
        # Prepend DEBUG_ to the last path component
        parts = output_path_str.rsplit("/", 1)
        if len(parts) == 2:
            output_path_str = f"{parts[0]}/DEBUG_{parts[1]}"
        else:
            output_path_str = f"DEBUG_{output_path_str}"

    run_dir = Path(output_path_str)
    preexisted = run_dir.exists()
    typer.echo(
        typer.style(
            f"Output directory: {run_dir}", fg=typer.colors.GREEN, bold=True
        )
    )
    if create:
        run_dir = utils.ensure_dir_exists(run_dir, create=True)
    elif run_dir.exists() and not run_dir.is_dir():
        raise FileNotFoundError(f"Path {run_dir} is not a directory")
    return run_dir, preexisted


def _handle_resume_validation(
    run_dir: Path,
    run_cfg: ResolvedRunConfig,
    env_spec: EnvironmentSpecType,
    resume: bool,
    overwrite: bool,
) -> None:
    """Validate configuration for resume mode.

    Args:
        run_dir: Path to existing run directory
        run_cfg: Current resolved run configuration
        env_spec: Current environment specification
        resume: Whether resume mode is enabled
        overwrite: Whether overwrite mode is enabled

    Raises:
        typer.Exit: If configuration mismatches detected during resume
    """
    if not resume or overwrite:
        return

    mismatches = _validate_resume_config(run_dir, run_cfg, env_spec)
    if mismatches:
        typer.echo(
            typer.style(
                "Cannot resume with different configuration:",
                fg=typer.colors.RED,
                bold=True,
            )
        )
        for field, saved, current in mismatches:
            typer.echo(f"  {field}: saved='{saved}' vs current='{current}'")
        typer.echo("\nUse --overwrite to start fresh with new configuration.")
        raise typer.Exit(1)


def _handle_early_completion(
    problem_names: list[str],
    run_dir: Path,
    problems_base_path: Path,
    console: Console,
    evaluate: bool,
    requested: list[str],
    catalog_integrity_check: Callable[[], None] | None = None,
) -> bool:
    """Handle case when all problems are already completed.

    Args:
        problem_names: List of problems still needing execution
        run_dir: The run output directory
        problems_base_path: Base path to problem definitions
        console: Rich console for output
        evaluate: Whether evaluation is enabled
        requested: Original list of requested problems

    Returns:
        True if caller should return (nothing to do), False to continue.
    """
    if problem_names:
        return False

    typer.echo(
        typer.style(
            "Nothing to do: all requested problems are already completed.",
            fg=typer.colors.GREEN,
            bold=True,
        )
    )
    postprocessing: PostprocessingResult | None = None
    try:
        if evaluate:
            postprocessing = _create_checkpoint_results_and_summary(
                run_dir=run_dir,
                problems_base_path=problems_base_path,
                problem_names=requested,
                console=console,
            )
        if catalog_integrity_check is not None:
            catalog_integrity_check()
    except BaseException as exc:
        _finalize_after_primary_error(run_dir, exc)
        raise

    status = (
        "incomplete_postprocessing"
        if postprocessing is not None and not postprocessing.successful
        else "completed"
    )
    details = {
        "no_work": True,
        "postprocessing": (
            postprocessing.model_dump()
            if postprocessing is not None
            else None
        ),
    }
    try:
        _finalize_run_provenance_or_raise(
            run_dir,
            status=status,
            details=details,
        )
    except BaseException as exc:  # noqa: BLE001
        typer.echo(
            typer.style(
                f"Failed to finalize benchmark provenance: {exc}",
                fg=typer.colors.RED,
                bold=True,
            )
        )
        raise typer.Exit(1) from exc
    if status != "completed":
        typer.echo(
            typer.style(
                "Postprocessing is incomplete; see "
                f"{run_dir / POSTPROCESSING_FILENAME}.",
                fg=typer.colors.RED,
                bold=True,
            )
        )
    _exit_on_incomplete_run(status)
    return True


def _validate_preexisting_output_provenance(
    run_dir: Path,
    *,
    run_dir_preexisted: bool,
    is_resuming: bool,
    profile: str | None,
) -> bool:
    """Validate a protected output exactly once before run mutations.

    Explicit resume is not the only way a finalized benchmark directory can
    be reopened: a named output path or ``--overwrite`` reaches the same
    directory through the normal config path. Any existing named-profile run,
    or any directory that already owns provenance, therefore receives the
    same fail-closed checksum gate.
    """
    if not run_dir_preexisted:
        return False
    provenance_path = run_dir / PROVENANCE_FILENAME
    protected = (
        is_resuming
        or profile is not None
        or provenance_path.exists()
        or provenance_path.is_symlink()
    )
    if not protected:
        return False
    if run_dir.is_symlink():
        raise ProvenanceIntegrityError(
            f"protected output directory is an unsupported symlink: {run_dir}"
        )
    validate_resumable_provenance(run_dir)
    return True


def _validate_resume_flags(
    resume: Path | None,
    config: Path | None,
    agent_config_path: str | None,
    environment_config_path: str | None,
    prompt_template_path: str | None,
    model_override: str | None,
    overrides: list[str] | None,
) -> None:
    """Validate that --resume is not combined with conflicting options.

    Args:
        resume: The --resume path (None if not specified)
        config: The --config path
        agent_config_path: The --agent flag
        environment_config_path: The --environment flag
        prompt_template_path: The --prompt flag
        model_override: The --model flag
        overrides: Positional config overrides

    Raises:
        typer.Exit: If conflicting options are specified with --resume
    """
    if resume is None:
        return

    conflicts: list[str] = []

    if config is not None:
        conflicts.append("--config")
    if agent_config_path is not None:
        conflicts.append("--agent")
    if environment_config_path is not None:
        conflicts.append("--environment")
    if prompt_template_path is not None:
        conflicts.append("--prompt")
    if model_override is not None:
        conflicts.append("--model")
    if overrides:
        conflicts.append(
            f"config overrides ({', '.join(overrides[:3])}{'...' if len(overrides) > 3 else ''})"
        )

    if conflicts:
        typer.echo(
            typer.style(
                f"Cannot use {', '.join(conflicts)} with --resume.\n"
                "--resume loads the saved configuration from the run directory.",
                fg=typer.colors.RED,
                bold=True,
            )
        )
        raise typer.Exit(1)


def _create_task_config(
    problem_base_path: Path,
    run_dir: Path,
    env_spec: EnvironmentSpecType,
    agent_config: AgentConfigBase,
    model_def: ModelDefinition,
    credential: ProviderCredential,
    run_cfg: ResolvedRunConfig,
    seed: int | None,
    verbosity: int,
    debug: bool,
    evaluate: bool,
    live_progress: bool,
    image_name: str,
    resume: bool,
    *,
    concurrent_evaluation: bool = False,
) -> problem_runner.RunTaskConfig:
    """Create task configuration for problem execution.

    Args:
        problem_base_path: Base path to problem definitions
        run_dir: The run output directory
        env_spec: Environment specification
        agent_config: Agent configuration
        model_def: Model definition from catalog
        credential: Provider credential
        run_cfg: Resolved run configuration
        seed: Random seed
        verbosity: Verbosity level
        debug: Debug mode flag
        evaluate: Whether to run evaluation
        live_progress: Whether to show live progress
        image_name: Docker image name
        resume: Whether to resume from checkpoints

    Returns:
        Configured RunTaskConfig
    """
    return problem_runner.RunTaskConfig(
        problem_base_path=problem_base_path,
        run_dir=run_dir,
        env_spec=env_spec,
        agent_config=agent_config,
        model_def=model_def,
        credential=credential,
        thinking_preset=run_cfg.thinking,
        thinking_max_tokens=run_cfg.thinking_max_tokens,
        prompt_template=run_cfg.prompt_content,
        pass_policy=run_cfg.pass_policy,
        seed=seed,
        verbosity=verbosity,
        debug=debug,
        disable_evaluation=not evaluate,
        concurrent_evaluation=concurrent_evaluation,
        live_progress=live_progress,
        image=image_name,
        resume=resume,
        one_shot=run_cfg.one_shot,
    )


def register(app: typer.Typer, name: str) -> None:
    app.command(
        name,
        help="Runs a model with an agent on the benchmark. Uses unified config system with hydra-style overrides.",
    )(run_agent)


def _report_results(results: list[problem_runner.TaskResult]) -> None:
    """Report the results of running problems.

    Args:
        results: List of TaskResult objects
    """
    logger = get_logger(__name__)
    successful = sum(1 for r in results if r.success)
    failed = len(results) - successful

    logger.info(
        "Agent runs completed",
        total=len(results),
        successful=successful,
        failed=failed,
    )

    if failed > 0:
        typer.echo(
            typer.style(
                f"\nCompleted with {failed} failure(s) out of "
                f"{len(results)} problems.",
                fg=typer.colors.YELLOW,
                bold=True,
            )
        )
        for result in results:
            if not result.success:
                summary = result.error_message or "Unknown error"
                if result.error_type:
                    summary = f"{result.error_type}: {summary}"
                typer.echo(
                    typer.style(
                        f"  - {result.problem_name}: {summary}",
                        fg=typer.colors.RED,
                    )
                )
                if result.error_traceback:
                    typer.echo(
                        typer.style(result.error_traceback, fg=typer.colors.RED)
                    )
    else:
        typer.echo(
            typer.style(
                f"\nAll {len(results)} problems completed successfully!",
                fg=typer.colors.GREEN,
                bold=True,
            )
        )


def _create_checkpoint_results_and_summary(
    run_dir: Path,
    problems_base_path: Path,
    problem_names: list[str],
    console: Console,
) -> PostprocessingResult:
    """Generate reports and return strict, durable completeness evidence."""
    errors: list[dict[str, str]] = []
    config: dict[str, Any] = {}
    configured_problem_names = list(dict.fromkeys(problem_names))
    require_checkpoint_evidence = False
    config_path = run_dir / CONFIG_FILENAME
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            loaded_config = yaml.safe_load(handle)
        if not isinstance(loaded_config, dict):
            raise ValueError("saved run config is not a mapping")
        config = loaded_config
        raw_problem_names = config.get("problems")
        if raw_problem_names is not None:
            if not isinstance(raw_problem_names, list) or not all(
                isinstance(name, str) and name
                for name in raw_problem_names
            ):
                raise ValueError(
                    "saved run config 'problems' must be a list of names"
                )
            configured_problem_names = list(dict.fromkeys(raw_problem_names))
        require_checkpoint_evidence = bool(config.get("profile"))
    except (OSError, ValueError, yaml.YAMLError) as exc:
        errors.append(
            {
                "kind": "invalid_run_config",
                "subject": str(config_path),
                "message": str(exc),
            }
        )

    expected_checkpoints = 0
    if config:
        try:
            expected_checkpoints = count_expected_checkpoints(
                config, problems_base_path
            )
        except ValueError as exc:
            errors.append(
                {
                    "kind": "invalid_expected_shape",
                    "subject": "configured problems",
                    "message": str(exc),
                }
            )

    results_file = run_dir / CHECKPOINT_RESULTS_FILENAME
    all_reports: list[dict[str, object]] = []
    expected_report_keys: set[tuple[str, str]] = set()

    for problem_name in configured_problem_names:
        problem_dir = run_dir / problem_name
        try:
            problem = ProblemConfig.from_yaml(problems_base_path / problem_name)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Skipping checkpoint report generation for problem",
                problem=problem_name,
                error=str(exc),
            )
            errors.append(
                {
                    "kind": "problem_config_error",
                    "subject": problem_name,
                    "message": str(exc),
                }
            )
            continue
        expected_report_keys.update(
            (problem_name, checkpoint_name)
            for checkpoint_name in problem.checkpoints
        )

        if not problem_dir.is_dir():
            errors.append(
                {
                    "kind": "missing_problem_output",
                    "subject": problem_name,
                    "message": f"Problem output directory not found: {problem_dir}",
                }
            )
            continue

        if require_checkpoint_evidence:
            required_artifacts = (
                ("evaluation", EVALUATION_FILENAME),
                ("inference", INFERENCE_RESULT_FILENAME),
            )
            for checkpoint_name in problem.checkpoints:
                checkpoint_dir = problem_dir / checkpoint_name
                for artifact_kind, filename in required_artifacts:
                    artifact = checkpoint_dir / filename
                    subject = f"{problem_name}/{checkpoint_name}/{filename}"
                    if artifact.is_symlink():
                        errors.append(
                            {
                                "kind": "symlink_checkpoint_evidence",
                                "subject": subject,
                                "message": (
                                    "Named-profile checkpoint evidence must "
                                    "be a regular file"
                                ),
                            }
                        )
                    elif not artifact.is_file():
                        errors.append(
                            {
                                "kind": f"missing_{artifact_kind}_evidence",
                                "subject": subject,
                                "message": (
                                    "Required named-profile checkpoint evidence "
                                    f"not found: {artifact}"
                                ),
                            }
                        )

        try:
            reports, report_errors = evaluation_entry.create_problem_reports(
                problem_dir, problem
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to create checkpoint reports",
                problem=problem_name,
                error=str(exc),
            )
            errors.append(
                {
                    "kind": "problem_report_error",
                    "subject": problem_name,
                    "message": str(exc),
                }
            )
            continue

        if require_checkpoint_evidence:
            for report in reports:
                domain_errors = _named_checkpoint_domain_errors(report)
                if domain_errors:
                    checkpoint_name = str(report.get("checkpoint", "unknown"))
                    errors.extend(
                        {
                            "kind": "invalid_checkpoint_metric_domain",
                            "subject": f"{problem_name}/{checkpoint_name}",
                            "message": message,
                        }
                        for message in domain_errors
                    )
                    continue
                all_reports.append(report)
        else:
            all_reports.extend(reports)
        errors.extend(
            {
                "kind": "checkpoint_report_error",
                "subject": f"{problem_name}/{checkpoint_name}",
                "message": message,
            }
            for checkpoint_name, message in report_errors
        )

    report_keys = [
        (str(report.get("problem", "")), str(report.get("checkpoint", "")))
        for report in all_reports
    ]
    if len(set(report_keys)) != len(report_keys):
        errors.append(
            {
                "kind": "duplicate_checkpoint_report",
                "subject": str(results_file),
                "message": "Generated checkpoint reports contain duplicate keys",
            }
        )
    produced_report_keys = set(report_keys)
    missing_report_keys = sorted(expected_report_keys - produced_report_keys)
    unexpected_report_keys = sorted(
        produced_report_keys - expected_report_keys
    )
    if missing_report_keys:
        errors.append(
            {
                "kind": "missing_checkpoint_reports",
                "subject": str(results_file),
                "message": json.dumps(missing_report_keys),
            }
        )
    if unexpected_report_keys:
        errors.append(
            {
                "kind": "unexpected_checkpoint_reports",
                "subject": str(results_file),
                "message": json.dumps(unexpected_report_keys),
            }
        )
    if (
        expected_checkpoints
        and len(expected_report_keys) != expected_checkpoints
    ):
        errors.append(
            {
                "kind": "expected_checkpoint_shape_mismatch",
                "subject": "configured problems",
                "message": (
                    f"Resolved {len(expected_report_keys)} checkpoint keys, "
                    f"expected {expected_checkpoints}"
                ),
            }
        )
    if expected_checkpoints and len(report_keys) != expected_checkpoints:
        errors.append(
            {
                "kind": "checkpoint_report_count_mismatch",
                "subject": str(results_file),
                "message": (
                    f"Expected {expected_checkpoints} checkpoint reports, "
                    f"generated {len(report_keys)}"
                ),
            }
        )

    # This file describes exactly this saved config. Do not preserve stale rows
    # from a previous, larger problem selection.
    results_file.unlink(missing_ok=True)
    if all_reports:
        update_results_jsonl(results_file, all_reports)
        logger.info(
            "Updated checkpoint results",
            path=str(results_file),
            report_count=len(all_reports),
        )
    else:
        logger.info(
            "No checkpoint reports generated",
            run_directory=str(run_dir),
        )

    summary_path = run_dir / SUMMARY_FILENAME
    summary_path.unlink(missing_ok=True)
    summary = None
    if config and expected_checkpoints:
        try:
            summary = display_and_save_summary(
                results_file,
                run_dir,
                config,
                console,
                expected_checkpoints,
                expected_problem_names=configured_problem_names,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to generate strict run summary",
                error=str(exc),
                exc_info=True,
            )
            errors.append(
                {
                    "kind": "summary_generation_error",
                    "subject": str(summary_path),
                    "message": str(exc),
                }
            )
    if summary is None or not summary_path.is_file():
        errors.append(
            {
                "kind": "missing_run_summary",
                "subject": str(summary_path),
                "message": "No run summary was generated",
            }
        )

    scb_check: dict[str, Any] | None = None
    if summary is not None:
        coverage = summary.scb_check
        scb_check = coverage.model_dump(mode="json")
        coverage_complete = (
            coverage.expected_checkpoints == expected_checkpoints
            and coverage.measured_checkpoints == expected_checkpoints
            and coverage.failed_checkpoints == 0
            and coverage.missing_snapshot_checkpoints == 0
            and coverage.missing_metadata_checkpoints == 0
            and coverage.missing_checkpoint_records == 0
            and coverage.unmeasured_checkpoints == 0
            and coverage.coverage_pct == 100.0
            and coverage.resolved_versions == [coverage.requested_version]
        )
        if not coverage_complete:
            errors.append(
                {
                    "kind": "incomplete_scb_check_coverage",
                    "subject": "scb-check",
                    "message": json.dumps(scb_check, sort_keys=True),
                }
            )
        if summary.expected_problems != len(configured_problem_names):
            errors.append(
                {
                    "kind": "problem_denominator_mismatch",
                    "subject": str(summary_path),
                    "message": (
                        f"Expected denominator {len(configured_problem_names)}, "
                        f"summary has {summary.expected_problems}"
                    ),
                }
            )

    observed_problem_names = sorted(
        {problem for problem, _ in report_keys if problem}
    )
    result = PostprocessingResult(
        status="completed" if not errors else "incomplete",
        expected_problem_names=configured_problem_names,
        observed_problem_names=observed_problem_names,
        executed_problem_names=list(dict.fromkeys(problem_names)),
        expected_checkpoints=expected_checkpoints,
        report_count=len(report_keys),
        summary_created=summary is not None and summary_path.is_file(),
        scb_check=scb_check,
        errors=errors,
    )
    _write_postprocessing_result(run_dir, result)
    if errors:
        logger.error(
            "Benchmark postprocessing incomplete",
            error_count=len(errors),
            expected_checkpoints=expected_checkpoints,
            report_count=len(report_keys),
            artifact=str(run_dir / POSTPROCESSING_FILENAME),
        )
    return result


def _final_run_status(
    results: list[problem_runner.TaskResult],
    postprocessing: PostprocessingResult | None,
) -> str:
    """Map execution and postprocessing completeness to provenance status."""
    problem_errors = not all(result.success for result in results)
    postprocessing_errors = (
        postprocessing is not None and not postprocessing.successful
    )
    if problem_errors and postprocessing_errors:
        return "incomplete_problem_execution_and_postprocessing"
    if problem_errors:
        return "incomplete_problem_execution"
    if postprocessing_errors:
        return "incomplete_postprocessing"
    return "completed"


def _finalize_after_primary_error(
    run_dir: Path,
    primary_error: BaseException,
) -> None:
    """Best-effort failure provenance without obscuring the primary error."""
    try:
        finalize_run_provenance(
            run_dir,
            status="failed",
            error_type=type(primary_error).__name__,
            checksum_artifacts=False,
        )
    except BaseException as finalization_error:  # noqa: BLE001
        primary_error.add_note(
            "Secondary provenance finalization failure: "
            f"{type(finalization_error).__qualname__}: {finalization_error}"
        )
        logger.error(
            "Provenance finalization failed while preserving run error",
            primary_error_type=type(primary_error).__qualname__,
            primary_error_message=str(primary_error),
            finalization_error_type=type(finalization_error).__qualname__,
            finalization_error_message=str(finalization_error),
            exc_info=True,
        )


def _finalize_initialization_error(
    *,
    repository_root: Path,
    run_dir: Path,
    primary_error: BaseException,
    source_image_name: str,
    base_image_name: str,
    agent_image_name: str,
    preflight: dict[str, Any] | None,
    executed_problem_names: list[str] | None = None,
) -> None:
    """Capture the latest initialization evidence, then finalize failure."""
    try:
        refresh_run_provenance_context(
            repository_root=repository_root,
            run_dir=run_dir,
            source_image_name=source_image_name,
            base_image_name=base_image_name,
            agent_image_name=agent_image_name,
            preflight=preflight,
            executed_problem_names=executed_problem_names,
        )
    except BaseException as refresh_error:  # noqa: BLE001
        primary_error.add_note(
            "Secondary provenance context refresh failure: "
            f"{type(refresh_error).__qualname__}: {refresh_error}"
        )
    _finalize_after_primary_error(run_dir, primary_error)


def _finalize_run_provenance_or_raise(
    run_dir: Path,
    *,
    status: str,
    details: dict[str, Any] | None,
) -> None:
    """Finalize full provenance, leaving a durable marker if hashing fails."""
    try:
        finalize_run_provenance(
            run_dir,
            status=status,
            details=details,
        )
    except BaseException as finalization_error:  # noqa: BLE001
        fallback_details = {
            "requested_status": status,
            "finalization_error_type": type(finalization_error).__qualname__,
            "finalization_error_message": str(finalization_error),
        }
        try:
            finalize_run_provenance(
                run_dir,
                status="incomplete_provenance",
                error_type=type(finalization_error).__qualname__,
                details=fallback_details,
                checksum_artifacts=False,
            )
        except BaseException as fallback_error:  # noqa: BLE001
            finalization_error.add_note(
                "Secondary fallback provenance failure: "
                f"{type(fallback_error).__qualname__}: {fallback_error}"
            )
        raise


def _exit_on_incomplete_run(status: str) -> None:
    """Return nonzero when execution or postprocessing integrity is incomplete."""
    if status == "completed":
        return
    typer.echo(
        typer.style(
            f"Benchmark run is incomplete: {status}",
            fg=typer.colors.RED,
            bold=True,
        )
    )
    raise typer.Exit(1)


def _configure_named_profile_temp_root(
    repository_root: Path,
    profile: str | None,
) -> Path | None:
    """Prepare the host temp root inherited by named-profile workers."""
    if profile is None:
        return None
    try:
        return configure_named_profile_temp_root(repository_root)
    except (NotADirectoryError, OSError, ValueError) as exc:
        typer.echo(typer.style(str(exc), fg=typer.colors.RED, bold=True))
        raise typer.Exit(1) from exc


def run_agent(
    ctx: typer.Context,
    # Config file (optional)
    config: Path | None = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to run configuration YAML file",
    ),
    # Override flags (all optional, override config file values)
    agent_config_path: str | None = typer.Option(
        None,
        "--agent",
        "-a",
        help="Override agent config (bare name or path)",
    ),
    environment_config_path: str | None = typer.Option(
        None,
        "--environment",
        "-e",
        help="Override environment config (bare name or path)",
    ),
    prompt_template_path: str | None = typer.Option(
        None,
        "--prompt",
        "-p",
        help="Override prompt template (bare name or path)",
    ),
    model_override: str | None = typer.Option(
        None,
        "--model",
        "-m",
        help="Override model: '{provider}/{model}'",
    ),
    # CLI-only flags (not in config)
    provider_api_key_env: str | None = typer.Option(
        None,
        "--provider-api-key-env",
        "-key",
        help="Override the environment variable used to resolve the provider API key.",
    ),
    problem_names: list[str] = typer.Option(
        [],
        "--problem",
        help="Name of the specific problems to run",
    ),
    num_workers: int = typer.Option(
        1,
        "--num-workers",
        "-n",
        help="Number of parallel workers for running problems",
    ),
    evaluate: bool = typer.Option(  # noqa: FBT001, FBT002
        True,  # noqa: FBT003
        "--evaluate/--no-evaluate",
        help="Whether to run evaluation",
    ),
    concurrent_evaluation: bool = typer.Option(  # noqa: FBT001, FBT002
        False,  # noqa: FBT003
        "--concurrent-evaluation/--no-concurrent-evaluation",
        help="Evaluate each checkpoint concurrently with the next "
        "checkpoint's solve (at most one solve + one eval at a time) so eval "
        "doesn't block progress. Scores unchanged for the ANY_CASE pass "
        "policy; cannot early-stop on test failures.",
    ),
    live_progress: bool = typer.Option(  # noqa: FBT001, FBT002
        True,  # noqa: FBT003
        "--live-progress/--no-live-progress",
        help="Whether to show live progress",
    ),
    resume: Path | None = typer.Option(
        None,
        "--resume",
        help="Resume from an existing run directory (loads saved config). "
        "Cannot be used with --config, --agent, --environment, --prompt, --model, or config overrides.",
    ),
    dry_run: bool = typer.Option(  # noqa: FBT001, FBT002
        False,  # noqa: FBT003
        "--dry-run",
        help="Preview what would be done without making changes (use with --resume)",
    ),
    # Config overrides via positional arguments
    overrides: list[str] | None = typer.Argument(
        None,
        help="Config overrides in key=value format (e.g., thinking=medium model.name=opus-4)",
    ),
) -> None:
    """Run the agent with unified config system.

    Examples:
        # Using a config file
        slop-code run --config my_run.yaml

        # Override values from config
        slop-code run --config my_run.yaml model.name=opus-4 thinking=high

        # Using flags only (defaults apply for unspecified values)
        slop-code run --agent claude_code --model anthropic/sonnet-4.5

        # Mix of flags and overrides
        slop-code run --model anthropic/sonnet-4.5 thinking=medium pass_policy=ALL_CASES

        # Resume from an existing run directory (loads saved config)
        slop-code run --resume outputs/sonnet-4/my-run/

        # Resume with different worker count
        slop-code run --resume outputs/my-run/ --num-workers 4
    """
    # 0. Validate --resume is not combined with conflicting options
    _validate_resume_flags(
        resume=resume,
        config=config,
        agent_config_path=agent_config_path,
        environment_config_path=environment_config_path,
        prompt_template_path=prompt_template_path,
        model_override=model_override,
        overrides=overrides,
    )

    # Track if we're in resume mode for later use
    is_resuming = resume is not None
    existing_integrity_validated = False

    # 1. Load config - either from run directory or via normal config loading
    if resume is not None:
        # Validate run directory exists
        if not resume.exists():
            typer.echo(
                typer.style(
                    f"Run directory not found: {resume}",
                    fg=typer.colors.RED,
                    bold=True,
                )
            )
            raise typer.Exit(1)

        # Validate the directory tree before loading config.yaml. Incomplete
        # provenance is resumable, but it must not be able to redirect that
        # first read through a symlink or another unsupported filesystem node.
        try:
            existing_integrity_validated = (
                _validate_preexisting_output_provenance(
                    resume,
                    run_dir_preexisted=True,
                    is_resuming=True,
                    profile=None,
                )
            )
        except ProvenanceIntegrityError as exc:
            typer.echo(typer.style(str(exc), fg=typer.colors.RED, bold=True))
            raise typer.Exit(1) from exc

        # Load saved config from run directory
        try:
            run_cfg = load_config_from_run_dir(resume)
        except (FileNotFoundError, ValueError) as exc:
            typer.echo(typer.style(str(exc), fg=typer.colors.RED, bold=True))
            raise typer.Exit(1) from exc

        # Use the run directory as output path (preexisted is always True)
        run_dir = resume
        run_dir_preexisted = True

        typer.echo(
            typer.style(
                f"Resuming from: {resume}",
                fg=typer.colors.CYAN,
                bold=True,
            )
        )
    else:
        # Normal config loading path
        try:
            cli_flags = _build_cli_flags(
                agent_config_path,
                environment_config_path,
                prompt_template_path,
                model_override,
            )
        except ValueError as exc:
            typer.echo(typer.style(str(exc), fg=typer.colors.RED, bold=True))
            raise typer.Exit(1) from exc

        run_cfg = _load_and_validate_run_config(config, cli_flags, overrides)

        # Resolve output directory
        run_dir, run_dir_preexisted = _resolve_output_directory(
            run_cfg.output_path,
            ctx.obj.debug,
            create=(
                not dry_run
                and getattr(run_cfg, "profile", None) is None
            ),
        )

    # Check every protected pre-existing output before preflight, logging, or
    # output filtering can rewrite a file covered by its finalized manifest.
    try:
        if not is_resuming:
            existing_integrity_validated = (
                _validate_preexisting_output_provenance(
                    run_dir,
                    run_dir_preexisted=run_dir_preexisted,
                    is_resuming=False,
                    profile=getattr(run_cfg, "profile", None),
                )
            )
    except ProvenanceIntegrityError as exc:
        typer.echo(typer.style(str(exc), fg=typer.colors.RED, bold=True))
        raise typer.Exit(1) from exc

    # 2. Resolve managed problem catalog
    try:
        scbench_home = Path(ctx.obj.scbench_home)
        if is_resuming:
            catalog_manifest = problem_catalog.validate_resume_catalog(
                run_dir, scbench_home
            )
            problem_root = problem_catalog.get_problem_root(
                scbench_home, bootstrap=False
            )
        elif dry_run:
            # A preview must never bootstrap or update external catalog state.
            problem_root = problem_catalog.get_problem_root(
                scbench_home, bootstrap=False
            )
            if problem_catalog.get_override_problem_root() is not None:
                # Override resolution is read-only and synthesizes its manifest
                # directly from the explicitly supplied local tree.
                catalog_manifest = problem_catalog.ensure_catalog_installed(
                    scbench_home
                )
            else:
                installed_manifest = problem_catalog.load_manifest(scbench_home)
                if installed_manifest is None:
                    raise problem_catalog.CatalogError(
                        "Problem catalog is not installed. "
                        "Run `slop-code sync` before --dry-run."
                    )
                catalog_manifest = installed_manifest
        else:
            catalog_manifest = problem_catalog.ensure_catalog_installed(
                scbench_home
            )
            problem_root = problem_catalog.get_problem_root(
                scbench_home, bootstrap=False
            )
    except problem_catalog.CatalogError as exc:
        typer.echo(typer.style(str(exc), fg=typer.colors.RED, bold=True))
        raise typer.Exit(1) from exc

    # 3. Resolve environment, model, and credentials
    env_spec, model_def, credential = _resolve_environment_and_credentials(
        run_cfg, provider_api_key_env
    )

    # 4. Build agent config
    agent_config = _build_agent_config(run_cfg)

    # 5. Echo model info
    typer.echo(f"Using model: {run_cfg.model.provider}/{run_cfg.model.name}")
    if provider_api_key_env:
        typer.echo(
            f"Using provider API key env override: {provider_api_key_env}"
        )

    # 6. Resolve problem names from CLI and config
    problem_names_resolved = _resolve_problem_names(
        list(problem_names), list(run_cfg.problems), is_resuming=is_resuming
    )

    # 7. Setup logging
    console = Console()
    run_logger = get_logger(__name__)

    # 8. Discover problems if not specified
    if not problem_names_resolved:
        problem_names_resolved = _discover_problems(problem_root)
        typer.echo(
            typer.style(
                f"Found {len(problem_names_resolved):,} problems",
                fg=typer.colors.GREEN,
                bold=True,
            )
        )

    # 9. Validate problem paths exist
    _validate_problem_paths(problem_names_resolved, problem_root)

    # Capture full resolved problem list before any filtering (for saving to config)
    full_problem_list = list(problem_names_resolved)
    repository_root = Path(__file__).resolve().parents[4]
    source_image_name = (
        env_spec.docker.image
        if isinstance(env_spec, docker_runtime.DockerEnvironmentSpec)
        else ""
    )
    base_image_name = (
        env_spec.get_base_image()
        if isinstance(env_spec, docker_runtime.DockerEnvironmentSpec)
        else ""
    )
    planned_agent_image_name = _planned_agent_image_name(
        env_spec,
        agent_config,
    )

    if not dry_run:

        def _initialize_provenance(
            storage_dir: Path,
            identity_dir: Path,
        ) -> None:
            start_run_provenance(
                repository_root=repository_root,
                run_dir=storage_dir,
                identity_run_dir=identity_dir,
                profile=getattr(run_cfg, "profile", None),
                model_provider=run_cfg.model.provider,
                model_name=run_cfg.model.name,
                agent_type=agent_config.type,
                agent_version=agent_config.version,
                thinking=run_cfg.thinking,
                seed=ctx.obj.seed,
                problem_names=full_problem_list,
                executed_problem_names=full_problem_list,
                catalog_version=catalog_manifest.version,
                catalog_commit=catalog_manifest.commit,
                num_workers=num_workers,
                evaluate=evaluate,
                environment_name=env_spec.name,
                source_image_name=source_image_name,
                base_image_name=base_image_name,
                agent_image_name=planned_agent_image_name,
                preflight=None,
                require_existing=existing_integrity_validated,
            )

        try:
            if (
                not run_dir_preexisted
                and getattr(run_cfg, "profile", None) is not None
            ):
                _publish_new_owned_run_directory(
                    run_dir,
                    _initialize_provenance,
                )
            else:
                _initialize_provenance(run_dir, run_dir)
        except (OSError, ProvenanceIntegrityError) as exc:
            typer.echo(typer.style(str(exc), fg=typer.colors.RED, bold=True))
            raise typer.Exit(1) from exc

        try:
            common.setup_command_logging(
                log_dir=run_dir,
                verbosity=ctx.obj.verbosity,
                log_file_name="run_agent.log",
                console=console,
                add_multiproc_info=num_workers > 1,
            )
            run_logger.info(
                "Starting agent run",
                agent_config=str(run_cfg.agent_config_path or "inline"),
                environment_config=str(
                    run_cfg.environment_config_path or "inline"
                ),
                prompt_path=str(run_cfg.prompt_path),
                model=f"{run_cfg.model.provider}/{run_cfg.model.name}",
                thinking=run_cfg.thinking,
                pass_policy=run_cfg.pass_policy.value,
                problem_names=problem_names_resolved,
                one_shot=run_cfg.one_shot.enabled,
            )
        except BaseException as exc:
            _finalize_initialization_error(
                repository_root=repository_root,
                run_dir=run_dir,
                primary_error=exc,
                source_image_name=source_image_name,
                base_image_name=base_image_name,
                agent_image_name=planned_agent_image_name,
                preflight=None,
                executed_problem_names=full_problem_list,
            )
            raise

    preflight: dict[str, Any] | None = None
    try:
        preflight = run_named_profile_preflight(
            repository_root=repository_root,
            run_dir=run_dir,
            catalog_root=problem_root,
            context=NamedProfileContext(
                profile=getattr(run_cfg, "profile", None),
                model_provider=run_cfg.model.provider,
                model_name=run_cfg.model.name,
                agent_type=agent_config.type,
                agent_version=agent_config.version,
                agent_config_path=run_cfg.agent_config_path,
                agent_config=run_cfg.agent,
                thinking=run_cfg.thinking,
                environment_config_path=run_cfg.environment_config_path,
                environment=run_cfg.environment,
                environment_name=env_spec.name,
                source_image=source_image_name,
                prompt_path=run_cfg.prompt_path,
                prompt_content=run_cfg.prompt_content,
                pass_policy=run_cfg.pass_policy.value,
                one_shot=run_cfg.one_shot.enabled,
                seed=ctx.obj.seed,
                evaluate=evaluate,
                num_workers=num_workers,
                concurrent_evaluation=concurrent_evaluation,
                problem_names=full_problem_list,
                catalog_version=catalog_manifest.version,
                catalog_commit=catalog_manifest.commit,
            ),
            persist=not dry_run,
            verify_evaluator=not dry_run,
        )
        catalog_integrity_check: Callable[[], None] | None = None
        if preflight is not None and not dry_run:
            catalog_evidence = preflight.get("catalog")
            execution_root = (
                catalog_evidence.get("execution_root")
                if isinstance(catalog_evidence, dict)
                else None
            )
            expected_root = (
                run_dir.resolve() / STAGED_CATALOG_PATH
            ).resolve()
            if not isinstance(execution_root, str) or (
                Path(execution_root).resolve() != expected_root
            ):
                raise ProvenanceIntegrityError(
                    "preflight catalog execution root is not the owned stage"
                )
            problem_root = expected_root
            evaluator_evidence = preflight.get("evaluator")
            execution_project = (
                evaluator_evidence.get("execution_project")
                if isinstance(evaluator_evidence, dict)
                else None
            )
            expected_evaluator = (
                run_dir.resolve() / STAGED_EVALUATOR_PATH
            ).resolve()
            if not isinstance(execution_project, str) or (
                Path(execution_project).resolve() != expected_evaluator
            ):
                raise ProvenanceIntegrityError(
                    "preflight evaluator execution root is not the owned stage"
                )
            os.environ[SCB_CHECK_PROJECT_ENV] = str(expected_evaluator)

            def _check_staged_catalog_integrity() -> None:
                verify_staged_catalog(repository_root, problem_root)
                verify_staged_evaluator(repository_root, expected_evaluator)

            catalog_integrity_check = _check_staged_catalog_integrity

        if not dry_run:
            refresh_run_provenance_context(
                repository_root=repository_root,
                run_dir=run_dir,
                source_image_name=source_image_name,
                base_image_name=base_image_name,
                agent_image_name=planned_agent_image_name,
                preflight=preflight,
                executed_problem_names=full_problem_list,
            )
    except BaseException as exc:
        failed_preflight = (
            exc.evidence
            if isinstance(exc, ScbenchV2PreflightError)
            else preflight
        )
        if not dry_run:
            _finalize_initialization_error(
                repository_root=repository_root,
                run_dir=run_dir,
                primary_error=exc,
                source_image_name=source_image_name,
                base_image_name=base_image_name,
                agent_image_name=planned_agent_image_name,
                preflight=failed_preflight,
                executed_problem_names=full_problem_list,
            )
        if isinstance(exc, ScbenchV2PreflightError):
            typer.echo(typer.style(str(exc), fg=typer.colors.RED, bold=True))
            raise typer.Exit(1) from exc
        raise

    # 10. Handle pre-existing run directory
    requested = list(problem_names_resolved)
    if run_dir_preexisted:
        if ctx.obj.overwrite:
            typer.echo(
                typer.style(
                    f"--overwrite set: rerunning all {len(requested):,} problem(s) in-place (run directory not deleted).",
                    fg=typer.colors.YELLOW,
                    bold=True,
                )
            )

        try:
            # Validate config matches saved config when resuming.
            _handle_resume_validation(
                run_dir,
                run_cfg,
                env_spec,
                is_resuming,
                ctx.obj.overwrite,
            )

            to_run, skipped, rerun_reasons = _filter_problems_for_execution(
                run_dir,
                problem_names_resolved,
                problem_root,
                run_cfg.prompt_content,
                env_spec,
                overwrite=ctx.obj.overwrite,
                resume=is_resuming or dry_run,
                require_evaluation=evaluate,
                read_only=dry_run,
            )
        except BaseException as exc:
            if not dry_run:
                _finalize_initialization_error(
                    repository_root=repository_root,
                    run_dir=run_dir,
                    primary_error=exc,
                    source_image_name=source_image_name,
                    base_image_name=base_image_name,
                    agent_image_name=planned_agent_image_name,
                    preflight=preflight,
                    executed_problem_names=full_problem_list,
                )
            raise

        if not ctx.obj.overwrite:
            # Log why problems are being rerun
            spec_changed = [
                p for p, r in rerun_reasons.items() if "spec_changed" in r
            ]
            if spec_changed:
                typer.echo(
                    typer.style(
                        f"Spec changed for {len(spec_changed)} problem(s): {', '.join(spec_changed[:3])}{'...' if len(spec_changed) > 3 else ''}",
                        fg=typer.colors.YELLOW,
                        bold=True,
                    )
                )
            typer.echo(
                typer.style(
                    f"Output directory exists: {len(skipped):,} done, {len(to_run):,} to run.",
                    fg=typer.colors.YELLOW,
                    bold=True,
                )
            )

        problem_names_resolved = to_run

        # Preview before no-work postprocessing or provenance mutation.
        if dry_run:
            _preview_dry_run(
                problem_names_resolved,
                run_dir,
                problem_root,
                run_cfg.prompt_content,
                env_spec,
                require_evaluation=evaluate,
            )
            return

        try:
            refresh_run_provenance_context(
                repository_root=repository_root,
                run_dir=run_dir,
                source_image_name=source_image_name,
                base_image_name=base_image_name,
                agent_image_name=planned_agent_image_name,
                preflight=preflight,
                executed_problem_names=problem_names_resolved,
            )
        except BaseException as exc:
            _finalize_after_primary_error(run_dir, exc)
            raise
        if _handle_early_completion(
            problem_names_resolved,
            run_dir,
            problem_root,
            console,
            evaluate,
            requested,
            catalog_integrity_check=catalog_integrity_check,
        ):
            return

    # 11. Handle dry-run mode
    if dry_run:
        _preview_dry_run(
            problem_names_resolved,
            run_dir,
            problem_root,
            run_cfg.prompt_content,
            env_spec,
            require_evaluation=evaluate,
        )
        return

    # 12. Update config with resolved/merged problems for future resumes
    run_cfg.problems = full_problem_list

    image_name = planned_agent_image_name
    try:
        # Named benchmark profiles use host-created workspaces as Docker bind
        # mounts. Configure a repository-visible root before image preparation
        # and before multiprocessing workers inherit the environment.
        profile_temp_root = _configure_named_profile_temp_root(
            repository_root,
            getattr(run_cfg, "profile", None),
        )
        if profile_temp_root is not None:
            evaluator = (
                preflight.get("evaluator") if preflight is not None else None
            )
            project_hash = (
                evaluator.get("project_sha256")
                if isinstance(evaluator, dict)
                else None
            )
            lock_hash = (
                evaluator.get("lock_sha256")
                if isinstance(evaluator, dict)
                else None
            )
            if not isinstance(project_hash, str) or not isinstance(
                lock_hash, str
            ):
                raise ProvenanceIntegrityError(
                    "verified evaluator hashes are missing"
                )
            evaluator_env_parent = profile_temp_root / "scb-check-envs"
            if evaluator_env_parent.is_symlink():
                raise ProvenanceIntegrityError(
                    "evaluator environment parent is a symlink"
                )
            evaluator_env_parent.mkdir(exist_ok=True)
            evaluator_env = evaluator_env_parent / (
                f"{project_hash[:16]}-{lock_hash[:16]}"
            )
            if evaluator_env.is_symlink():
                raise ProvenanceIntegrityError(
                    "evaluator environment is a symlink"
                )
            os.environ[SCB_CHECK_VENV_ENV] = str(evaluator_env)
            run_logger.info(
                "Configured named-profile temporary root",
                path=str(profile_temp_root),
                evaluator_environment=str(evaluator_env),
            )

        # 13. Save configuration evidence, then build the Docker image.
        image_name = _prepare_run_artifacts(
            run_dir,
            env_spec,
            agent_config,
            run_cfg,
            catalog_manifest,
        )
        refresh_run_provenance_context(
            repository_root=repository_root,
            run_dir=run_dir,
            source_image_name=source_image_name,
            base_image_name=base_image_name,
            agent_image_name=image_name,
            preflight=preflight,
            executed_problem_names=problem_names_resolved,
        )
    except BaseException as exc:
        _finalize_initialization_error(
            repository_root=repository_root,
            run_dir=run_dir,
            primary_error=exc,
            source_image_name=source_image_name,
            base_image_name=base_image_name,
            agent_image_name=image_name,
            preflight=preflight,
            executed_problem_names=problem_names_resolved,
        )
        raise
    postprocessing: PostprocessingResult | None = None
    try:
        run_logger.info(
            "Starting agent runs",
            num_problems=len(problem_names_resolved),
            num_workers=num_workers,
        )

        # 14. Create task config
        task_config = _create_task_config(
            problem_base_path=problem_root,
            run_dir=run_dir,
            env_spec=env_spec,
            agent_config=agent_config,
            model_def=model_def,
            credential=credential,
            run_cfg=run_cfg,
            seed=ctx.obj.seed,
            verbosity=ctx.obj.verbosity,
            debug=ctx.obj.debug,
            evaluate=evaluate,
            concurrent_evaluation=concurrent_evaluation,
            live_progress=live_progress,
            image_name=image_name,
            resume=is_resuming,
        )

        # 15. Run problems
        results = problem_runner.run_problems(
            problem_names=problem_names_resolved,
            config=task_config,
            num_workers=num_workers,
            console=console,
        )

        # 16. Report results
        _report_results(results)

        # 17. Create summary if evaluating
        if evaluate:
            postprocessing = _create_checkpoint_results_and_summary(
                run_dir=run_dir,
                problems_base_path=problem_root,
                problem_names=problem_names_resolved,
                console=console,
            )
        else:
            run_logger.info(
                "Evaluation disabled; skipping checkpoint result generation",
                run_directory=str(run_dir),
            )
        if catalog_integrity_check is not None:
            catalog_integrity_check()
    except BaseException as exc:
        _finalize_after_primary_error(run_dir, exc)
        raise

    final_status = _final_run_status(results, postprocessing)
    provenance_details = (
        {"postprocessing": postprocessing.model_dump()}
        if postprocessing is not None
        else None
    )
    try:
        _finalize_run_provenance_or_raise(
            run_dir,
            status=final_status,
            details=provenance_details,
        )
    except BaseException as exc:  # noqa: BLE001
        typer.echo(
            typer.style(
                f"Failed to finalize benchmark provenance: {exc}",
                fg=typer.colors.RED,
                bold=True,
            )
        )
        raise typer.Exit(1) from exc
    _exit_on_incomplete_run(final_status)
