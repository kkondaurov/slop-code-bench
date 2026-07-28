#!/usr/bin/env python3
"""Verify that the checked-out suite matches the SCBench paper-v1 package."""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import tomllib
from pathlib import Path
from typing import Any

import yaml

EXPECTED_SOURCE_COMMIT = (
    "21dd1f58f408cd89b7acccdf5db969905aece51b"
)
EXPECTED_PROMPT_SHA256 = (
    "663d8920c1f88f30f9fc51ff47328a083a61fd13e6db082a9707613a6aa725e3"
)
EXPECTED_AST_GREP_VERSION = "0.42.0"
EXPECTED_AST_GREP_REQUIREMENT = "ast-grep-cli==0.42.0"
EXPECTED_LOCK_EXCLUDE_NEWER = "2026-03-24T21:59:00Z"
EXPECTED_PROBLEMS = {
    "circuit_eval": 8,
    "code_search": 5,
    "dag_execution": 3,
    "database_migration": 5,
    "dynamic_buffer": 4,
    "dynamic_config_service_api": 4,
    "etl_pipeline": 5,
    "eve_industry": 6,
    "eve_jump_planner": 3,
    "eve_market_tools": 4,
    "eve_route_planner": 3,
    "execution_server": 6,
    "file_backup": 4,
    "file_merger": 4,
    "file_query_tool": 5,
    "layered_config_synthesizer": 4,
    "log_query": 5,
    "metric_transform_lang": 5,
    "migrate_configs": 5,
    "trajectory_api": 5,
}
EXPECTED_MATRIX = [
    {
        "paper_label": "Sonnet 4.5",
        "model": "sonnet-4.5",
        "provider": "anthropic",
        "agent": "claude_code",
        "harness": "Claude Code",
        "cli_version": "2.0.65",
        "reasoning": "high",
    },
    {
        "paper_label": "Sonnet 4.6",
        "model": "sonnet-4.6",
        "provider": "anthropic",
        "agent": "claude_code",
        "harness": "Claude Code",
        "cli_version": "2.1.44",
        "reasoning": "high",
    },
    {
        "paper_label": "Opus 4.5",
        "model": "opus-4.5",
        "provider": "anthropic",
        "agent": "claude_code",
        "harness": "Claude Code",
        "cli_version": "2.0.51",
        "reasoning": "high",
    },
    {
        "paper_label": "Opus 4.6",
        "model": "opus-4.6",
        "provider": "anthropic",
        "agent": "claude_code",
        "harness": "Claude Code",
        "cli_version": "2.1.32",
        "reasoning": "high",
    },
    {
        "paper_label": "GPT 5.1 Codex Max",
        "model": "gpt-5.1-codex-max",
        "provider": "openai",
        "agent": "codex",
        "harness": "Codex CLI",
        "cli_version": "0.65.0",
        "reasoning": "high",
    },
    {
        "paper_label": "GPT 5.2",
        "model": "gpt-5.2",
        "provider": "openai",
        "agent": "codex",
        "harness": "Codex CLI",
        "cli_version": "0.71.0",
        "reasoning": "high",
    },
    {
        "paper_label": "GPT 5.2 Codex",
        "model": "gpt-5.2-codex",
        "provider": "openai",
        "agent": "codex",
        "harness": "Codex CLI",
        "cli_version": "0.80.0",
        "reasoning": "high",
    },
    {
        "paper_label": "GPT 5.3 Spark",
        "model": "gpt-5.3-codex-spark",
        "provider": "openai",
        "agent": "codex",
        "harness": "Codex CLI",
        "cli_version": "0.100.0",
        "reasoning": "high",
    },
    {
        "paper_label": "GPT 5.3 Codex",
        "model": "gpt-5.3-codex",
        "provider": "openai",
        "agent": "codex",
        "harness": "Codex CLI",
        "cli_version": "0.98.0",
        "reasoning": "high",
    },
    {
        "paper_label": "GPT 5.4",
        "model": "gpt-5.4",
        "provider": "openai",
        "agent": "codex",
        "harness": "Codex CLI",
        "cli_version": "0.110.0",
        "reasoning": "high",
    },
    {
        "paper_label": "GLM 4.7",
        "model": "glm-4.7",
        "provider": "zhipu-coding-plan",
        "agent": "claude_code",
        "harness": "Claude Code",
        "cli_version": "2.0.76",
        "reasoning": "high",
    },
]
EXPECTED_BASE_CONFIGS = {
    "codex": {
        "path": "configs/runs/paper-v1-codex.yaml",
        "agent": "codex",
        "model": "gpt-5.4",
        "provider": "openai",
        "version": "0.110.0",
    },
    "claude_code": {
        "path": "configs/runs/paper-v1-claude-code.yaml",
        "agent": "claude_code",
        "model": "opus-4.6",
        "provider": "anthropic",
        "version": "2.1.32",
    },
}


def _read_yaml(path: Path, errors: list[str]) -> dict[str, Any] | None:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        errors.append(f"cannot read YAML {path}: {exc}")
        return None
    if not isinstance(value, dict):
        errors.append(f"expected a YAML mapping in {path}")
        return None
    return value


def load_manifest(root: Path) -> dict[str, Any]:
    """Load the paper-v1 manifest or raise a useful exception."""
    path = root / "configs" / "paper-v1" / "manifest.yaml"
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a YAML mapping in {path}")
    return value


def _nested(value: dict[str, Any], *keys: str) -> Any:
    current: Any = value
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _expect(
    errors: list[str], actual: Any, expected: Any, label: str
) -> None:
    if actual != expected:
        errors.append(f"{label}: expected {expected!r}, got {actual!r}")


def verify_manifest(manifest: dict[str, Any]) -> list[str]:
    """Validate the immutable facts recorded in the manifest."""
    errors: list[str] = []
    _expect(errors, manifest.get("schema_version"), 1, "schema_version")
    _expect(
        errors,
        _nested(manifest, "paper", "arxiv_id"),
        "2603.24755",
        "paper.arxiv_id",
    )
    _expect(
        errors,
        _nested(manifest, "paper", "version"),
        "v1",
        "paper.version",
    )
    _expect(
        errors,
        _nested(manifest, "source", "commit"),
        EXPECTED_SOURCE_COMMIT,
        "source.commit",
    )
    _expect(
        errors,
        _nested(manifest, "corpus", "problem_count"),
        len(EXPECTED_PROBLEMS),
        "corpus.problem_count",
    )
    _expect(
        errors,
        _nested(manifest, "corpus", "track"),
        "python",
        "corpus.track",
    )
    _expect(
        errors,
        _nested(manifest, "corpus", "root"),
        "problems",
        "corpus.root",
    )
    _expect(
        errors,
        _nested(manifest, "corpus", "checkpoint_count"),
        sum(EXPECTED_PROBLEMS.values()),
        "corpus.checkpoint_count",
    )
    _expect(
        errors,
        _nested(manifest, "corpus", "problems"),
        EXPECTED_PROBLEMS,
        "corpus.problems",
    )
    _expect(
        errors,
        _nested(manifest, "prompt", "sha256"),
        EXPECTED_PROMPT_SHA256,
        "prompt.sha256",
    )
    _expect(
        errors,
        _nested(manifest, "prompt", "name"),
        "just-solve",
        "prompt.name",
    )
    _expect(
        errors,
        _nested(manifest, "prompt", "path"),
        "configs/prompts/just-solve.jinja",
        "prompt.path",
    )
    _expect(
        errors,
        _nested(manifest, "metrics", "ast_grep_rules", "count"),
        137,
        "metrics.ast_grep_rules.count",
    )
    _expect(
        errors,
        _nested(manifest, "tooling", "lock_exclude_newer"),
        EXPECTED_LOCK_EXCLUDE_NEWER,
        "tooling.lock_exclude_newer",
    )
    _expect(
        errors,
        _nested(manifest, "tooling", "ast_grep", "version"),
        EXPECTED_AST_GREP_VERSION,
        "tooling.ast_grep.version",
    )
    _expect(
        errors,
        _nested(manifest, "tooling", "ast_grep", "release_date"),
        "2026-03-16",
        "tooling.ast_grep.release_date",
    )
    _expect(
        errors,
        _nested(manifest, "tooling", "ast_grep", "executable"),
        "sg",
        "tooling.ast_grep.executable",
    )
    _expect(
        errors,
        _nested(manifest, "tooling", "ast_grep", "distribution"),
        "ast-grep-cli",
        "tooling.ast_grep.distribution",
    )
    _expect(
        errors,
        _nested(manifest, "tooling", "ast_grep", "requirement"),
        EXPECTED_AST_GREP_REQUIREMENT,
        "tooling.ast_grep.requirement",
    )
    expected_protocol = {
        "environment": "docker-python3.12-uv",
        "execution_user": "non-root",
        "initial_workspace": "empty except declared static problem assets",
        "checkpoint_order": "sequential",
        "fresh_container_per_checkpoint": True,
        "carry_forward": (
            "prior checkpoint working-directory snapshot only"
        ),
        "timeout_seconds_per_checkpoint": 7200,
        "turn_limit": 0,
        "cost_limit_usd": 0,
        "net_cost_limit_usd": 0,
        "reasoning": "high",
        "pass_policy": "any-case",
        "reset_between_checkpoints": [
            "installed packages",
            "shell history",
            "agent session data",
            "conversation context",
        ],
        "hidden_tests": {
            "visible_to_agent": False,
            "feedback_visible_to_agent": False,
            "materialized_after_agent_session": True,
        },
        "correctness": {
            "strict": "all checkpoint tests, including regressions",
            "isolated": (
                "all current-checkpoint non-regression tests"
            ),
            "core": "current-checkpoint core tests only",
        },
    }
    protocol = manifest.get("protocol")
    if not isinstance(protocol, dict):
        errors.append("protocol: expected a mapping")
    else:
        for key, expected in expected_protocol.items():
            _expect(errors, protocol.get(key), expected, f"protocol.{key}")
    _expect(
        errors,
        manifest.get("model_cli_matrix"),
        EXPECTED_MATRIX,
        "model_cli_matrix",
    )
    expected_paths = {
        name: config["path"]
        for name, config in EXPECTED_BASE_CONFIGS.items()
    }
    _expect(
        errors,
        manifest.get("base_configs"),
        expected_paths,
        "base_configs",
    )
    return errors


def _checkpoint_names(count: int, suffix: str) -> set[str]:
    return {f"checkpoint_{index}{suffix}" for index in range(1, count + 1)}


def verify_corpus(root: Path) -> list[str]:
    """Validate the exact paper corpus, checkpoint specs, and tests."""
    errors: list[str] = []
    problems_root = root / "problems"
    if not problems_root.is_dir():
        return [f"missing corpus directory: {problems_root}"]

    actual_problems = {
        path.name
        for path in problems_root.iterdir()
        if path.is_dir() and (path / "config.yaml").is_file()
    }
    expected_names = set(EXPECTED_PROBLEMS)
    if actual_problems != expected_names:
        missing = sorted(expected_names - actual_problems)
        extra = sorted(actual_problems - expected_names)
        errors.append(
            f"problem corpus mismatch: missing={missing}, extra={extra}"
        )

    for problem_name, checkpoint_count in EXPECTED_PROBLEMS.items():
        problem_dir = problems_root / problem_name
        config_path = problem_dir / "config.yaml"
        if not config_path.is_file():
            errors.append(f"{problem_name}: missing config.yaml")
            continue

        config = _read_yaml(config_path, errors)
        if config is not None:
            checkpoints = config.get("checkpoints")
            if not isinstance(checkpoints, dict):
                errors.append(
                    f"{problem_name}: config checkpoints is not a mapping"
                )
            else:
                expected_config_names = [
                    f"checkpoint_{index}"
                    for index in range(1, checkpoint_count + 1)
                ]
                _expect(
                    errors,
                    list(checkpoints),
                    expected_config_names,
                    f"{problem_name}.config.checkpoints",
                )
                for index, checkpoint_name in enumerate(
                    expected_config_names, start=1
                ):
                    checkpoint = checkpoints.get(checkpoint_name)
                    if isinstance(checkpoint, dict):
                        _expect(
                            errors,
                            checkpoint.get("order"),
                            index,
                            f"{problem_name}.{checkpoint_name}.order",
                        )

        spec_names = {
            path.name for path in problem_dir.glob("checkpoint_*.md")
        }
        expected_specs = _checkpoint_names(checkpoint_count, ".md")
        if spec_names != expected_specs:
            errors.append(
                f"{problem_name}: checkpoint specs mismatch; "
                f"expected={sorted(expected_specs)}, got={sorted(spec_names)}"
            )

        tests_dir = problem_dir / "tests"
        test_names = {
            path.name for path in tests_dir.glob("test_checkpoint_*.py")
        }
        expected_tests = _checkpoint_names(checkpoint_count, ".py")
        expected_tests = {
            f"test_{name}" for name in expected_tests
        }
        if test_names != expected_tests:
            errors.append(
                f"{problem_name}: checkpoint tests mismatch; "
                f"expected={sorted(expected_tests)}, got={sorted(test_names)}"
            )

    return errors


def verify_rules(root: Path, manifest: dict[str, Any]) -> list[str]:
    """Validate the consolidated 137-rule AST-grep file."""
    errors: list[str] = []
    relative_path = _nested(
        manifest, "metrics", "ast_grep_rules", "path"
    )
    if relative_path != "configs/slop_rules.yaml":
        return [
            "metrics.ast_grep_rules.path: expected "
            f"'configs/slop_rules.yaml', got {relative_path!r}"
        ]
    rules_path = root / relative_path
    try:
        rules = [
            rule
            for rule in yaml.safe_load_all(
                rules_path.read_text(encoding="utf-8")
            )
            if rule is not None
        ]
    except (OSError, yaml.YAMLError) as exc:
        return [f"cannot read AST-grep rules {rules_path}: {exc}"]

    _expect(errors, len(rules), 137, "AST-grep rule count")
    invalid = [rule for rule in rules if not isinstance(rule, dict)]
    if invalid:
        errors.append("AST-grep rules must all be mappings")
        return errors
    rule_ids = [rule.get("id") for rule in rules]
    if any(not isinstance(rule_id, str) or not rule_id for rule_id in rule_ids):
        errors.append("every AST-grep rule must have a non-empty string id")
    if len(set(rule_ids)) != len(rule_ids):
        errors.append("AST-grep rule ids must be unique")
    non_python = [
        rule.get("id") for rule in rules if rule.get("language") != "python"
    ]
    if non_python:
        errors.append(f"non-Python AST-grep rules found: {non_python}")
    return errors


def verify_ast_grep_tool(
    root: Path, manifest: dict[str, Any]
) -> list[str]:
    """Require the pinned AST-grep executable and scan the full ruleset."""
    errors: list[str] = []
    executable_name = _nested(
        manifest, "tooling", "ast_grep", "executable"
    )
    version = _nested(manifest, "tooling", "ast_grep", "version")
    if executable_name != "sg":
        return [
            "tooling.ast_grep.executable: expected 'sg', "
            f"got {executable_name!r}"
        ]
    override = os.environ.get("AST_GREP_BIN")
    executable = override or shutil.which("sg")
    if executable is None:
        return [
            "missing executable AST-grep pin; run "
            "`UV_NO_CONFIG=1 uv sync --frozen`"
        ]
    executable_path = Path(executable)
    if (
        not executable_path.is_file()
        or not os.access(executable_path, os.X_OK)
    ):
        return [f"AST_GREP_BIN is not executable: {executable_path}"]
    try:
        version_result = subprocess.run(  # noqa: S603
            [str(executable_path), "--version"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return [f"cannot execute pinned AST-grep: {exc}"]
    version_output = (
        f"{version_result.stdout}\n{version_result.stderr}".strip()
    )
    if (
        version_result.returncode != 0
        or str(version) not in version_output
    ):
        errors.append(
            f"AST-grep version mismatch: expected {version}, "
            f"got {version_output!r}"
        )

    rules_path = root / "configs" / "slop_rules.yaml"
    smoke_source = root / "scripts" / "verify_paper_v1.py"
    try:
        scan_result = subprocess.run(  # noqa: S603
            [
                str(executable_path),
                "scan",
                "--json=stream",
                "-r",
                str(rules_path),
                str(smoke_source),
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        errors.append(f"AST-grep ruleset smoke scan failed: {exc}")
        return errors
    if scan_result.returncode != 0:
        errors.append(
            "AST-grep cannot scan the paper-v1 ruleset: "
            f"{scan_result.stderr.strip()}"
        )
    return errors


def verify_ast_grep_dependency(root: Path) -> list[str]:
    """Require an exact project and lockfile pin for the AST-grep CLI."""
    errors: list[str] = []
    pyproject_path = root / "pyproject.toml"
    lock_path = root / "uv.lock"
    try:
        pyproject = tomllib.loads(
            pyproject_path.read_text(encoding="utf-8")
        )
    except (OSError, tomllib.TOMLDecodeError) as exc:
        return [f"cannot read {pyproject_path}: {exc}"]
    dependencies = pyproject.get("project", {}).get("dependencies", [])
    if EXPECTED_AST_GREP_REQUIREMENT not in dependencies:
        errors.append(
            "pyproject.toml must contain exact dependency "
            f"{EXPECTED_AST_GREP_REQUIREMENT!r}"
        )

    try:
        lock = tomllib.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        errors.append(f"cannot read {lock_path}: {exc}")
        return errors
    locked_versions = {
        str(package.get("version"))
        for package in lock.get("package", [])
        if package.get("name") == "ast-grep-cli"
    }
    if locked_versions != {EXPECTED_AST_GREP_VERSION}:
        errors.append(
            "uv.lock ast-grep-cli version mismatch: expected "
            f"{EXPECTED_AST_GREP_VERSION}, got {sorted(locked_versions)}"
        )
    lock_cutoff = lock.get("options", {}).get("exclude-newer")
    if lock_cutoff != EXPECTED_LOCK_EXCLUDE_NEWER:
        errors.append(
            "uv.lock artifact cutoff mismatch: expected "
            f"{EXPECTED_LOCK_EXCLUDE_NEWER}, got {lock_cutoff!r}"
        )
    return errors


def verify_prompt(root: Path, manifest: dict[str, Any]) -> list[str]:
    """Validate the exact baseline prompt bytes used by paper-v1."""
    errors: list[str] = []
    relative_path = _nested(manifest, "prompt", "path")
    if relative_path != "configs/prompts/just-solve.jinja":
        return [
            "prompt.path: expected 'configs/prompts/just-solve.jinja', "
            f"got {relative_path!r}"
        ]
    prompt_path = root / relative_path
    try:
        digest = hashlib.sha256(prompt_path.read_bytes()).hexdigest()
    except OSError as exc:
        return [f"cannot read prompt {prompt_path}: {exc}"]
    _expect(errors, digest, EXPECTED_PROMPT_SHA256, "just-solve SHA-256")
    return errors


def verify_matrix_catalog() -> list[str]:
    """Check that every paper matrix model/provider remains resolvable."""
    from slop_code.agent_runner import ModelCatalog
    from slop_code.agent_runner import ProviderCatalog

    errors: list[str] = []
    for row in EXPECTED_MATRIX:
        if ModelCatalog.get(row["model"]) is None:
            errors.append(f"unknown model config: {row['model']}")
        if ProviderCatalog.get(row["provider"]) is None:
            errors.append(f"unknown provider config: {row['provider']}")
    return errors


def verify_base_configs(
    root: Path, manifest: dict[str, Any]
) -> list[str]:
    """Resolve and type-check the two no-credential paper run presets."""
    from slop_code.agent_runner import ModelCatalog
    from slop_code.agent_runner import ProviderCatalog
    from slop_code.agent_runner import build_agent_config
    from slop_code.entrypoints.config import RunConfig
    from slop_code.entrypoints.config import load_run_config
    from slop_code.entrypoints.config.loader import resolve_environment

    errors: list[str] = []
    manifest_configs = manifest.get("base_configs")
    if not isinstance(manifest_configs, dict):
        return ["base_configs: expected a mapping"]

    for name, expected in EXPECTED_BASE_CONFIGS.items():
        relative_path = manifest_configs.get(name)
        config_path = root / str(relative_path)
        raw = _read_yaml(config_path, errors)
        if raw is None:
            continue
        try:
            RunConfig.model_validate(raw)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{name}: raw RunConfig is invalid: {exc}")
            continue
        try:
            resolved = load_run_config(config_path)
            agent_config = build_agent_config(resolved.agent)
            environment_source = (
                resolved.environment_config_path or resolved.environment
            )
            resolve_environment(environment_source)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{name}: cannot resolve base config: {exc}")
            continue

        _expect(
            errors,
            agent_config.type,
            expected["agent"],
            f"{name}.agent.type",
        )
        _expect(
            errors,
            agent_config.version,
            expected["version"],
            f"{name}.agent.version",
        )
        _expect(
            errors,
            getattr(agent_config, "timeout", None),
            7200,
            f"{name}.agent.timeout",
        )
        limits = agent_config.cost_limits
        _expect(errors, limits.step_limit, 0, f"{name}.step_limit")
        _expect(errors, limits.cost_limit, 0, f"{name}.cost_limit")
        _expect(
            errors,
            limits.net_cost_limit,
            0,
            f"{name}.net_cost_limit",
        )
        _expect(
            errors,
            resolved.model.name,
            expected["model"],
            f"{name}.model.name",
        )
        _expect(
            errors,
            resolved.model.provider,
            expected["provider"],
            f"{name}.model.provider",
        )
        _expect(errors, resolved.thinking, "high", f"{name}.thinking")
        _expect(
            errors,
            resolved.pass_policy.value,
            "any-case",
            f"{name}.pass_policy",
        )
        _expect(
            errors,
            resolved.problems,
            list(EXPECTED_PROBLEMS),
            f"{name}.problems",
        )
        _expect(
            errors,
            resolved.prompt_path,
            root / "configs" / "prompts" / "just-solve.jinja",
            f"{name}.prompt_path",
        )
        _expect(
            errors,
            resolved.one_shot.enabled,
            expected=False,
            label=f"{name}.one_shot.enabled",
        )
        _expect(
            errors,
            resolved.save_dir,
            "outputs/paper-v1",
            f"{name}.save_dir",
        )
        if not resolved.output_path.startswith("outputs/paper-v1/"):
            errors.append(
                f"{name}.output_path is outside outputs/paper-v1: "
                f"{resolved.output_path}"
            )
        if "${" in resolved.output_path:
            errors.append(
                f"{name}.output_path has unresolved interpolation: "
                f"{resolved.output_path}"
            )
        if ModelCatalog.get(resolved.model.name) is None:
            errors.append(f"{name}: unknown model {resolved.model.name}")
        if ProviderCatalog.get(resolved.model.provider) is None:
            errors.append(
                f"{name}: unknown provider {resolved.model.provider}"
            )

    return errors


def verify_repository(
    root: Path, *, check_tools: bool = True
) -> list[str]:
    """Return all paper-v1 packaging errors found under ``root``."""
    errors: list[str] = []
    manifest_path = root / "configs" / "paper-v1" / "manifest.yaml"
    manifest = _read_yaml(manifest_path, errors)
    if manifest is None:
        return errors

    errors.extend(verify_manifest(manifest))
    errors.extend(verify_corpus(root))
    errors.extend(verify_rules(root, manifest))
    errors.extend(verify_ast_grep_dependency(root))
    if check_tools:
        errors.extend(verify_ast_grep_tool(root, manifest))
    errors.extend(verify_prompt(root, manifest))
    errors.extend(verify_matrix_catalog())
    errors.extend(verify_base_configs(root, manifest))
    return errors


def main() -> int:
    """Run the paper-v1 verifier from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root (defaults to the script's parent repository)",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    errors = verify_repository(root)
    if errors:
        print(f"paper-v1 verification failed ({len(errors)} errors):")
        for error in errors:
            print(f"- {error}")
        return 1

    print(
        "paper-v1 verification passed: "
        "20 problems, 93 checkpoints, 137 rules, ast-grep 0.42.0, "
        "11 model/CLI rows"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
