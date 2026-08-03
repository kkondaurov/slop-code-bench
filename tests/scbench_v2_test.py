from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from slop_code.scbench_v2 import EVALUATOR_COMMAND
from slop_code.scbench_v2 import HASH_ALGORITHM
from slop_code.scbench_v2 import MANIFEST_ID
from slop_code.scbench_v2 import PAPER_ARXIV_ID
from slop_code.scbench_v2 import PAPER_URL
from slop_code.scbench_v2 import PAPER_VERSION
from slop_code.scbench_v2 import PREFLIGHT_FILENAME
from slop_code.scbench_v2 import SOURCE_IMAGE_REFERENCE
from slop_code.scbench_v2 import SOURCE_IMAGE_ROLE
from slop_code.scbench_v2 import STAGED_CATALOG_PATH
from slop_code.scbench_v2 import STAGED_EVALUATOR_PATH
from slop_code.scbench_v2 import SUITE_REVISION
from slop_code.scbench_v2 import NamedProfileContext
from slop_code.scbench_v2 import ScbenchV2CatalogIntegrityError
from slop_code.scbench_v2 import ScbenchV2PreflightError
from slop_code.scbench_v2 import build_catalog_lock
from slop_code.scbench_v2 import run_named_profile_preflight
from slop_code.scbench_v2 import sha256_file
from slop_code.scbench_v2 import verify_catalog_content
from slop_code.scbench_v2 import verify_staged_catalog
from slop_code.scbench_v2 import verify_staged_evaluator


def _write_yaml(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def _make_problem(catalog: Path, name: str, checkpoints: int = 1) -> None:
    problem = catalog / name
    problem.mkdir(parents=True)
    _write_yaml(
        problem / "config.yaml",
        {
            "checkpoints": {
                f"checkpoint_{index}": {} for index in range(1, checkpoints + 1)
            }
        },
    )
    (problem / "checkpoint_1.md").write_text(name, encoding="utf-8")


def _fixture_repo(tmp_path: Path) -> tuple[Path, Path, NamedProfileContext]:
    root = tmp_path / "repo"
    catalog = tmp_path / "catalog"
    for name in ("alpha", "beta", "gamma"):
        _make_problem(catalog, name)

    agent_path = root / "configs/agents/codex-test.yaml"
    agent = {
        "type": "codex",
        "binary": "codex",
        "version": "1.2.3",
        "npm_package_integrity": "sha512-package",
        "npm_linux_arm64_integrity": "sha512-arm64",
        "npm_linux_x64_integrity": "sha512-x64",
        "timeout": 7200,
        "extra_args": [],
        "env": {},
        "cost_limits": {
            "cost_limit": 0,
            "step_limit": 0,
            "net_cost_limit": 0,
            "max_retries": 2,
        },
    }
    _write_yaml(agent_path, agent)

    environment_path = root / "configs/environments/scb-v2.yaml"
    environment = {
        "type": "docker",
        "name": "python-test",
        "docker": {
            "image": "example.invalid/image@sha256:abc",
            "prebuilt_image": "sha256:base",
            "expected_image_id": "sha256:base",
            "expected_architecture": "arm64",
            "workdir": "/workspace",
            "mount_workspace": True,
        },
        "environment": {"env": {"UV_NO_CONFIG": "1"}},
    }
    _write_yaml(environment_path, environment)

    prompt_path = root / "configs/prompts/just-solve.jinja"
    prompt_path.parent.mkdir(parents=True)
    prompt_path.write_text("solve this\n", encoding="utf-8")
    model_path = root / "configs/models/model.yaml"
    _write_yaml(model_path, {"internal_name": "model", "pricing": {}})
    providers_path = root / "configs/providers.yaml"
    _write_yaml(providers_path, {"test_auth": {"type": "file"}})

    evaluator_dir = root / "configs/scbench-v2/evaluator"
    evaluator_dir.mkdir(parents=True)
    evaluator_project = evaluator_dir / "pyproject.toml"
    evaluator_project.write_text("[project]\nname='eval'\n", encoding="utf-8")
    evaluator_lock = evaluator_dir / "uv.lock"
    evaluator_lock.write_text("version = 1\n", encoding="utf-8")

    setup_template = (
        root
        / "src/slop_code/execution/docker_runtime/setup_base.docker.j2"
    )
    setup_template.parent.mkdir(parents=True)
    setup_template.write_text(
        "minio_release='RELEASE.test'; \\\n"
        "arm64) minio_sha256='"
        + "b" * 64
        + "' ;;\n",
        encoding="utf-8",
    )
    loader = root / "scripts/load-base.sh"
    loader.parent.mkdir()
    loader.write_text("#!/bin/sh\n", encoding="utf-8")
    checksum_file = root / "configs/scbench-v2/release-assets.sha256"
    checksum_file.write_text(
        f"{'c' * 64}  base.tar.zst\n",
        encoding="ascii",
    )

    lock = build_catalog_lock(
        catalog,
        source={
            "repository": "https://example.invalid/catalog",
            "release": "v1",
            "commit": "a" * 40,
        },
    )
    lock_path = root / "configs/scbench-v2/content-lock.json"
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "id": MANIFEST_ID,
        "suite_revision": SUITE_REVISION,
        "paper": {
            "arxiv_id": PAPER_ARXIV_ID,
            "version": PAPER_VERSION,
            "url": PAPER_URL,
        },
        "catalog": {
            "repository": "https://example.invalid/catalog",
            "release": "v1",
            "commit": "a" * 40,
            "content_lock": "configs/scbench-v2/content-lock.json",
            "content_lock_sha256": sha256_file(lock_path),
            "hash_algorithm": HASH_ALGORITHM,
            "content_lock_schema_version": lock["schema_version"],
            "problem_count": lock["problem_count"],
            "checkpoint_count": lock["checkpoint_count"],
        },
        "protocol": {
            "environment": "scb-v2",
            "environment_sha256": sha256_file(environment_path),
            "prompt": "just-solve",
            "prompt_sha256": sha256_file(prompt_path),
            "seed": 42,
            "evaluate": True,
            "num_workers": 2,
            "concurrent_evaluation": False,
            "pass_policy": "any-case",
            "one_shot": False,
            "timeout_seconds_per_checkpoint": 7200,
            "cost_limit_usd": 0,
            "step_limit": 0,
            "net_cost_limit_usd": 0,
            "providers_config": "configs/providers.yaml",
            "providers_config_sha256": sha256_file(providers_path),
            "source_image": {
                "reference": SOURCE_IMAGE_REFERENCE,
                "pinned": "example.invalid/image@sha256:abc",
                "digest": "sha256:abc",
                "role": SOURCE_IMAGE_ROLE,
            },
            "prebuilt_base": {
                "reference": "sha256:base",
                "image_id": "sha256:base",
                "platform": "linux/arm64",
                "architecture": "arm64",
                "bundled_tools": {
                    "minio": {
                        "release": "RELEASE.test",
                        "architecture": "arm64",
                        "sha256": "b" * 64,
                        "runtime": "linux/arm64",
                    }
                },
                "archive": {
                    "path": "outputs/base.tar.zst",
                    "checksum_file": (
                        "configs/scbench-v2/release-assets.sha256"
                    ),
                    "sha256": "c" * 64,
                    "loader": "scripts/load-base.sh",
                },
            },
        },
        "quality_evaluator": {
            "primary": {
                "package": "scb-check==0.1.3",
                "project": "configs/scbench-v2/evaluator/pyproject.toml",
                "project_sha256": sha256_file(evaluator_project),
                "lock": "configs/scbench-v2/evaluator/uv.lock",
                "lock_sha256": sha256_file(evaluator_lock),
                "command": EVALUATOR_COMMAND,
            }
        },
        "profiles": {
            "test-profile": {
                "provider": "test_auth",
                "model": "model",
                "agent_type": "codex",
                "cli_version": "1.2.3",
                "reasoning": "high",
                "agent_config": "configs/agents/codex-test.yaml",
                "agent_config_sha256": sha256_file(agent_path),
                "model_config": "configs/models/model.yaml",
                "model_config_sha256": sha256_file(model_path),
                "config": "configs/runs/test-profile.yaml",
                "diagnostic_config": (
                    "configs/runs/test-profile-diagnostic.yaml"
                ),
                "release_evidence": {
                    "package_integrity": "sha512-package",
                    "platform_alias_targets": {
                        "linux_arm64": "@openai/codex@1.2.3-linux-arm64",
                        "linux_x64": "@openai/codex@1.2.3-linux-x64",
                    },
                    "platform_integrity": {
                        "linux_arm64": "sha512-arm64",
                        "linux_x64": "sha512-x64",
                    },
                },
            }
        },
        "diagnostic_subset": {
            "problems": {"alpha": 1, "beta": 1},
            "checkpoint_count": 2,
        },
    }
    _write_yaml(root / "configs/runs/test-profile.yaml", {"profile": "test"})
    _write_yaml(
        root / "configs/runs/test-profile-diagnostic.yaml",
        {"profile": "test"},
    )
    _write_yaml(root / "configs/scbench-v2/manifest.yaml", manifest)

    context = NamedProfileContext(
        profile="test-profile",
        model_provider="test_auth",
        model_name="model",
        agent_type="codex",
        agent_version="1.2.3",
        agent_config_path=agent_path,
        agent_config=agent,
        thinking="high",
        environment_config_path=environment_path,
        environment=environment,
        environment_name="python-test",
        source_image="example.invalid/image@sha256:abc",
        prompt_path=prompt_path,
        prompt_content="solve this\n",
        pass_policy="any-case",  # noqa: S106 - benchmark policy, not a secret.
        one_shot=False,
        seed=42,
        evaluate=True,
        num_workers=2,
        concurrent_evaluation=False,
        problem_names=["alpha", "beta", "gamma"],
        catalog_version="v1",
        catalog_commit="a" * 40,
    )
    return root, catalog, context


def _evaluator(root: Path) -> dict[str, object]:
    return {
        "status": "verified",
        "requirement": "scb-check==0.1.3",
        "project_sha256": sha256_file(
            root / "configs/scbench-v2/evaluator/pyproject.toml"
        ),
        "lock_sha256": sha256_file(
            root / "configs/scbench-v2/evaluator/uv.lock"
        ),
    }


def test_named_profile_preflight_persists_verified_evidence(
    tmp_path: Path,
) -> None:
    root, catalog, context = _fixture_repo(tmp_path)
    run_dir = tmp_path / "run"

    evidence = run_named_profile_preflight(
        repository_root=root,
        run_dir=run_dir,
        catalog_root=catalog,
        context=context,
        evaluator_preflight=lambda: _evaluator(root),
    )

    assert evidence is not None
    assert evidence["status"] == "verified"
    assert evidence["profile"]["variant"] == "full"
    assert evidence["catalog"]["status"] == "verified"
    staged_root = run_dir / STAGED_CATALOG_PATH
    assert Path(evidence["catalog"]["execution_root"]) == staged_root
    assert (staged_root / "alpha/checkpoint_1.md").read_text() == "alpha"
    assert staged_root.stat().st_mode & 0o222 == 0
    staged_evaluator = run_dir / STAGED_EVALUATOR_PATH
    assert Path(evidence["evaluator"]["execution_project"]) == (
        staged_evaluator
    )
    assert staged_evaluator.stat().st_mode & 0o200
    assert (staged_evaluator / "uv.lock").stat().st_mode & 0o222 == 0
    (catalog / "alpha/checkpoint_1.md").write_text(
        "changed after preflight",
        encoding="utf-8",
    )
    verified = verify_staged_catalog(root, staged_root)
    assert verified["tree_sha256"] == evidence["catalog"]["expected"][
        "tree_sha256"
    ]
    saved = json.loads((run_dir / PREFLIGHT_FILENAME).read_text())
    assert saved["current_attempt_id"] == "attempt-000001"
    assert saved["attempts"][0]["status"] == "verified"


def test_named_profile_accepts_declared_capability_subset(
    tmp_path: Path,
) -> None:
    root, catalog, context = _fixture_repo(tmp_path)
    manifest_path = root / "configs/scbench-v2/manifest.yaml"
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    manifest["capability_subset"] = {
        "id": "capability-11",
        "problems": {"alpha": 1, "gamma": 1},
        "checkpoint_count": 2,
    }
    _write_yaml(manifest_path, manifest)

    evidence = run_named_profile_preflight(
        repository_root=root,
        run_dir=tmp_path / "run",
        catalog_root=catalog,
        context=replace(context, problem_names=["alpha", "gamma"]),
        persist=False,
        verify_evaluator=False,
    )

    assert evidence is not None
    assert evidence["status"] == "verified"
    assert evidence["profile"]["variant"] == "capability-11"


def test_named_profile_preflight_preserves_prior_attempts(
    tmp_path: Path,
) -> None:
    root, catalog, context = _fixture_repo(tmp_path)
    run_dir = tmp_path / "run"

    first = run_named_profile_preflight(
        repository_root=root,
        run_dir=run_dir,
        catalog_root=catalog,
        context=context,
        evaluator_preflight=lambda: _evaluator(root),
    )
    second = run_named_profile_preflight(
        repository_root=root,
        run_dir=run_dir,
        catalog_root=catalog,
        context=context,
        evaluator_preflight=lambda: _evaluator(root),
    )

    saved = json.loads((run_dir / PREFLIGHT_FILENAME).read_text())
    assert saved["current_attempt_id"] == "attempt-000002"
    assert [attempt["attempt_id"] for attempt in saved["attempts"]] == [
        "attempt-000001",
        "attempt-000002",
    ]
    assert saved["attempts"][0]["started_at"] == first["started_at"]
    assert saved["attempts"][1]["started_at"] == second["started_at"]


def test_named_profile_preflight_migrates_single_attempt_artifact(
    tmp_path: Path,
) -> None:
    root, catalog, context = _fixture_repo(tmp_path)
    run_dir = tmp_path / "run"
    legacy = run_named_profile_preflight(
        repository_root=root,
        run_dir=run_dir,
        catalog_root=catalog,
        context=context,
        evaluator_preflight=lambda: _evaluator(root),
        persist=False,
    )
    run_dir.mkdir()
    (run_dir / PREFLIGHT_FILENAME).write_text(
        json.dumps(legacy),
        encoding="utf-8",
    )

    run_named_profile_preflight(
        repository_root=root,
        run_dir=run_dir,
        catalog_root=catalog,
        context=context,
        evaluator_preflight=lambda: _evaluator(root),
    )

    saved = json.loads((run_dir / PREFLIGHT_FILENAME).read_text())
    assert saved["current_attempt_id"] == "attempt-000002"
    assert saved["attempts"][0]["started_at"] == legacy["started_at"]


def test_read_only_preflight_does_not_write_or_invoke_evaluator(
    tmp_path: Path,
) -> None:
    root, catalog, context = _fixture_repo(tmp_path)
    called = False

    def evaluator() -> dict[str, object]:
        nonlocal called
        called = True
        return _evaluator(root)

    evidence = run_named_profile_preflight(
        repository_root=root,
        run_dir=tmp_path / "run",
        catalog_root=catalog,
        context=context,
        evaluator_preflight=evaluator,
        persist=False,
        verify_evaluator=False,
    )

    assert evidence is not None
    assert evidence["status"] == "verified"
    assert evidence["evaluator"]["status"] == "not_run"
    assert called is False
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize(
    ("field", "value"),
    [("thinking", "xhigh"), ("num_workers", 1), ("evaluate", False)],
)
def test_named_profile_rejects_semantic_drift(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    root, catalog, context = _fixture_repo(tmp_path)
    drifted = replace(context, **{field: value})

    with pytest.raises(ScbenchV2PreflightError, match="preflight failed"):
        run_named_profile_preflight(
            repository_root=root,
            run_dir=tmp_path / "run",
            catalog_root=catalog,
            context=drifted,
            evaluator_preflight=lambda: _evaluator(root),
        )

    saved = json.loads((tmp_path / "run" / PREFLIGHT_FILENAME).read_text())
    attempt = saved["attempts"][-1]
    assert attempt["status"] == "failed"
    assert attempt["profile"]["checks"][field]["status"] == "mismatch"


def test_named_profile_rejects_catalog_byte_tamper(tmp_path: Path) -> None:
    root, catalog, context = _fixture_repo(tmp_path)
    (catalog / "alpha/checkpoint_1.md").write_text("tampered", encoding="utf-8")

    with pytest.raises(ScbenchV2PreflightError, match="tree_sha256 mismatch"):
        run_named_profile_preflight(
            repository_root=root,
            run_dir=tmp_path / "run",
            catalog_root=catalog,
            context=context,
            evaluator_preflight=lambda: _evaluator(root),
        )

    saved = json.loads((tmp_path / "run" / PREFLIGHT_FILENAME).read_text())
    attempt = saved["attempts"][-1]
    assert attempt["catalog"]["status"] == "failed"
    assert (
        attempt["catalog"]["actual"]["tree_sha256"]
        != attempt["catalog"]["expected"]["tree_sha256"]
    )


@pytest.mark.parametrize(
    ("section", "field", "value", "message"),
    [
        ("catalog", "problem_count", 99, "catalog.problem_count"),
        (
            "diagnostic_subset",
            "checkpoint_count",
            99,
            "diagnostic.checkpoint_count",
        ),
        (
            "protocol",
            "timeout_seconds_per_checkpoint",
            1,
            "protocol.timeout_seconds_per_checkpoint",
        ),
    ],
)
def test_named_profile_rejects_manifest_semantic_drift(
    tmp_path: Path,
    section: str,
    field: str,
    value: object,
    message: str,
) -> None:
    root, catalog, context = _fixture_repo(tmp_path)
    manifest_path = root / "configs/scbench-v2/manifest.yaml"
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    manifest[section][field] = value
    _write_yaml(manifest_path, manifest)

    with pytest.raises(ScbenchV2PreflightError, match=message):
        run_named_profile_preflight(
            repository_root=root,
            run_dir=tmp_path / "run",
            catalog_root=catalog,
            context=context,
            evaluator_preflight=lambda: _evaluator(root),
        )


def test_named_profile_rejects_suite_revision_drift(tmp_path: Path) -> None:
    root, catalog, context = _fixture_repo(tmp_path)
    manifest_path = root / "configs/scbench-v2/manifest.yaml"
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    manifest["suite_revision"] = "v2"
    _write_yaml(manifest_path, manifest)

    with pytest.raises(
        ScbenchV2PreflightError,
        match="manifest.suite_revision",
    ):
        run_named_profile_preflight(
            repository_root=root,
            run_dir=tmp_path / "run",
            catalog_root=catalog,
            context=context,
            evaluator_preflight=lambda: _evaluator(root),
        )


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("paper", "arxiv_id"), "wrong", "paper.arxiv_id"),
        (
            ("catalog", "repository"),
            "https://example.invalid/drift",
            "catalog.source.repository",
        ),
        (
            ("protocol", "source_image", "digest"),
            "sha256:drift",
            "protocol.source_image.digest",
        ),
        (
            ("quality_evaluator", "primary", "command"),
            "run-something-else",
            "evaluator.command",
        ),
        (
            (
                "protocol",
                "prebuilt_base",
                "bundled_tools",
                "minio",
                "sha256",
            ),
            "0" * 64,
            "environment.minio.sha256",
        ),
    ],
)
def test_named_profile_rejects_published_identity_drift(
    tmp_path: Path,
    path: tuple[str, ...],
    value: object,
    message: str,
) -> None:
    root, catalog, context = _fixture_repo(tmp_path)
    manifest_path = root / "configs/scbench-v2/manifest.yaml"
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    target = manifest
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = value
    _write_yaml(manifest_path, manifest)

    with pytest.raises(ScbenchV2PreflightError, match=message):
        run_named_profile_preflight(
            repository_root=root,
            run_dir=tmp_path / "run",
            catalog_root=catalog,
            context=context,
            evaluator_preflight=lambda: _evaluator(root),
        )


def test_named_profile_rejects_unsafe_manifest_path(tmp_path: Path) -> None:
    root, catalog, context = _fixture_repo(tmp_path)
    manifest_path = root / "configs/scbench-v2/manifest.yaml"
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    manifest["quality_evaluator"]["primary"]["project"] = "../outside.toml"
    _write_yaml(manifest_path, manifest)

    with pytest.raises(ScbenchV2PreflightError, match="unsafe.*manifest path"):
        run_named_profile_preflight(
            repository_root=root,
            run_dir=tmp_path / "run",
            catalog_root=catalog,
            context=context,
            evaluator_preflight=lambda: _evaluator(root),
        )


def test_named_profile_rejects_symlinked_manifest_input(tmp_path: Path) -> None:
    root, catalog, context = _fixture_repo(tmp_path)
    project = root / "configs/scbench-v2/evaluator/pyproject.toml"
    outside = tmp_path / "outside.toml"
    project.replace(outside)
    project.symlink_to(outside)

    with pytest.raises(ScbenchV2PreflightError, match="traverses a symlink"):
        run_named_profile_preflight(
            repository_root=root,
            run_dir=tmp_path / "run",
            catalog_root=catalog,
            context=context,
            evaluator_preflight=lambda: _evaluator(root),
        )


def test_named_profile_rejects_release_integrity_drift(tmp_path: Path) -> None:
    root, catalog, context = _fixture_repo(tmp_path)
    manifest_path = root / "configs/scbench-v2/manifest.yaml"
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    release = manifest["profiles"]["test-profile"]["release_evidence"]
    release["platform_integrity"]["linux_arm64"] = "sha512-drift"
    _write_yaml(manifest_path, manifest)

    with pytest.raises(
        ScbenchV2PreflightError,
        match="agent.linux_arm64_integrity",
    ):
        run_named_profile_preflight(
            repository_root=root,
            run_dir=tmp_path / "run",
            catalog_root=catalog,
            context=context,
            evaluator_preflight=lambda: _evaluator(root),
        )


def test_catalog_rejects_top_level_problem_symlink(tmp_path: Path) -> None:
    catalog = tmp_path / "catalog"
    catalog.mkdir()
    outside = tmp_path / "outside"
    _make_problem(outside, "problem")
    (catalog / "problem").symlink_to(outside / "problem", target_is_directory=True)

    with pytest.raises(ValueError, match="top-level symlink"):
        build_catalog_lock(catalog)


def test_staged_catalog_tamper_fails_final_verification(tmp_path: Path) -> None:
    root, catalog, context = _fixture_repo(tmp_path)
    run_dir = tmp_path / "run"
    run_named_profile_preflight(
        repository_root=root,
        run_dir=run_dir,
        catalog_root=catalog,
        context=context,
        evaluator_preflight=lambda: _evaluator(root),
    )
    staged_file = run_dir / STAGED_CATALOG_PATH / "alpha/checkpoint_1.md"
    staged_file.chmod(0o644)
    staged_file.write_text("tampered", encoding="utf-8")

    with pytest.raises(
        ScbenchV2CatalogIntegrityError,
        match="staged SCBench v2 catalog integrity failed",
    ):
        verify_staged_catalog(root, run_dir / STAGED_CATALOG_PATH)


def test_staged_evaluator_tamper_fails_final_verification(
    tmp_path: Path,
) -> None:
    root, catalog, context = _fixture_repo(tmp_path)
    run_dir = tmp_path / "run"
    run_named_profile_preflight(
        repository_root=root,
        run_dir=run_dir,
        catalog_root=catalog,
        context=context,
        evaluator_preflight=lambda: _evaluator(root),
    )
    staged_lock = run_dir / STAGED_EVALUATOR_PATH / "uv.lock"
    staged_lock.chmod(0o644)
    staged_lock.write_text("tampered\n", encoding="utf-8")

    with pytest.raises(
        ScbenchV2CatalogIntegrityError,
        match="staged SCBench v2 evaluator integrity failed",
    ):
        verify_staged_evaluator(root, run_dir / STAGED_EVALUATOR_PATH)


def test_catalog_executable_mode_drift_changes_content_lock(tmp_path: Path) -> None:
    root, catalog, _ = _fixture_repo(tmp_path)
    del root
    lock = build_catalog_lock(catalog)
    checkpoint = catalog / "alpha" / "checkpoint_1.md"
    checkpoint.chmod(checkpoint.stat().st_mode | 0o111)

    _, errors = verify_catalog_content(catalog, lock)

    assert any("tree_sha256 mismatch" in error for error in errors)


def test_catalog_file_to_directory_type_drift_changes_content_lock(
    tmp_path: Path,
) -> None:
    _, catalog, _ = _fixture_repo(tmp_path)
    lock = build_catalog_lock(catalog)
    checkpoint = catalog / "alpha" / "checkpoint_1.md"
    checkpoint.unlink()
    checkpoint.mkdir()

    _, errors = verify_catalog_content(catalog, lock)

    assert any("tree_sha256 mismatch" in error for error in errors)


def test_catalog_empty_directory_addition_changes_content_lock(
    tmp_path: Path,
) -> None:
    _, catalog, _ = _fixture_repo(tmp_path)
    lock = build_catalog_lock(catalog)
    (catalog / "alpha" / "empty-added-directory").mkdir()

    _, errors = verify_catalog_content(catalog, lock)

    assert any("tree_sha256 mismatch" in error for error in errors)


def test_staged_catalog_chmod_only_tamper_fails_final_verification(
    tmp_path: Path,
) -> None:
    root, catalog, context = _fixture_repo(tmp_path)
    run_dir = tmp_path / "run"
    run_named_profile_preflight(
        repository_root=root,
        run_dir=run_dir,
        catalog_root=catalog,
        context=context,
        evaluator_preflight=lambda: _evaluator(root),
    )
    staged_file = run_dir / STAGED_CATALOG_PATH / "alpha/checkpoint_1.md"
    staged_file.chmod(staged_file.stat().st_mode | 0o200)

    with pytest.raises(
        ScbenchV2CatalogIntegrityError,
        match="staged catalog entry is writable",
    ):
        verify_staged_catalog(root, run_dir / STAGED_CATALOG_PATH)


def test_preflight_writer_rejects_symlink_target_without_touching_source(
    tmp_path: Path,
) -> None:
    root, catalog, context = _fixture_repo(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_text("untouched\n", encoding="utf-8")
    (run_dir / PREFLIGHT_FILENAME).symlink_to(outside)

    with pytest.raises(ScbenchV2PreflightError, match="preflight evidence path"):
        run_named_profile_preflight(
            repository_root=root,
            run_dir=run_dir,
            catalog_root=catalog,
            context=context,
            evaluator_preflight=lambda: _evaluator(root),
        )

    assert outside.read_text(encoding="utf-8") == "untouched\n"


def test_unnamed_run_does_not_invoke_v2_preflight(tmp_path: Path) -> None:
    root, catalog, context = _fixture_repo(tmp_path)
    called = False

    def evaluator() -> dict[str, object]:
        nonlocal called
        called = True
        return {}

    result = run_named_profile_preflight(
        repository_root=root,
        run_dir=tmp_path / "run",
        catalog_root=catalog,
        context=replace(context, profile=None),
        evaluator_preflight=evaluator,
    )

    assert result is None
    assert called is False
    assert not (tmp_path / "run" / PREFLIGHT_FILENAME).exists()
