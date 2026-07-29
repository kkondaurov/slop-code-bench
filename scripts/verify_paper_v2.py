#!/usr/bin/env python3
"""Verify the pinned SCBench v2 problem catalog and experiment profiles."""

from __future__ import annotations

import argparse
import json
import os
import re
import stat
import sys
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any

import yaml

from slop_code.metrics.checkpoint.driver import SCB_CHECK_REQUIREMENT
from slop_code.scbench_v2 import EVALUATOR_COMMAND
from slop_code.scbench_v2 import EVALUATOR_LOCK
from slop_code.scbench_v2 import EVALUATOR_PROJECT
from slop_code.scbench_v2 import HASH_ALGORITHM
from slop_code.scbench_v2 import MANIFEST_ID
from slop_code.scbench_v2 import MANIFEST_SCHEMA_VERSION
from slop_code.scbench_v2 import PAPER_ARXIV_ID
from slop_code.scbench_v2 import PAPER_URL
from slop_code.scbench_v2 import PAPER_VERSION
from slop_code.scbench_v2 import SOURCE_IMAGE_REFERENCE
from slop_code.scbench_v2 import SOURCE_IMAGE_ROLE
from slop_code.scbench_v2 import build_catalog_lock
from slop_code.scbench_v2 import sha256_file
from slop_code.scbench_v2 import verify_catalog_content

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOCK_PATH = ROOT / "configs" / "scbench-v2" / "content-lock.json"
EXPECTED_RELEASE = "v1.0"
EXPECTED_COMMIT = "4d38d300059667d57e43c31969bc455f5c338b52"
EXPECTED_CATALOG_REPOSITORY = "https://github.com/gabeorlanski/scb-problems"
REPRODUCIBILITY_REPOSITORY = (
    "https://github.com/kkondaurov/slop-code-bench"
)
UPSTREAM_RUNNER_REPOSITORY = (
    "https://github.com/SprocketLab/slop-code-bench"
)
RUNNER_BASE_COMMIT = "13de1a7a6b8b3dc5cc532a0c322a0997afa5bec7"
RELEASE_TAG = "scbench-v2-repro.2"
RELEASE_URL = f"{REPRODUCIBILITY_REPOSITORY}/releases/tag/{RELEASE_TAG}"
PREBUILT_ARCHIVE_NAME = (
    "slopcodebench-base-scb-v2-linux-arm64-image-d2b862aad2bf.tar.zst"
)
PREBUILT_ARCHIVE_URL = (
    f"{REPRODUCIBILITY_REPOSITORY}/releases/download/{RELEASE_TAG}/"
    f"{PREBUILT_ARCHIVE_NAME}"
)
PREBUILT_CHECKSUM_URL = (
    f"{REPRODUCIBILITY_REPOSITORY}/releases/download/{RELEASE_TAG}/"
    "release-assets.sha256"
)
PREBUILT_PUBLICATION_STATUS = "published"
SETUP_BASE_TEMPLATE = (
    "src/slop_code/execution/docker_runtime/setup_base.docker.j2"
)


def _manifest_relative_path(value: str) -> PurePosixPath:
    parsed = PurePosixPath(value)
    if (
        not value
        or "\\" in value
        or parsed.is_absolute()
        or value != parsed.as_posix()
        or any(part in {"", ".", ".."} for part in parsed.parts)
    ):
        raise ValueError(f"unsafe suite manifest path: {value!r}")
    return parsed


def _safe_manifest_path(
    root: Path,
    value: str,
    *,
    must_exist: bool = True,
) -> Path:
    """Resolve a manifest-owned relative path without following symlinks."""
    parsed = _manifest_relative_path(value)
    root = root.resolve(strict=True)
    current = root
    for index, part in enumerate(parsed.parts):
        current /= part
        try:
            current_stat = current.lstat()
        except FileNotFoundError:
            if not must_exist and index == len(parsed.parts) - 1:
                return current
            raise
        if stat.S_ISLNK(current_stat.st_mode):
            raise ValueError(f"suite manifest path traverses symlink: {value!r}")
    resolved = current.resolve(strict=must_exist)
    if not resolved.is_relative_to(root):
        raise ValueError(f"suite manifest path escapes root: {value!r}")
    return resolved


def load_lock(path: Path = DEFAULT_LOCK_PATH) -> dict[str, Any]:
    """Load the checked-in catalog content lock."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def build_lock(catalog_root: Path) -> dict[str, Any]:
    """Build lock data for inspection; checked-in locks are edited deliberately."""
    return build_catalog_lock(
        catalog_root,
        source={
            "repository": EXPECTED_CATALOG_REPOSITORY,
            "release": EXPECTED_RELEASE,
            "commit": EXPECTED_COMMIT,
        },
    )


def verify_managed_manifest(catalog_root: Path) -> list[str]:
    """Verify sync metadata when the catalog is a managed installation."""
    manifest_path = catalog_root.parent / "manifest.json"
    if not manifest_path.is_file():
        return [
            f"managed catalog manifest is missing: {manifest_path}; "
            "run `UV_NO_CONFIG=1 uv run --frozen slop-code sync "
            f"{EXPECTED_RELEASE}`"
        ]
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"cannot read managed catalog manifest {manifest_path}: {exc}"]
    errors: list[str] = []
    if not isinstance(manifest, dict):
        return [f"managed catalog manifest is not an object: {manifest_path}"]
    if manifest.get("version") != EXPECTED_RELEASE:
        errors.append(
            "catalog release mismatch: expected "
            f"{EXPECTED_RELEASE}, got {manifest.get('version')!r}"
        )
    if manifest.get("commit") != EXPECTED_COMMIT:
        errors.append(
            "catalog commit mismatch: expected "
            f"{EXPECTED_COMMIT}, got {manifest.get('commit')!r}"
        )
    return errors


def verify_catalog(catalog_root: Path, lock: dict[str, Any]) -> list[str]:
    """Compare an installed catalog with the checked-in lock."""
    try:
        _, errors = verify_catalog_content(catalog_root, lock)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        return [f"cannot inspect catalog {catalog_root}: {exc}"]
    source = lock.get("source")
    if not isinstance(source, dict):
        errors.append("content lock source must be an object")
    else:
        if source.get("release") != EXPECTED_RELEASE:
            errors.append("content lock has an unexpected release")
        if source.get("commit") != EXPECTED_COMMIT:
            errors.append("content lock has an unexpected commit")
    return errors


def verify_suite_manifest(
    lock_path: Path,
    lock: dict[str, Any],
    *,
    root: Path = ROOT,
) -> list[str]:
    """Bind the checked-in lock bytes and algorithm to the suite manifest."""
    errors: list[str] = []
    try:
        manifest_path = _safe_manifest_path(
            root,
            "configs/scbench-v2/manifest.yaml",
        )
        manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, yaml.YAMLError) as exc:
        return [f"cannot safely read suite manifest: {exc}"]
    if not isinstance(manifest, dict):
        return ["suite manifest must be an object"]
    try:
        json.dumps(manifest, allow_nan=False)
    except (TypeError, ValueError):
        errors.append(
            "suite manifest must contain only JSON-compatible values"
        )
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        errors.append("suite manifest schema version mismatch")
    if manifest.get("id") != MANIFEST_ID:
        errors.append("suite manifest identity mismatch")
    paper = manifest.get("paper")
    if not isinstance(paper, dict):
        errors.append("suite manifest paper identity must be an object")
    else:
        for field, expected in (
            ("arxiv_id", PAPER_ARXIV_ID),
            ("version", PAPER_VERSION),
            ("url", PAPER_URL),
        ):
            if paper.get(field) != expected:
                errors.append(f"suite manifest paper {field} mismatch")
    runner = manifest.get("runner")
    if not isinstance(runner, dict):
        errors.append("suite manifest runner identity must be an object")
    else:
        expected_runner = {
            "repository": REPRODUCIBILITY_REPOSITORY,
            "upstream_repository": UPSTREAM_RUNNER_REPOSITORY,
            "base_commit": RUNNER_BASE_COMMIT,
            "release_tag": RELEASE_TAG,
            "release_url": RELEASE_URL,
        }
        for field, expected in expected_runner.items():
            if runner.get(field) != expected:
                errors.append(f"suite manifest runner {field} mismatch")
    catalog = manifest.get("catalog") if isinstance(manifest, dict) else None
    if not isinstance(catalog, dict):
        return [*errors, "suite manifest catalog must be an object"]
    try:
        relative_lock_path = lock_path.relative_to(root).as_posix()
    except ValueError:
        errors.append("content lock path escapes repository root")
        relative_lock_path = str(lock_path)
    if catalog.get("content_lock") != relative_lock_path:
        errors.append("suite manifest content-lock path mismatch")
    lock_value = catalog.get("content_lock")
    if isinstance(lock_value, str):
        try:
            manifest_lock_path = _safe_manifest_path(root, lock_value)
        except (OSError, ValueError) as exc:
            errors.append(f"unsafe suite content-lock path: {exc}")
        else:
            if manifest_lock_path != lock_path.resolve():
                errors.append("suite manifest content-lock identity mismatch")
    else:
        errors.append("suite manifest content-lock path is invalid")
    if catalog.get("content_lock_schema_version") != lock.get("schema_version"):
        errors.append("suite manifest content-lock schema mismatch")
    if catalog.get("hash_algorithm") != HASH_ALGORITHM:
        errors.append("suite manifest catalog hash algorithm mismatch")
    if lock.get("hash_algorithm") != HASH_ALGORITHM:
        errors.append("content lock hash algorithm mismatch")
    if catalog.get("content_lock_sha256") != sha256_file(lock_path):
        errors.append("suite manifest content-lock digest mismatch")
    for field in ("problem_count", "checkpoint_count"):
        if catalog.get(field) != lock.get(field):
            errors.append(f"suite manifest catalog {field} mismatch")
    lock_source = lock.get("source")
    if not isinstance(lock_source, dict):
        errors.append("content lock source must be an object")
    else:
        expected_source = {
            "repository": EXPECTED_CATALOG_REPOSITORY,
            "release": EXPECTED_RELEASE,
            "commit": EXPECTED_COMMIT,
        }
        for field, expected in expected_source.items():
            if lock_source.get(field) != expected:
                errors.append(f"content lock source {field} mismatch")
            if catalog.get(field) != lock_source.get(field):
                errors.append(f"suite manifest catalog {field} mismatch")

    problems = lock.get("problems")
    diagnostic = manifest.get("diagnostic_subset")
    diagnostic_problems = (
        diagnostic.get("problems") if isinstance(diagnostic, dict) else None
    )
    if not isinstance(problems, dict) or not isinstance(
        diagnostic_problems, dict
    ):
        errors.append("suite manifest diagnostic subset is invalid")
    else:
        expected_diagnostic = {
            name: problems.get(name) for name in diagnostic_problems
        }
        if diagnostic_problems != expected_diagnostic:
            errors.append("suite manifest diagnostic problem counts mismatch")
        diagnostic_values = list(diagnostic_problems.values())
        if not all(type(value) is int and value >= 0 for value in diagnostic_values):
            errors.append("suite manifest diagnostic counts are invalid")
            diagnostic_count = None
        else:
            diagnostic_count = sum(diagnostic_values)
        if diagnostic.get("checkpoint_count") != diagnostic_count:
            errors.append("suite manifest diagnostic checkpoint count mismatch")

    protocol = manifest.get("protocol")
    if not isinstance(protocol, dict):
        errors.append("suite manifest protocol must be an object")
        return errors
    environment: dict[str, Any] | None = None
    for path_key, hash_key, relative_builder in (
        (
            "environment",
            "environment_sha256",
            lambda value: f"configs/environments/{value}.yaml",
        ),
        (
            "prompt",
            "prompt_sha256",
            lambda value: f"configs/prompts/{value}.jinja",
        ),
        (
            "providers_config",
            "providers_config_sha256",
            lambda value: str(value),
        ),
    ):
        value = protocol.get(path_key)
        if not isinstance(value, str):
            errors.append(f"suite manifest protocol {path_key} is invalid")
            continue
        try:
            path = _safe_manifest_path(root, relative_builder(value))
            actual_hash = sha256_file(path)
            if path_key == "environment":
                loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    environment = loaded
                else:
                    errors.append("suite environment must be an object")
        except (OSError, ValueError, yaml.YAMLError) as exc:
            errors.append(f"cannot safely hash suite {path_key}: {exc}")
            continue
        if protocol.get(hash_key) != actual_hash:
            errors.append(f"suite manifest protocol {hash_key} mismatch")

    docker = environment.get("docker") if isinstance(environment, dict) else None
    source_image = protocol.get("source_image")
    if not isinstance(docker, dict) or not isinstance(source_image, dict):
        errors.append("suite manifest source image binding is invalid")
    else:
        pinned = docker.get("image")
        digest = (
            pinned.split("@", 1)[1]
            if isinstance(pinned, str) and "@" in pinned
            else None
        )
        expected_source = {
            "reference": SOURCE_IMAGE_REFERENCE,
            "pinned": pinned,
            "digest": digest,
            "role": SOURCE_IMAGE_ROLE,
        }
        for field, expected in expected_source.items():
            if source_image.get(field) != expected:
                errors.append(f"suite manifest source image {field} mismatch")

    prebuilt = protocol.get("prebuilt_base")
    if not isinstance(docker, dict) or not isinstance(prebuilt, dict):
        errors.append("suite manifest prebuilt base binding is invalid")
    else:
        expected_prebuilt = {
            "reference": docker.get("prebuilt_image"),
            "image_id": docker.get("expected_image_id"),
            "architecture": docker.get("expected_architecture"),
        }
        for field, expected in expected_prebuilt.items():
            if prebuilt.get(field) != expected:
                errors.append(f"suite manifest prebuilt base {field} mismatch")
        expected_architecture = docker.get("expected_architecture")
        if prebuilt.get("platform") != f"linux/{expected_architecture}":
            errors.append("suite manifest prebuilt base platform mismatch")
        if prebuilt.get("reference") != prebuilt.get("image_id"):
            errors.append("suite manifest prebuilt image identity mismatch")
        if (
            prebuilt.get("publication_status")
            != PREBUILT_PUBLICATION_STATUS
        ):
            errors.append(
                "suite manifest prebuilt publication status mismatch"
            )

        archive = prebuilt.get("archive")
        if not isinstance(archive, dict):
            errors.append("suite manifest prebuilt archive is invalid")
        else:
            expected_archive_publication = {
                "download_url": PREBUILT_ARCHIVE_URL,
                "checksum_url": PREBUILT_CHECKSUM_URL,
            }
            for field, expected in expected_archive_publication.items():
                if archive.get(field) != expected:
                    errors.append(
                        f"suite prebuilt archive {field} mismatch"
                    )
            resolved_archive_paths: dict[str, Path] = {}
            for field, must_exist in (
                ("path", False),
                ("loader", True),
                ("checksum_file", True),
            ):
                value = archive.get(field)
                if not isinstance(value, str):
                    errors.append(f"suite prebuilt archive {field} is invalid")
                    continue
                try:
                    resolved_archive_paths[field] = _safe_manifest_path(
                        root,
                        value,
                        must_exist=must_exist,
                    )
                except (OSError, ValueError) as exc:
                    errors.append(
                        f"unsafe suite prebuilt archive {field}: {exc}"
                    )
            archive_path_value = archive.get("path")
            archive_sha256 = archive.get("sha256")
            checksum_path = resolved_archive_paths.get("checksum_file")
            if (
                isinstance(archive_path_value, str)
                and isinstance(archive_sha256, str)
                and checksum_path is not None
            ):
                expected_checksum_line = (
                    f"{archive_sha256}  "
                    f"{PurePosixPath(archive_path_value).name}\n"
                )
                try:
                    checksum_text = checksum_path.read_text(encoding="ascii")
                except (OSError, UnicodeError) as exc:
                    errors.append(
                        f"cannot read suite archive checksum file: {exc}"
                    )
                else:
                    if checksum_text != expected_checksum_line:
                        errors.append(
                            "suite archive checksum file content mismatch"
                        )

        bundled_tools = prebuilt.get("bundled_tools")
        minio = (
            bundled_tools.get("minio")
            if isinstance(bundled_tools, dict)
            else None
        )
        if not isinstance(minio, dict):
            errors.append("suite manifest bundled MinIO identity is invalid")
        else:
            try:
                setup_template = _safe_manifest_path(
                    root,
                    SETUP_BASE_TEMPLATE,
                ).read_text(encoding="utf-8")
            except (OSError, ValueError) as exc:
                errors.append(f"cannot inspect base setup template: {exc}")
            else:
                release_match = re.search(
                    r"minio_release='([^']+)'",
                    setup_template,
                )
                architecture = prebuilt.get("architecture")
                sha_match = (
                    re.search(
                        rf"{re.escape(str(architecture))}\) "
                        r"minio_sha256='([0-9a-f]{64})'",
                        setup_template,
                    )
                    if isinstance(architecture, str)
                    else None
                )
                expected_minio = {
                    "release": (
                        release_match.group(1) if release_match else None
                    ),
                    "architecture": architecture,
                    "sha256": sha_match.group(1) if sha_match else None,
                    "runtime": (
                        f"linux/{architecture}"
                        if isinstance(architecture, str)
                        else None
                    ),
                }
                for field, expected in expected_minio.items():
                    if minio.get(field) != expected:
                        errors.append(
                            f"suite manifest bundled MinIO {field} mismatch"
                        )

    quality = manifest.get("quality_evaluator")
    primary = quality.get("primary") if isinstance(quality, dict) else None
    if not isinstance(primary, dict):
        errors.append("suite manifest primary evaluator is invalid")
    else:
        evaluator_semantics = {
            "package": SCB_CHECK_REQUIREMENT,
            "project": EVALUATOR_PROJECT,
            "lock": EVALUATOR_LOCK,
            "command": EVALUATOR_COMMAND,
        }
        for field, expected in evaluator_semantics.items():
            actual = primary.get(field)
            if field == "command" and isinstance(actual, str):
                actual = " ".join(actual.split())
            if actual != expected:
                errors.append(f"suite evaluator {field} mismatch")
        for path_key, hash_key in (
            ("project", "project_sha256"),
            ("lock", "lock_sha256"),
        ):
            value = primary.get(path_key)
            if not isinstance(value, str):
                errors.append(f"suite evaluator {path_key} is invalid")
                continue
            try:
                evaluator_path = _safe_manifest_path(root, value)
                actual_hash = sha256_file(evaluator_path)
            except (OSError, ValueError) as exc:
                errors.append(
                    f"cannot safely hash suite evaluator {path_key}: {exc}"
                )
                continue
            if primary.get(hash_key) != actual_hash:
                errors.append(f"suite evaluator {hash_key} mismatch")

    profiles = manifest.get("profiles")
    if not isinstance(profiles, dict):
        errors.append("suite manifest profiles must be an object")
        return errors
    for profile_name, profile in profiles.items():
        if not isinstance(profile, dict):
            errors.append(f"suite profile {profile_name} is invalid")
            continue
        agent_value = profile.get("agent_config")
        if not isinstance(agent_value, str):
            errors.append(f"suite profile {profile_name} agent path is invalid")
            continue
        try:
            agent_path = _safe_manifest_path(root, agent_value)
            agent = yaml.safe_load(agent_path.read_text(encoding="utf-8"))
            agent_hash = sha256_file(agent_path)
        except (OSError, ValueError, yaml.YAMLError) as exc:
            errors.append(
                f"cannot safely read suite agent for {profile_name}: {exc}"
            )
            continue
        if not isinstance(agent, dict):
            errors.append(f"suite profile {profile_name} agent is invalid")
            continue
        if profile.get("agent_config_sha256") != agent_hash:
            errors.append(f"suite profile {profile_name} agent digest mismatch")

        model_value = profile.get("model_config")
        if not isinstance(model_value, str):
            errors.append(f"suite profile {profile_name} model path is invalid")
        else:
            try:
                model_path = _safe_manifest_path(root, model_value)
                model_hash = sha256_file(model_path)
            except (OSError, ValueError) as exc:
                errors.append(
                    f"cannot safely read suite model for {profile_name}: {exc}"
                )
            else:
                if profile.get("model_config_sha256") != model_hash:
                    errors.append(
                        f"suite profile {profile_name} model digest mismatch"
                    )

        expected_problem_sets = {
            "config": list(lock.get("problems", {})),
            "diagnostic_config": list(diagnostic_problems or {}),
        }
        for config_field, expected_problems in expected_problem_sets.items():
            config_value = profile.get(config_field)
            if not isinstance(config_value, str):
                errors.append(
                    f"suite profile {profile_name} {config_field} path is invalid"
                )
                continue
            try:
                config_path = _safe_manifest_path(root, config_value)
                run_config = yaml.safe_load(
                    config_path.read_text(encoding="utf-8")
                )
            except (OSError, ValueError, yaml.YAMLError) as exc:
                errors.append(
                    f"cannot safely read suite profile {config_field} for "
                    f"{profile_name}: {exc}"
                )
                continue
            if not isinstance(run_config, dict):
                errors.append(
                    f"suite profile {profile_name} {config_field} is invalid"
                )
                continue
            expected_config = {
                "profile": profile_name,
                "agent": Path(agent_value).stem,
                "environment": protocol.get("environment"),
                "prompt": protocol.get("prompt"),
                "thinking": profile.get("reasoning"),
                "pass_policy": protocol.get("pass_policy"),
                "problems": expected_problems,
            }
            for field, expected in expected_config.items():
                if run_config.get(field) != expected:
                    errors.append(
                        f"suite profile {profile_name} {config_field} "
                        f"{field} mismatch"
                    )
            model = run_config.get("model")
            if not isinstance(model, dict) or model != {
                "provider": profile.get("provider"),
                "name": profile.get("model"),
            }:
                errors.append(
                    f"suite profile {profile_name} {config_field} model mismatch"
                )
        if profile.get("cli_version") != agent.get("version"):
            errors.append(f"suite profile {profile_name} CLI version mismatch")
        if protocol.get("timeout_seconds_per_checkpoint") != agent.get(
            "timeout"
        ):
            errors.append(f"suite profile {profile_name} timeout mismatch")
        limits = agent.get("cost_limits")
        for protocol_key, agent_key in (
            ("cost_limit_usd", "cost_limit"),
            ("step_limit", "step_limit"),
            ("net_cost_limit_usd", "net_cost_limit"),
        ):
            if not isinstance(limits, dict) or protocol.get(
                protocol_key
            ) != limits.get(agent_key):
                errors.append(
                    f"suite profile {profile_name} {protocol_key} mismatch"
                )
        release = profile.get("release_evidence")
        platform_integrity = (
            release.get("platform_integrity")
            if isinstance(release, dict)
            else None
        )
        alias_targets = (
            release.get("platform_alias_targets")
            if isinstance(release, dict)
            else None
        )
        if not isinstance(release, dict):
            errors.append(
                f"suite profile {profile_name} release evidence is invalid"
            )
            continue
        if release.get("package_integrity") != agent.get(
            "npm_package_integrity"
        ):
            errors.append(
                f"suite profile {profile_name} package integrity mismatch"
            )
        for platform_name, agent_key in (
            ("linux_arm64", "npm_linux_arm64_integrity"),
            ("linux_x64", "npm_linux_x64_integrity"),
        ):
            if not isinstance(platform_integrity, dict) or (
                platform_integrity.get(platform_name) != agent.get(agent_key)
            ):
                errors.append(
                    f"suite profile {profile_name} {platform_name} "
                    "integrity mismatch"
                )
            expected_alias = (
                f"@openai/codex@{agent.get('version')}-"
                f"{platform_name.replace('_', '-')}"
            )
            if not isinstance(alias_targets, dict) or alias_targets.get(
                platform_name
            ) != expected_alias:
                errors.append(
                    f"suite profile {profile_name} {platform_name} alias mismatch"
                )
    return errors


def resolve_catalog_root(explicit: Path | None) -> tuple[Path, bool]:
    """Resolve the managed root, preserving explicit/override semantics."""
    if explicit is not None:
        return explicit.expanduser(), False
    override = os.getenv("SCBENCH_PROBLEMS_PATH")
    if override:
        return Path(override).expanduser(), False
    home = Path(os.getenv("SCBENCH_HOME", "~/.cache/scbench")).expanduser()
    return home / "problems", True


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalog-root",
        type=Path,
        help="catalog problem root (default: managed SCBench catalog)",
    )
    parser.add_argument(
        "--lock",
        type=Path,
        default=DEFAULT_LOCK_PATH,
        help="content-lock JSON path",
    )
    parser.add_argument(
        "--print-lock",
        action="store_true",
        help="print lock data for deliberate review; does not edit files",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    catalog_root, managed = resolve_catalog_root(args.catalog_root)
    if args.print_lock:
        try:
            print(json.dumps(build_lock(catalog_root), indent=2) + "\n")
        except (OSError, ValueError, yaml.YAMLError) as exc:
            print(
                f"SCBench v2 catalog inspection failed: {exc}", file=sys.stderr
            )
            return 1
        return 0

    try:
        lock = load_lock(args.lock)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"SCBench v2 lock verification failed: {exc}", file=sys.stderr)
        return 1

    errors = verify_catalog(catalog_root, lock)
    errors.extend(verify_suite_manifest(args.lock.resolve(), lock))
    if managed:
        errors.extend(verify_managed_manifest(catalog_root))
    if errors:
        print("SCBench v2 catalog verification failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    print(
        "SCBench v2 catalog verified: "
        f"{lock['problem_count']} problems, "
        f"{lock['checkpoint_count']} checkpoints, "
        f"{lock['tree_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
