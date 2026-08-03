"""Fail-closed validation for the named SCBench v2 experiment profiles."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import stat
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC
from datetime import datetime
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any

import yaml

from slop_code.common.atomic import atomic_write_text

PREFLIGHT_FILENAME = "scbench_v2_preflight.json"
PREFLIGHT_HISTORY_SCHEMA_VERSION = 2
MANIFEST_PATH = Path("configs/scbench-v2/manifest.yaml")
CONTENT_LOCK_PATH = Path("configs/scbench-v2/content-lock.json")
STAGED_CATALOG_PATH = Path("inputs/scbench-v2-catalog")
STAGED_EVALUATOR_PATH = Path("inputs/scbench-v2-evaluator")
SCB_CHECK_PROJECT_ENV = "SLOP_CODE_SCB_CHECK_PROJECT_DIR"
SCB_CHECK_VENV_ENV = "SLOP_CODE_SCB_CHECK_VENV_DIR"
HASH_ALGORITHM = "sha256-length-prefixed-mode-v2"
CATALOG_HASH_DOMAIN = b"slop-code.catalog-tree.v2\0"
MANIFEST_SCHEMA_VERSION = 1
MANIFEST_ID = "scbench-v2"
SUITE_REVISION = "v2.2"
CAPABILITY_SUBSET_ID = "capability-11"
PAPER_ARXIV_ID = "2603.24755"
PAPER_VERSION = "v2"
PAPER_URL = "https://arxiv.org/html/2603.24755v2"
SOURCE_IMAGE_REFERENCE = "ghcr.io/astral-sh/uv:python3.12-trixie-slim"
SOURCE_IMAGE_ROLE = "immutable parent input to the base-image setup recipe"
EVALUATOR_PROJECT = "configs/scbench-v2/evaluator/pyproject.toml"
EVALUATOR_LOCK = "configs/scbench-v2/evaluator/uv.lock"
EVALUATOR_COMMAND = (
    "UV_NO_CONFIG=1 uv run --frozen "
    "--project configs/scbench-v2/evaluator scb-check"
)
SETUP_BASE_TEMPLATE = (
    "src/slop_code/execution/docker_runtime/setup_base.docker.j2"
)
EPHEMERAL_PARTS = frozenset(
    {
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "__pycache__",
    }
)
EPHEMERAL_NAMES = frozenset({".DS_Store"})
EPHEMERAL_SUFFIXES = frozenset({".pyc", ".pyo"})


class ScbenchV2PreflightError(RuntimeError):
    """Raised when a named experiment profile no longer matches its lock."""

    def __init__(self, message: str, evidence: dict[str, Any]) -> None:
        super().__init__(message)
        self.evidence = evidence


class ScbenchV2CatalogIntegrityError(RuntimeError):
    """Raised when a staged catalog no longer matches the content lock."""


@dataclass(frozen=True)
class NamedProfileContext:
    """Resolved fields whose benchmark semantics must remain immutable."""

    profile: str | None
    model_provider: str
    model_name: str
    agent_type: str
    agent_version: str | None
    agent_config_path: Path | None
    agent_config: dict[str, Any]
    thinking: str | None
    environment_config_path: Path | None
    environment: dict[str, Any]
    environment_name: str
    source_image: str
    prompt_path: Path
    prompt_content: str
    pass_policy: str
    one_shot: bool
    seed: int | None
    evaluate: bool
    num_workers: int
    concurrent_evaluation: bool
    problem_names: list[str]
    catalog_version: str
    catalog_commit: str


def _utc_now() -> str:
    return datetime.now(tz=UTC).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest for a regular file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _mapping_sha256(value: dict[str, Any]) -> str:
    encoded = json.dumps(
        value,
        default=str,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_mapping(path: Path, *, yaml_file: bool) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    value = yaml.safe_load(text) if yaml_file else json.loads(text)
    if not isinstance(value, dict):
        raise ValueError(f"expected an object in {path}")
    return value


def discover_catalog_problems(catalog_root: Path) -> list[Path]:
    """Return flat problem directories in deterministic order."""
    if not catalog_root.is_dir():
        return []
    problems: list[Path] = []
    for child in catalog_root.iterdir():
        if child.is_symlink():
            raise ValueError(
                f"catalog contains unsupported top-level symlink: {child}"
            )
        if child.is_dir() and (child / "config.yaml").is_file():
            problems.append(child)
    return sorted(problems, key=lambda path: path.name)


def catalog_checkpoint_counts(catalog_root: Path) -> dict[str, int]:
    """Read checkpoint counts from every installed problem config."""
    counts: dict[str, int] = {}
    for problem_dir in discover_catalog_problems(catalog_root):
        value = yaml.safe_load(
            (problem_dir / "config.yaml").read_text(encoding="utf-8")
        )
        if not isinstance(value, dict):
            raise ValueError(f"invalid config mapping: {problem_dir}")
        checkpoints = value.get("checkpoints")
        if not isinstance(checkpoints, dict):
            raise ValueError(
                f"invalid checkpoints mapping: {problem_dir / 'config.yaml'}"
            )
        counts[problem_dir.name] = len(checkpoints)
    return counts


def catalog_files(catalog_root: Path) -> list[Path]:
    """Return all content-locked files in deterministic order."""
    files: list[Path] = []
    for problem_dir in discover_catalog_problems(catalog_root):
        for path in problem_dir.rglob("*"):
            relative = path.relative_to(catalog_root)
            if any(part in EPHEMERAL_PARTS for part in relative.parts):
                continue
            if path.name in EPHEMERAL_NAMES:
                continue
            if path.suffix in EPHEMERAL_SUFFIXES:
                continue
            if path.is_symlink():
                raise ValueError(
                    f"catalog contains unsupported symlink: {path}"
                )
            if path.is_file():
                files.append(path)
            elif not path.is_dir():
                raise ValueError(
                    f"catalog contains unsupported filesystem type: {path}"
                )
    return sorted(
        files,
        key=lambda path: path.relative_to(catalog_root).as_posix(),
    )


def _catalog_directories(catalog_root: Path) -> list[Path]:
    """Return every non-ephemeral catalog directory, parents first."""
    directories: set[Path] = set()
    for problem_dir in discover_catalog_problems(catalog_root):
        directories.add(problem_dir)
        for path in problem_dir.rglob("*"):
            relative = path.relative_to(catalog_root)
            if any(part in EPHEMERAL_PARTS for part in relative.parts):
                continue
            if path.name in EPHEMERAL_NAMES:
                continue
            if path.suffix in EPHEMERAL_SUFFIXES:
                continue
            if path.is_symlink():
                raise ValueError(
                    f"catalog contains unsupported symlink: {path}"
                )
            if path.is_dir():
                directories.add(path)
            elif not path.is_file():
                raise ValueError(
                    f"catalog contains unsupported filesystem type: {path}"
                )
    return sorted(
        directories,
        key=lambda path: (
            len(path.relative_to(catalog_root).parts),
            path.relative_to(catalog_root).as_posix(),
        ),
    )


def catalog_tree_sha256(catalog_root: Path) -> tuple[str, int]:
    """Hash catalog types, paths, semantic modes, and bytes deterministically."""
    digest = hashlib.sha256()
    digest.update(CATALOG_HASH_DOMAIN)
    files = catalog_files(catalog_root)
    directories = _catalog_directories(catalog_root)

    def field(value: bytes) -> None:
        digest.update(len(value).to_bytes(8, byteorder="big"))
        digest.update(value)

    for path in [*directories, *files]:
        relative = path.relative_to(catalog_root).as_posix().encode("utf-8")
        mode = stat.S_IMODE(path.stat(follow_symlinks=False).st_mode) & ~0o222
        field(b"directory" if path.is_dir() else b"file")
        field(relative)
        field(mode.to_bytes(4, byteorder="big"))
        if path.is_dir():
            field(b"")
            continue
        size = path.stat(follow_symlinks=False).st_size
        digest.update(size.to_bytes(8, byteorder="big"))
        bytes_seen = 0
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                bytes_seen += len(chunk)
                if bytes_seen > size:
                    raise ValueError(
                        f"catalog file changed while hashing: {path}"
                    )
                digest.update(chunk)
        if bytes_seen != size:
            raise ValueError(f"catalog file changed while hashing: {path}")
    return digest.hexdigest(), len(files)


def build_catalog_lock(
    catalog_root: Path,
    source: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the content facts used for catalog verification."""
    counts = catalog_checkpoint_counts(catalog_root)
    tree_sha256, file_count = catalog_tree_sha256(catalog_root)
    return {
        "schema_version": 2,
        "source": dict(source or {}),
        "hash_algorithm": HASH_ALGORITHM,
        "scope": (
            "type, relative path, read/execute mode, and bytes for all "
            "non-ephemeral entries under the problem directories"
        ),
        "problem_count": len(counts),
        "checkpoint_count": sum(counts.values()),
        "file_count": file_count,
        "tree_sha256": tree_sha256,
        "problems": counts,
    }


def verify_catalog_content(
    catalog_root: Path,
    expected: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    """Return actual catalog facts and all deviations from the lock."""
    source = expected.get("source")
    actual = build_catalog_lock(
        catalog_root,
        source if isinstance(source, dict) else None,
    )
    errors: list[str] = []
    for key in (
        "schema_version",
        "hash_algorithm",
        "problem_count",
        "checkpoint_count",
        "file_count",
        "tree_sha256",
        "problems",
    ):
        if actual.get(key) != expected.get(key):
            errors.append(
                f"catalog {key} mismatch: expected {expected.get(key)!r}, "
                f"got {actual.get(key)!r}"
            )
    return actual, errors


def _make_catalog_read_only(catalog_root: Path) -> None:
    """Remove write bits from every staged catalog entry and directory."""
    entries = list(catalog_root.rglob("*"))
    for path in entries:
        if path.is_symlink():
            raise ValueError(f"staged catalog contains symlink: {path}")
        if path.is_file():
            path.chmod(stat.S_IMODE(path.stat().st_mode) & ~0o222)
    directories = [path for path in entries if path.is_dir()]
    for path in sorted(directories, key=lambda value: len(value.parts), reverse=True):
        path.chmod(stat.S_IMODE(path.stat().st_mode) & ~0o222)
    catalog_root.chmod(stat.S_IMODE(catalog_root.stat().st_mode) & ~0o222)


def _verify_staged_catalog_content(
    catalog_root: Path,
    expected: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    """Verify locked bytes and reject every untracked staged filesystem entry."""
    actual, errors = verify_catalog_content(catalog_root, expected)
    locked_files = {
        path.relative_to(catalog_root) for path in catalog_files(catalog_root)
    }
    allowed_directories = {
        parent
        for relative in locked_files
        for parent in relative.parents
        if parent != Path()
    }
    for path in catalog_root.rglob("*"):
        relative = path.relative_to(catalog_root)
        if path.is_symlink():
            errors.append(
                f"staged catalog contains unsupported symlink: {relative}"
            )
        elif path.is_file() and relative not in locked_files:
            errors.append(f"staged catalog contains untracked file: {relative}")
        elif path.is_dir() and relative not in allowed_directories:
            errors.append(
                f"staged catalog contains untracked directory: {relative}"
            )
        elif not path.is_file() and not path.is_dir():
            errors.append(
                f"staged catalog contains unsupported entry: {relative}"
            )
        if not path.is_symlink() and path.stat().st_mode & 0o222:
            errors.append(f"staged catalog entry is writable: {relative}")
    if catalog_root.stat().st_mode & 0o222:
        errors.append("staged catalog root is writable")
    return actual, errors


def _stage_verified_catalog(
    source_root: Path,
    run_dir: Path,
    expected: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    """Create or validate the content-locked run-local execution catalog."""
    run_root = run_dir.resolve()
    staged_parent = run_root / STAGED_CATALOG_PATH.parent
    if staged_parent.is_symlink():
        raise ValueError(f"staged catalog parent is a symlink: {staged_parent}")
    staged_root = run_root / STAGED_CATALOG_PATH
    if staged_root.is_symlink():
        raise ValueError(f"staged catalog path is a symlink: {staged_root}")
    if staged_root.exists():
        actual, errors = _verify_staged_catalog_content(staged_root, expected)
        if errors:
            raise ValueError("; ".join(errors))
        _make_catalog_read_only(staged_root)
        return staged_root, actual

    staged_parent.mkdir(parents=True, exist_ok=True)
    if staged_parent.resolve() != staged_parent:
        raise ValueError(
            f"staged catalog parent escapes the run directory: {staged_parent}"
        )
    temporary = Path(
        tempfile.mkdtemp(
            prefix=".scbench-v2-catalog-",
            dir=staged_root.parent,
        )
    )
    try:
        source_files = catalog_files(source_root)
        for source_directory in _catalog_directories(source_root):
            destination_directory = temporary / source_directory.relative_to(
                source_root
            )
            destination_directory.mkdir(parents=True, exist_ok=True)
            shutil.copystat(
                source_directory,
                destination_directory,
                follow_symlinks=False,
            )
        for source in source_files:
            relative = source.relative_to(source_root)
            destination = temporary / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination, follow_symlinks=False)
        _make_catalog_read_only(temporary)
        actual, errors = _verify_staged_catalog_content(temporary, expected)
        if errors:
            raise ValueError("; ".join(errors))
        temporary.replace(staged_root)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    _make_catalog_read_only(staged_root)
    return staged_root, actual


def verify_staged_catalog(
    repository_root: Path,
    catalog_root: Path,
) -> dict[str, Any]:
    """Revalidate the run-local catalog before accepting run artifacts."""
    root = repository_root.resolve()
    manifest = _load_mapping(root / MANIFEST_PATH, yaml_file=True)
    catalog = manifest.get("catalog")
    if not isinstance(catalog, dict):
        raise ScbenchV2CatalogIntegrityError(
            "SCBench v2 manifest has no catalog mapping"
        )
    lock_value = catalog.get("content_lock")
    if lock_value != CONTENT_LOCK_PATH.as_posix():
        raise ScbenchV2CatalogIntegrityError(
            f"unexpected staged catalog lock path: {lock_value!r}"
        )
    lock_path = root / CONTENT_LOCK_PATH
    lock_sha256 = sha256_file(lock_path)
    if catalog.get("content_lock_sha256") != lock_sha256:
        raise ScbenchV2CatalogIntegrityError(
            "staged catalog content-lock digest no longer matches the manifest"
        )
    expected = _load_mapping(lock_path, yaml_file=False)
    if catalog.get("hash_algorithm") != expected.get("hash_algorithm"):
        raise ScbenchV2CatalogIntegrityError(
            "staged catalog hash algorithm no longer matches the manifest"
        )
    actual, errors = _verify_staged_catalog_content(catalog_root, expected)
    if errors:
        raise ScbenchV2CatalogIntegrityError(
            "staged SCBench v2 catalog integrity failed: " + "; ".join(errors)
        )
    return actual


def _verify_staged_evaluator_content(
    evaluator_root: Path,
    primary: dict[str, Any],
) -> list[str]:
    """Return deviations from the two-file immutable evaluator project."""
    errors: list[str] = []
    expected_files = {"pyproject.toml", "uv.lock"}
    if evaluator_root.is_symlink() or not evaluator_root.is_dir():
        return [f"staged evaluator root is unsafe: {evaluator_root}"]
    entries = list(evaluator_root.iterdir())
    actual_names = {entry.name for entry in entries}
    unexpected = actual_names - expected_files
    missing = expected_files - actual_names
    if unexpected or missing:
        errors.append(
            "staged evaluator entries mismatch: "
            f"missing={sorted(missing)!r}, unexpected={sorted(unexpected)!r}"
        )
    for name, manifest_key in (
        ("pyproject.toml", "project_sha256"),
        ("uv.lock", "lock_sha256"),
    ):
        path = evaluator_root / name
        if path.is_symlink() or not path.is_file():
            errors.append(f"staged evaluator input is unsafe: {name}")
            continue
        actual_hash = sha256_file(path)
        if actual_hash != primary.get(manifest_key):
            errors.append(
                f"staged evaluator {name} digest mismatch: expected "
                f"{primary.get(manifest_key)!r}, got {actual_hash!r}"
            )
        if path.stat(follow_symlinks=False).st_mode & 0o222:
            errors.append(f"staged evaluator input is writable: {name}")
    return errors


def _make_evaluator_read_only(evaluator_root: Path) -> None:
    for name in ("pyproject.toml", "uv.lock"):
        path = evaluator_root / name
        if path.is_symlink():
            raise ValueError(f"staged evaluator input is a symlink: {path}")
        path.chmod(stat.S_IMODE(path.stat().st_mode) & ~0o222)


def _stage_verified_evaluator(
    source_root: Path,
    run_dir: Path,
    primary: dict[str, Any],
) -> Path:
    """Create or verify the run-local frozen evaluator project."""
    staged_root = run_dir.resolve() / STAGED_EVALUATOR_PATH
    staged_parent = staged_root.parent
    if staged_parent.is_symlink() or staged_root.is_symlink():
        raise ValueError("staged evaluator path contains a symlink")
    if staged_root.exists():
        errors = _verify_staged_evaluator_content(staged_root, primary)
        if errors:
            raise ValueError("; ".join(errors))
        return staged_root

    staged_parent.mkdir(parents=True, exist_ok=True)
    if staged_parent.resolve() != staged_parent:
        raise ValueError(
            f"staged evaluator parent escapes the run directory: {staged_parent}"
        )
    temporary = Path(
        tempfile.mkdtemp(
            prefix=".scbench-v2-evaluator-",
            dir=staged_parent,
        )
    )
    try:
        for name in ("pyproject.toml", "uv.lock"):
            source = source_root / name
            if source.is_symlink() or not source.is_file():
                raise ValueError(f"evaluator input is unsafe: {source}")
            shutil.copy2(source, temporary / name, follow_symlinks=False)
        _make_evaluator_read_only(temporary)
        errors = _verify_staged_evaluator_content(temporary, primary)
        if errors:
            raise ValueError("; ".join(errors))
        temporary.replace(staged_root)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return staged_root


def verify_staged_evaluator(
    repository_root: Path,
    evaluator_root: Path,
) -> None:
    """Revalidate the frozen evaluator after all checkpoint measurements."""
    root = repository_root.resolve()
    manifest = _load_mapping(root / MANIFEST_PATH, yaml_file=True)
    quality = manifest.get("quality_evaluator")
    primary = quality.get("primary") if isinstance(quality, dict) else None
    if not isinstance(primary, dict):
        raise ScbenchV2CatalogIntegrityError(
            "SCBench v2 manifest has no primary evaluator"
        )
    errors = _verify_staged_evaluator_content(evaluator_root, primary)
    if errors:
        raise ScbenchV2CatalogIntegrityError(
            "staged SCBench v2 evaluator integrity failed: "
            + "; ".join(errors)
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
        raise ValueError(f"unsafe SCBench v2 manifest path: {value!r}")
    return parsed


def _resolved_path(root: Path, value: str) -> Path:
    """Resolve a checked-in manifest path without traversing symlinks."""
    parsed = _manifest_relative_path(value)
    root = root.resolve(strict=True)
    current = root
    for part in parsed.parts:
        current /= part
        try:
            current_stat = current.lstat()
        except FileNotFoundError as exc:
            raise ValueError(
                f"SCBench v2 manifest path does not exist: {value!r}"
            ) from exc
        if stat.S_ISLNK(current_stat.st_mode):
            raise ValueError(
                f"SCBench v2 manifest path traverses a symlink: {value!r}"
            )
    resolved = current.resolve(strict=True)
    if not resolved.is_relative_to(root):
        raise ValueError(f"SCBench v2 manifest path escapes root: {value!r}")
    return resolved


def _path_text(root: Path, path: Path | None) -> str | None:
    if path is None:
        return None
    resolved = path.resolve()
    try:
        return resolved.relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _add_check(
    checks: dict[str, dict[str, Any]],
    errors: list[str],
    field: str,
    expected: Any,
    actual: Any,
) -> None:
    matches = actual == expected
    checks[field] = {
        "expected": expected,
        "actual": actual,
        "status": "verified" if matches else "mismatch",
    }
    if not matches:
        errors.append(f"{field}: expected {expected!r}, got {actual!r}")


def _profile_evidence(
    root: Path,
    manifest: dict[str, Any],
    lock: dict[str, Any],
    context: NamedProfileContext,
) -> dict[str, Any]:
    profiles = manifest.get("profiles")
    if not isinstance(profiles, dict):
        raise ValueError("SCBench v2 manifest has no profiles mapping")
    expected = profiles.get(context.profile)
    if not isinstance(expected, dict):
        raise ValueError(f"unknown SCBench v2 profile: {context.profile!r}")
    protocol = manifest.get("protocol")
    if not isinstance(protocol, dict):
        raise ValueError("SCBench v2 manifest has no protocol mapping")
    catalog = manifest.get("catalog")
    if not isinstance(catalog, dict):
        raise ValueError("SCBench v2 manifest has no catalog mapping")

    # Validate every filesystem claim used to describe this profile, including
    # paths that are provenance-only rather than execution inputs.
    declared_existing_paths: list[str] = []
    for container, key in (
        (catalog, "content_lock"),
        (expected, "agent_config"),
        (expected, "model_config"),
        (expected, "config"),
        (expected, "diagnostic_config"),
    ):
        path_value = container.get(key)
        if not isinstance(path_value, str):
            raise ValueError(f"SCBench v2 manifest path {key!r} is invalid")
        declared_existing_paths.append(path_value)
    capability_config = expected.get("capability_config")
    if capability_config is not None:
        if not isinstance(capability_config, str):
            raise ValueError(
                "SCBench v2 manifest path 'capability_config' is invalid"
            )
        declared_existing_paths.append(capability_config)
    providers_path = protocol.get("providers_config")
    if not isinstance(providers_path, str):
        raise ValueError("SCBench v2 providers path is invalid")
    declared_existing_paths.append(providers_path)
    quality = manifest.get("quality_evaluator")
    primary = quality.get("primary") if isinstance(quality, dict) else None
    if not isinstance(primary, dict):
        raise ValueError("SCBench v2 manifest has no primary evaluator")
    for key in ("project", "lock"):
        path_value = primary.get(key)
        if not isinstance(path_value, str):
            raise ValueError(f"SCBench v2 evaluator {key} path is invalid")
        declared_existing_paths.append(path_value)
    prebuilt = protocol.get("prebuilt_base")
    archive = prebuilt.get("archive") if isinstance(prebuilt, dict) else None
    if isinstance(archive, dict):
        archive_path = archive.get("path")
        if not isinstance(archive_path, str):
            raise ValueError("SCBench v2 prebuilt archive path is invalid")
        _manifest_relative_path(archive_path)
        for key in ("loader", "checksum_file"):
            existing_path = archive.get(key)
            if not isinstance(existing_path, str):
                raise ValueError(
                    f"SCBench v2 prebuilt {key} path is invalid"
                )
            declared_existing_paths.append(existing_path)
    for path_value in declared_existing_paths:
        _resolved_path(root, path_value)

    locked_problems = lock.get("problems")
    if not isinstance(locked_problems, dict):
        raise ValueError("SCBench v2 content lock has no problems mapping")
    full_problems = list(locked_problems)
    diagnostic = manifest.get("diagnostic_subset")
    diagnostic_mapping = (
        diagnostic.get("problems") if isinstance(diagnostic, dict) else None
    )
    if not isinstance(diagnostic_mapping, dict):
        raise ValueError(
            "SCBench v2 manifest has no diagnostic problem mapping"
        )
    diagnostic_problems = list(diagnostic_mapping)
    capability = manifest.get("capability_subset")
    capability_mapping = (
        capability.get("problems") if isinstance(capability, dict) else None
    )
    capability_id = (
        capability.get("id") if isinstance(capability, dict) else None
    )
    capability_problems = (
        list(capability_mapping)
        if isinstance(capability_mapping, dict)
        else []
    )
    if context.problem_names == full_problems:
        variant = "full"
    elif context.problem_names == diagnostic_problems:
        variant = "diagnostic"
    elif (
        isinstance(capability_id, str)
        and capability_problems
        and context.problem_names == capability_problems
    ):
        variant = capability_id
    else:
        variant = None

    checks: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    paper = manifest.get("paper")
    lock_source = lock.get("source")
    _add_check(
        checks,
        errors,
        "manifest.schema_version",
        MANIFEST_SCHEMA_VERSION,
        manifest.get("schema_version"),
    )
    _add_check(
        checks,
        errors,
        "manifest.id",
        MANIFEST_ID,
        manifest.get("id"),
    )
    _add_check(
        checks,
        errors,
        "manifest.suite_revision",
        SUITE_REVISION,
        manifest.get("suite_revision"),
    )
    if not isinstance(paper, dict):
        errors.append("manifest paper identity must be a mapping")
    else:
        for field, expected_value in (
            ("arxiv_id", PAPER_ARXIV_ID),
            ("version", PAPER_VERSION),
            ("url", PAPER_URL),
        ):
            _add_check(
                checks,
                errors,
                f"paper.{field}",
                expected_value,
                paper.get(field),
            )
    if not isinstance(lock_source, dict):
        errors.append("content lock source must be a mapping")
    else:
        for field in ("repository", "release", "commit"):
            _add_check(
                checks,
                errors,
                f"catalog.source.{field}",
                lock_source.get(field),
                catalog.get(field),
            )
    _add_check(
        checks,
        errors,
        "model.provider",
        expected.get("provider"),
        context.model_provider,
    )
    _add_check(
        checks,
        errors,
        "model.name",
        expected.get("model"),
        context.model_name,
    )
    _add_check(
        checks,
        errors,
        "agent.type",
        expected.get("agent_type"),
        context.agent_type,
    )
    _add_check(
        checks,
        errors,
        "agent.version",
        expected.get("cli_version"),
        context.agent_version,
    )
    _add_check(
        checks,
        errors,
        "thinking",
        expected.get("reasoning"),
        context.thinking,
    )
    _add_check(
        checks,
        errors,
        "pass_policy",
        protocol.get("pass_policy"),
        context.pass_policy,
    )
    _add_check(
        checks,
        errors,
        "one_shot",
        protocol.get("one_shot"),
        context.one_shot,
    )
    _add_check(checks, errors, "seed", protocol.get("seed"), context.seed)
    _add_check(
        checks,
        errors,
        "evaluate",
        protocol.get("evaluate"),
        context.evaluate,
    )
    _add_check(
        checks,
        errors,
        "num_workers",
        protocol.get("num_workers"),
        context.num_workers,
    )
    _add_check(
        checks,
        errors,
        "concurrent_evaluation",
        protocol.get("concurrent_evaluation"),
        context.concurrent_evaluation,
    )
    _add_check(
        checks,
        errors,
        "catalog.version",
        catalog.get("release"),
        context.catalog_version,
    )
    _add_check(
        checks,
        errors,
        "catalog.commit",
        catalog.get("commit"),
        context.catalog_commit,
    )
    _add_check(
        checks,
        errors,
        "problem_set",
        variant or "full, diagnostic, or capability subset",
        variant or context.problem_names,
    )
    _add_check(
        checks,
        errors,
        "catalog.problem_count",
        lock.get("problem_count"),
        catalog.get("problem_count"),
    )
    _add_check(
        checks,
        errors,
        "catalog.checkpoint_count",
        lock.get("checkpoint_count"),
        catalog.get("checkpoint_count"),
    )
    _add_check(
        checks,
        errors,
        "catalog.content_lock_schema_version",
        lock.get("schema_version"),
        catalog.get("content_lock_schema_version"),
    )
    _add_check(
        checks,
        errors,
        "catalog.hash_algorithm",
        lock.get("hash_algorithm"),
        catalog.get("hash_algorithm"),
    )
    locked_diagnostic = {
        name: locked_problems.get(name) for name in diagnostic_mapping
    }
    _add_check(
        checks,
        errors,
        "diagnostic.problems",
        locked_diagnostic,
        diagnostic_mapping,
    )
    diagnostic_count = sum(
        value
        for value in diagnostic_mapping.values()
        if type(value) is int
    )
    _add_check(
        checks,
        errors,
        "diagnostic.checkpoint_count",
        diagnostic_count,
        diagnostic.get("checkpoint_count")
        if isinstance(diagnostic, dict)
        else None,
    )
    if isinstance(capability_mapping, dict):
        _add_check(
            checks,
            errors,
            "capability.id",
            CAPABILITY_SUBSET_ID,
            capability_id,
        )
        locked_capability = {
            name: locked_problems.get(name) for name in capability_mapping
        }
        _add_check(
            checks,
            errors,
            "capability.problems",
            locked_capability,
            capability_mapping,
        )
        capability_count = sum(
            value
            for value in capability_mapping.values()
            if type(value) is int
        )
        _add_check(
            checks,
            errors,
            "capability.checkpoint_count",
            capability_count,
            capability.get("checkpoint_count")
            if isinstance(capability, dict)
            else None,
        )

    expected_agent_path = expected.get("agent_config")
    actual_agent_path = _path_text(root, context.agent_config_path)
    _add_check(
        checks,
        errors,
        "agent.config_path",
        expected_agent_path,
        actual_agent_path,
    )
    expected_environment = protocol.get("environment")
    environment_path = (
        f"configs/environments/{expected_environment}.yaml"
        if isinstance(expected_environment, str)
        else None
    )
    _add_check(
        checks,
        errors,
        "environment.config_path",
        environment_path,
        _path_text(root, context.environment_config_path),
    )
    prompt_name = protocol.get("prompt")
    prompt_path = (
        f"configs/prompts/{prompt_name}.jinja"
        if isinstance(prompt_name, str)
        else None
    )
    _add_check(
        checks,
        errors,
        "prompt.path",
        prompt_path,
        _path_text(root, context.prompt_path),
    )

    if isinstance(expected_agent_path, str):
        canonical_agent_path = _resolved_path(root, expected_agent_path)
        canonical_agent = _load_mapping(
            canonical_agent_path,
            yaml_file=True,
        )
        _add_check(
            checks,
            errors,
            "agent.config_sha256",
            expected.get("agent_config_sha256"),
            sha256_file(canonical_agent_path),
        )
        _add_check(
            checks,
            errors,
            "agent.resolved_config_sha256",
            _mapping_sha256(canonical_agent),
            _mapping_sha256(context.agent_config),
        )
        _add_check(
            checks,
            errors,
            "agent.timeout",
            canonical_agent.get("timeout"),
            context.agent_config.get("timeout"),
        )
        _add_check(
            checks,
            errors,
            "agent.cost_limits",
            canonical_agent.get("cost_limits"),
            context.agent_config.get("cost_limits"),
        )
        canonical_cost_limits = canonical_agent.get("cost_limits")
        _add_check(
            checks,
            errors,
            "protocol.timeout_seconds_per_checkpoint",
            canonical_agent.get("timeout"),
            protocol.get("timeout_seconds_per_checkpoint"),
        )
        for protocol_key, agent_key in (
            ("cost_limit_usd", "cost_limit"),
            ("step_limit", "step_limit"),
            ("net_cost_limit_usd", "net_cost_limit"),
        ):
            _add_check(
                checks,
                errors,
                f"protocol.{protocol_key}",
                canonical_cost_limits.get(agent_key)
                if isinstance(canonical_cost_limits, dict)
                else None,
                protocol.get(protocol_key),
            )
        release_evidence = expected.get("release_evidence")
        if not isinstance(release_evidence, dict):
            errors.append("profile release_evidence must be a mapping")
        else:
            _add_check(
                checks,
                errors,
                "agent.npm_package_integrity",
                release_evidence.get("package_integrity"),
                canonical_agent.get("npm_package_integrity"),
            )
            platform_integrity = release_evidence.get("platform_integrity")
            alias_targets = release_evidence.get("platform_alias_targets")
            for platform_name, config_key in (
                ("linux_arm64", "npm_linux_arm64_integrity"),
                ("linux_x64", "npm_linux_x64_integrity"),
            ):
                _add_check(
                    checks,
                    errors,
                    f"agent.{platform_name}_integrity",
                    platform_integrity.get(platform_name)
                    if isinstance(platform_integrity, dict)
                    else None,
                    canonical_agent.get(config_key),
                )
                _add_check(
                    checks,
                    errors,
                    f"agent.{platform_name}_alias",
                    (
                        f"@openai/codex@{context.agent_version}-"
                        f"{platform_name.replace('_', '-')}"
                    ),
                    alias_targets.get(platform_name)
                    if isinstance(alias_targets, dict)
                    else None,
                )
    if isinstance(environment_path, str):
        canonical_environment_path = _resolved_path(root, environment_path)
        canonical_environment = _load_mapping(
            canonical_environment_path,
            yaml_file=True,
        )
        _add_check(
            checks,
            errors,
            "environment.config_sha256",
            protocol.get("environment_sha256"),
            sha256_file(canonical_environment_path),
        )
        _add_check(
            checks,
            errors,
            "environment.resolved_config_sha256",
            _mapping_sha256(canonical_environment),
            _mapping_sha256(context.environment),
        )
        docker = canonical_environment.get("docker")
        expected_image = (
            docker.get("image") if isinstance(docker, dict) else None
        )
        _add_check(
            checks,
            errors,
            "environment.source_image",
            expected_image,
            context.source_image,
        )
        source_image = protocol.get("source_image")
        if not isinstance(source_image, dict):
            errors.append("protocol source_image must be a mapping")
        else:
            pinned_digest = (
                expected_image.split("@", 1)[1]
                if isinstance(expected_image, str) and "@" in expected_image
                else None
            )
            for field, expected_value in (
                ("reference", SOURCE_IMAGE_REFERENCE),
                ("pinned", expected_image),
                ("digest", pinned_digest),
                ("role", SOURCE_IMAGE_ROLE),
            ):
                _add_check(
                    checks,
                    errors,
                    f"protocol.source_image.{field}",
                    expected_value,
                    source_image.get(field),
                )
        prebuilt_base = protocol.get("prebuilt_base")
        if isinstance(prebuilt_base, dict):
            _add_check(
                checks,
                errors,
                "environment.prebuilt_image",
                prebuilt_base.get("reference"),
                docker.get("prebuilt_image")
                if isinstance(docker, dict)
                else None,
            )
            _add_check(
                checks,
                errors,
                "environment.expected_image_id",
                prebuilt_base.get("image_id"),
                docker.get("expected_image_id")
                if isinstance(docker, dict)
                else None,
            )
            _add_check(
                checks,
                errors,
                "environment.expected_architecture",
                prebuilt_base.get("architecture"),
                docker.get("expected_architecture")
                if isinstance(docker, dict)
                else None,
            )
            architecture = prebuilt_base.get("architecture")
            _add_check(
                checks,
                errors,
                "environment.prebuilt_platform",
                f"linux/{architecture}",
                prebuilt_base.get("platform"),
            )
            _add_check(
                checks,
                errors,
                "environment.prebuilt_identity",
                prebuilt_base.get("image_id"),
                prebuilt_base.get("reference"),
            )
            bundled_tools = prebuilt_base.get("bundled_tools")
            minio = (
                bundled_tools.get("minio")
                if isinstance(bundled_tools, dict)
                else None
            )
            if not isinstance(minio, dict):
                errors.append("prebuilt base MinIO identity must be a mapping")
            else:
                setup_template = _resolved_path(
                    root,
                    SETUP_BASE_TEMPLATE,
                ).read_text(encoding="utf-8")
                release_match = re.search(
                    r"minio_release='([^']+)'",
                    setup_template,
                )
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
                for field, expected_value in expected_minio.items():
                    _add_check(
                        checks,
                        errors,
                        f"environment.minio.{field}",
                        expected_value,
                        minio.get(field),
                    )
            archive = prebuilt_base.get("archive")
            if not isinstance(archive, dict):
                errors.append("prebuilt base archive must be a mapping")
            else:
                archive_path = archive.get("path")
                checksum_path = archive.get("checksum_file")
                archive_sha256 = archive.get("sha256")
                if not all(
                    isinstance(value, str)
                    for value in (
                        archive_path,
                        checksum_path,
                        archive_sha256,
                    )
                ):
                    errors.append("prebuilt archive checksum identity is invalid")
                else:
                    checksum_text = _resolved_path(
                        root,
                        checksum_path,
                    ).read_text(encoding="ascii")
                    expected_checksum = (
                        f"{archive_sha256}  "
                        f"{PurePosixPath(archive_path).name}\n"
                    )
                    _add_check(
                        checks,
                        errors,
                        "environment.archive.checksum_file",
                        expected_checksum,
                        checksum_text,
                    )
        _add_check(
            checks,
            errors,
            "environment.name",
            canonical_environment.get("name"),
            context.environment_name,
        )
    if isinstance(prompt_path, str):
        canonical_prompt_path = _resolved_path(root, prompt_path)
        _add_check(
            checks,
            errors,
            "prompt.sha256",
            protocol.get("prompt_sha256"),
            sha256_file(canonical_prompt_path),
        )
        _add_check(
            checks,
            errors,
            "prompt.resolved_sha256",
            protocol.get("prompt_sha256"),
            _sha256_text(context.prompt_content),
        )

    model_config_path = expected.get("model_config")
    if isinstance(model_config_path, str):
        actual_model_config_path = f"configs/models/{context.model_name}.yaml"
        _add_check(
            checks,
            errors,
            "model.config_path",
            model_config_path,
            actual_model_config_path,
        )
        _add_check(
            checks,
            errors,
            "model.config_sha256",
            expected.get("model_config_sha256"),
            sha256_file(_resolved_path(root, model_config_path)),
        )
    providers_config_path = protocol.get("providers_config")
    if isinstance(providers_config_path, str):
        actual_providers_config_path = "configs/providers.yaml"
        _add_check(
            checks,
            errors,
            "providers.config_path",
            providers_config_path,
            actual_providers_config_path,
        )
        _add_check(
            checks,
            errors,
            "providers.config_sha256",
            protocol.get("providers_config_sha256"),
            sha256_file(_resolved_path(root, providers_config_path)),
        )

    quality = manifest.get("quality_evaluator")
    primary = quality.get("primary") if isinstance(quality, dict) else None
    if not isinstance(primary, dict):
        errors.append("quality evaluator primary must be a mapping")
    else:
        from slop_code.metrics.checkpoint.driver import SCB_CHECK_REQUIREMENT

        semantic_values = (
            ("package", SCB_CHECK_REQUIREMENT),
            ("project", EVALUATOR_PROJECT),
            ("lock", EVALUATOR_LOCK),
            ("command", EVALUATOR_COMMAND),
        )
        for field, expected_value in semantic_values:
            actual_value = primary.get(field)
            if field == "command" and isinstance(actual_value, str):
                actual_value = " ".join(actual_value.split())
            _add_check(
                checks,
                errors,
                f"evaluator.{field}",
                expected_value,
                actual_value,
            )
        for path_key, hash_key in (
            ("project", "project_sha256"),
            ("lock", "lock_sha256"),
        ):
            path_value = primary.get(path_key)
            if isinstance(path_value, str):
                _add_check(
                    checks,
                    errors,
                    f"evaluator.{hash_key}",
                    primary.get(hash_key),
                    sha256_file(_resolved_path(root, path_value)),
                )

    return {
        "name": context.profile,
        "variant": variant,
        "status": "verified" if not errors else "failed",
        "checks": checks,
        "errors": errors,
    }


def _catalog_evidence(
    root: Path,
    manifest: dict[str, Any],
    catalog_root: Path,
) -> tuple[dict[str, Any], list[str]]:
    catalog = manifest.get("catalog")
    if not isinstance(catalog, dict):
        raise ValueError("SCBench v2 manifest has no catalog mapping")
    lock_value = catalog.get("content_lock")
    if not isinstance(lock_value, str):
        raise ValueError("SCBench v2 manifest has no content-lock path")
    lock_path = _resolved_path(root, lock_value)
    expected = _load_mapping(lock_path, yaml_file=False)
    actual, errors = verify_catalog_content(catalog_root, expected)
    lock_sha256 = sha256_file(lock_path)
    if catalog.get("content_lock_sha256") != lock_sha256:
        errors.append(
            "catalog content-lock digest mismatch: expected "
            f"{catalog.get('content_lock_sha256')!r}, got {lock_sha256!r}"
        )
    if catalog.get("hash_algorithm") != expected.get("hash_algorithm"):
        errors.append(
            "catalog manifest hash algorithm does not match the content lock"
        )
    evidence = {
        "status": "verified" if not errors else "failed",
        "root": str(catalog_root.resolve()),
        "lock_path": lock_value,
        "lock_sha256": lock_sha256,
        "expected": {
            key: expected.get(key)
            for key in (
                "hash_algorithm",
                "problem_count",
                "checkpoint_count",
                "file_count",
                "tree_sha256",
            )
        },
        "actual": {
            key: actual.get(key)
            for key in (
                "hash_algorithm",
                "problem_count",
                "checkpoint_count",
                "file_count",
                "tree_sha256",
            )
        },
        "errors": errors,
    }
    return evidence, errors


def _preflight_attempts(target: Path) -> list[dict[str, Any]]:
    """Load prior attempts without silently discarding malformed evidence."""
    if not target.exists():
        return []
    value = _load_mapping(target, yaml_file=False)
    if value.get("schema_version") == PREFLIGHT_HISTORY_SCHEMA_VERSION:
        attempts = value.get("attempts")
        current_attempt_id = value.get("current_attempt_id")
        if not isinstance(attempts, list) or not all(
            isinstance(attempt, dict) for attempt in attempts
        ):
            raise ValueError(f"invalid preflight attempt history: {target}")
        attempt_ids = [attempt.get("attempt_id") for attempt in attempts]
        if (
            not attempts
            or not all(
                isinstance(attempt_id, str) for attempt_id in attempt_ids
            )
            or len(set(attempt_ids)) != len(attempt_ids)
            or current_attempt_id != attempt_ids[-1]
        ):
            raise ValueError(f"invalid preflight current pointer: {target}")
        return attempts

    # Migrate the original single-attempt artifact without losing it.
    if isinstance(value.get("started_at"), str) and isinstance(
        value.get("status"), str
    ):
        return [{"attempt_id": "attempt-000001", **value}]
    raise ValueError(f"unrecognized preflight evidence schema: {target}")


def _write_preflight(run_dir: Path, evidence: dict[str, Any]) -> None:
    """Append an immutable attempt and atomically advance the current pointer."""
    if run_dir.is_symlink():
        raise OSError(f"preflight run directory is a symlink: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    target = run_dir / PREFLIGHT_FILENAME
    if target.is_symlink():
        raise OSError(f"preflight evidence path is a symlink: {target}")
    attempts = _preflight_attempts(target)
    attempt_id = f"attempt-{len(attempts) + 1:06d}"
    attempt = json.loads(json.dumps({"attempt_id": attempt_id, **evidence}))
    history = {
        "schema_version": PREFLIGHT_HISTORY_SCHEMA_VERSION,
        "current_attempt_id": attempt_id,
        "attempts": [*attempts, attempt],
    }
    atomic_write_text(
        target,
        json.dumps(history, indent=2, sort_keys=True) + "\n",
    )


def run_named_profile_preflight(
    *,
    repository_root: Path,
    run_dir: Path,
    catalog_root: Path,
    context: NamedProfileContext,
    evaluator_preflight: Callable[[], dict[str, Any]] | None = None,
    persist: bool = True,
    verify_evaluator: bool = True,
) -> dict[str, Any] | None:
    """Validate immutable inputs for a named v2 profile.

    Normal runs verify the frozen evaluator and append evidence. Dry-run
    callers can disable both operations while retaining read-only profile and
    catalog validation.
    """
    if context.profile is None:
        return None

    root = repository_root.resolve()
    evidence: dict[str, Any] = {
        "schema_version": 1,
        "started_at": _utc_now(),
        "finished_at": None,
        "status": "running",
        "profile": None,
        "catalog": None,
        "evaluator": {"status": "not_run"},
        "errors": [],
    }
    try:
        manifest = _load_mapping(
            _resolved_path(root, MANIFEST_PATH.as_posix()),
            yaml_file=True,
        )
        lock = _load_mapping(
            _resolved_path(root, CONTENT_LOCK_PATH.as_posix()),
            yaml_file=False,
        )
        profile = _profile_evidence(root, manifest, lock, context)
        evidence["profile"] = profile
        catalog, catalog_errors = _catalog_evidence(
            root,
            manifest,
            catalog_root,
        )
        evidence["catalog"] = catalog
        errors = [*profile["errors"], *catalog_errors]
        if errors:
            raise ValueError("; ".join(errors))

        if verify_evaluator:
            if evaluator_preflight is None:
                from slop_code.metrics.checkpoint.driver import (
                    scb_check_preflight,
                )

                evaluator_preflight = scb_check_preflight
            evidence["evaluator"] = evaluator_preflight()
            if evidence["evaluator"].get("status") != "verified":
                raise ValueError(
                    "scb-check evaluator preflight was not verified"
                )
            quality = manifest.get("quality_evaluator")
            primary = (
                quality.get("primary") if isinstance(quality, dict) else None
            )
            if not isinstance(primary, dict):
                raise ValueError("SCBench v2 manifest has no primary evaluator")
            evaluator_errors = []
            for field in ("project_sha256", "lock_sha256"):
                expected_hash = primary.get(field)
                actual_hash = evidence["evaluator"].get(field)
                if actual_hash != expected_hash:
                    evaluator_errors.append(
                        f"evaluator {field}: expected {expected_hash!r}, "
                        f"got {actual_hash!r}"
                    )
            if evidence["evaluator"].get("requirement") != primary.get(
                "package"
            ):
                evaluator_errors.append(
                    "evaluator requirement does not match manifest"
                )
            if evaluator_errors:
                raise ValueError("; ".join(evaluator_errors))
        else:
            evidence["evaluator"] = {
                "status": "not_run",
                "reason": "read_only_validation",
            }
        if persist:
            quality = manifest.get("quality_evaluator")
            primary = (
                quality.get("primary")
                if isinstance(quality, dict)
                else None
            )
            if not isinstance(primary, dict):
                raise ValueError("SCBench v2 manifest has no primary evaluator")
            project_value = primary.get("project")
            lock_value = primary.get("lock")
            if not isinstance(project_value, str) or not isinstance(
                lock_value, str
            ):
                raise ValueError(
                    "SCBench v2 manifest evaluator paths are invalid"
                )
            evaluator_source = _resolved_path(root, project_value).parent
            if _resolved_path(root, lock_value).parent != evaluator_source:
                raise ValueError(
                    "SCBench v2 evaluator project and lock do not share a root"
                )
            staged_evaluator = _stage_verified_evaluator(
                evaluator_source,
                run_dir,
                primary,
            )
            evidence["evaluator"]["execution_project"] = str(
                staged_evaluator
            )
            evidence["evaluator"]["staged"] = {
                "status": "verified",
                "project_sha256": primary.get("project_sha256"),
                "lock_sha256": primary.get("lock_sha256"),
            }
            staged_root, staged_actual = _stage_verified_catalog(
                catalog_root,
                run_dir,
                lock,
            )
            catalog["execution_root"] = str(staged_root)
            catalog["staged"] = {
                "status": "verified",
                "root": str(staged_root),
                "actual": {
                    key: staged_actual.get(key)
                    for key in (
                        "hash_algorithm",
                        "problem_count",
                        "checkpoint_count",
                        "file_count",
                        "tree_sha256",
                    )
                },
            }
        else:
            catalog["execution_root"] = None
            catalog["staged"] = {
                "status": "not_created",
                "reason": "read_only_validation",
            }
    except Exception as exc:  # noqa: BLE001 - all preflight drift fails closed.
        evidence["status"] = "failed"
        evidence["errors"] = [str(exc)]
        evidence["finished_at"] = _utc_now()
        if persist:
            try:
                _write_preflight(run_dir, evidence)
            except (OSError, ValueError) as write_exc:
                evidence["errors"].append(
                    f"cannot persist preflight evidence: {write_exc}"
                )
        raise ScbenchV2PreflightError(
            f"SCBench v2 profile preflight failed: {exc}",
            evidence,
        ) from exc

    evidence["status"] = "verified"
    evidence["finished_at"] = _utc_now()
    if persist:
        try:
            _write_preflight(run_dir, evidence)
        except (OSError, ValueError) as exc:
            evidence["status"] = "failed"
            evidence["errors"] = [f"cannot persist preflight evidence: {exc}"]
            raise ScbenchV2PreflightError(
                f"SCBench v2 profile preflight failed: {exc}",
                evidence,
            ) from exc
    return evidence
