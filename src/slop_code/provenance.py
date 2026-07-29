"""Create credential-safe, machine-readable provenance for benchmark runs."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
from datetime import UTC
from datetime import datetime
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any
from urllib.parse import urlsplit
from urllib.parse import urlunsplit

import yaml

from slop_code.common.atomic import UnsafeAtomicWriteError
from slop_code.common.atomic import _open_directory_without_symlinks
from slop_code.common.atomic import atomic_write_text

PROVENANCE_FILENAME = "provenance.json"
PROVENANCE_SCHEMA_VERSION = 5
ARTIFACT_MANIFEST_SCHEMA_VERSION = 2
LEGACY_ARTIFACT_MANIFEST_SCHEMA_VERSION = 1
SUPPORTED_LEGACY_PROVENANCE_SCHEMAS = frozenset({2, 4})
ARTIFACT_EXCLUSIONS = (PROVENANCE_FILENAME, "run_agent.log")
LEGACY_ARTIFACT_EXCLUSIONS = (PROVENANCE_FILENAME, "*.log")
PROVENANCE_STATUSES = frozenset(
    {
        "running",
        "completed",
        "failed",
        "interrupted_before_next_invocation",
        "incomplete_problem_execution",
        "incomplete_postprocessing",
        "incomplete_problem_execution_and_postprocessing",
        "incomplete_provenance",
    }
)
_SUITE_MANIFEST = Path("configs/scbench-v2/manifest.yaml")
_SUITE_CONTENT_LOCK = Path("configs/scbench-v2/content-lock.json")
_SENSITIVE_NAME = re.compile(
    r"(?i)(api[_-]?key|access[_-]?token|auth(?:orization)?[_-]?token|"
    r"authorization|password|secret|credential|cookie|header)"
)
_ENV_OVERRIDE_NAME = re.compile(r"(?i)(?:^|[._-])env(?:$|[._-])")
_SECRET_VALUE = re.compile(
    r"(?i)^(?:sk-|sess-|gh[opusr]_|xox[baprs]-|Bearer\s+)[^\s]+$"
)
_SECRET_FRAGMENT = re.compile(
    r"(?i)(?:authorization\s*:\s*(?:bearer|basic)\s+\S+|"
    r"(?:api[_-]?key|access[_-]?token|auth[_-]?token|password|secret)="
    r"[^&\s]+)"
)
_SCP_REMOTE = re.compile(
    r"^(?:[^/@:\s]+(?::[^/@\s]+)?)@(?P<host>[^:\s]+):(?P<path>.+)$"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class ProvenanceIntegrityError(RuntimeError):
    """Raised when existing benchmark provenance cannot be trusted."""


def _utc_now() -> str:
    return datetime.now(tz=UTC).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a regular file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _optional_file_sha256(path: Path) -> str | None:
    try:
        return sha256_file(path)
    except OSError:
        return None


def sanitize_invocation(arguments: list[str]) -> list[str]:
    """Redact likely credential values from command-line arguments."""
    sanitized: list[str] = []
    redact_next = False
    for argument in arguments:
        if redact_next:
            sanitized.append("<redacted>")
            redact_next = False
            continue

        if "=" in argument:
            name, value = argument.split("=", 1)
            if (
                _SENSITIVE_NAME.search(name)
                or _ENV_OVERRIDE_NAME.search(name)
                or _SECRET_VALUE.match(value)
                or _SECRET_FRAGMENT.search(value)
            ):
                sanitized.append(f"{name}=<redacted>")
                continue
            sanitized_url = _sanitize_remote_url(value)
            if sanitized_url != value:
                sanitized.append(f"{name}={sanitized_url}")
                continue

        if argument.startswith("-") and _SENSITIVE_NAME.search(argument):
            sanitized.append(argument)
            redact_next = True
            continue

        if _SECRET_VALUE.match(argument) or _SECRET_FRAGMENT.search(argument):
            sanitized.append("<redacted>")
            continue

        sanitized_url = _sanitize_remote_url(argument)
        sanitized.append(sanitized_url or "<redacted>")
    return sanitized


def _sanitize_remote_url(value: str | None) -> str | None:
    if not value:
        return value
    scp_match = _SCP_REMOTE.match(value)
    if scp_match:
        return f"{scp_match.group('host')}:{scp_match.group('path')}"
    parsed = urlsplit(value)
    if not parsed.scheme or not parsed.netloc:
        return value
    hostname = parsed.hostname or ""
    if parsed.port is not None:
        hostname = f"{hostname}:{parsed.port}"
    return urlunsplit((parsed.scheme, hostname, parsed.path, "", ""))


def _run_command(
    arguments: list[str],
    *,
    cwd: Path | None = None,
    timeout: int = 15,
) -> dict[str, str | int | None]:
    try:
        result = subprocess.run(  # noqa: S603
            arguments,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "value": None,
            "returncode": None,
            "error": type(exc).__name__,
        }
    output = result.stdout.strip() or result.stderr.strip()
    return {
        "value": output or None,
        "returncode": result.returncode,
        "error": None if result.returncode == 0 else "command_failed",
    }


def _git_output(root: Path, *arguments: str) -> str | None:
    result = _run_command(["git", *arguments], cwd=root)
    if result["returncode"] != 0:
        return None
    value = result["value"]
    return value if isinstance(value, str) else None


def _git_diff_sha256(root: Path) -> tuple[bool | None, str | None]:
    git = shutil.which("git")
    if git is None:
        return None, None
    try:
        status = subprocess.run(  # noqa: S603
            [git, "status", "--porcelain=v1", "-z", "--untracked-files=all"],
            cwd=root,
            capture_output=True,
            check=False,
            timeout=15,
        )
        if status.returncode != 0:
            return None, None
        if not status.stdout:
            return False, None
        diff = subprocess.run(  # noqa: S603
            [git, "diff", "--binary", "HEAD", "--"],
            cwd=root,
            capture_output=True,
            check=False,
            timeout=30,
        )
        untracked = subprocess.run(  # noqa: S603
            [git, "ls-files", "--others", "--exclude-standard", "-z"],
            cwd=root,
            capture_output=True,
            check=False,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None, None
    if diff.returncode != 0 or untracked.returncode != 0:
        return None, None

    digest = hashlib.sha256()
    digest.update(b"status\0")
    digest.update(status.stdout)
    digest.update(b"diff\0")
    digest.update(diff.stdout)
    for raw_path in sorted(filter(None, untracked.stdout.split(b"\0"))):
        digest.update(b"untracked\0")
        digest.update(raw_path)
        path = root / os.fsdecode(raw_path)
        if path.is_file() and not path.is_symlink():
            digest.update(sha256_file(path).encode("ascii"))
    return True, digest.hexdigest()


def _load_suite_manifest(root: Path) -> dict[str, Any] | None:
    try:
        value = yaml.safe_load(
            (root / _SUITE_MANIFEST).read_text(encoding="utf-8")
        )
    except (OSError, yaml.YAMLError):
        return None
    return value if isinstance(value, dict) else None


def _git_repository_metadata(root: Path) -> dict[str, Any]:
    head = _git_output(root, "rev-parse", "HEAD")
    branch = _git_output(root, "symbolic-ref", "--short", "-q", "HEAD")
    tags_raw = _git_output(root, "tag", "--points-at", "HEAD")
    dirty, diff_sha256 = _git_diff_sha256(root)
    manifest = _load_suite_manifest(root) or {}
    runner = manifest.get("runner")
    declared_base = (
        runner.get("base_commit") if isinstance(runner, dict) else None
    )
    base_is_ancestor: bool | None = None
    if isinstance(declared_base, str):
        ancestor = _run_command(
            ["git", "merge-base", "--is-ancestor", declared_base, "HEAD"],
            cwd=root,
        )
        if ancestor["returncode"] in {0, 1}:
            base_is_ancestor = ancestor["returncode"] == 0
    return {
        "head": {
            "sha": head,
            "branch": branch,
            "tags": sorted(tags_raw.splitlines()) if tags_raw else [],
            "dirty": dirty,
            "diff_sha256": diff_sha256,
        },
        "fork": {
            "remote": "origin",
            "url": _sanitize_remote_url(
                _git_output(root, "remote", "get-url", "origin")
            ),
            "sha": head,
        },
        "upstream": {
            "remote": "upstream",
            "url": _sanitize_remote_url(
                _git_output(root, "remote", "get-url", "upstream")
            ),
            "remote_head_sha": _git_output(
                root, "rev-parse", "--verify", "upstream/main"
            ),
            "declared_base_sha": declared_base,
            "declared_base_is_ancestor": base_is_ancestor,
        },
    }


def _image_metadata(image_name: str) -> dict[str, Any]:
    if not image_name:
        return {"name": image_name, "available": False}
    inspected = _run_command(
        ["docker", "image", "inspect", image_name], timeout=30
    )
    value = inspected["value"]
    if inspected["returncode"] != 0 or not isinstance(value, str):
        return {
            "name": image_name,
            "available": False,
            "error": inspected["error"],
        }
    try:
        image = json.loads(value)[0]
    except (json.JSONDecodeError, IndexError, TypeError):
        return {
            "name": image_name,
            "available": False,
            "error": "invalid_inspect_output",
        }
    return {
        "name": image_name,
        "available": True,
        "id": image.get("Id"),
        "repo_digests": sorted(image.get("RepoDigests") or []),
        "created": image.get("Created"),
        "os": image.get("Os"),
        "architecture": image.get("Architecture"),
    }


def _container_tool_versions(
    image_name: str, agent_type: str
) -> dict[str, Any]:
    if not image_name:
        return {"value": None, "returncode": None, "error": "no_image"}
    agent_commands = {
        "claude_code": "claude --version",
        "codex": "codex --version",
        "cursor_cli": "agent --version",
        "gemini": "gemini --version",
        "opencode": "opencode --version",
    }
    commands = [
        "python --version",
        "uv --version",
        "git --version",
        "rg --version",
    ]
    agent_command = agent_commands.get(agent_type)
    if agent_command:
        commands.append(agent_command)
    return _run_command(
        [
            "docker",
            "run",
            "--rm",
            "--entrypoint",
            "sh",
            image_name,
            "-lc",
            "; ".join(commands),
        ],
        timeout=60,
    )


def _host_metadata() -> dict[str, Any]:
    return {
        "os": platform.system(),
        "os_release": platform.release(),
        "architecture": platform.machine(),
        "python": sys.version.splitlines()[0],
        "uv": _run_command(["uv", "--version"]),
        "docker": _run_command(["docker", "--version"]),
    }


def _relative_or_absolute(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _input_metadata(root: Path, run_dir: Path) -> dict[str, Any]:
    content_lock_path = root / _SUITE_CONTENT_LOCK
    content_tree_sha256 = None
    try:
        content_lock = json.loads(content_lock_path.read_text(encoding="utf-8"))
        if isinstance(content_lock, dict):
            content_tree_sha256 = content_lock.get("tree_sha256")
    except (OSError, json.JSONDecodeError):
        pass
    paths = {
        "suite_manifest": root / _SUITE_MANIFEST,
        "suite_content_lock": content_lock_path,
        "pyproject": root / "pyproject.toml",
        "uv_lock": root / "uv.lock",
        "resolved_config": run_dir / "config.yaml",
        "resolved_environment": run_dir / "environment.yaml",
        "problem_catalog": run_dir / "problem_catalog.json",
        "scbench_v2_preflight": run_dir / "scbench_v2_preflight.json",
    }
    inputs = {
        key: {
            "path": _relative_or_absolute(root, path),
            "sha256": _optional_file_sha256(path),
        }
        for key, path in paths.items()
    }
    inputs["suite_content_tree_sha256"] = content_tree_sha256
    return inputs


def _write_provenance(run_dir: Path, value: dict[str, Any]) -> None:
    target = run_dir / PROVENANCE_FILENAME
    try:
        atomic_write_text(
            target,
            json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n",
        )
    except UnsafeAtomicWriteError as exc:
        raise ProvenanceIntegrityError(
            f"refusing unsafe provenance write: {exc}"
        ) from exc


def _is_timestamp(value: object) -> bool:
    if not isinstance(value, str) or not value:
        return False
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return parsed.tzinfo is not None


def _artifact_manifest(
    value: object,
    *,
    legacy: bool,
) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ProvenanceIntegrityError("provenance artifacts must be an object")

    schema_version = value.get("schema_version")
    if legacy and schema_version is None:
        schema_version = (
            ARTIFACT_MANIFEST_SCHEMA_VERSION
            if value.get("excluded") == list(ARTIFACT_EXCLUSIONS)
            else LEGACY_ARTIFACT_MANIFEST_SCHEMA_VERSION
        )
    if schema_version not in {
        LEGACY_ARTIFACT_MANIFEST_SCHEMA_VERSION,
        ARTIFACT_MANIFEST_SCHEMA_VERSION,
    }:
        raise ProvenanceIntegrityError(
            "unsupported provenance artifact manifest schema: "
            f"{schema_version!r}"
        )
    if value.get("algorithm") != "sha256":
        raise ProvenanceIntegrityError(
            "provenance artifact manifest must use sha256"
        )
    expected_exclusions = (
        LEGACY_ARTIFACT_EXCLUSIONS
        if schema_version == LEGACY_ARTIFACT_MANIFEST_SCHEMA_VERSION
        else ARTIFACT_EXCLUSIONS
    )
    if value.get("excluded") != list(expected_exclusions):
        raise ProvenanceIntegrityError(
            "provenance artifact exclusions do not match the supported policy"
        )
    files = value.get("files")
    if not isinstance(files, dict):
        raise ProvenanceIntegrityError(
            "provenance artifact manifest files must be an object"
        )
    for relative, digest in files.items():
        if not isinstance(relative, str) or not relative:
            raise ProvenanceIntegrityError(
                "provenance artifact paths must be non-empty strings"
            )
        parsed = PurePosixPath(relative)
        if parsed.is_absolute() or ".." in parsed.parts or relative != parsed.as_posix():
            raise ProvenanceIntegrityError(
                f"unsafe provenance artifact path: {relative!r}"
            )
        if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
            raise ProvenanceIntegrityError(
                f"invalid provenance artifact digest for {relative!r}"
            )

    complete = value.get("complete")
    if legacy and complete is None:
        complete = value.get("reason") is None
    if type(complete) is not bool:
        raise ProvenanceIntegrityError(
            "provenance artifact manifest must state boolean completeness"
        )
    reason = value.get("reason")
    if complete and reason is not None:
        raise ProvenanceIntegrityError(
            "complete provenance artifact manifest cannot have a failure reason"
        )
    if not complete and (not isinstance(reason, str) or not reason):
        raise ProvenanceIntegrityError(
            "incomplete provenance artifact manifest must state a reason"
        )
    if not complete and files:
        raise ProvenanceIntegrityError(
            "incomplete provenance artifact manifest cannot claim file hashes"
        )
    return {
        "schema_version": schema_version,
        "algorithm": "sha256",
        "excluded": list(expected_exclusions),
        "files": dict(files),
        "complete": complete,
        **({"reason": reason} if reason is not None else {}),
    }


def _migrate_provenance(value: dict[str, Any]) -> dict[str, Any]:
    """Normalize only the two provenance schemas emitted by this fork."""
    schema_version = value.get("schema_version")
    if schema_version == PROVENANCE_SCHEMA_VERSION:
        return value
    if schema_version not in SUPPORTED_LEGACY_PROVENANCE_SCHEMAS:
        raise ProvenanceIntegrityError(
            "unsupported provenance schema version: "
            f"{schema_version!r}; supported legacy versions are "
            f"{sorted(SUPPORTED_LEGACY_PROVENANCE_SCHEMAS)}"
        )

    # JSON round-tripping gives migration its own tree and rejects values that
    # could not have come from the on-disk JSON document.
    migrated = json.loads(json.dumps(value, allow_nan=False))
    migrated["migrated_from_schema_version"] = schema_version
    if schema_version == 2:
        migrated["suite"] = migrated.pop("paper", None)
        migrated.setdefault("preflight", None)
    migrated["artifacts"] = _artifact_manifest(
        migrated.get("artifacts"),
        legacy=True,
    )

    invocations = migrated.get("invocations")
    if isinstance(invocations, list):
        for invocation in invocations:
            if not isinstance(invocation, dict):
                continue
            invocation["migrated_from_schema_version"] = schema_version
            invocation["artifacts"] = _artifact_manifest(
                invocation.get("artifacts"),
                legacy=True,
            )
            context = invocation.get("context")
            if schema_version == 2 and isinstance(context, dict):
                context["suite"] = context.pop("paper", None)
                context.setdefault("preflight", None)
    migrated["schema_version"] = PROVENANCE_SCHEMA_VERSION
    return migrated


def _validate_run_identity(
    run: object,
    run_dir: Path,
    *,
    legacy_schema: int | None,
) -> None:
    if not isinstance(run, dict):
        raise ProvenanceIntegrityError("provenance run identity must be an object")
    directory = run.get("directory")
    if not isinstance(directory, str) or Path(directory).resolve() != run_dir:
        raise ProvenanceIntegrityError(
            "provenance run directory does not match the resumed directory"
        )
    model = run.get("model")
    agent = run.get("agent")
    if not isinstance(model, dict) or not all(
        isinstance(model.get(key), str) and model[key]
        for key in ("provider", "name")
    ):
        raise ProvenanceIntegrityError("provenance model identity is invalid")
    if not isinstance(agent, dict) or not isinstance(agent.get("type"), str):
        raise ProvenanceIntegrityError("provenance agent identity is invalid")
    if agent.get("version") is not None and not isinstance(
        agent.get("version"), str
    ):
        raise ProvenanceIntegrityError("provenance agent version is invalid")
    problems = run.get("problems")
    if (
        not isinstance(problems, list)
        or not problems
        or not all(isinstance(problem, str) and problem for problem in problems)
        or len(set(problems)) != len(problems)
    ):
        raise ProvenanceIntegrityError("provenance problem identity is invalid")
    if not isinstance(run.get("environment"), str):
        raise ProvenanceIntegrityError("provenance environment identity is invalid")
    catalog = run.get("catalog")
    if legacy_schema != 2 and (
        not isinstance(catalog, dict)
        or not all(
            isinstance(catalog.get(key), str) and catalog[key]
            for key in ("version", "commit")
        )
    ):
        raise ProvenanceIntegrityError("provenance catalog identity is invalid")


def _validate_context(
    context: object,
    run_dir: Path,
    *,
    legacy_schema: int | None,
) -> None:
    if not isinstance(context, dict):
        raise ProvenanceIntegrityError("provenance invocation context is invalid")
    profile = context.get("profile")
    if profile is not None and not isinstance(profile, str):
        raise ProvenanceIntegrityError("provenance profile identity is invalid")
    suite = context.get("suite")
    if suite is not None and not isinstance(suite, dict):
        raise ProvenanceIntegrityError("provenance suite identity is invalid")
    for key in ("repository", "inputs", "host", "docker"):
        if not isinstance(context.get(key), dict):
            raise ProvenanceIntegrityError(
                f"provenance context {key!r} must be an object"
            )
    preflight = context.get("preflight")
    if preflight is not None and not isinstance(preflight, dict):
        raise ProvenanceIntegrityError("provenance preflight evidence is invalid")
    _validate_run_identity(
        context.get("run"),
        run_dir,
        legacy_schema=legacy_schema,
    )


def _validate_provenance(value: dict[str, Any], run_dir: Path) -> None:
    if value.get("schema_version") != PROVENANCE_SCHEMA_VERSION:
        raise ProvenanceIntegrityError("provenance migration did not reach v5")
    migrated_from = value.get("migrated_from_schema_version")
    if migrated_from is not None and migrated_from not in (
        SUPPORTED_LEGACY_PROVENANCE_SCHEMAS
    ):
        raise ProvenanceIntegrityError("invalid provenance migration marker")

    top_context = {
        key: value.get(key)
        for key in (
            "profile",
            "suite",
            "repository",
            "inputs",
            "run",
            "host",
            "docker",
            "preflight",
        )
    }
    _validate_context(
        top_context,
        run_dir,
        legacy_schema=migrated_from if isinstance(migrated_from, int) else None,
    )

    invocations = value.get("invocations")
    if not isinstance(invocations, list) or not invocations:
        raise ProvenanceIntegrityError(
            "existing provenance has no invocation history"
        )
    for position, invocation in enumerate(invocations, start=1):
        if not isinstance(invocation, dict):
            raise ProvenanceIntegrityError(
                f"provenance invocation {position} is not an object"
            )
        legacy_schema = invocation.get("migrated_from_schema_version")
        if legacy_schema is not None and legacy_schema not in (
            SUPPORTED_LEGACY_PROVENANCE_SCHEMAS
        ):
            raise ProvenanceIntegrityError(
                f"provenance invocation {position} has an invalid migration marker"
            )
        if not _is_timestamp(invocation.get("started_at")):
            raise ProvenanceIntegrityError(
                f"provenance invocation {position} has an invalid start time"
            )
        status = invocation.get("status")
        if status not in PROVENANCE_STATUSES:
            raise ProvenanceIntegrityError(
                f"provenance invocation {position} has an invalid status"
            )
        finished_at = invocation.get("finished_at")
        if status == "running":
            if position != len(invocations):
                raise ProvenanceIntegrityError(
                    f"historical provenance invocation {position} is still running"
                )
            if finished_at is not None:
                raise ProvenanceIntegrityError(
                    f"running provenance invocation {position} is already finished"
                )
        elif not _is_timestamp(finished_at):
            raise ProvenanceIntegrityError(
                f"provenance invocation {position} has an invalid finish time"
            )
        if invocation.get("error_type") is not None and not isinstance(
            invocation.get("error_type"), str
        ):
            raise ProvenanceIntegrityError(
                f"provenance invocation {position} has an invalid error type"
            )
        arguments = invocation.get("arguments")
        problems = invocation.get("problems_executed")
        if not isinstance(arguments, list) or not all(
            isinstance(argument, str) for argument in arguments
        ):
            raise ProvenanceIntegrityError(
                f"provenance invocation {position} arguments are invalid"
            )
        if not isinstance(problems, list) or not all(
            isinstance(problem, str) and problem for problem in problems
        ):
            raise ProvenanceIntegrityError(
                f"provenance invocation {position} problem list is invalid"
            )
        if type(invocation.get("num_workers")) is not int or invocation[
            "num_workers"
        ] < 1:
            raise ProvenanceIntegrityError(
                f"provenance invocation {position} worker count is invalid"
            )
        if type(invocation.get("evaluate")) is not bool:
            raise ProvenanceIntegrityError(
                f"provenance invocation {position} evaluate flag is invalid"
            )
        _validate_context(
            invocation.get("context"),
            run_dir,
            legacy_schema=legacy_schema if isinstance(legacy_schema, int) else None,
        )
        artifacts = _artifact_manifest(
            invocation.get("artifacts"),
            legacy=False,
        )
        if status == "running" and artifacts is not None:
            raise ProvenanceIntegrityError(
                f"running provenance invocation {position} claims artifacts"
            )
        if (
            status not in {"running", "interrupted_before_next_invocation"}
            and artifacts is None
        ):
            raise ProvenanceIntegrityError(
                f"finalized provenance invocation {position} has no artifact manifest"
            )

    current = invocations[-1]
    if value.get("final_status") != current.get("status"):
        raise ProvenanceIntegrityError(
            "provenance final status does not match the current invocation"
        )
    if current.get("context") != top_context:
        raise ProvenanceIntegrityError(
            "provenance top-level identity does not match the current invocation"
        )
    top_artifacts = _artifact_manifest(value.get("artifacts"), legacy=False)
    current_artifacts = _artifact_manifest(
        current.get("artifacts"),
        legacy=False,
    )
    if top_artifacts != current_artifacts:
        raise ProvenanceIntegrityError(
            "provenance top-level artifacts do not match the current invocation"
        )
    if current.get("status") == "completed" and (
        top_artifacts is None or top_artifacts.get("complete") is not True
    ):
        raise ProvenanceIntegrityError(
            "completed provenance has no complete artifact manifest"
        )


def _load_existing(
    run_dir: Path,
    *,
    required: bool = False,
) -> dict[str, Any] | None:
    """Load structurally valid provenance without laundering corruption."""
    path = run_dir / PROVENANCE_FILENAME
    if path.is_symlink():
        raise ProvenanceIntegrityError(
            f"provenance path is an unsupported symlink: {path}"
        )
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        if required:
            raise ProvenanceIntegrityError(
                f"required provenance file is missing: {path}"
            ) from exc
        return None
    except OSError as exc:
        raise ProvenanceIntegrityError(
            f"cannot read existing provenance {path}: {exc}"
        ) from exc
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ProvenanceIntegrityError(
            f"existing provenance is malformed and was preserved: {path}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise ProvenanceIntegrityError(
            f"existing provenance is not a JSON object: {path}"
        )
    try:
        migrated = _migrate_provenance(value)
        _validate_provenance(migrated, run_dir)
    except ProvenanceIntegrityError as exc:
        raise ProvenanceIntegrityError(
            f"existing provenance is semantically invalid and was preserved: "
            f"{path}: {exc}"
        ) from exc
    return migrated


def validate_resumable_provenance(run_dir: Path) -> None:
    """Fail before resume mutates a run with missing or corrupt provenance."""
    resolved_run_dir = run_dir.resolve()
    existing = _load_existing(resolved_run_dir, required=True)
    if existing is None:  # pragma: no cover - required=True fails first.
        raise ProvenanceIntegrityError("required provenance unexpectedly missing")
    artifacts = _artifact_manifest(existing.get("artifacts"), legacy=False)
    exclusions = (
        tuple(artifacts["excluded"])
        if artifacts is not None
        else ARTIFACT_EXCLUSIONS
    )
    # Even an interrupted or failure-finalized invocation without a complete
    # checksum manifest must not be allowed to smuggle symlinks, sockets, or
    # other unsupported entries into paths that resume will read or replace.
    # The scan also pins every directory and file while inspecting it.
    actual = artifact_checksums(
        resolved_run_dir,
        exclusions=exclusions,
    )
    # Running and interrupted invocations legitimately have no complete
    # manifest. Every finalized complete manifest, regardless of run outcome,
    # must bind the exact current artifact set before resume may mutate it.
    if artifacts is None or artifacts.get("complete") is not True:
        return
    expected = artifacts["files"]
    if actual != expected:
        missing = sorted(set(expected) - set(actual))
        unexpected = sorted(set(actual) - set(expected))
        changed = sorted(
            path
            for path in set(expected) & set(actual)
            if expected[path] != actual[path]
        )
        raise ProvenanceIntegrityError(
            "run artifacts changed after provenance finalization; refusing "
            f"resume (missing={missing}, unexpected={unexpected}, "
            f"changed={changed})"
        )


def start_run_provenance(
    *,
    repository_root: Path,
    run_dir: Path,
    profile: str | None,
    model_provider: str,
    model_name: str,
    agent_type: str,
    agent_version: str | None,
    thinking: str | None,
    seed: int | None,
    problem_names: list[str],
    catalog_version: str,
    catalog_commit: str,
    executed_problem_names: list[str] | None = None,
    num_workers: int,
    evaluate: bool,
    environment_name: str,
    source_image_name: str,
    base_image_name: str,
    agent_image_name: str,
    invocation: list[str] | None = None,
    preflight: dict[str, Any] | None = None,
    require_existing: bool = False,
    identity_run_dir: Path | None = None,
) -> None:
    """Initialize or append to a run's provenance record.

    ``identity_run_dir`` supports atomic publication of a new named run: the
    first provenance document is written into a same-parent staging directory
    while already naming the final run directory as its durable identity.
    Existing runs may never change that identity.
    """
    root = repository_root.resolve()
    run_dir = run_dir.resolve()
    identity_dir = (
        identity_run_dir.resolve()
        if identity_run_dir is not None
        else run_dir
    )
    provenance_path = run_dir / PROVENANCE_FILENAME
    provenance_exists = provenance_path.exists() or provenance_path.is_symlink()
    must_load_existing = require_existing or provenance_exists
    if must_load_existing:
        # This is intentionally performed here as well as at the CLI boundary:
        # callers must not bypass a completed artifact manifest merely by
        # spelling the operation as a fresh or overwrite run.
        validate_resumable_provenance(run_dir)
    existing = _load_existing(run_dir, required=must_load_existing)
    if existing is not None and identity_dir != run_dir:
        raise ProvenanceIntegrityError(
            "an existing run cannot be republished under a new identity"
        )
    invocations = existing.get("invocations", []) if existing else []
    if not isinstance(invocations, list):
        invocations = []

    repository = _git_repository_metadata(root)
    inputs = _input_metadata(root, identity_dir)
    run = {
        "directory": str(identity_dir),
        "model": {"provider": model_provider, "name": model_name},
        "agent": {"type": agent_type, "version": agent_version},
        "thinking": thinking,
        "seed": seed,
        "problems": list(problem_names),
        "catalog": {
            "version": catalog_version,
            "commit": catalog_commit,
        },
        "environment": environment_name,
    }
    if existing is not None:
        previous_run = existing.get("run")
        legacy_schema = existing.get("migrated_from_schema_version")
        identity_keys = {
            "directory",
            "model",
            "agent",
            "thinking",
            "seed",
            "problems",
            "environment",
        }
        if legacy_schema != 2:
            identity_keys.add("catalog")
        identity_matches = isinstance(previous_run, dict) and all(
            previous_run.get(key) == run.get(key) for key in identity_keys
        )
        if not identity_matches or existing.get("profile") != profile:
            raise ProvenanceIntegrityError(
                "resume identity does not match the existing provenance"
            )
    host = _host_metadata()
    docker = {
        "source_image": _image_metadata(source_image_name),
        "base_image": _image_metadata(base_image_name),
        "agent_image": _image_metadata(agent_image_name),
        "agent_image_tool_versions": _container_tool_versions(
            agent_image_name, agent_type
        ),
    }
    if existing and invocations:
        previous = invocations[-1]
        if isinstance(previous, dict):
            if (
                previous.get("status") == "running"
                or previous.get("finished_at") is None
            ):
                previous["finished_at"] = _utc_now()
                previous["status"] = "interrupted_before_next_invocation"
                previous["error_type"] = "UncleanShutdown"
            if "context" not in previous:
                previous["context"] = {
                    key: existing.get(key)
                    for key in (
                        "profile",
                        "suite",
                        "repository",
                        "inputs",
                        "run",
                        "host",
                        "docker",
                        "preflight",
                    )
                }
    suite = _load_suite_manifest(root)
    context = {
        "profile": profile,
        "suite": suite,
        "repository": repository,
        "inputs": inputs,
        "run": run,
        "host": host,
        "docker": docker,
        "preflight": preflight,
    }
    invocations.append(
        {
            "started_at": _utc_now(),
            "finished_at": None,
            "status": "running",
            "error_type": None,
            "arguments": sanitize_invocation(invocation or sys.argv),
            "num_workers": num_workers,
            "evaluate": evaluate,
            "problems_executed": list(
                executed_problem_names
                if executed_problem_names is not None
                else problem_names
            ),
            "context": context,
            "artifacts": None,
        }
    )
    provenance = {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        **context,
        "invocations": invocations,
        "final_status": "running",
        "artifacts": None,
    }
    _write_provenance(run_dir, provenance)


def refresh_run_provenance_context(
    *,
    repository_root: Path,
    run_dir: Path,
    source_image_name: str,
    base_image_name: str,
    agent_image_name: str,
    preflight: dict[str, Any] | None,
    executed_problem_names: list[str] | None = None,
) -> None:
    """Refresh evidence owned by the current in-progress invocation.

    A named run starts before preflight and image construction so every later
    mutation is attributable to an invocation. This refresh records the
    staged-input hashes and resolved image metadata without creating a second
    invocation or rewriting historical context.
    """
    root = repository_root.resolve()
    run_dir = run_dir.resolve()
    provenance = _load_existing(run_dir, required=True)
    if provenance is None:  # pragma: no cover - required=True fails first.
        raise ProvenanceIntegrityError("required provenance unexpectedly missing")
    invocations = provenance.get("invocations")
    if not isinstance(invocations, list) or not invocations:
        raise ProvenanceIntegrityError("provenance has no current invocation")
    current = invocations[-1]
    if not isinstance(current, dict) or current.get("status") != "running":
        raise ProvenanceIntegrityError(
            "only a running provenance invocation can be refreshed"
        )
    current_context = current.get("context")
    if not isinstance(current_context, dict):
        raise ProvenanceIntegrityError(
            "current provenance invocation context is invalid"
        )
    run = current_context.get("run")
    if not isinstance(run, dict):
        raise ProvenanceIntegrityError("current provenance run identity is invalid")
    agent = run.get("agent")
    if not isinstance(agent, dict) or not isinstance(agent.get("type"), str):
        raise ProvenanceIntegrityError("current provenance agent identity is invalid")

    context = {
        **current_context,
        "suite": _load_suite_manifest(root),
        "repository": _git_repository_metadata(root),
        "inputs": _input_metadata(root, run_dir),
        "host": _host_metadata(),
        "docker": {
            "source_image": _image_metadata(source_image_name),
            "base_image": _image_metadata(base_image_name),
            "agent_image": _image_metadata(agent_image_name),
            "agent_image_tool_versions": _container_tool_versions(
                agent_image_name,
                agent["type"],
            ),
        },
        "preflight": preflight,
    }
    current["context"] = context
    if executed_problem_names is not None:
        current["problems_executed"] = list(executed_problem_names)
    for key, value in context.items():
        provenance[key] = value
    _write_provenance(run_dir, provenance)


def _is_stable_artifact(
    relative_path: Path,
    *,
    exclusions: tuple[str, ...],
) -> bool:
    if exclusions == LEGACY_ARTIFACT_EXCLUSIONS:
        return (
            relative_path.name != PROVENANCE_FILENAME
            and relative_path.suffix != ".log"
        )
    if exclusions != ARTIFACT_EXCLUSIONS:
        raise ProvenanceIntegrityError(
            "unsupported run artifact exclusion policy"
        )
    return relative_path not in {
        Path(PROVENANCE_FILENAME),
        Path("run_agent.log"),
    }


def _stat_identity(value: os.stat_result) -> tuple[int, ...]:
    """Return fields that must remain stable while bytes are consumed."""
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _hash_artifact_descriptor(
    descriptor: int,
    *,
    relative: Path,
    before: os.stat_result,
) -> str:
    if not stat.S_ISREG(before.st_mode):
        raise ProvenanceIntegrityError(
            "run artifact has an unsupported filesystem type: "
            f"{relative.as_posix()}"
        )
    digest = hashlib.sha256()
    with os.fdopen(os.dup(descriptor), "rb", closefd=True) as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    after = os.fstat(descriptor)
    if _stat_identity(after) != _stat_identity(before):
        raise ProvenanceIntegrityError(
            "run artifact changed while hashing: "
            f"{relative.as_posix()}"
        )
    return digest.hexdigest()


def _artifact_checksums_from_directory(
    directory_fd: int,
    *,
    prefix: Path,
    exclusions: tuple[str, ...],
) -> dict[str, str]:
    directory_before = os.fstat(directory_fd)
    entries = list(os.scandir(directory_fd))
    entries.sort(key=lambda entry: os.fsencode(entry.name))
    checksums: dict[str, str] = {}
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    directory_flags |= getattr(os, "O_NOFOLLOW", 0)
    file_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)

    for entry in entries:
        relative = prefix / entry.name
        try:
            before = os.stat(
                entry.name,
                dir_fd=directory_fd,
                follow_symlinks=False,
            )
        except FileNotFoundError as exc:
            raise ProvenanceIntegrityError(
                "run artifact disappeared while hashing: "
                f"{relative.as_posix()}"
            ) from exc
        if stat.S_ISLNK(before.st_mode):
            raise ProvenanceIntegrityError(
                "run artifact is an unsupported symlink: "
                f"{relative.as_posix()}"
            )
        if stat.S_ISDIR(before.st_mode):
            try:
                child_fd = os.open(
                    entry.name,
                    directory_flags,
                    dir_fd=directory_fd,
                )
            except OSError as exc:
                raise ProvenanceIntegrityError(
                    "run artifact directory changed while hashing: "
                    f"{relative.as_posix()}"
                ) from exc
            try:
                child_before = os.fstat(child_fd)
                if _stat_identity(child_before) != _stat_identity(before):
                    raise ProvenanceIntegrityError(
                        "run artifact directory changed while hashing: "
                        f"{relative.as_posix()}"
                    )
                checksums.update(
                    _artifact_checksums_from_directory(
                        child_fd,
                        prefix=relative,
                        exclusions=exclusions,
                    )
                )
                child_after = os.fstat(child_fd)
                path_after = os.stat(
                    entry.name,
                    dir_fd=directory_fd,
                    follow_symlinks=False,
                )
                if (
                    _stat_identity(child_after) != _stat_identity(child_before)
                    or _stat_identity(path_after) != _stat_identity(child_after)
                ):
                    raise ProvenanceIntegrityError(
                        "run artifact directory changed while hashing: "
                        f"{relative.as_posix()}"
                    )
            finally:
                os.close(child_fd)
            continue
        if not stat.S_ISREG(before.st_mode):
            raise ProvenanceIntegrityError(
                "run artifact has an unsupported filesystem type: "
                f"{relative.as_posix()}"
            )
        try:
            file_fd = os.open(
                entry.name,
                file_flags,
                dir_fd=directory_fd,
            )
        except OSError as exc:
            raise ProvenanceIntegrityError(
                "run artifact changed while hashing: "
                f"{relative.as_posix()}"
            ) from exc
        try:
            opened = os.fstat(file_fd)
            if _stat_identity(opened) != _stat_identity(before):
                raise ProvenanceIntegrityError(
                    "run artifact changed while hashing: "
                    f"{relative.as_posix()}"
                )
            digest = _hash_artifact_descriptor(
                file_fd,
                relative=relative,
                before=opened,
            )
            path_after = os.stat(
                entry.name,
                dir_fd=directory_fd,
                follow_symlinks=False,
            )
            if _stat_identity(path_after) != _stat_identity(os.fstat(file_fd)):
                raise ProvenanceIntegrityError(
                    "run artifact changed while hashing: "
                    f"{relative.as_posix()}"
                )
        finally:
            os.close(file_fd)
        if _is_stable_artifact(relative, exclusions=exclusions):
            checksums[relative.as_posix()] = digest

    directory_after = os.fstat(directory_fd)
    if _stat_identity(directory_after) != _stat_identity(directory_before):
        raise ProvenanceIntegrityError(
            "run artifact directory changed while hashing: "
            f"{prefix.as_posix() or '.'}"
        )
    return checksums


def artifact_checksums(
    run_dir: Path,
    *,
    exclusions: tuple[str, ...] = ARTIFACT_EXCLUSIONS,
) -> dict[str, str]:
    """Hash stable artifacts through pinned, no-follow descriptors."""
    absolute = run_dir.absolute()
    try:
        root_fd = _open_directory_without_symlinks(absolute)
    except OSError as exc:
        raise ProvenanceIntegrityError(
            f"cannot safely open run artifact directory: {absolute}: {exc}"
        ) from exc
    try:
        root_before = os.fstat(root_fd)
        checksums = _artifact_checksums_from_directory(
            root_fd,
            prefix=Path(),
            exclusions=exclusions,
        )
        root_after = absolute.stat(follow_symlinks=False)
        if _stat_identity(root_after) != _stat_identity(root_before):
            raise ProvenanceIntegrityError(
                f"run artifact root changed while hashing: {absolute}"
            )
        return dict(sorted(checksums.items()))
    except ProvenanceIntegrityError:
        raise
    except OSError as exc:
        raise ProvenanceIntegrityError(
            f"run artifacts changed while hashing: {exc}"
        ) from exc
    finally:
        os.close(root_fd)


def finalize_run_provenance(
    run_dir: Path,
    *,
    status: str,
    error_type: str | None = None,
    details: dict[str, Any] | None = None,
    checksum_artifacts: bool = True,
) -> None:
    """Finalize the current invocation and optionally checksum artifacts.

    Artifact hashing is skipped while preserving an already-active primary
    failure. This keeps failure finalization bounded by the provenance file
    itself and prevents a disappearing or unreadable artifact from masking the
    actual benchmark error.
    """
    run_dir = run_dir.resolve()
    provenance = _load_existing(run_dir, required=True)
    if provenance is None:  # pragma: no cover - required=True is fail-closed.
        raise ProvenanceIntegrityError(
            f"required provenance file is missing: {run_dir / PROVENANCE_FILENAME}"
        )
    invocations = provenance.get("invocations")
    if isinstance(invocations, list) and invocations:
        current = invocations[-1]
        if isinstance(current, dict):
            current["finished_at"] = _utc_now()
            current["status"] = status
            current["error_type"] = error_type
            current["details"] = details
    provenance["final_status"] = status
    provenance["final_details"] = details
    artifacts = (
        {
            "schema_version": ARTIFACT_MANIFEST_SCHEMA_VERSION,
            "algorithm": "sha256",
            "excluded": list(ARTIFACT_EXCLUSIONS),
            "files": artifact_checksums(run_dir),
            "complete": True,
        }
        if checksum_artifacts
        else {
            "schema_version": ARTIFACT_MANIFEST_SCHEMA_VERSION,
            "algorithm": "sha256",
            "excluded": list(ARTIFACT_EXCLUSIONS),
            "files": {},
            "complete": False,
            "reason": "skipped_during_failure_finalization",
        }
    )
    provenance["artifacts"] = artifacts
    if isinstance(invocations, list) and invocations:
        current = invocations[-1]
        if isinstance(current, dict):
            current["artifacts"] = artifacts
    _write_provenance(run_dir, provenance)
