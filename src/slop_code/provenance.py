"""Create credential-safe, machine-readable provenance for benchmark runs."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from datetime import UTC
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
from urllib.parse import urlunsplit

import yaml

PROVENANCE_FILENAME = "provenance.json"
PROVENANCE_SCHEMA_VERSION = 2
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
    """Redact likely credential values from a command-line invocation."""
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
        if sanitized_url != argument:
            sanitized.append(sanitized_url or "<redacted>")
            continue

        sanitized.append(argument)
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


def _git_repository_metadata(root: Path) -> dict[str, Any]:
    head = _git_output(root, "rev-parse", "HEAD")
    branch = _git_output(root, "symbolic-ref", "--short", "-q", "HEAD")
    tags_raw = _git_output(root, "tag", "--points-at", "HEAD")
    dirty, diff_sha256 = _git_diff_sha256(root)
    source_commit = None
    manifest_path = root / "configs" / "paper-v1" / "manifest.yaml"
    try:
        manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        if isinstance(manifest, dict):
            source = manifest.get("source")
            if isinstance(source, dict):
                source_commit = source.get("commit")
    except (OSError, yaml.YAMLError):
        pass

    source_is_ancestor: bool | None = None
    if isinstance(source_commit, str):
        ancestor = _run_command(
            ["git", "merge-base", "--is-ancestor", source_commit, "HEAD"],
            cwd=root,
        )
        if ancestor["returncode"] in {0, 1}:
            source_is_ancestor = ancestor["returncode"] == 0

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
            "paper_source_sha": source_commit,
            "paper_source_is_ancestor": source_is_ancestor,
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
        values = json.loads(value)
        image = values[0]
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


def _container_tool_versions(image_name: str) -> dict[str, Any]:
    if not image_name:
        return {"value": None, "returncode": None, "error": "no_image"}
    return _run_command(
        [
            "docker",
            "run",
            "--rm",
            "--entrypoint",
            "sh",
            image_name,
            "-lc",
            "python --version; uv --version; git --version; rg --version",
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


def _input_metadata(root: Path, run_dir: Path) -> dict[str, Any]:
    lock_path = root / "configs" / "paper-v1" / "content-lock.json"
    content_tree_sha256 = None
    try:
        content_lock = json.loads(lock_path.read_text(encoding="utf-8"))
        if isinstance(content_lock, dict):
            content_tree_sha256 = content_lock.get("tree_sha256")
    except (OSError, json.JSONDecodeError):
        pass
    paths = {
        "paper_manifest": root / "configs" / "paper-v1" / "manifest.yaml",
        "paper_content_lock": lock_path,
        "uv_lock": root / "uv.lock",
        "resolved_config": run_dir / "config.yaml",
        "resolved_environment": run_dir / "environment.yaml",
    }
    return {
        key: {
            "path": str(path.relative_to(root))
            if path.is_relative_to(root)
            else str(path),
            "sha256": _optional_file_sha256(path),
        }
        for key, path in paths.items()
    } | {"paper_content_tree_sha256": content_tree_sha256}


def _write_provenance(run_dir: Path, value: dict[str, Any]) -> None:
    target = run_dir / PROVENANCE_FILENAME
    temporary = run_dir / f".{PROVENANCE_FILENAME}.tmp"
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)


def _load_existing(run_dir: Path) -> dict[str, Any] | None:
    path = run_dir / PROVENANCE_FILENAME
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


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
    executed_problem_names: list[str] | None = None,
    num_workers: int,
    evaluate: bool,
    environment_name: str,
    base_image_name: str,
    agent_image_name: str,
    invocation: list[str] | None = None,
) -> None:
    """Initialize or append to a run's provenance record."""
    root = repository_root.resolve()
    run_dir = run_dir.resolve()
    existing = _load_existing(run_dir)
    invocations = existing.get("invocations", []) if existing else []
    if not isinstance(invocations, list):
        invocations = []

    manifest_path = root / "configs" / "paper-v1" / "manifest.yaml"
    paper: dict[str, Any] | None = None
    if profile == "paper-v1":
        try:
            manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
            if isinstance(manifest, dict):
                paper_value = manifest.get("paper")
                if isinstance(paper_value, dict):
                    paper = {
                        "arxiv_id": paper_value.get("arxiv_id"),
                        "version": paper_value.get("version"),
                        "url": paper_value.get("url"),
                    }
        except (OSError, yaml.YAMLError):
            pass

    repository = _git_repository_metadata(root)
    inputs = _input_metadata(root, run_dir)
    run = {
        "directory": str(run_dir),
        "model": {
            "provider": model_provider,
            "name": model_name,
        },
        "agent": {
            "type": agent_type,
            "version": agent_version,
        },
        "thinking": thinking,
        "seed": seed,
        "problems": list(problem_names),
        "environment": environment_name,
    }
    host = _host_metadata()
    docker = {
        "base_image": _image_metadata(base_image_name),
        "agent_image": _image_metadata(agent_image_name),
        "agent_image_tool_versions": _container_tool_versions(
            agent_image_name
        ),
    }
    if existing and invocations:
        previous = invocations[-1]
        if isinstance(previous, dict) and "context" not in previous:
            previous["context"] = {
                key: existing.get(key)
                for key in (
                    "profile",
                    "paper",
                    "repository",
                    "inputs",
                    "run",
                    "host",
                    "docker",
                )
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
            "context": {
                "profile": profile,
                "paper": paper,
                "repository": repository,
                "inputs": inputs,
                "run": run,
                "host": host,
                "docker": docker,
            },
            "artifacts": None,
        }
    )

    provenance = {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "profile": profile,
        "paper": paper,
        "repository": repository,
        "inputs": inputs,
        "run": run,
        "host": host,
        "docker": docker,
        "invocations": invocations,
        "final_status": "running",
        "artifacts": None,
    }
    _write_provenance(run_dir, provenance)


def _is_stable_artifact(relative_path: Path) -> bool:
    if relative_path.name in {
        PROVENANCE_FILENAME,
        f".{PROVENANCE_FILENAME}.tmp",
    }:
        return False
    return relative_path.suffix != ".log"


def artifact_checksums(run_dir: Path) -> dict[str, str]:
    """Hash stable run artifacts, excluding logs and provenance itself."""
    checksums: dict[str, str] = {}
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(run_dir)
        if _is_stable_artifact(relative):
            checksums[relative.as_posix()] = sha256_file(path)
    return checksums


def finalize_run_provenance(
    run_dir: Path,
    *,
    status: str,
    error_type: str | None = None,
) -> None:
    """Finalize the current invocation and checksum stable run artifacts."""
    run_dir = run_dir.resolve()
    provenance = _load_existing(run_dir)
    if provenance is None:
        return
    invocations = provenance.get("invocations")
    if isinstance(invocations, list) and invocations:
        current = invocations[-1]
        if isinstance(current, dict):
            current["finished_at"] = _utc_now()
            current["status"] = status
            current["error_type"] = error_type
    provenance["final_status"] = status
    artifacts = {
        "algorithm": "sha256",
        "excluded": [PROVENANCE_FILENAME, "*.log"],
        "files": artifact_checksums(run_dir),
    }
    provenance["artifacts"] = artifacts
    if isinstance(invocations, list) and invocations:
        current = invocations[-1]
        if isinstance(current, dict):
            current["artifacts"] = artifacts
    _write_provenance(run_dir, provenance)
