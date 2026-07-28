from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
from unittest.mock import MagicMock
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "scripts" / "verify_paper_v1.py"
SPEC = importlib.util.spec_from_file_location(
    "verify_paper_v1_module", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_repository_matches_paper_v1_package() -> None:
    assert MODULE.verify_repository(ROOT, check_tools=False) == []


def test_manifest_source_drift_is_reported() -> None:
    manifest = copy.deepcopy(MODULE.load_manifest(ROOT))
    manifest["source"]["commit"] = "0" * 40

    errors = MODULE.verify_manifest(manifest)

    assert any("source.commit" in error for error in errors)


def test_manifest_ast_grep_version_drift_is_reported() -> None:
    manifest = copy.deepcopy(MODULE.load_manifest(ROOT))
    manifest["tooling"]["ast_grep"]["version"] = "999.0.0"

    errors = MODULE.verify_manifest(manifest)

    assert any("tooling.ast_grep.version" in error for error in errors)


def test_missing_ast_grep_tool_is_reported(tmp_path: Path) -> None:
    manifest = {
        "tooling": {
            "ast_grep": {
                "executable": "sg",
                "version": MODULE.EXPECTED_AST_GREP_VERSION,
            }
        }
    }

    with (
        patch.dict("os.environ", {}, clear=True),
        patch.object(MODULE.shutil, "which", return_value=None),
    ):
        errors = MODULE.verify_ast_grep_tool(tmp_path, manifest)

    assert any("uv sync --frozen" in error for error in errors)


def test_prompt_byte_drift_is_reported(tmp_path: Path) -> None:
    prompt_path = tmp_path / "configs" / "prompts" / "just-solve.jinja"
    prompt_path.parent.mkdir(parents=True)
    prompt_path.write_text("changed prompt\n", encoding="utf-8")
    manifest = {
        "prompt": {
            "path": "configs/prompts/just-solve.jinja",
        }
    }

    errors = MODULE.verify_prompt(tmp_path, manifest)

    assert any("just-solve SHA-256" in error for error in errors)


def _make_minimal_content_lock_root(tmp_path: Path) -> None:
    for relative in MODULE.CONTENT_LOCK_STATIC_PATHS:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"fixture:{relative}\n".encode())
    problem_file = tmp_path / "problems" / "example" / "config.yaml"
    problem_file.parent.mkdir(parents=True)
    problem_file.write_text("checkpoints: {}\n", encoding="utf-8")


def test_content_lock_detects_byte_drift(tmp_path: Path) -> None:
    _make_minimal_content_lock_root(tmp_path)
    MODULE.write_content_lock(tmp_path)
    changed = tmp_path / "problems" / "example" / "config.yaml"
    changed.write_text("checkpoints: changed\n", encoding="utf-8")

    errors = MODULE.verify_content_lock(tmp_path)

    assert any("byte mismatch" in error for error in errors)


def test_content_lock_detects_new_non_ephemeral_input(tmp_path: Path) -> None:
    _make_minimal_content_lock_root(tmp_path)
    MODULE.write_content_lock(tmp_path)
    extra = tmp_path / "problems" / "example" / "new_asset.json"
    extra.write_text("{}\n", encoding="utf-8")

    errors = MODULE.verify_content_lock(tmp_path)

    assert any("path set mismatch" in error for error in errors)


def test_content_lock_ignores_declared_ephemera(tmp_path: Path) -> None:
    _make_minimal_content_lock_root(tmp_path)
    MODULE.write_content_lock(tmp_path)
    cache = tmp_path / "problems" / "example" / "__pycache__" / "value.pyc"
    cache.parent.mkdir(parents=True)
    cache.write_bytes(b"cache")

    assert MODULE.verify_content_lock(tmp_path) == []


def test_lock_rejects_relative_cutoff_and_late_artifact(
    tmp_path: Path,
) -> None:
    (tmp_path / "pyproject.toml").write_text(
        """
[project]
dependencies = ["ast-grep-cli==0.42.0"]

[tool.uv]
exclude-newer = "2026-03-24T21:59:00Z"
exclude-newer-span = "1 day"
""".strip()
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "uv.lock").write_text(
        """
[options]
exclude-newer = "2026-03-24T21:59:00Z"
exclude-newer-span = "1 day"

[[package]]
name = "ast-grep-cli"
version = "0.42.0"
sdist = { url = "https://example.test/pkg", upload-time = "2026-03-25T00:00:00Z" }
""".strip()
        + "\n",
        encoding="utf-8",
    )

    errors = MODULE.verify_ast_grep_dependency(tmp_path)

    assert sum("exclude-newer-span" in error for error in errors) == 2
    assert any("uploaded after the paper cutoff" in error for error in errors)


def test_publication_ready_requires_clean_worktree(tmp_path: Path) -> None:
    manifest = {"source": {"commit": "a" * 40}}
    ancestor = MagicMock(returncode=0, stdout="", stderr="")
    dirty = MagicMock(returncode=0, stdout=" M file.py\n", stderr="")
    tagged = MagicMock(
        returncode=0,
        stdout=f"{MODULE.EXPECTED_PUBLICATION_TAG}\n",
        stderr="",
    )
    with patch.object(
        MODULE.subprocess,
        "run",
        side_effect=[ancestor, dirty, tagged],
    ):
        errors = MODULE.verify_git_provenance(
            tmp_path,
            manifest,
            publication_ready=True,
        )

    assert errors == [
        "publication-ready verification requires a clean worktree"
    ]


def test_publication_ready_requires_release_tag(tmp_path: Path) -> None:
    manifest = {"source": {"commit": "a" * 40}}
    success = MagicMock(returncode=0, stdout="", stderr="")
    with patch.object(
        MODULE.subprocess,
        "run",
        side_effect=[success, success, success],
    ):
        errors = MODULE.verify_git_provenance(
            tmp_path,
            manifest,
            publication_ready=True,
        )

    assert errors == [
        "publication-ready verification requires HEAD at tag "
        f"{MODULE.EXPECTED_PUBLICATION_TAG}"
    ]
