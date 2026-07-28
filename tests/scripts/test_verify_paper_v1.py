from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
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
