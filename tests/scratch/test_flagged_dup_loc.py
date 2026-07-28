from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

MODULE_PATH = (
    Path(__file__).resolve().parents[2] / "scratch" / "flagged_dup_loc.py"
)
if not MODULE_PATH.exists():
    pytest.skip(
        "paper-v1 source snapshot omits scratch/flagged_dup_loc.py",
        allow_module_level=True,
    )
SPEC = importlib.util.spec_from_file_location(
    "flagged_dup_loc_module", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")


def test_summarize_checkpoint_dedupes_overlapping_flags_and_reports_duplicate_pct(
    tmp_path: Path,
) -> None:
    checkpoint_dir = tmp_path / "checkpoint_1"
    quality_dir = checkpoint_dir / "quality_analysis"
    quality_dir.mkdir(parents=True)

    (checkpoint_dir / "a.py").write_text(
        "\n".join(
            [
                "# comment",
                "def f():",
                "    x = 1",
                "    y = 2",
                "",
                "def g():",
                "    z = 3",
            ]
        )
        + "\n"
    )
    (checkpoint_dir / "b.py").write_text(
        "\n".join(
            [
                '"""module docstring"""',
                "def h():",
                "    return 1",
            ]
        )
        + "\n"
    )

    _write_jsonl(
        quality_dir / "files.jsonl",
        [
            {
                "file_path": "a.py",
                "loc": 5,
                "total_lines": 7,
                "clone_lines": 2,
                "clone_ratio": 0.4,
            },
            {
                "file_path": "b.py",
                "loc": 2,
                "total_lines": 3,
                "clone_lines": 1,
                "clone_ratio": 0.5,
            },
        ],
    )
    _write_jsonl(
        quality_dir / "ast_grep.jsonl",
        [
            {"file_path": "a.py", "line": 1, "end_line": 3},
            {"file_path": "a.py", "line": 2, "end_line": 2},
            {"file_path": "a.py", "line": 5, "end_line": 6},
            {"file_path": "b.py", "line": 0, "end_line": 2},
        ],
    )

    summary = MODULE.summarize_checkpoint(checkpoint_dir)

    assert summary["total_loc"] == 7
    assert summary["flagged_sloc_lines"] == 7
    assert summary["flagged_pct"] == pytest.approx(100.0)
    assert summary["duplicate_loc_lines"] == 3
    assert summary["duplicate_pct"] == pytest.approx((3 / 7) * 100, rel=1e-6)


def test_summarize_checkpoint_reads_files_from_snapshot_dir(
    tmp_path: Path,
) -> None:
    checkpoint_dir = tmp_path / "checkpoint_1"
    quality_dir = checkpoint_dir / "quality_analysis"
    snapshot_dir = checkpoint_dir / "snapshot"
    quality_dir.mkdir(parents=True)
    snapshot_dir.mkdir()

    (snapshot_dir / "main.py").write_text(
        "\n".join(
            [
                "def f():",
                "    return 1",
            ]
        )
        + "\n"
    )

    _write_jsonl(
        quality_dir / "files.jsonl",
        [
            {
                "file_path": "main.py",
                "loc": 2,
                "total_lines": 2,
                "clone_lines": 1,
                "clone_ratio": 0.5,
            }
        ],
    )
    _write_jsonl(
        quality_dir / "ast_grep.jsonl",
        [{"file_path": "main.py", "line": 0, "end_line": 1}],
    )

    summary = MODULE.summarize_checkpoint(checkpoint_dir)

    assert summary["total_loc"] == 2
    assert summary["flagged_sloc_lines"] == 2
