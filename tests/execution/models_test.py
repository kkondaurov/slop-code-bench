from __future__ import annotations

from slop_code.execution.models import SnapshotConfig


def test_snapshot_globs_have_deterministic_json_serialization() -> None:
    snapshot = SnapshotConfig(
        keep_globs={"z-last", "a-first"},
        ignore_globs={"*.pyc", ".venv/*", "venv/*"},
    )

    dumped = snapshot.model_dump(mode="json")

    assert dumped["keep_globs"] == ["a-first", "z-last"]
    assert dumped["ignore_globs"] == ["*.pyc", ".venv/*", "venv/*"]
    assert snapshot.keep_globs == {"a-first", "z-last"}
    assert snapshot.ignore_globs == {"*.pyc", ".venv/*", "venv/*"}
