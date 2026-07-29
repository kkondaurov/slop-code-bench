from __future__ import annotations

from pathlib import Path

import pytest

from slop_code.common.atomic import UnsafeAtomicWriteError
from slop_code.common.atomic import atomic_write_text


def test_atomic_write_text_replaces_regular_file(tmp_path: Path) -> None:
    target = tmp_path / "evidence.json"
    target.write_text("old", encoding="utf-8")

    atomic_write_text(target, "new\n")

    assert target.read_text(encoding="utf-8") == "new\n"
    assert not list(tmp_path.glob(".evidence.json.*.tmp"))


def test_atomic_write_text_rejects_target_symlink(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.write_text("untouched", encoding="utf-8")
    target = tmp_path / "evidence.json"
    target.symlink_to(outside)

    with pytest.raises(UnsafeAtomicWriteError, match="symlink target"):
        atomic_write_text(target, "poison")

    assert outside.read_text(encoding="utf-8") == "untouched"


def test_atomic_write_text_rejects_parent_symlink(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    parent = tmp_path / "linked"
    parent.symlink_to(outside, target_is_directory=True)

    with pytest.raises(UnsafeAtomicWriteError, match="symlink parent"):
        atomic_write_text(parent / "evidence.json", "poison")

    assert not (outside / "evidence.json").exists()


def test_atomic_write_text_rejects_ancestor_symlink(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    nested = outside / "nested"
    nested.mkdir(parents=True)
    linked = tmp_path / "linked"
    linked.symlink_to(outside, target_is_directory=True)

    with pytest.raises(UnsafeAtomicWriteError, match="unsafe parent path"):
        atomic_write_text(linked / "nested" / "evidence.json", "poison")

    assert not (nested / "evidence.json").exists()


def test_atomic_write_text_cannot_follow_precreated_temp_symlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outside = tmp_path / "outside"
    outside.write_text("untouched", encoding="utf-8")
    monkeypatch.setattr("secrets.token_hex", lambda _: "fixed")
    trap = tmp_path / ".evidence.json.fixed.tmp"
    trap.symlink_to(outside)

    with pytest.raises(FileExistsError):
        atomic_write_text(tmp_path / "evidence.json", "poison")

    assert outside.read_text(encoding="utf-8") == "untouched"
