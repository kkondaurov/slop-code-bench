from __future__ import annotations

from pathlib import Path

import pytest

from slop_code.common.temp import TEMP_ROOT_ENV
from slop_code.common.temp import temporary_directory


def test_temporary_directory_uses_default_when_unset(monkeypatch) -> None:
    monkeypatch.delenv(TEMP_ROOT_ENV, raising=False)

    with temporary_directory() as temp_dir:
        assert Path(temp_dir).is_dir()


def test_temporary_directory_uses_configured_root(
    monkeypatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "docker-shared"
    monkeypatch.setenv(TEMP_ROOT_ENV, str(root))

    temp_dir = temporary_directory()
    path = Path(temp_dir.name)
    try:
        assert path.parent == root
        assert path.is_dir()
    finally:
        temp_dir.cleanup()

    assert not path.exists()


def test_temporary_directory_creates_configured_root(
    monkeypatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "missing" / "docker-shared"
    monkeypatch.setenv(TEMP_ROOT_ENV, str(root))

    with temporary_directory() as temp_dir:
        assert Path(temp_dir).parent == root


def test_temporary_directory_rejects_empty_root(monkeypatch) -> None:
    monkeypatch.setenv(TEMP_ROOT_ENV, "")

    with pytest.raises(ValueError, match=TEMP_ROOT_ENV):
        temporary_directory()


def test_temporary_directory_rejects_relative_root(monkeypatch) -> None:
    monkeypatch.setenv(TEMP_ROOT_ENV, "relative/docker-shared")

    with pytest.raises(ValueError, match=TEMP_ROOT_ENV):
        temporary_directory()


def test_temporary_directory_rejects_file_root(
    monkeypatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "not-a-directory"
    root.write_text("file")
    monkeypatch.setenv(TEMP_ROOT_ENV, str(root))

    with pytest.raises(NotADirectoryError, match=TEMP_ROOT_ENV):
        temporary_directory()
