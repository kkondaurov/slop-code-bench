from __future__ import annotations

import os
from pathlib import Path

import pytest

from slop_code.common.temp import NAMED_PROFILE_TEMP_DIR
from slop_code.common.temp import TEMP_ROOT_ENV
from slop_code.common.temp import configure_named_profile_temp_root
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


def test_named_profile_configures_repository_visible_root(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv(TEMP_ROOT_ENV, raising=False)

    root = configure_named_profile_temp_root(tmp_path)

    assert root == (tmp_path / NAMED_PROFILE_TEMP_DIR).resolve()
    assert root.is_dir()
    assert Path(os.environ[TEMP_ROOT_ENV]) == root


def test_named_profile_tolerates_concurrent_first_component_creation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv(TEMP_ROOT_ENV, raising=False)
    repository = tmp_path / "repository"
    repository.mkdir()
    raced_component = repository / NAMED_PROFILE_TEMP_DIR.parts[0]
    original_mkdir = Path.mkdir
    injected_race = False

    def mkdir_after_other_run(
        path: Path,
        *args: object,
        **kwargs: object,
    ) -> None:
        nonlocal injected_race
        if path == raced_component and not injected_race:
            injected_race = True
            original_mkdir(path)
        original_mkdir(path, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "mkdir", mkdir_after_other_run)

    root = configure_named_profile_temp_root(repository)

    assert injected_race
    assert root == (repository / NAMED_PROFILE_TEMP_DIR).resolve()
    assert root.is_dir()


def test_named_profile_preserves_explicit_root(
    monkeypatch,
    tmp_path: Path,
) -> None:
    explicit_root = tmp_path / "explicit-docker-root"
    monkeypatch.setenv(TEMP_ROOT_ENV, str(explicit_root))

    root = configure_named_profile_temp_root(tmp_path / "repository")

    assert root == explicit_root.resolve()
    assert Path(os.environ[TEMP_ROOT_ENV]) == root


def test_named_profile_validates_explicit_root(
    monkeypatch,
    tmp_path: Path,
) -> None:
    del tmp_path
    monkeypatch.setenv(TEMP_ROOT_ENV, "relative/docker-root")

    with pytest.raises(ValueError, match=TEMP_ROOT_ENV):
        configure_named_profile_temp_root(Path("/repository"))


def test_named_profile_rejects_symlink_in_automatic_root(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv(TEMP_ROOT_ENV, raising=False)
    repository = tmp_path / "repository"
    repository.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (repository / "tmp").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="cannot traverse a symlink"):
        configure_named_profile_temp_root(repository)

    assert TEMP_ROOT_ENV not in os.environ


def test_named_profile_allows_explicit_symlinked_root(
    monkeypatch,
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    real_root = tmp_path / "docker-shared"
    real_root.mkdir()
    linked_root = tmp_path / "linked-root"
    linked_root.symlink_to(real_root, target_is_directory=True)
    monkeypatch.setenv(TEMP_ROOT_ENV, str(linked_root))

    configured = configure_named_profile_temp_root(repository)

    assert configured == real_root.resolve()
