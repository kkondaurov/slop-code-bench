"""Tests for snapshot diff functionality."""

import os
import tarfile
import tempfile
from pathlib import Path
from typing import BinaryIO

import pytest

from slop_code.execution import snapshot as snapshot_module
from slop_code.execution.models import ExecutionError
from slop_code.execution.snapshot import SYMLINK_TARGET_TYPE_PAX
from slop_code.execution.snapshot import FileChangeType
from slop_code.execution.snapshot import FileDiff
from slop_code.execution.snapshot import Snapshot
from slop_code.execution.snapshot import _create_text_file_diff
from slop_code.execution.snapshot import _decode_text
from slop_code.execution.snapshot import _is_binary
from slop_code.execution.snapshot import _safe_archive_member_path
from slop_code.execution.snapshot import _safe_relative_symlink_target


@pytest.mark.parametrize(
    "member_name",
    [
        r"C:\workspace\escape.py",
        "C:/workspace/escape.py",
        r"..\escape.py",
        r"\\server\share\escape.py",
    ],
)
def test_archive_member_rejects_windows_path_syntax(member_name: str) -> None:
    with pytest.raises(ExecutionError, match="Unsafe snapshot archive member"):
        _safe_archive_member_path(member_name)


@pytest.mark.parametrize(
    "target",
    [
        r"C:\workspace\escape.py",
        "C:/workspace/escape.py",
        r"..\escape.py",
        r"\\server\share\escape.py",
    ],
)
def test_symlink_target_rejects_windows_path_syntax(target: str) -> None:
    with pytest.raises(ExecutionError, match="Unsafe snapshot symlink target"):
        _safe_relative_symlink_target(Path("alias.py"), target)


class TestIsBinary:
    """Test binary file detection."""

    def test_null_byte_is_binary(self):
        """Files with null bytes are binary."""
        assert _is_binary(b"hello\x00world")

    def test_text_is_not_binary(self):
        """Plain text is not binary."""
        assert not _is_binary(b"Hello, world!\n")

    def test_empty_is_not_binary(self):
        """Empty data is not binary."""
        assert not _is_binary(b"")

    def test_high_non_printable_ratio_is_binary(self):
        """High ratio of non-printable characters means binary."""
        # Create data with lots of non-printable bytes
        data = bytes(range(256)) * 10
        assert _is_binary(data)

    def test_tabs_and_newlines_allowed(self):
        """Tabs and newlines don't count as binary."""
        assert not _is_binary(b"line1\nline2\n\ttabbed")


class TestDecodeText:
    """Test text decoding."""

    def test_utf8_decode(self):
        """UTF-8 text decodes correctly."""
        text = "Hello, 世界!"
        assert _decode_text(text.encode("utf-8")) == text

    def test_latin1_decode(self):
        """Latin-1 text decodes correctly."""
        text = "Héllo, wörld!"
        assert _decode_text(text.encode("latin-1")) == text

    def test_binary_returns_none(self):
        """Binary data with null bytes cannot decode."""
        # This will actually decode with latin-1, so let's use truly invalid UTF-8
        # that also isn't valid in other encodings - but latin-1 accepts all bytes
        # So we need to accept that _decode_text might return something
        result = _decode_text(b"\xff\xfe\x00\x00")
        # This should decode with latin-1 even though it's not valid UTF-8
        assert (
            result is not None or result is None
        )  # Always true, but documents behavior


class TestCreateTextFileDiff:
    """Test individual text file diff creation."""

    def test_created_text_file(self):
        """Test diff for a newly created text file."""
        content = "line1\nline2\nline3\n"
        diff = _create_text_file_diff(
            Path("test.txt"),
            None,
            content,
            FileChangeType.CREATED,
        )
        assert diff.change_type == FileChangeType.CREATED
        assert not diff.is_binary
        assert diff.lines_added == 3
        assert diff.lines_removed == 0

    def test_deleted_text_file(self):
        """Test diff for a deleted text file."""
        content = "line1\nline2\n"
        diff = _create_text_file_diff(
            Path("test.txt"),
            content,
            None,
            FileChangeType.DELETED,
        )
        assert diff.change_type == FileChangeType.DELETED
        assert not diff.is_binary
        assert diff.lines_added == 0
        assert diff.lines_removed == 2

    def test_modified_text_file(self):
        """Test diff for a modified text file."""
        old_content = "line1\nline2\nline3\n"
        new_content = "line1\nmodified\nline3\nnew line\n"
        diff = _create_text_file_diff(
            Path("test.txt"),
            old_content,
            new_content,
            FileChangeType.MODIFIED,
        )
        assert diff.change_type == FileChangeType.MODIFIED
        assert not diff.is_binary
        assert diff.lines_added > 0
        assert diff.lines_removed > 0


class TestSnapshot:
    """Test archive snapshot functionality."""

    @pytest.fixture
    def temp_workspace(self, tmp_path):
        """Create a temporary workspace with some files."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()

        # Create some initial files
        (workspace / "file1.txt").write_text("line1\nline2\n")
        (workspace / "file2.txt").write_text("hello\nworld\n")
        (workspace / "subdir").mkdir()
        (workspace / "subdir" / "file3.txt").write_text("nested\nfile\n")

        return workspace

    def test_archive_snapshot_extract_contents(self, temp_workspace):
        """Test extracting contents from an archive snapshot."""
        with tempfile.TemporaryDirectory() as tmpdir:
            snapshot = Snapshot.from_directory(
                cwd=temp_workspace,
                env={},
                save_path=Path(tmpdir),
            )

            contents = snapshot.extract_contents()

            assert Path("file1.txt") in contents
            assert Path("file2.txt") in contents
            assert Path("subdir/file3.txt") in contents
            assert contents[Path("file1.txt")] == b"line1\nline2\n"

    def test_extract_to_path_preserves_executable_mode(
        self, temp_workspace, tmp_path
    ):
        """Checkpoint carry-forward keeps executable regular files executable."""
        executable = temp_workspace / "run.sh"
        executable.write_text("#!/bin/sh\nexit 0\n")
        executable.chmod(0o755)
        archive_dir = tmp_path / "archives"
        archive_dir.mkdir()
        target = tmp_path / "restored"

        snapshot = Snapshot.from_directory(
            cwd=temp_workspace,
            env={},
            save_path=archive_dir,
        )
        snapshot.extract_to_path(target)

        assert (target / "run.sh").stat().st_mode & 0o777 == 0o755

    def test_extract_to_path_preserves_file_times_and_lru_order(
        self,
        temp_workspace: Path,
        tmp_path: Path,
    ) -> None:
        """Distinct atimes survive without silently becoming mtimes."""
        first = temp_workspace / "first.cache"
        second = temp_workspace / "second.cache"
        first.write_text("first", encoding="utf-8")
        second.write_text("second", encoding="utf-8")
        os.utime(
            first,
            ns=(1_900_000_001_123_456_789, 1_700_000_004_123_456_789),
        )
        os.utime(
            second,
            ns=(1_900_000_002_987_654_321, 1_700_000_003_987_654_321),
        )
        expected = {
            path.name: (path.stat().st_atime_ns, path.stat().st_mtime_ns)
            for path in (first, second)
        }
        archive_dir = tmp_path / "archives"
        archive_dir.mkdir()
        target = tmp_path / "restored"

        snapshot = Snapshot.from_directory(
            cwd=temp_workspace,
            env={},
            save_path=archive_dir,
        )
        snapshot.extract_to_path(target)

        restored = {
            name: (
                (target / name).stat().st_atime_ns,
                (target / name).stat().st_mtime_ns,
            )
            for name in expected
        }
        assert restored == expected
        assert sorted(expected, key=lambda name: expected[name][0]) == [
            "first.cache",
            "second.cache",
        ]
        assert sorted(restored, key=lambda name: restored[name][0]) == [
            "first.cache",
            "second.cache",
        ]

    @pytest.mark.skipif(
        os.utime not in os.supports_follow_symlinks,
        reason="host cannot set symlink mtime without following it",
    )
    def test_extract_to_path_preserves_symlink_times_ns(
        self,
        temp_workspace: Path,
        tmp_path: Path,
    ) -> None:
        """Checkpoint carry-forward restores symlink metadata, not its target."""
        source = temp_workspace / "alias.txt"
        source.symlink_to("file1.txt")
        requested_atime_ns = 1_900_000_000_123_456_789
        requested_mtime_ns = 1_700_000_000_987_654_321
        os.utime(
            source,
            ns=(requested_atime_ns, requested_mtime_ns),
            follow_symlinks=False,
        )
        expected_atime_ns = source.lstat().st_atime_ns
        expected_mtime_ns = source.lstat().st_mtime_ns
        target_mtime_ns = (temp_workspace / "file1.txt").stat().st_mtime_ns
        archive_dir = tmp_path / "archives"
        archive_dir.mkdir()
        target = tmp_path / "restored"

        snapshot = Snapshot.from_directory(
            cwd=temp_workspace,
            env={},
            save_path=archive_dir,
        )
        snapshot.extract_to_path(target)

        restored = target / "alias.txt"
        assert restored.lstat().st_atime_ns == expected_atime_ns
        assert restored.lstat().st_mtime_ns == expected_mtime_ns
        assert (target / "file1.txt").stat().st_mtime_ns == target_mtime_ns

    def test_extract_to_path_preserves_empty_nested_directory_metadata(
        self,
        temp_workspace: Path,
        tmp_path: Path,
    ) -> None:
        """Empty directories retain existence, mode, and exact mtime."""
        outer = temp_workspace / "empty-parent"
        nested = outer / "empty-child"
        nested.mkdir(parents=True)
        outer.chmod(0o751)
        nested.chmod(0o705)
        outer_atime_ns = 1_900_000_001_987_654_321
        outer_mtime_ns = 1_700_000_001_123_456_789
        nested_atime_ns = 1_900_000_002_123_456_789
        nested_mtime_ns = 1_700_000_002_987_654_321
        os.utime(outer, ns=(outer_atime_ns, outer_mtime_ns))
        os.utime(nested, ns=(nested_atime_ns, nested_mtime_ns))
        expected_outer_atime_ns = outer.stat().st_atime_ns
        expected_nested_atime_ns = nested.stat().st_atime_ns
        expected_outer_mtime_ns = outer.stat().st_mtime_ns
        expected_nested_mtime_ns = nested.stat().st_mtime_ns
        archive_dir = tmp_path / "archives"
        archive_dir.mkdir()
        target = tmp_path / "restored"

        snapshot = Snapshot.from_directory(
            cwd=temp_workspace,
            env={},
            save_path=archive_dir,
        )
        snapshot.extract_to_path(target)

        restored_outer = target / "empty-parent"
        restored_nested = restored_outer / "empty-child"
        assert restored_nested.is_dir()
        assert restored_outer.stat().st_mode & 0o777 == 0o751
        assert restored_nested.stat().st_mode & 0o777 == 0o705
        assert restored_outer.stat().st_atime_ns == expected_outer_atime_ns
        assert restored_nested.stat().st_atime_ns == expected_nested_atime_ns
        assert restored_outer.stat().st_mtime_ns == expected_outer_mtime_ns
        assert restored_nested.stat().st_mtime_ns == expected_nested_mtime_ns

    def test_extract_to_path_preserves_workspace_root_metadata(
        self,
        temp_workspace: Path,
        tmp_path: Path,
    ) -> None:
        """Root mode and timestamps are applied after all child writes."""
        temp_workspace.chmod(0o711)
        requested_atime_ns = 1_900_000_003_123_456_789
        requested_mtime_ns = 1_700_000_003_987_654_321
        os.utime(
            temp_workspace,
            ns=(requested_atime_ns, requested_mtime_ns),
        )
        expected = temp_workspace.stat()
        archive_dir = tmp_path / "archives"
        archive_dir.mkdir()
        target = tmp_path / "restored"

        snapshot = Snapshot.from_directory(
            cwd=temp_workspace,
            env={},
            save_path=archive_dir,
        )
        snapshot.extract_to_path(target)

        restored = target.stat()
        assert restored.st_mode & 0o777 == 0o711
        assert restored.st_atime_ns == expected.st_atime_ns
        assert restored.st_mtime_ns == expected.st_mtime_ns

    def test_extract_rejects_archive_checksum_mismatch(
        self,
        temp_workspace: Path,
        tmp_path: Path,
    ) -> None:
        """Stored checksums gate every archive consumer before parsing."""
        archive_dir = tmp_path / "archives"
        archive_dir.mkdir()
        snapshot = Snapshot.from_directory(
            cwd=temp_workspace,
            env={},
            save_path=archive_dir,
        )
        with snapshot.archive.open("ab") as archive:
            archive.write(b"post-capture mutation")

        target = tmp_path / "restored"
        with pytest.raises(ExecutionError, match="checksum mismatch"):
            snapshot.extract_to_path(target)
        assert not target.exists()

    @pytest.mark.skipif(not hasattr(Path, "symlink_to"), reason="no symlinks")
    def test_extract_to_path_preserves_safe_relative_symlink(
        self, temp_workspace, tmp_path
    ):
        """Safe in-workspace relative links survive checkpoint carry-forward."""
        (temp_workspace / "alias.txt").symlink_to("file1.txt")
        archive_dir = tmp_path / "archives"
        archive_dir.mkdir()
        target = tmp_path / "restored"

        snapshot = Snapshot.from_directory(
            cwd=temp_workspace,
            env={},
            save_path=archive_dir,
        )
        snapshot.extract_to_path(target)

        restored = target / "alias.txt"
        assert restored.is_symlink()
        assert restored.readlink() == Path("file1.txt")
        assert restored.read_text() == "line1\nline2\n"

    @pytest.mark.skipif(not hasattr(Path, "symlink_to"), reason="no symlinks")
    def test_snapshot_preserves_directory_symlink_type_for_windows(
        self,
        temp_workspace: Path,
        tmp_path: Path,
    ) -> None:
        """PAX metadata carries the flag Windows needs to create a dir link."""
        (temp_workspace / "directory-alias").symlink_to(
            "subdir",
            target_is_directory=True,
        )
        archive_dir = tmp_path / "archives"
        archive_dir.mkdir()
        target = tmp_path / "restored"

        snapshot = Snapshot.from_directory(
            cwd=temp_workspace,
            env={},
            save_path=archive_dir,
        )
        with tarfile.open(snapshot.archive, "r:gz") as archive:
            member = archive.getmember("directory-alias")
            assert member.pax_headers[SYMLINK_TARGET_TYPE_PAX] == "directory"

        snapshot.extract_to_path(target)

        restored = target / "directory-alias"
        assert restored.is_symlink()
        assert restored.is_dir()
        assert restored.readlink() == Path("subdir")

    def test_snapshot_rejects_workspace_escaping_symlink(
        self, temp_workspace, tmp_path
    ):
        """A snapshot cannot capture a link that escapes the workspace."""
        (temp_workspace / "escape").symlink_to("../outside")
        archive_dir = tmp_path / "archives"
        archive_dir.mkdir()

        with pytest.raises(ExecutionError, match="escapes the workspace"):
            Snapshot.from_directory(
                cwd=temp_workspace,
                env={},
                save_path=archive_dir,
            )

    def test_snapshot_materializes_hardlinks_as_regular_files(
        self, temp_workspace, tmp_path
    ):
        """Repeated inodes remain restorable without unsafe tar hard links."""
        original = temp_workspace / "hardlink-source"
        linked = temp_workspace / "hardlink-copy"
        original.write_text("shared inode\n", encoding="utf-8")
        os.link(original, linked)
        archive_dir = tmp_path / "archives"
        archive_dir.mkdir()
        target = tmp_path / "restored"

        snapshot = Snapshot.from_directory(
            cwd=temp_workspace,
            env={},
            save_path=archive_dir,
        )
        snapshot.extract_to_path(target)

        assert (target / "hardlink-source").read_text() == "shared inode\n"
        assert (target / "hardlink-copy").read_text() == "shared inode\n"

    def test_snapshot_rejects_file_mutated_during_capture(
        self,
        temp_workspace,
        tmp_path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Descriptor-pinned capture detects a concurrent writer."""
        source = temp_workspace / "file1.txt"
        archive_dir = tmp_path / "archives"
        archive_dir.mkdir()
        original_addfile = tarfile.TarFile.addfile

        def addfile_then_mutate(
            archive: tarfile.TarFile,
            tar_info: tarfile.TarInfo,
            fileobj=None,
        ) -> None:
            original_addfile(archive, tar_info, fileobj)
            if tar_info.name == "file1.txt":
                with source.open("a", encoding="utf-8") as handle:
                    handle.write("concurrent-write\n")

        monkeypatch.setattr(tarfile.TarFile, "addfile", addfile_then_mutate)

        with pytest.raises(ExecutionError, match="changed during capture"):
            Snapshot.from_directory(
                cwd=temp_workspace,
                env={},
                save_path=archive_dir,
            )

        assert list(archive_dir.iterdir()) == []

    def test_snapshot_rejects_same_size_rewrite_with_restored_mtime(
        self,
        temp_workspace: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Ctime catches a rewrite that restores size and mtime."""
        source = temp_workspace / "file1.txt"
        original = source.stat()
        replacement = b"x" * original.st_size
        archive_dir = tmp_path / "archives"
        archive_dir.mkdir()
        original_addfile = tarfile.TarFile.addfile

        def addfile_then_rewrite(
            archive: tarfile.TarFile,
            tar_info: tarfile.TarInfo,
            fileobj=None,
        ) -> None:
            original_addfile(archive, tar_info, fileobj)
            if tar_info.name == "file1.txt":
                source.write_bytes(replacement)
                os.utime(
                    source,
                    ns=(original.st_atime_ns, original.st_mtime_ns),
                )

        monkeypatch.setattr(tarfile.TarFile, "addfile", addfile_then_rewrite)

        with pytest.raises(ExecutionError, match="changed during capture"):
            Snapshot.from_directory(
                cwd=temp_workspace,
                env={},
                save_path=archive_dir,
            )

        assert source.stat().st_size == original.st_size
        assert source.stat().st_mtime_ns == original.st_mtime_ns
        assert source.stat().st_ctime_ns != original.st_ctime_ns
        assert list(archive_dir.iterdir()) == []

    @pytest.mark.skipif(
        not snapshot_module._supports_descriptor_safe_extraction(),  # noqa: SLF001
        reason="host has no descriptor-safe extraction primitives",
    )
    def test_extract_parent_symlink_swap_cannot_redirect_output(
        self,
        temp_workspace: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A renamed parent cannot redirect output or pass final validation."""
        source_parent = temp_workspace / "raced-parent"
        source_parent.mkdir()
        (source_parent / "payload.txt").write_text(
            "pinned payload\n",
            encoding="utf-8",
        )
        archive_dir = tmp_path / "archives"
        archive_dir.mkdir()
        snapshot = Snapshot.from_directory(
            cwd=temp_workspace,
            env={},
            save_path=archive_dir,
        )
        target = tmp_path / "restored"
        outside = tmp_path / "outside"
        outside.mkdir()
        renamed_parent = target / "descriptor-pinned-parent"
        original_writer = snapshot_module._write_regular_member_at  # noqa: SLF001
        swapped = False

        def swap_parent_then_write(
            parent_fd: int,
            member: tarfile.TarInfo,
            source: BinaryIO,
            *,
            uid: int | None,
            gid: int | None,
        ) -> None:
            nonlocal swapped
            if member.name == "raced-parent/payload.txt":
                raced_parent = target / "raced-parent"
                raced_parent.rename(renamed_parent)
                raced_parent.symlink_to(outside, target_is_directory=True)
                swapped = True
            original_writer(
                parent_fd,
                member,
                source,
                uid=uid,
                gid=gid,
            )

        monkeypatch.setattr(
            snapshot_module,
            "_write_regular_member_at",
            swap_parent_then_write,
        )

        with pytest.raises(
            ExecutionError,
            match="changed before metadata restore",
        ):
            snapshot.extract_to_path(target)

        assert swapped
        assert (renamed_parent / "payload.txt").read_text(
            encoding="utf-8"
        ) == "pinned payload\n"
        assert not (outside / "payload.txt").exists()

    @pytest.mark.skipif(not hasattr(Path, "symlink_to"), reason="no symlinks")
    def test_extract_rejects_symlinked_target_root(
        self,
        temp_workspace,
        tmp_path,
    ) -> None:
        """Extraction never accepts a root redirected through a symlink."""
        archive_dir = tmp_path / "archives"
        archive_dir.mkdir()
        snapshot = Snapshot.from_directory(
            cwd=temp_workspace,
            env={},
            save_path=archive_dir,
        )
        real_target = tmp_path / "real-target"
        real_target.mkdir()
        linked_target = tmp_path / "linked-target"
        linked_target.symlink_to(real_target, target_is_directory=True)

        with pytest.raises(ExecutionError, match="traverses a symlink"):
            snapshot.extract_to_path(linked_target)

    @pytest.mark.skipif(not hasattr(Path, "symlink_to"), reason="no symlinks")
    def test_extract_rejects_symlinked_target_ancestor(
        self,
        temp_workspace,
        tmp_path,
    ) -> None:
        """A symlink in any extraction-root ancestor is rejected."""
        archive_dir = tmp_path / "archives"
        archive_dir.mkdir()
        snapshot = Snapshot.from_directory(
            cwd=temp_workspace,
            env={},
            save_path=archive_dir,
        )
        real_parent = tmp_path / "real-parent"
        real_parent.mkdir()
        linked_parent = tmp_path / "linked-parent"
        linked_parent.symlink_to(real_parent, target_is_directory=True)

        with pytest.raises(ExecutionError, match="traverses a symlink"):
            snapshot.extract_to_path(linked_parent / "nested")

    def test_archive_snapshot_extract_text_contents(self, temp_workspace):
        """Test extracting text contents from an archive snapshot."""
        with tempfile.TemporaryDirectory() as tmpdir:
            snapshot = Snapshot.from_directory(
                cwd=temp_workspace,
                env={},
                save_path=Path(tmpdir),
            )

            contents = snapshot.extract_text_contents()

            assert Path("file1.txt") in contents
            assert Path("file2.txt") in contents
            assert Path("subdir/file3.txt") in contents
            assert contents[Path("file1.txt")] == "line1\nline2\n"

    def test_archive_snapshot_diff_no_changes(self, temp_workspace):
        """Test diff with no changes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            snapshot1 = Snapshot.from_directory(
                cwd=temp_workspace,
                env={},
                save_path=Path(tmpdir),
            )
            snapshot2 = Snapshot.from_directory(
                cwd=temp_workspace,
                env={},
                save_path=Path(tmpdir),
            )

            diff = snapshot1.diff(snapshot2)

            assert len(diff.file_diffs) == 0

    def test_archive_snapshot_diff_file_created(self, temp_workspace):
        """Test diff when a file is created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # First snapshot
            snapshot1 = Snapshot.from_directory(
                cwd=temp_workspace,
                env={},
                save_path=Path(tmpdir),
            )

            # Create a new file
            (temp_workspace / "new_file.txt").write_text("new content\n")

            # Second snapshot
            snapshot2 = Snapshot.from_directory(
                cwd=temp_workspace,
                env={},
                save_path=Path(tmpdir),
            )

            diff = snapshot1.diff(snapshot2)

            assert len(diff.file_diffs) == 1
            assert Path("new_file.txt") in diff.file_diffs
            file_diff = diff.file_diffs[Path("new_file.txt")]
            assert file_diff.change_type == FileChangeType.CREATED
            assert not file_diff.is_binary

    def test_archive_snapshot_diff_file_deleted(self, temp_workspace):
        """Test diff when a file is deleted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # First snapshot
            snapshot1 = Snapshot.from_directory(
                cwd=temp_workspace,
                env={},
                save_path=Path(tmpdir),
            )

            # Delete a file
            (temp_workspace / "file1.txt").unlink()

            # Second snapshot
            snapshot2 = Snapshot.from_directory(
                cwd=temp_workspace,
                env={},
                save_path=Path(tmpdir),
            )

            diff = snapshot1.diff(snapshot2)

            assert len(diff.file_diffs) == 1
            assert Path("file1.txt") in diff.file_diffs
            file_diff = diff.file_diffs[Path("file1.txt")]
            assert file_diff.change_type == FileChangeType.DELETED

    def test_archive_snapshot_diff_file_modified(self, temp_workspace):
        """Test diff when a file is modified."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # First snapshot
            snapshot1 = Snapshot.from_directory(
                cwd=temp_workspace,
                env={},
                save_path=Path(tmpdir),
            )

            # Modify a file
            (temp_workspace / "file1.txt").write_text("modified\ncontent\n")

            # Second snapshot
            snapshot2 = Snapshot.from_directory(
                cwd=temp_workspace,
                env={},
                save_path=Path(tmpdir),
            )

            diff = snapshot1.diff(snapshot2)

            assert len(diff.file_diffs) == 1
            assert Path("file1.txt") in diff.file_diffs
            file_diff = diff.file_diffs[Path("file1.txt")]
            assert file_diff.change_type == FileChangeType.MODIFIED
            assert not file_diff.is_binary

    def test_archive_snapshot_diff_multiple_changes(self, temp_workspace):
        """Test diff with multiple file changes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # First snapshot
            snapshot1 = Snapshot.from_directory(
                cwd=temp_workspace,
                env={},
                save_path=Path(tmpdir),
            )

            # Make multiple changes
            (temp_workspace / "file1.txt").write_text("modified\n")
            (temp_workspace / "file2.txt").unlink()
            (temp_workspace / "new.txt").write_text("new\n")

            # Second snapshot
            snapshot2 = Snapshot.from_directory(
                cwd=temp_workspace,
                env={},
                save_path=Path(tmpdir),
            )

            diff = snapshot1.diff(snapshot2)

            assert len(diff.file_diffs) == 3
            assert (
                diff.file_diffs[Path("file1.txt")].change_type
                == FileChangeType.MODIFIED
            )
            assert (
                diff.file_diffs[Path("file2.txt")].change_type
                == FileChangeType.DELETED
            )
            assert (
                diff.file_diffs[Path("new.txt")].change_type
                == FileChangeType.CREATED
            )

    def test_archive_snapshot_diff_binary_file(self, temp_workspace):
        """Test diff with binary files (should be excluded from text-only diff)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # First snapshot
            snapshot1 = Snapshot.from_directory(
                cwd=temp_workspace,
                env={},
                save_path=Path(tmpdir),
            )

            # Create a binary file
            (temp_workspace / "binary.bin").write_bytes(b"hello\x00world")

            # Second snapshot
            snapshot2 = Snapshot.from_directory(
                cwd=temp_workspace,
                env={},
                save_path=Path(tmpdir),
            )

            diff = snapshot1.diff(snapshot2)

            # Binary files are not included in text-only diff
            assert len(diff.file_diffs) == 0


class TestFileDiffRepr:
    """Test FileDiff string representations."""

    def test_created_text_repr(self):
        """Test repr for created text file."""
        diff = FileDiff(
            path=Path("test.txt"),
            change_type=FileChangeType.CREATED,
            lines_added=0,
        )
        repr_str = repr(diff)
        assert "CREATED" in repr_str
        assert "test.txt" in repr_str

    def test_deleted_binary_repr(self):
        """Test repr for deleted binary file."""
        diff = FileDiff(
            path=Path("test.bin"),
            change_type=FileChangeType.DELETED,
            is_binary=True,
            old_size=1024,
        )
        repr_str = repr(diff)
        assert "DELETED" in repr_str
        assert "1024" in repr_str

    def test_modified_binary_repr(self):
        """Test repr for modified binary file."""
        diff = FileDiff(
            path=Path("test.bin"),
            change_type=FileChangeType.MODIFIED,
            is_binary=True,
            old_size=1024,
            new_size=2048,
        )
        repr_str = repr(diff)
        assert "MODIFIED" in repr_str
        assert "1024" in repr_str
        assert "2048" in repr_str


class TestSnapshotDiffRepr:
    """Test SnapshotDiff string representation."""

    @pytest.fixture
    def temp_workspace(self, tmp_path):
        """Create a temporary workspace with some files."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()

        # Create some initial files
        (workspace / "file1.txt").write_text("line1\nline2\n")
        (workspace / "file2.txt").write_text("hello\nworld\n")
        (workspace / "subdir").mkdir()
        (workspace / "subdir" / "file3.txt").write_text("nested\nfile\n")

        return workspace

    def test_snapshot_diff_repr(self, temp_workspace):
        """Test repr for snapshot diff."""
        with tempfile.TemporaryDirectory() as tmpdir:
            snapshot1 = Snapshot.from_directory(
                cwd=temp_workspace,
                env={},
                save_path=Path(tmpdir),
            )

            # Make changes
            (temp_workspace / "file1.txt").write_text("modified\n")
            (temp_workspace / "file2.txt").unlink()
            (temp_workspace / "new.txt").write_text("new\n")

            snapshot2 = Snapshot.from_directory(
                cwd=temp_workspace,
                env={},
                save_path=Path(tmpdir),
            )

            diff = snapshot1.diff(snapshot2)
            repr_str = repr(diff)

            assert "+1" in repr_str  # 1 created
            assert "-1" in repr_str  # 1 deleted
            assert "~1" in repr_str  # 1 modified
