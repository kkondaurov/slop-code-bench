"""Session management for execution environments.

This module provides high-level session management that coordinates workspaces,
runtimes, and execution lifecycle:

- **Session**: Main session class managing workspace and runtime lifecycle
- **SessionError**: Session-specific exception
- Context manager support for automatic setup and cleanup
- Runtime spawning and management
- Input file materialization and content retrieval
- Static asset placeholder resolution
- Checkpoint completion and snapshot management

Sessions provide a convenient interface for managing complete execution
environments with proper resource cleanup and state management.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

from slop_code.execution.assets import ResolvedStaticAsset
from slop_code.execution.file_ops import InputFile
from slop_code.execution.file_ops import materialize_input_files
from slop_code.execution.models import EnvironmentSpec
from slop_code.execution.placeholders import resolve_static_placeholders
from slop_code.execution.protocols import ExecRuntime
from slop_code.execution.protocols import StreamingRuntime
from slop_code.execution.runtime import spawn_exec_runtime
from slop_code.execution.runtime import spawn_streaming_runtime
from slop_code.execution.snapshot import Snapshot
from slop_code.execution.snapshot import SnapshotDiff
from slop_code.execution.workspace import Workspace
from slop_code.logging import get_logger

logger = get_logger(__name__)


class SessionError(Exception):
    """Exception raised by the Session class."""


def cleanup_preserving_primary(
    cleanup: Callable[[], None],
    primary_error: BaseException | None,
    *,
    phase: str,
) -> None:
    """Run cleanup without replacing an already-active failure."""
    try:
        cleanup()
    except BaseException as cleanup_error:  # noqa: BLE001
        if primary_error is None:
            raise
        if not isinstance(cleanup_error, Exception):
            cleanup_error.add_note(
                f"Earlier {phase} failure: "
                f"{type(primary_error).__qualname__}: {primary_error}"
            )
            logger.error(
                "Cancellation/control flow interrupted cleanup",
                phase=phase,
                primary_error_type=type(primary_error).__qualname__,
                primary_error_message=str(primary_error),
                cleanup_error_type=type(cleanup_error).__qualname__,
                cleanup_error_message=str(cleanup_error),
                exc_info=True,
            )
            raise
        primary_error.add_note(
            f"Secondary {phase} cleanup failure: "
            f"{type(cleanup_error).__qualname__}: {cleanup_error}"
        )
        logger.error(
            "Cleanup failed while preserving an active error",
            phase=phase,
            primary_error_type=type(primary_error).__qualname__,
            primary_error_message=str(primary_error),
            cleanup_error_type=type(cleanup_error).__qualname__,
            cleanup_error_message=str(cleanup_error),
            exc_info=True,
        )


class Session:
    """Manages an execution session with workspace and runtime lifecycle.

    A session provides a high-level interface for managing execution
    environments, including workspace setup, runtime spawning, and cleanup.
    """

    def __init__(
        self,
        spec: EnvironmentSpec,
        workspace: Workspace,
        static_assets: dict[str, ResolvedStaticAsset] | None = None,
        is_agent_infer: bool = False,
    ):
        """Initialize a new session.

        Args:
            spec: Environment specification for execution
            workspace: Workspace instance for file management
            static_assets: Optional static assets available to the session
            is_agent_infer: Whether this is an agent inference session
        """
        logger.debug(
            "Initializing session",
            spec_type=spec.type,
            is_agent_infer=is_agent_infer,
            static_assets=list((static_assets or {}).keys()),
            verbose=True,
        )
        self.workspace = workspace
        self.static_assets = static_assets
        self.spec = spec
        self._streaming_runtimes: list[StreamingRuntime] = []
        self._exec_runtimes: list[ExecRuntime] = []
        self.is_agent_infer = is_agent_infer

    def spawn(
        self,
        ports: dict[int, int] | None = None,
        mounts: dict[str, dict[str, str] | str] | None = None,
        env_vars: dict[str, str] | None = None,
        setup_command: str | None = None,
        disable_setup: bool = False,
        **runtime_kwargs: Any,
    ) -> StreamingRuntime:
        """Spawn a new streaming runtime for interactive execution.

        Used by agents for long-running sessions with streamed output.

        Args:
            ports: Port mappings
            mounts: Volume mounts
            env_vars: Environment variables for runtime
            setup_command: Additional setup command
            disable_setup: Whether to disable setup commands
            **runtime_kwargs: Additional runtime-specific arguments

        Returns:
            New StreamingRuntime instance configured for this session
        """
        logger.debug(
            "Spawning new streaming runtime",
            verbose=True,
        )

        runtime = spawn_streaming_runtime(
            environment=self.spec,
            working_dir=self.workspace.working_dir,
            static_assets=self.static_assets,
            ports=ports or {},
            mounts=mounts or {},
            env_vars=env_vars or {},
            setup_command=setup_command,
            is_evaluation=not self.is_agent_infer,
            disable_setup=disable_setup,
            **runtime_kwargs,
        )

        self._streaming_runtimes.append(runtime)
        return runtime

    def exec(
        self,
        command: str,
        ports: dict[int, int] | None = None,
        mounts: dict[str, dict[str, str] | str] | None = None,
        env_vars: dict[str, str] | None = None,
        setup_command: str | None = None,
        disable_setup: bool = False,
        **runtime_kwargs: Any,
    ) -> ExecRuntime:
        """Spawn a new execution runtime for one-shot execution.

        Used by evaluation for single command execution with buffered output.
        The command is set at spawn time and executed via .execute().

        Args:
            command: Command to execute (immutable)
            ports: Port mappings
            mounts: Volume mounts
            env_vars: Environment variables for runtime
            setup_command: Additional setup command
            disable_setup: Whether to disable setup commands
            **runtime_kwargs: Additional runtime-specific arguments

        Returns:
            New ExecRuntime instance configured for this session
        """
        logger.debug(
            "Spawning new exec runtime",
            command=command[:100],
            verbose=True,
        )

        runtime = spawn_exec_runtime(
            environment=self.spec,
            working_dir=self.workspace.working_dir,
            command=command,
            static_assets=self.static_assets,
            ports=ports or {},
            mounts=mounts or {},
            env_vars=env_vars or {},
            setup_command=setup_command,
            is_evaluation=not self.is_agent_infer,
            disable_setup=disable_setup,
            **runtime_kwargs,
        )

        self._exec_runtimes.append(runtime)
        return runtime

    def prepare(self) -> None:
        """Prepare the session for execution."""
        logger.debug("Preparing session", verbose=True)
        self.workspace.prepare()

    def materialize_assets(self) -> None:
        """Materialize static assets into workspace."""
        self.workspace.materialize_assets()

    def cleanup(self) -> None:
        """Clean up all session resources."""
        logger.debug(
            "Cleaning up session",
            num_streaming_runtimes=len(self._streaming_runtimes),
            num_exec_runtimes=len(self._exec_runtimes),
            verbose=True,
        )
        try:
            self.cleanup_runtimes()
        except BaseException as error:  # noqa: BLE001
            if not isinstance(error, Exception):
                raise
            logger.warning(
                "Runtime cleanup failed; retrying retained runtimes once",
                error_type=type(error).__qualname__,
                error_message=str(error),
            )
            try:
                self.cleanup_runtimes()
            except BaseException as retry_error:  # noqa: BLE001
                if not isinstance(retry_error, Exception):
                    retry_error.add_note(
                        "An earlier runtime cleanup attempt also failed: "
                        f"{type(error).__qualname__}: {error}"
                    )
                    raise
                error.add_note(
                    "Runtime cleanup retry also failed: "
                    f"{type(retry_error).__qualname__}: {retry_error}"
                )
                raise error

        cleanup_error: BaseException | None = None
        try:
            self.workspace.cleanup()
        except BaseException as error:  # noqa: BLE001
            if cleanup_error is None:
                cleanup_error = error
            else:
                logger.error(
                    "Workspace cleanup failed after runtime cleanup error",
                    error_type=type(error).__qualname__,
                    error_message=str(error),
                    exc_info=True,
                )
        try:
            self.workspace.cleanup_snapshot()
        except BaseException as error:  # noqa: BLE001
            if cleanup_error is None:
                cleanup_error = error
            else:
                logger.error(
                    "Snapshot cleanup failed after an earlier cleanup error",
                    error_type=type(error).__qualname__,
                    error_message=str(error),
                    exc_info=True,
                )
        if cleanup_error is not None:
            raise cleanup_error

    def cleanup_runtimes(self) -> None:
        """Stop Session-owned runtimes while preserving the host workspace."""
        logger.debug(
            "Cleaning up session runtimes",
            num_streaming_runtimes=len(self._streaming_runtimes),
            num_exec_runtimes=len(self._exec_runtimes),
            verbose=True,
        )
        cleanup_error: BaseException | None = None
        streaming_runtimes = self._streaming_runtimes
        exec_runtimes = self._exec_runtimes
        self._streaming_runtimes = []
        self._exec_runtimes = []

        for runtime in streaming_runtimes:
            try:
                runtime.cleanup()
            except BaseException as error:  # noqa: BLE001
                # Retain ownership so a later Session.cleanup() can retry the
                # stop/remove operation. In particular, a Docker container
                # with a host bind mount must never become unreachable here.
                self._streaming_runtimes.append(runtime)
                if cleanup_error is None:
                    cleanup_error = error
                else:
                    logger.error(
                        "Additional runtime cleanup failed",
                        error_type=type(error).__qualname__,
                        error_message=str(error),
                        exc_info=True,
                    )
        for runtime in exec_runtimes:
            try:
                runtime.cleanup()
            except BaseException as error:  # noqa: BLE001
                self._exec_runtimes.append(runtime)
                if cleanup_error is None:
                    cleanup_error = error
                else:
                    logger.error(
                        "Additional runtime cleanup failed",
                        error_type=type(error).__qualname__,
                        error_message=str(error),
                        exc_info=True,
                    )
        if cleanup_error is not None:
            raise cleanup_error

    def __enter__(self) -> "Session":
        """Context manager entry."""
        self.prepare()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Context manager exit."""
        cleanup_preserving_primary(
            self.cleanup,
            exc_value,
            phase="session",
        )

    @property
    def working_dir(self) -> Path:
        """Get the working directory for this session."""
        return self.workspace.working_dir

    def reset(self) -> None:
        """Reset the workspace to its initial state."""
        logger.debug("Resetting session workspace", verbose=True)
        self.workspace.reset()

    def restore_from_snapshot_dir(self, snapshot_dir: Path) -> None:
        """Restore workspace state from a snapshot directory.

        This copies all files from the snapshot directory into the workspace,
        used when resuming from a previous checkpoint.

        Args:
            snapshot_dir: Path to directory containing snapshot files

        Raises:
            SessionError: If snapshot directory does not exist
        """
        import shutil

        if not snapshot_dir.exists():
            raise SessionError(
                f"Snapshot directory does not exist: {snapshot_dir}"
            )

        logger.info(
            "Restoring workspace from snapshot directory",
            snapshot_dir=str(snapshot_dir),
            working_dir=str(self.working_dir),
        )

        # Copy all files from snapshot to workspace
        for item in snapshot_dir.iterdir():
            dest = self.working_dir / item.name
            if item.is_dir():
                if dest.exists():
                    shutil.rmtree(dest)
                shutil.copytree(item, dest)
            else:
                shutil.copy2(item, dest)

        # Re-materialize static assets (they may have been in snapshot ignores)
        self.workspace.materialize_assets()

        logger.debug(
            "Snapshot restoration complete",
            snapshot_dir=str(snapshot_dir),
        )

    def materialize_input_files(self, files: list[InputFile]) -> None:
        """Materialize input files into the workspace.

        Args:
            files: List of input files to write to workspace
        """
        materialize_input_files(files, self.workspace.working_dir)

    def get_file_contents(self, files: list[str]) -> dict[str, str | bytes]:
        """Get contents of files from the workspace.

        Args:
            files: Relative paths or glob patterns to read.

        Returns:
            Dictionary mapping matched file paths to their contents.
        """
        return self.workspace.get_file_contents(files)

    def resolve_static_placeholders(
        self, data: dict[str, Any]
    ) -> dict[str, Any]:
        """Resolve static asset placeholders in data.

        Args:
            data: Data structure potentially containing placeholders

        Returns:
            Data with placeholders resolved to appropriate paths
        """
        return resolve_static_placeholders(
            data,
            self.static_assets or {},
            is_docker=self.spec.type == "docker",
        )

    def finish_checkpoint(self, output_dir: Path) -> SnapshotDiff:
        """Finish a checkpoint and save results.

        Args:
            output_dir: Directory to save checkpoint results

        Returns:
            SnapshotDiff showing changes made during the checkpoint

        Raises:
            SessionError: If not in agent infer mode
        """
        if not self.is_agent_infer:
            raise SessionError(
                "Cannot finish checkpoint in non-agent infer mode"
            )

        logger.debug(
            "Finishing checkpoint",
            output_dir=output_dir,
            verbose=True,
        )

        try:
            # The workspace is a host bind mount for Docker agents. Stop every
            # process/container that can still mutate it before reading bytes.
            self.cleanup_runtimes()
        except BaseException:
            # A failed runtime remains Session-owned for a later cleanup retry.
            # Its bind-mounted process may still be writing, so fail closed
            # instead of producing a racy snapshot that looks durable.
            logger.error(
                "Runtime cleanup failed; refusing live workspace snapshot",
                output_dir=str(output_dir),
                exc_info=True,
            )
            raise

        old_snapshot: Snapshot | None = None
        snapshot_error: BaseException | None = None
        try:
            old_snapshot = self.workspace.update_snapshot()
            new_snapshot = self.workspace.initial_snapshot
            output_dir.mkdir(parents=True, exist_ok=True)
            new_snapshot.extract_to_path(output_dir)

            archive_filename = new_snapshot.archive.name
            archive_in_output = output_dir / archive_filename
            if archive_in_output.exists():
                logger.debug(
                    "Removing snapshot archive from output directory",
                    archive=str(archive_in_output),
                    verbose=True,
                )
                archive_in_output.unlink()

            diff = SnapshotDiff.from_snapshots(old_snapshot, new_snapshot)
        except BaseException as error:  # noqa: BLE001
            snapshot_error = error
            raise
        finally:
            if old_snapshot is not None:
                cleanup_preserving_primary(
                    old_snapshot.cleanup,
                    snapshot_error,
                    phase="previous checkpoint snapshot",
                )

        return diff

    @classmethod
    def from_environment_spec(
        cls,
        spec: EnvironmentSpec,
        base_dir: Path | None,
        static_assets: dict[str, ResolvedStaticAsset] | None = None,
        image_name: str | None = None,
        is_agent_infer: bool = False,
    ) -> "Session":
        """Create a session from an environment specification.

        Args:
            spec: Environment specification
            base_dir: Optional base directory to initialize from
            static_assets: Optional static assets
            image_name: Optional image name to use for the session
            is_agent_infer: Whether this is an agent inference session

        Returns:
            New Session instance
        """
        logger.debug(
            "Creating session from environment spec",
            spec=spec.type,
            image_name=image_name,
            base_dir=base_dir,
            static_assets=list((static_assets or {}).keys()),
            verbose=True,
        )

        def snapshot_fn(cwd: Path) -> Snapshot:
            return Snapshot.from_environment_spec(
                cwd=cwd,
                env_spec=spec,
                static_assets=static_assets,
            )

        snapshot = None
        if base_dir is not None:
            logger.debug(
                "Creating initial snapshot from base directory",
                base_dir=base_dir,
                verbose=True,
            )
            snapshot = snapshot_fn(base_dir)

        workspace = Workspace(
            initial_snapshot=snapshot,
            snapshot_fn=snapshot_fn,
            static_assets=static_assets,
            is_agent_infer=is_agent_infer,
        )
        return cls(
            spec=spec,
            workspace=workspace,
            static_assets=static_assets,
            is_agent_infer=is_agent_infer,
        )
