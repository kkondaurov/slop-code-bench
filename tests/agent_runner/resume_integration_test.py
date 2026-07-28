"""Integration tests for checkpoint resume functionality.

These tests verify that resume commands work correctly in a real Docker
environment with proper user permissions.
"""

from __future__ import annotations

import queue
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import yaml

from slop_code.agent_runner.agent import Agent
from slop_code.agent_runner.runner import AgentRunner
from slop_code.execution.docker_runtime import DockerEnvironmentSpec
from slop_code.execution.docker_runtime import DockerExecRuntime
from slop_code.execution.runtime import RuntimeResult


def run_docker_command(
    env: DockerEnvironmentSpec,
    working_dir: Path,
    command: str,
    timeout: int = 120,
) -> RuntimeResult:
    """Run a single command in Docker and return the result."""
    runtime = DockerExecRuntime.spawn(
        environment=env,
        working_dir=working_dir,
        command=command,
        static_assets=None,
        ports={},
        mounts={},
        env_vars={},
        setup_command=None,
        user=env.get_eval_user(),
        is_evaluation=False,
        disable_setup=True,
    )
    try:
        return runtime.execute(env={}, stdin=None, timeout=timeout)
    finally:
        runtime.cleanup()


@pytest.fixture
def docker_python_env() -> DockerEnvironmentSpec:
    """Load the docker-python3.12-uv environment spec."""
    config_path = (
        Path(__file__).parent.parent.parent
        / "configs"
        / "environments"
        / "docker-python3.12-uv.yaml"
    )
    with config_path.open() as f:
        config = yaml.safe_load(f)
    return DockerEnvironmentSpec(**config)


def cleanup_as_root(path: Path) -> None:
    """Clean up a directory that may have root-owned files using docker."""
    if not path.exists():
        return
    # Use docker to remove files as root
    subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{path}:/cleanup",
            "alpine:latest",
            "rm",
            "-rf",
            "/cleanup",
        ],
        capture_output=True,
        check=False,
    )
    # Remove the now-empty directory
    if path.exists():
        path.rmdir()


@pytest.fixture
def workspace_dir(docker_shared_tmp_path: Path):
    """Create a workspace directory that gets properly cleaned up."""
    workspace = docker_shared_tmp_path / "workspace"
    workspace.mkdir()
    yield workspace
    # Clean up any root-owned files created by docker
    cleanup_as_root(workspace)


@pytest.mark.integration
class TestResumeCommandsIntegration:
    """Integration tests for resume commands in Docker."""

    def test_resume_commands_create_venv(
        self,
        docker_python_env: DockerEnvironmentSpec,
        workspace_dir: Path,
        tmp_path: Path,
    ) -> None:
        """Test that resume commands create a working venv."""
        # Create a minimal snapshot with a requirements.txt
        snapshot_dir = tmp_path / "snapshot"
        snapshot_dir.mkdir()
        (snapshot_dir / "requirements.txt").write_text("requests==2.31.0\n")
        (snapshot_dir / "main.py").write_text("print('hello')\n")

        # Copy snapshot to workspace
        import shutil

        for item in snapshot_dir.iterdir():
            if item.is_dir():
                shutil.copytree(item, workspace_dir / item.name)
            else:
                shutil.copy2(item, workspace_dir / item.name)

        # Run resume commands
        resume_commands = docker_python_env.get_resume_commands()
        assert len(resume_commands) >= 2, "Expected at least 2 resume commands"

        for cmd in resume_commands:
            result = run_docker_command(
                docker_python_env, workspace_dir, cmd, timeout=120
            )
            # Commands use || true so should always succeed
            assert result.exit_code == 0, (
                f"Resume command failed: {cmd}\n"
                f"stdout: {result.stdout}\n"
                f"stderr: {result.stderr}"
            )

        # Verify venv was created
        result = run_docker_command(
            docker_python_env,
            workspace_dir,
            "test -d .venv && echo 'venv exists'",
            timeout=10,
        )
        assert "venv exists" in result.stdout, "venv directory not created"

        # Verify pip works in the venv
        result = run_docker_command(
            docker_python_env,
            workspace_dir,
            ".venv/bin/pip --version",
            timeout=30,
        )
        assert result.exit_code == 0, (
            f"pip not working in venv: {result.stderr}"
        )
        assert "pip" in result.stdout

        # Verify requests was installed
        result = run_docker_command(
            docker_python_env,
            workspace_dir,
            ".venv/bin/pip show requests",
            timeout=30,
        )
        assert result.exit_code == 0, f"requests not installed: {result.stderr}"
        assert "requests" in result.stdout.lower()

    def test_checkpoint_lifecycle_uses_fresh_container_and_snapshot_only(
        self,
        docker_python_env: DockerEnvironmentSpec,
        docker_shared_tmp_path: Path,
    ) -> None:
        """Only snapshot files carry into a fresh checkpoint container."""
        image = docker_python_env.get_base_image()
        image_check = subprocess.run(  # noqa: S603,S607
            ["docker", "image", "inspect", image],  # noqa: S607
            capture_output=True,
            check=False,
        )
        if image_check.returncode != 0:
            pytest.skip(f"Docker image is not available: {image}")

        snapshot_config = docker_python_env.snapshot.model_copy(
            update={
                "archive_save_dir": docker_shared_tmp_path / "archives"
            }
        )
        setup_config = docker_python_env.setup.model_copy(
            update={"resume_commands": []}
        )
        environment = docker_python_env.model_copy(
            update={"setup": setup_config, "snapshot": snapshot_config}
        )
        problem = SimpleNamespace(
            name="checkpoint_lifecycle",
            path=docker_shared_tmp_path,
            static_assets={},
        )
        run_spec = SimpleNamespace(
            problem=problem,
            environment=environment,
        )
        agent = MagicMock(spec=Agent)
        agent_runner = AgentRunner(
            run_spec=run_spec,
            agent=agent,
            output_path=docker_shared_tmp_path / "output",
            progress_queue=queue.Queue(),
        )
        snapshot_dir = docker_shared_tmp_path / "checkpoint_1_snapshot"

        try:
            agent_runner._setup_for_checkpoint(
                SimpleNamespace(name="checkpoint_1"),
                prior_snapshot_dir=None,
            )
            first_runtime = agent_runner.session.spawn(disable_setup=True)
            first_events = list(
                first_runtime.stream(
                    "printf 'checkpoint one\\n' > carried.txt && "
                    "mkdir -p .evaluation_tests && "
                    "printf hidden > .evaluation_tests/hidden.txt && "
                    "printf private > /tmp/checkpoint-only",
                    env={},
                    timeout=30,
                )
            )
            first_result = first_events[-1].result
            assert first_result is not None
            assert first_result.exit_code == 0, first_result.stderr
            first_container_id = first_runtime.container.id
            agent_runner.session.finish_checkpoint(snapshot_dir)
            agent_runner._has_executed_checkpoint = True
            agent_runner._cleanup_checkpoint_session()

            assert (snapshot_dir / "carried.txt").read_text() == (
                "checkpoint one\n"
            )
            assert not (
                snapshot_dir / ".evaluation_tests" / "hidden.txt"
            ).exists()

            agent_runner._setup_for_checkpoint(
                SimpleNamespace(name="checkpoint_2"),
                prior_snapshot_dir=snapshot_dir,
            )
            second_runtime = agent_runner.session.spawn(disable_setup=True)
            second_events = list(
                second_runtime.stream(
                    "test \"$(cat carried.txt)\" = 'checkpoint one' && "
                    "test ! -e /tmp/checkpoint-only && "
                    "test ! -e .evaluation_tests/hidden.txt",
                    env={},
                    timeout=30,
                )
            )
            second_result = second_events[-1].result
            assert second_result is not None
            assert second_result.exit_code == 0, second_result.stderr
            second_container_id = second_runtime.container.id

            assert second_container_id != first_container_id
        finally:
            agent_runner._cleanup_checkpoint_session()

    def test_resume_commands_handle_missing_requirements(
        self,
        docker_python_env: DockerEnvironmentSpec,
        workspace_dir: Path,
        tmp_path: Path,
    ) -> None:
        """Test that resume commands don't fail when requirements.txt is missing."""
        # No requirements.txt - simulates a problem that doesn't need deps
        (workspace_dir / "main.py").write_text("print('no deps')\n")

        # All resume commands should succeed (they use || true)
        for cmd in docker_python_env.get_resume_commands():
            result = run_docker_command(
                docker_python_env, workspace_dir, cmd, timeout=120
            )
            assert result.exit_code == 0, (
                f"Resume command should not fail even without "
                f"requirements.txt: {cmd}\nstderr: {result.stderr}"
            )
