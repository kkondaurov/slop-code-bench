from __future__ import annotations

from pathlib import Path

from slop_code.execution.docker_runtime.exec import DockerExecRuntime
from slop_code.execution.docker_runtime.images import make_base_image
from slop_code.execution.docker_runtime.models import DockerConfig
from slop_code.execution.docker_runtime.models import DockerEnvironmentSpec
from slop_code.execution.models import CommandConfig

from .conftest import docker_available
from .conftest import test_image_available as requires_test_image


def test_rendered_base_image_installs_git_and_ripgrep() -> None:
    environment = DockerEnvironmentSpec(
        type="docker",
        name="python3.12",
        commands=CommandConfig(command="python"),
        docker=DockerConfig(
            image="python:3.12-slim",
            workdir="/workspace",
        ),
    )

    dockerfile = make_base_image(environment)
    install_line = next(
        line
        for line in dockerfile.splitlines()
        if line.startswith("RUN apt-get install -y bzip2")
    )
    installed_packages = set(install_line.split()[4:])

    assert {"git", "ripgrep"} <= installed_packages


@docker_available
@requires_test_image
def test_base_image_exposes_git_and_ripgrep(tmp_path: Path) -> None:
    environment = DockerEnvironmentSpec(
        type="docker",
        name="python3.12",
        commands=CommandConfig(command="python"),
        docker=DockerConfig(
            image="python:3.12-slim",
            workdir="/workspace",
        ),
    )
    runtime = DockerExecRuntime.spawn(
        environment=environment,
        working_dir=tmp_path,
        command="git --version && rg --version",
        disable_setup=True,
    )

    try:
        result = runtime.execute(env={}, stdin=None, timeout=30)
    finally:
        runtime.cleanup()

    assert result.exit_code == 0, result.stderr
    assert result.stdout.startswith("git version ")
    assert "ripgrep " in result.stdout
