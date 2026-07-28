"""Shared test fixtures and utilities."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import docker
import pytest
from dotenv import dotenv_values

from slop_code.common.temp import TEMP_ROOT_ENV
from slop_code.common.temp import temporary_directory

_LOCAL_ENV = dotenv_values(Path(__file__).resolve().parents[1] / ".env")
for _name in ("DOCKER_HOST", TEMP_ROOT_ENV):
    _value = _LOCAL_ENV.get(_name)
    if isinstance(_value, str):
        os.environ.setdefault(_name, _value)


def is_docker_available() -> bool:
    """Check if Docker is available and running.

    Returns:
        True if Docker is available, False otherwise
    """
    try:
        client = docker.from_env()
        client.ping()
        client.close()
        return True
    except docker.errors.DockerException:
        return False


@pytest.fixture(scope="session")
def docker_available() -> bool:
    """Session-scoped fixture that checks if Docker is available."""
    return is_docker_available()


@pytest.fixture(scope="session")
def skip_if_no_docker():
    """Skip test if Docker is not available."""
    if not is_docker_available():
        pytest.skip("Docker is not available")


@pytest.fixture
def docker_shared_tmp_path() -> Iterator[Path]:
    """Create a Docker-bindable temp path when a shared root is configured."""
    with temporary_directory() as temp_dir:
        yield Path(temp_dir)
