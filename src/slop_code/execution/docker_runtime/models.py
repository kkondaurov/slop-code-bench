from __future__ import annotations

import os
import platform
import re
from typing import Literal

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import model_validator

from slop_code.execution.models import EnvironmentSpec
from slop_code.logging import get_logger

logger = get_logger(__name__)

IMAGE_NAME_PREFIX = "slop-code"
SHA256_IMAGE_ID_PATTERN = re.compile(r"sha256:[0-9a-f]{64}\Z")


class DockerConfig(BaseModel):
    """Docker-specific configuration for container execution.

    Attributes:
        image: Container image used for execution
        binary: Docker CLI binary used to launch containers
        workdir: Working directory inside the container
        mount_workspace: Whether to bind-mount workspace into container
        extra_mounts: Additional host-to-container mount mappings
        network: Docker network to attach the container to
        user: User specifier for docker run (e.g. '1000:1000')
        keep_container_after_clean: Prevents container removal after cleanup
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    image: str = Field(
        description="Base container image used for execution.",
    )
    binary: str = Field(
        default="docker",
        description="Docker CLI binary used to launch containers.",
    )
    workdir: str = Field(
        default="/workspace",
        description="Working directory inside the container.",
    )
    mount_workspace: bool = Field(
        default=True,
        description="Whether to bind-mount workspace into container.",
    )
    extra_mounts: dict[str, str | dict[str, str]] = Field(
        default_factory=dict,
        description="Additional host-to-container mount mappings.",
    )
    network: str | None = Field(
        default=None,
        description="Docker network to attach the container to.",
    )
    user: str | None = Field(
        default=None,
        description="User specifier for docker run (e.g. '1000:1000').",
    )
    prebuilt_image: str | None = Field(
        default=None,
        description=(
            "Immutable prebuilt image ID or registry digest to use directly "
            "instead of executing the base-image setup recipe."
        ),
    )
    expected_image_id: str | None = Field(
        default=None,
        description=(
            "Exact Docker image ID required for the resolved prebuilt image."
        ),
    )
    expected_architecture: Literal["amd64", "arm64"] | None = Field(
        default=None,
        description=(
            "Architecture required for the resolved prebuilt image."
        ),
    )

    @model_validator(mode="after")
    def validate_prebuilt_image_lock(self) -> DockerConfig:
        """Require a complete, immutable lock for direct prebuilt images."""
        values = (
            self.prebuilt_image,
            self.expected_image_id,
            self.expected_architecture,
        )
        if all(value is None for value in values):
            return self
        if any(value is None for value in values):
            raise ValueError(
                "prebuilt_image, expected_image_id, and "
                "expected_architecture must be configured together"
            )
        reference = self.prebuilt_image
        expected_id = self.expected_image_id
        if reference is None or expected_id is None:
            raise ValueError("Prebuilt image lock is incomplete")
        if SHA256_IMAGE_ID_PATTERN.fullmatch(expected_id) is None:
            raise ValueError(
                "expected_image_id must be a lowercase sha256 Docker image ID"
            )
        if (
            SHA256_IMAGE_ID_PATTERN.fullmatch(reference) is None
            and "@sha256:" not in reference
        ):
            raise ValueError(
                "prebuilt_image must be a sha256 image ID or an immutable "
                "registry digest reference"
            )
        return self


class DockerEnvironmentSpec(EnvironmentSpec):
    """Container-based execution configuration.

    Attributes:
        docker: Docker-specific configuration (image, workdir, mounts, etc.)
    """

    type: Literal["docker"] = "docker"  # type: ignore[assignment]
    docker: DockerConfig

    def get_eval_user(self) -> str:
        if self.docker.user:
            return self.docker.user
        return "1000:1000"

    def get_actual_user(self) -> str:
        """Resolve the user to run as outside evaluation contexts."""
        if self.docker.user:
            return self.docker.user
        uid = os.getenv("HUID")
        gid = os.getenv("HGID")
        if uid and gid:
            return f"{uid}:{gid}"
        return "0:0"

    def get_effective_address(self, address: str) -> str:
        """Get the address to pass to commands inside the container.

        When using ``bridge`` networking and the caller requests a loopback
        address, we bind to all interfaces (``0.0.0.0``) so that the service is
        reachable via port mapping from the host.
        """
        if self.effective_network_mode() == "bridge" and address in (
            "127.0.0.1",
            "localhost",
        ):
            return "0.0.0.0"
        return address

    def effective_network_mode(self) -> str:
        desired = (
            self.docker.network if self.docker.network is not None else "bridge"
        )
        if desired == "host" and platform.system() != "Linux":
            logger.warning(
                "Host network mode unsupported on this platform; using bridge",
                platform=platform.system(),
            )
            return "bridge"
        return desired

    def get_base_image(self) -> str:
        return f"{IMAGE_NAME_PREFIX}:{self.name}"
