from __future__ import annotations

import hashlib
import io
import json
import os
import re
import stat
import tarfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Any

from docker.errors import BuildError
from docker.errors import DockerException
from docker.errors import ImageNotFound
from jinja2 import Template

from slop_code.execution.assets import ResolvedStaticAsset
from slop_code.execution.docker_runtime.models import IMAGE_NAME_PREFIX
from slop_code.execution.docker_runtime.models import DockerEnvironmentSpec
from slop_code.execution.models import EnvironmentSpec
from slop_code.logging import get_logger

logger = get_logger(__name__)

BASE_IMAGE_TEMPLATE = Path(__file__).parent / "setup_base.docker.j2"
AGENT_USER = "1000:1000"
BASE_IMAGE_HASH_LABEL = "io.slop-code.base-image-hash"
UPSTREAM_IMAGE_ID_LABEL = "io.slop-code.upstream-image-id"
IMAGE_SPEC_HASH_LABEL = "io.slop-code.image-spec-hash"
PARENT_IMAGE_ID_LABEL = "io.slop-code.parent-image-id"
BUILD_CONTEXT_HASH_LABEL = "io.slop-code.build-context-sha256"
FROM_INSTRUCTION_PATTERN = re.compile(
    r"^(?P<prefix>\s*FROM\s+)"
    r"(?P<platform>--platform=\S+\s+)?"
    r"(?P<image>\S+)"
    r"(?P<suffix>.*)$",
    flags=re.IGNORECASE,
)

if TYPE_CHECKING:
    import docker
    from docker.models.images import Image


@dataclass(frozen=True)
class _FrozenDockerContext:
    """Immutable Docker context bytes and the hash of those exact bytes."""

    data: bytes
    sha256: str

    def open(self) -> io.BytesIO:
        return io.BytesIO(self.data)


def _stat_signature(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _context_changed(source: Path) -> ValueError:
    return ValueError(f"Docker context changed while freezing: {source}")


def _assert_context_path_unchanged(
    source: Path,
    before: os.stat_result,
) -> None:
    try:
        after = source.lstat()
    except OSError as error:
        raise _context_changed(source) from error
    if _stat_signature(before) != _stat_signature(after):
        raise _context_changed(source)


def _canonical_tar_info(
    arc_name: str,
    source_stat: os.stat_result,
) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name=arc_name)
    info.mode = stat.S_IMODE(source_stat.st_mode)
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mtime = 0
    return info


def _directory_children(source: Path) -> tuple[str, ...]:
    try:
        return tuple(sorted(child.name for child in source.iterdir()))
    except OSError as error:
        raise _context_changed(source) from error


def _add_context_path(
    archive: tarfile.TarFile,
    source: Path,
    arc_name: str,
) -> None:
    """Add one path while detecting replacement or concurrent mutation."""
    try:
        before = source.lstat()
    except OSError as error:
        raise ValueError(
            f"Could not read Docker context path: {source}"
        ) from error

    info = _canonical_tar_info(arc_name, before)
    if stat.S_ISDIR(before.st_mode):
        children = _directory_children(source)
        info.type = tarfile.DIRTYPE
        archive.addfile(info)
        for child_name in children:
            _add_context_path(
                archive,
                source / child_name,
                f"{arc_name}/{child_name}",
            )
        if children != _directory_children(source):
            raise _context_changed(source)
        _assert_context_path_unchanged(source, before)
        return

    if stat.S_ISLNK(before.st_mode):
        try:
            link_target = str(source.readlink())
        except OSError as error:
            raise _context_changed(source) from error
        info.type = tarfile.SYMTYPE
        info.linkname = link_target
        archive.addfile(info)
        _assert_context_path_unchanged(source, before)
        try:
            final_target = str(source.readlink())
        except OSError as error:
            raise _context_changed(source) from error
        if final_target != link_target:
            raise _context_changed(source)
        return

    if not stat.S_ISREG(before.st_mode):
        raise ValueError(f"Unsupported Docker context file type: {source}")

    try:
        with source.open("rb") as handle:
            opened_before = os.fstat(handle.fileno())
            if _stat_signature(before) != _stat_signature(opened_before):
                raise _context_changed(source)
            data = handle.read()
            opened_after = os.fstat(handle.fileno())
    except OSError as error:
        raise _context_changed(source) from error
    if _stat_signature(before) != _stat_signature(opened_after):
        raise _context_changed(source)
    info.type = tarfile.REGTYPE
    info.size = len(data)
    archive.addfile(info, io.BytesIO(data))
    _assert_context_path_unchanged(source, before)


def _safe_context_arc_name(name: str) -> str:
    pure_name = PurePosixPath(name)
    if (
        pure_name.is_absolute()
        or not pure_name.parts
        or "\\" in name
        or any(part in {"", ".", ".."} for part in pure_name.parts)
    ):
        raise ValueError(f"Unsafe Docker context archive name: {name!r}")
    return pure_name.as_posix()


def _freeze_docker_context(
    docker_file: str,
    extra_arcs: dict[str, Path] | None = None,
) -> _FrozenDockerContext:
    context_tar = io.BytesIO()
    with tarfile.open(
        fileobj=context_tar,
        mode="w",
        format=tarfile.PAX_FORMAT,
    ) as archive:
        dockerfile_data = docker_file.encode("utf-8")
        dockerfile_info = tarfile.TarInfo(name="Dockerfile")
        dockerfile_info.mode = 0o644
        dockerfile_info.uid = 0
        dockerfile_info.gid = 0
        dockerfile_info.uname = ""
        dockerfile_info.gname = ""
        dockerfile_info.mtime = 0
        dockerfile_info.size = len(dockerfile_data)
        archive.addfile(dockerfile_info, io.BytesIO(dockerfile_data))

        for arc_name, arc_path in sorted((extra_arcs or {}).items()):
            safe_arc_name = _safe_context_arc_name(arc_name)
            logger.debug(
                "Adding extra arc",
                arc_name=safe_arc_name,
                arc_path=arc_path,
            )
            _add_context_path(archive, arc_path, safe_arc_name)

    data = context_tar.getvalue()
    return _FrozenDockerContext(
        data=data,
        sha256=hashlib.sha256(data).hexdigest(),
    )


def _make_docker_context(
    docker_file: str, extra_arcs: dict[str, Path] | None = None
) -> io.BytesIO:
    logger.debug(
        "Making docker context", docker_file=docker_file, extra_arcs=extra_arcs
    )

    return _freeze_docker_context(docker_file, extra_arcs).open()


def _find_image(image_name: str, client: docker.DockerClient) -> Image | None:
    logger.info("Checking if image exists", image_name=image_name)
    try:
        found_image = client.images.get(image_name)
    except ImageNotFound:
        logger.info("Image does not exist", image_name=image_name)
        return None
    logger.debug(
        "Found image", image_name=image_name, found_image=found_image.id
    )
    return found_image


def _build_image(
    image_name: str,
    client: docker.DockerClient,
    context_tar: io.BytesIO,
    *,
    labels: dict[str, str] | None = None,
) -> Image:
    logger.info("Building image", image_name=image_name)
    try:
        build_options: dict[str, Any] = {}
        if labels is not None:
            build_options["labels"] = labels
        image, build_logs = client.images.build(
            fileobj=context_tar,
            custom_context=True,
            tag=image_name,
            rm=True,
            **build_options,
        )
    except ImageNotFound:
        logger.warning(
            "Docker SDK could not inspect the built image by id; "
            "falling back to tag lookup",
            image_name=image_name,
        )
        return client.images.get(image_name)
    except BuildError as e:
        logger.error("Failed to build image", image_name=image_name, error=e)
        raise e
    for msg in build_logs:
        logger.debug(msg)
    logger.info(
        "Finished building image", image_name=image_name, image_id=image.id
    )
    return image


def get_submission_image_name(submission_path: Path) -> str:
    hashed_submission = hashlib.sha256(
        submission_path.as_posix().encode("utf-8")
    ).hexdigest()[:8]
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    return f"{IMAGE_NAME_PREFIX}:submission-{hashed_submission}-{timestamp}"


def _render_base_image(
    environment_spec: DockerEnvironmentSpec,
    base_image: str | None = None,
) -> str:
    return Template(BASE_IMAGE_TEMPLATE.read_text()).render(
        base_image=base_image or environment_spec.docker.image,
        env=environment_spec.environment.env,
    )


def _get_base_image_hash(
    environment_spec: DockerEnvironmentSpec,
    upstream_image_id: str | None = None,
) -> str:
    dockerfile = _render_base_image(
        environment_spec,
        base_image=upstream_image_id,
    )
    digest = hashlib.sha256()
    digest.update(dockerfile.encode("utf-8"))
    digest.update(b"\0")
    if upstream_image_id is not None:
        digest.update(upstream_image_id.encode("utf-8"))
    return digest.hexdigest()[:12]


def _get_image_labels(image: Image) -> dict[str, str]:
    attrs = getattr(image, "attrs", {})
    if not isinstance(attrs, dict):
        return {}
    config = attrs.get("Config")
    if not isinstance(config, dict):
        return {}
    labels = config.get("Labels")
    if not isinstance(labels, dict):
        return {}
    return {str(key): str(value) for key, value in labels.items()}


def _base_image_is_current(
    image: Image,
    environment_spec: DockerEnvironmentSpec,
    upstream_image_id: str,
) -> bool:
    labels = _get_image_labels(image)
    return labels.get(BASE_IMAGE_HASH_LABEL) == _get_base_image_hash(
        environment_spec,
        upstream_image_id,
    ) and labels.get(UPSTREAM_IMAGE_ID_LABEL) == upstream_image_id


def _resolve_upstream_image_id(
    client: docker.DockerClient,
    image_name: str,
) -> str:
    """Pull a mutable upstream reference and return its platform image ID.

    If the registry is temporarily unavailable, a locally cached copy keeps
    offline runs possible. The warning makes that weaker freshness guarantee
    explicit in logs and provenance still records the resolved local ID.
    """
    try:
        image = client.images.pull(image_name)
    except DockerException as pull_error:
        try:
            image = client.images.get(image_name)
        except ImageNotFound:
            raise pull_error
        logger.warning(
            "Could not refresh upstream image; using local cached parent",
            image_name=image_name,
            error=str(pull_error),
        )
    return str(image.id)


def _get_image_architecture(image: Image) -> str:
    attrs = getattr(image, "attrs", {})
    architecture = (
        attrs.get("Architecture") if isinstance(attrs, dict) else None
    )
    if not isinstance(architecture, str):
        image.reload()
        attrs = getattr(image, "attrs", {})
        architecture = (
            attrs.get("Architecture") if isinstance(attrs, dict) else None
        )
    if not isinstance(architecture, str) or not architecture:
        raise ValueError(
            f"Could not determine architecture for prebuilt image {image.id}"
        )
    return architecture


def _resolve_prebuilt_image(
    client: docker.DockerClient,
    environment_spec: DockerEnvironmentSpec,
    image_name: str,
) -> Image:
    """Resolve, validate, and locally tag an immutable prebuilt base image."""
    reference = environment_spec.docker.prebuilt_image
    expected_id = environment_spec.docker.expected_image_id
    expected_architecture = environment_spec.docker.expected_architecture
    if (
        reference is None
        or expected_id is None
        or expected_architecture is None
    ):
        raise ValueError("Prebuilt image lock is incomplete")

    try:
        image = client.images.get(reference)
    except ImageNotFound as get_error:
        if reference.startswith("sha256:"):
            raise ValueError(
                f"Prebuilt image {reference} is not loaded. Load the verified "
                "image archive before running this environment."
            ) from get_error
        try:
            image = client.images.pull(reference)
        except DockerException as pull_error:
            raise ValueError(
                f"Could not resolve prebuilt image {reference}"
            ) from pull_error

    actual_id = str(image.id)
    if actual_id != expected_id:
        raise ValueError(
            "Prebuilt image ID mismatch: "
            f"expected {expected_id}, got {actual_id}"
        )
    actual_architecture = _get_image_architecture(image)
    if actual_architecture != expected_architecture:
        raise ValueError(
            "Prebuilt image architecture mismatch: "
            f"expected {expected_architecture}, got {actual_architecture}"
        )

    image.tag(image_name)
    logger.info(
        "Using verified prebuilt base image",
        reference=reference,
        image_name=image_name,
        image_id=actual_id,
        architecture=actual_architecture,
    )
    return image


def _get_image_spec_hash(
    dockerfile: str,
    parent_image_id: str | None,
) -> str:
    """Hash an image recipe together with its resolved parent image."""
    digest = hashlib.sha256()
    digest.update(dockerfile.encode("utf-8"))
    digest.update(b"\0")
    if parent_image_id is not None:
        digest.update(parent_image_id.encode("utf-8"))
    return digest.hexdigest()[:12]


def _image_from_str_is_current(
    image: Image,
    dockerfile: str,
    parent_image_id: str | None,
) -> bool:
    labels = _get_image_labels(image)
    if labels.get(IMAGE_SPEC_HASH_LABEL) != _get_image_spec_hash(
        dockerfile,
        parent_image_id,
    ):
        return False
    if parent_image_id is None:
        return True
    return labels.get(PARENT_IMAGE_ID_LABEL) == parent_image_id


def _label_image_from_str(
    dockerfile: str,
    parent_image_id: str | None,
) -> str:
    labels = [
        "LABEL "
        f"{IMAGE_SPEC_HASH_LABEL}="
        f"{json.dumps(_get_image_spec_hash(dockerfile, parent_image_id))}"
    ]
    if parent_image_id is not None:
        labels.append(
            f"LABEL {PARENT_IMAGE_ID_LABEL}="
            f"{json.dumps(parent_image_id)}"
        )
    rendered_labels = "\n".join(labels)
    return f"{dockerfile.rstrip()}\n\n{rendered_labels}\n"


def _pin_first_from_image(
    dockerfile: str,
    parent_image_id: str | None,
) -> str:
    """Make the first build stage consume the resolved immutable parent.

    ``parent_image_id`` used to participate only in labels and cache keys. That
    records intent but does not prevent a mutable local tag in ``FROM`` from
    changing before Docker reads the context. Rewriting the actual instruction
    closes that gap while preserving an optional platform flag and stage alias.
    """
    if parent_image_id is None:
        return dockerfile

    lines = dockerfile.splitlines(keepends=True)
    for index, line in enumerate(lines):
        line_body = line.rstrip("\r\n")
        newline = line[len(line_body) :]
        match = FROM_INSTRUCTION_PATTERN.match(line_body)
        if match is None:
            continue
        platform_flag = match.group("platform") or ""
        lines[index] = (
            f"{match.group('prefix')}{platform_flag}{parent_image_id}"
            f"{match.group('suffix')}{newline}"
        )
        return "".join(lines)

    raise ValueError(
        "A parent_image_id was supplied but the Dockerfile has no FROM "
        "instruction to pin"
    )


def make_base_image(
    environment_spec: DockerEnvironmentSpec,
    upstream_image_id: str | None = None,
) -> str:
    # The resolved ID is not merely provenance: it is the image Docker must
    # actually consume. Otherwise another process can retag a mutable parent
    # between resolution and build while the resulting labels claim the old ID.
    dockerfile = _render_base_image(
        environment_spec,
        base_image=upstream_image_id,
    ).rstrip()
    labels = [
        f'LABEL {BASE_IMAGE_HASH_LABEL}="'
        f'{_get_base_image_hash(environment_spec, upstream_image_id)}"'
    ]
    if upstream_image_id is not None:
        labels.append(
            f'LABEL {UPSTREAM_IMAGE_ID_LABEL}="{upstream_image_id}"'
        )
    rendered_labels = "\n".join(labels)
    return f"{dockerfile}\n\n{rendered_labels}\n"


def make_submission_docker_file(
    env_spec: DockerEnvironmentSpec,
    base_image: str,
    static_assets: dict[str, ResolvedStaticAsset],
) -> str:
    lines = [
        f"FROM {base_image}",
        "",
        "# Set up workspace directory",
        f"RUN mkdir -p {env_spec.docker.workdir}",
    ]

    lines.append("# Copy static assets")
    if static_assets:
        lines.append("RUN mkdir -p /static")
        for asset in static_assets.values():
            lines.append(
                f"COPY --chown={AGENT_USER} {asset.save_path} /static/{str(asset.save_path)}"
            )

    lines.append("")
    lines.append("# Copy submission")
    lines.append(f"COPY --chown={AGENT_USER} submission submission")
    lines.append("ENV SUBMISSION_PATH=/submission")
    lines.append("WORKDIR /submission")

    lines.append(
        f"RUN {' && '.join(env_spec.get_setup_commands(is_evaluation=True))}"
    )

    lines.append(f"RUN chown -R {AGENT_USER} /submission")
    lines.append(f"WORKDIR {env_spec.docker.workdir}")
    lines.append(f"RUN chown -R {AGENT_USER} {env_spec.docker.workdir}")
    lines.append(f"RUN chown -R {AGENT_USER} /tmp")
    lines.append(f"USER {AGENT_USER}")

    return "\n".join(lines)


def build_base_image(
    client: docker.DockerClient,
    environment_spec: EnvironmentSpec,
    force_build: bool = False,  # noqa: FBT001, FBT002
) -> Image:
    if not isinstance(environment_spec, DockerEnvironmentSpec):
        raise ValueError("Environment spec must be a DockerEnvironmentSpec")

    image_name = environment_spec.get_base_image()
    logger.info(
        f"Building base image for environment '{environment_spec.name}'",
        image_name=image_name,
        base_image=environment_spec.docker.image,
    )
    if environment_spec.docker.prebuilt_image is not None:
        return _resolve_prebuilt_image(
            client,
            environment_spec,
            image_name,
        )

    upstream_image_id = _resolve_upstream_image_id(
        client,
        environment_spec.docker.image,
    )
    found_image = _find_image(image_name, client)

    if found_image is not None and not force_build:
        if _base_image_is_current(
            found_image,
            environment_spec,
            upstream_image_id,
        ):
            logger.info("Found current image and not forcing build")
            return found_image
        logger.info(
            "Found stale image; rebuilding",
            image_name=image_name,
        )

    context_tar = _make_docker_context(
        make_base_image(environment_spec, upstream_image_id)
    )
    return _build_image(image_name, client, context_tar)


def build_submission_image(
    client: docker.DockerClient,
    submission_path: Path,
    environment_spec: EnvironmentSpec,
    static_assets: dict[str, ResolvedStaticAsset],
    use_name: str | None = None,
    *,
    build_base: bool = False,
    force_build: bool = False,
) -> tuple[str, Image]:
    if not isinstance(environment_spec, DockerEnvironmentSpec):
        raise ValueError("Environment spec must be a DockerEnvironmentSpec")

    logger.info(
        f"Building submission image for {environment_spec.name}",
        submission_path=submission_path,
        static_assets=list(static_assets.keys()),
        build_base=build_base,
    )
    base_image = environment_spec.get_base_image()

    found_base_image = _find_image(base_image, client)
    if (
        found_base_image is None
        and environment_spec.docker.prebuilt_image is None
        and not build_base
        and not force_build
    ):
        raise ValueError(f"Base image {base_image} does not exist")

    # Resolve the upstream parent again even when a base tag already exists.
    # Validating an image against the parent ID stored on that same image would
    # let a mutable upstream reference self-certify forever.
    refreshed_base = build_base_image(
        client,
        environment_spec,
        force_build=force_build,
    )
    refreshed_base_id = str(refreshed_base.id)

    image_name = use_name or get_submission_image_name(submission_path)
    dockerfile = make_submission_docker_file(
        environment_spec,
        refreshed_base_id,
        static_assets,
    )
    context_arcs = {
        "submission": submission_path,
        **{
            asset.save_path.as_posix(): asset.absolute_path
            for asset in static_assets.values()
        },
    }
    frozen_context = _freeze_docker_context(
        _label_image_from_str(dockerfile, refreshed_base_id),
        context_arcs,
    )
    context_hash = frozen_context.sha256
    found_image = _find_image(image_name, client)
    if found_image is not None and not force_build:
        labels = _get_image_labels(found_image)
        if (
            _image_from_str_is_current(
                found_image,
                dockerfile,
                refreshed_base_id,
            )
            and labels.get(BUILD_CONTEXT_HASH_LABEL) == context_hash
        ):
            logger.info("Found current submission image and not forcing build")
            return image_name, found_image
        logger.info(
            "Found stale submission image; rebuilding",
            image_name=image_name,
        )

    return image_name, _build_image(
        image_name,
        client,
        frozen_context.open(),
        labels={BUILD_CONTEXT_HASH_LABEL: context_hash},
    )


def build_image_from_str(
    client: docker.DockerClient,
    image_name: str,
    dockerfile: str,
    *,
    force_build: bool = False,
    parent_image_id: str | None = None,
) -> Image:
    pinned_dockerfile = _pin_first_from_image(
        dockerfile,
        parent_image_id,
    )
    found_image = _find_image(image_name, client)
    if found_image is not None and not force_build:
        if _image_from_str_is_current(
            found_image,
            pinned_dockerfile,
            parent_image_id,
        ):
            logger.info("Found current image and not forcing build")
            return found_image
        logger.info(
            "Found stale dependent image; rebuilding",
            image_name=image_name,
        )

    context_tar = _make_docker_context(
        _label_image_from_str(pinned_dockerfile, parent_image_id)
    )
    return _build_image(image_name, client, context_tar)
