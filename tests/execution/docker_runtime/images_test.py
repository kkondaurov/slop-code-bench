"""Tests for Docker image build caching."""

from __future__ import annotations

import hashlib
import io
import tarfile
from pathlib import Path
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from docker.errors import ImageNotFound
from pydantic import ValidationError

from slop_code.execution.docker_runtime.images import BASE_IMAGE_HASH_LABEL
from slop_code.execution.docker_runtime.images import BUILD_CONTEXT_HASH_LABEL
from slop_code.execution.docker_runtime.images import IMAGE_SPEC_HASH_LABEL
from slop_code.execution.docker_runtime.images import PARENT_IMAGE_ID_LABEL
from slop_code.execution.docker_runtime.images import UPSTREAM_IMAGE_ID_LABEL
from slop_code.execution.docker_runtime.images import _build_image
from slop_code.execution.docker_runtime.images import _freeze_docker_context
from slop_code.execution.docker_runtime.images import _get_base_image_hash
from slop_code.execution.docker_runtime.images import _get_image_spec_hash
from slop_code.execution.docker_runtime.images import _label_image_from_str
from slop_code.execution.docker_runtime.images import _pin_first_from_image
from slop_code.execution.docker_runtime.images import build_base_image
from slop_code.execution.docker_runtime.images import build_image_from_str
from slop_code.execution.docker_runtime.images import build_submission_image
from slop_code.execution.docker_runtime.images import make_base_image
from slop_code.execution.docker_runtime.images import (
    make_submission_docker_file,
)
from slop_code.execution.docker_runtime.models import DockerEnvironmentSpec

PREBUILT_IMAGE_ID = (
    "sha256:d2b862aad2bf40fe80573d0facc462608ce2a2fe76b56927a36050dc02a44f14"
)


def _mock_image(
    hash_value: str | None,
    upstream_image_id: str | None = None,
) -> MagicMock:
    image = MagicMock()
    labels = {} if hash_value is None else {BASE_IMAGE_HASH_LABEL: hash_value}
    if upstream_image_id is not None:
        labels[UPSTREAM_IMAGE_ID_LABEL] = upstream_image_id
    image.attrs = {"Config": {"Labels": labels}}
    return image


def _mock_submission_image(
    *,
    dockerfile: str,
    parent_image_id: str,
    context_hash: str,
) -> MagicMock:
    image = MagicMock()
    image.attrs = {
        "Config": {
            "Labels": {
                IMAGE_SPEC_HASH_LABEL: _get_image_spec_hash(
                    dockerfile,
                    parent_image_id,
                ),
                PARENT_IMAGE_ID_LABEL: parent_image_id,
                BUILD_CONTEXT_HASH_LABEL: context_hash,
            }
        }
    }
    return image


def _submission_context_hash(
    dockerfile: str,
    parent_image_id: str,
    submission_path: Path,
) -> str:
    return _freeze_docker_context(
        _label_image_from_str(dockerfile, parent_image_id),
        {"submission": submission_path},
    ).sha256


def _prebuilt_spec(
    docker_spec: DockerEnvironmentSpec,
    *,
    reference: str = PREBUILT_IMAGE_ID,
    expected_image_id: str = PREBUILT_IMAGE_ID,
    expected_architecture: str = "arm64",
) -> DockerEnvironmentSpec:
    payload = docker_spec.model_dump()
    payload["docker"].update(
        {
            "prebuilt_image": reference,
            "expected_image_id": expected_image_id,
            "expected_architecture": expected_architecture,
        }
    )
    return DockerEnvironmentSpec.model_validate(payload)


def test_prebuilt_image_config_requires_complete_immutable_lock(
    docker_spec: DockerEnvironmentSpec,
) -> None:
    payload = docker_spec.model_dump()
    payload["docker"]["prebuilt_image"] = PREBUILT_IMAGE_ID

    with pytest.raises(ValidationError, match="configured together"):
        DockerEnvironmentSpec.model_validate(payload)

    payload["docker"].update(
        {
            "prebuilt_image": "registry.example/base:latest",
            "expected_image_id": PREBUILT_IMAGE_ID,
            "expected_architecture": "arm64",
        }
    )
    with pytest.raises(ValidationError, match="immutable registry digest"):
        DockerEnvironmentSpec.model_validate(payload)


def test_build_base_image_uses_loaded_verified_prebuilt_image(
    docker_spec: DockerEnvironmentSpec,
) -> None:
    spec = _prebuilt_spec(docker_spec)
    client = MagicMock()
    image = MagicMock(
        id=PREBUILT_IMAGE_ID,
        attrs={"Architecture": "arm64"},
    )
    client.images.get.return_value = image

    with patch(
        "slop_code.execution.docker_runtime.images._build_image"
    ) as build:
        result = build_base_image(client, spec)

    assert result is image
    client.images.get.assert_called_once_with(PREBUILT_IMAGE_ID)
    client.images.pull.assert_not_called()
    image.tag.assert_called_once_with(spec.get_base_image())
    build.assert_not_called()


def test_build_base_image_pulls_digest_pinned_prebuilt_reference(
    docker_spec: DockerEnvironmentSpec,
) -> None:
    reference = (
        "registry.example/base@"
        "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
    )
    spec = _prebuilt_spec(docker_spec, reference=reference)
    client = MagicMock()
    client.images.get.side_effect = ImageNotFound(reference)
    image = MagicMock(
        id=PREBUILT_IMAGE_ID,
        attrs={"Architecture": "arm64"},
    )
    client.images.pull.return_value = image

    result = build_base_image(client, spec)

    assert result is image
    client.images.pull.assert_called_once_with(reference)
    image.tag.assert_called_once_with(spec.get_base_image())


def test_build_base_image_rejects_unloaded_local_prebuilt_image(
    docker_spec: DockerEnvironmentSpec,
) -> None:
    spec = _prebuilt_spec(docker_spec)
    client = MagicMock()
    client.images.get.side_effect = ImageNotFound(PREBUILT_IMAGE_ID)

    with pytest.raises(ValueError, match="not loaded"):
        build_base_image(client, spec)

    client.images.pull.assert_not_called()


def test_build_base_image_rejects_prebuilt_image_id_mismatch(
    docker_spec: DockerEnvironmentSpec,
) -> None:
    spec = _prebuilt_spec(docker_spec)
    client = MagicMock()
    client.images.get.return_value = MagicMock(
        id=f"sha256:{'c' * 64}",
        attrs={"Architecture": "arm64"},
    )

    with pytest.raises(ValueError, match="image ID mismatch"):
        build_base_image(client, spec)


@pytest.mark.parametrize("architecture", ["amd64", None])
def test_build_base_image_rejects_wrong_or_unknown_prebuilt_architecture(
    docker_spec: DockerEnvironmentSpec,
    architecture: str | None,
) -> None:
    spec = _prebuilt_spec(docker_spec)
    client = MagicMock()
    attrs = {} if architecture is None else {"Architecture": architecture}
    image = MagicMock(id=PREBUILT_IMAGE_ID, attrs=attrs)
    client.images.get.return_value = image

    expected_error = (
        "Could not determine architecture"
        if architecture is None
        else "architecture mismatch"
    )
    with pytest.raises(ValueError, match=expected_error):
        build_base_image(client, spec)

    image.tag.assert_not_called()


def test_build_base_image_reuses_current_image(
    docker_spec: DockerEnvironmentSpec,
) -> None:
    client = MagicMock()
    upstream_image_id = "sha256:upstream-v1"
    client.images.pull.return_value.id = upstream_image_id
    current_hash = _get_base_image_hash(docker_spec, upstream_image_id)
    current_image = _mock_image(current_hash, upstream_image_id)

    with (
        patch(
            "slop_code.execution.docker_runtime.images._find_image",
            return_value=current_image,
        ),
        patch(
            "slop_code.execution.docker_runtime.images._build_image"
        ) as build,
    ):
        result = build_base_image(client, docker_spec)

    assert result is current_image
    build.assert_not_called()


def test_build_base_image_rebuilds_stale_image(
    docker_spec: DockerEnvironmentSpec,
) -> None:
    client = MagicMock()
    client.images.pull.return_value.id = "sha256:upstream-v1"
    stale_image = _mock_image(None)
    rebuilt_image = MagicMock()

    with (
        patch(
            "slop_code.execution.docker_runtime.images._find_image",
            return_value=stale_image,
        ),
        patch(
            "slop_code.execution.docker_runtime.images._build_image",
            return_value=rebuilt_image,
        ) as build,
    ):
        result = build_base_image(client, docker_spec)

    assert result is rebuilt_image
    build.assert_called_once()


def test_build_base_image_rebuilds_when_mutable_parent_tag_changes(
    docker_spec: DockerEnvironmentSpec,
) -> None:
    client = MagicMock()
    old_upstream_id = "sha256:upstream-v1"
    new_upstream_id = "sha256:upstream-v2"
    client.images.pull.return_value.id = new_upstream_id
    cached_image = _mock_image(
        _get_base_image_hash(docker_spec, old_upstream_id),
        old_upstream_id,
    )
    rebuilt_image = MagicMock()

    with (
        patch(
            "slop_code.execution.docker_runtime.images._find_image",
            return_value=cached_image,
        ),
        patch(
            "slop_code.execution.docker_runtime.images._build_image",
            return_value=rebuilt_image,
        ) as build,
    ):
        result = build_base_image(client, docker_spec)

    assert result is rebuilt_image
    dockerfile_context = build.call_args.args[2]
    with tarfile.open(fileobj=dockerfile_context, mode="r") as archive:
        assert set(archive.getnames()) == {
            "Dockerfile",
            "base_node_tools/package.json",
            "base_node_tools/package-lock.json",
        }
        dockerfile = archive.extractfile("Dockerfile")
        assert dockerfile is not None
        rendered = dockerfile.read().decode()
    assert f'{UPSTREAM_IMAGE_ID_LABEL}="{new_upstream_id}"' in rendered
    assert rendered.splitlines()[0] == f"FROM {new_upstream_id}"
    assert (
        _get_base_image_hash(docker_spec, new_upstream_id) in rendered
    )


def test_build_submission_image_revalidates_existing_base_image(
    docker_spec: DockerEnvironmentSpec,
    tmp_path: Path,
) -> None:
    client = MagicMock()
    submission_path = tmp_path / "submission"
    submission_path.mkdir()
    (submission_path / "solution.py").write_text("print('ok')\n")

    stale_base_image = _mock_image(None)
    submission_image = MagicMock()

    with (
        patch(
            "slop_code.execution.docker_runtime.images._find_image",
            side_effect=[stale_base_image, None],
        ),
        patch(
            "slop_code.execution.docker_runtime.images.build_base_image",
            return_value=MagicMock(),
        ) as build_base,
        patch(
            "slop_code.execution.docker_runtime.images._build_image",
            return_value=submission_image,
        ),
    ):
        image_name, built_image = build_submission_image(
            client,
            submission_path,
            docker_spec,
            {},
            build_base=False,
        )

    assert image_name.startswith("slop-code:submission-")
    assert built_image is submission_image
    build_base.assert_called_once_with(
        client,
        docker_spec,
        force_build=False,
    )


def test_build_submission_image_reuses_content_and_parent_matched_cache(
    docker_spec: DockerEnvironmentSpec,
    tmp_path: Path,
) -> None:
    client = MagicMock()
    submission_path = tmp_path / "submission"
    submission_path.mkdir()
    (submission_path / "solution.py").write_text("print('ok')\n")
    parent_id = "sha256:refreshed-base"
    refreshed_base = MagicMock(id=parent_id)
    dockerfile = make_submission_docker_file(
        docker_spec,
        parent_id,
        {},
    )
    context_hash = _submission_context_hash(
        dockerfile,
        parent_id,
        submission_path,
    )
    cached = _mock_submission_image(
        dockerfile=dockerfile,
        parent_image_id=parent_id,
        context_hash=context_hash,
    )

    with (
        patch(
            "slop_code.execution.docker_runtime.images._find_image",
            side_effect=[MagicMock(), cached],
        ),
        patch(
            "slop_code.execution.docker_runtime.images.build_base_image",
            return_value=refreshed_base,
        ),
        patch(
            "slop_code.execution.docker_runtime.images._build_image"
        ) as build,
    ):
        _, result = build_submission_image(
            client,
            submission_path,
            docker_spec,
            {},
            use_name="slop-code:fixed-submission",
        )

    assert result is cached
    build.assert_not_called()


def test_build_submission_image_rebuilds_cache_from_old_parent(
    docker_spec: DockerEnvironmentSpec,
    tmp_path: Path,
) -> None:
    client = MagicMock()
    submission_path = tmp_path / "submission"
    submission_path.mkdir()
    (submission_path / "solution.py").write_text("print('ok')\n")
    refreshed_parent_id = "sha256:refreshed-base"
    refreshed_base = MagicMock(id=refreshed_parent_id)
    dockerfile = make_submission_docker_file(
        docker_spec,
        refreshed_parent_id,
        {},
    )
    cached = _mock_submission_image(
        dockerfile=dockerfile,
        parent_image_id="sha256:old-base",
        context_hash=_submission_context_hash(
            dockerfile,
            "sha256:old-base",
            submission_path,
        ),
    )
    rebuilt = MagicMock()

    with (
        patch(
            "slop_code.execution.docker_runtime.images._find_image",
            side_effect=[MagicMock(), cached],
        ),
        patch(
            "slop_code.execution.docker_runtime.images.build_base_image",
            return_value=refreshed_base,
        ),
        patch(
            "slop_code.execution.docker_runtime.images._build_image",
            return_value=rebuilt,
        ) as build,
    ):
        _, result = build_submission_image(
            client,
            submission_path,
            docker_spec,
            {},
            use_name="slop-code:fixed-submission",
        )

    assert result is rebuilt
    build.assert_called_once()


def test_build_submission_image_rebuilds_cache_when_content_changes(
    docker_spec: DockerEnvironmentSpec,
    tmp_path: Path,
) -> None:
    client = MagicMock()
    submission_path = tmp_path / "submission"
    submission_path.mkdir()
    solution = submission_path / "solution.py"
    solution.write_text("print('first')\n")
    parent_id = "sha256:refreshed-base"
    refreshed_base = MagicMock(id=parent_id)
    dockerfile = make_submission_docker_file(
        docker_spec,
        parent_id,
        {},
    )
    cached = _mock_submission_image(
        dockerfile=dockerfile,
        parent_image_id=parent_id,
        context_hash=_submission_context_hash(
            dockerfile,
            parent_id,
            submission_path,
        ),
    )
    solution.write_text("print('changed')\n")
    rebuilt = MagicMock()

    with (
        patch(
            "slop_code.execution.docker_runtime.images._find_image",
            side_effect=[MagicMock(), cached],
        ),
        patch(
            "slop_code.execution.docker_runtime.images.build_base_image",
            return_value=refreshed_base,
        ),
        patch(
            "slop_code.execution.docker_runtime.images._build_image",
            return_value=rebuilt,
        ) as build,
    ):
        _, result = build_submission_image(
            client,
            submission_path,
            docker_spec,
            {},
            use_name="slop-code:fixed-submission",
        )

    assert result is rebuilt
    build.assert_called_once()


def test_build_submission_image_hashes_exact_bytes_sent_to_docker(
    docker_spec: DockerEnvironmentSpec,
    tmp_path: Path,
) -> None:
    client = MagicMock()
    submission_path = tmp_path / "submission"
    submission_path.mkdir()
    solution = submission_path / "solution.py"
    solution.write_text("print('frozen')\n")
    refreshed_base = MagicMock(id="sha256:refreshed-base")
    built_image = MagicMock()

    def find_image(
        image_name: str, unused_client: MagicMock
    ) -> MagicMock | None:
        del unused_client
        if image_name == docker_spec.get_base_image():
            return MagicMock()
        # This mutation happens after the context is frozen but before build.
        solution.write_text("print('late mutation')\n")
        return None

    with (
        patch(
            "slop_code.execution.docker_runtime.images._find_image",
            side_effect=find_image,
        ),
        patch(
            "slop_code.execution.docker_runtime.images.build_base_image",
            return_value=refreshed_base,
        ),
        patch(
            "slop_code.execution.docker_runtime.images._build_image",
            return_value=built_image,
        ) as build,
    ):
        _, result = build_submission_image(
            client,
            submission_path,
            docker_spec,
            {},
            use_name="slop-code:fixed-submission",
        )

    assert result is built_image
    context = build.call_args.args[2]
    context_bytes = context.getvalue()
    assert build.call_args.kwargs["labels"] == {
        BUILD_CONTEXT_HASH_LABEL: hashlib.sha256(context_bytes).hexdigest()
    }
    with tarfile.open(fileobj=io.BytesIO(context_bytes), mode="r:") as archive:
        frozen_dockerfile = archive.extractfile("Dockerfile")
        assert frozen_dockerfile is not None
        assert frozen_dockerfile.read().decode().splitlines()[0] == (
            f"FROM {refreshed_base.id}"
        )
        frozen_solution = archive.extractfile("submission/solution.py")
        assert frozen_solution is not None
        assert frozen_solution.read() == b"print('frozen')\n"


def test_freeze_docker_context_rejects_directory_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    submission_path = tmp_path / "submission"
    submission_path.mkdir()
    (submission_path / "solution.py").write_text("print('ok')\n")
    original_addfile = tarfile.TarFile.addfile

    def addfile_then_mutate(
        archive: tarfile.TarFile,
        tar_info: tarfile.TarInfo,
        fileobj=None,
    ) -> None:
        original_addfile(archive, tar_info, fileobj)
        if tar_info.name == "submission" and tar_info.isdir():
            (submission_path / "appeared-late.py").write_text("surprise\n")

    monkeypatch.setattr(tarfile.TarFile, "addfile", addfile_then_mutate)

    with pytest.raises(ValueError, match="changed while freezing"):
        _freeze_docker_context(
            "FROM scratch\n",
            {"submission": submission_path},
        )


def test_freeze_docker_context_rejects_directory_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    submission_path = tmp_path / "submission"
    submission_path.mkdir()
    solution = submission_path / "solution.py"
    solution.write_text("print('ok')\n")
    original_addfile = tarfile.TarFile.addfile

    def addfile_then_replace(
        archive: tarfile.TarFile,
        tar_info: tarfile.TarInfo,
        fileobj=None,
    ) -> None:
        original_addfile(archive, tar_info, fileobj)
        if tar_info.name == "submission" and tar_info.isdir():
            submission_path.rename(tmp_path / "replaced-submission")
            submission_path.mkdir()
            (submission_path / "solution.py").write_text("print('ok')\n")

    monkeypatch.setattr(tarfile.TarFile, "addfile", addfile_then_replace)

    with pytest.raises(ValueError, match="changed while freezing"):
        _freeze_docker_context(
            "FROM scratch\n",
            {"submission": submission_path},
        )


def test_build_image_falls_back_to_tag_lookup_after_sdk_image_not_found() -> (
    None
):
    client = MagicMock()
    tagged_image = MagicMock()
    client.images.build.side_effect = ImageNotFound("missing")
    client.images.get.return_value = tagged_image

    result = _build_image(
        "slop-code:test-image",
        client,
        io.BytesIO(b"docker context"),
    )

    assert result is tagged_image
    client.images.get.assert_called_once_with("slop-code:test-image")


def test_build_image_passes_external_labels_to_docker() -> None:
    client = MagicMock()
    image = MagicMock()
    client.images.build.return_value = (image, [])
    labels = {BUILD_CONTEXT_HASH_LABEL: "frozen-context-hash"}
    context = io.BytesIO(b"docker context")

    result = _build_image(
        "slop-code:test-image",
        client,
        context,
        labels=labels,
    )

    assert result is image
    client.images.build.assert_called_once_with(
        fileobj=context,
        custom_context=True,
        tag="slop-code:test-image",
        rm=True,
        labels=labels,
    )


def test_pin_first_from_image_preserves_platform_and_stage_alias() -> None:
    dockerfile = (
        "# syntax=docker/dockerfile:1\n"
        "FROM --platform=linux/arm64 mutable:latest AS agent\n"
        "RUN true\n"
    )

    pinned = _pin_first_from_image(
        dockerfile,
        "sha256:immutable-parent",
    )

    assert pinned.splitlines()[1] == (
        "FROM --platform=linux/arm64 sha256:immutable-parent AS agent"
    )


def test_pin_first_from_image_rejects_missing_from_instruction() -> None:
    with pytest.raises(ValueError, match="no FROM instruction"):
        _pin_first_from_image(
            "RUN true\n",
            "sha256:immutable-parent",
        )


def test_rendered_base_image_installs_expected_tools_and_native_minio(
    docker_spec: DockerEnvironmentSpec,
) -> None:
    dockerfile = make_base_image(docker_spec)

    assert "\n        git \\\n" in dockerfile
    assert "\n        ripgrep \\\n" in dockerfile
    assert "minio_release='RELEASE.2025-09-07T16-13-09Z'" in dockerfile
    assert 'minio_arch="$(dpkg --print-architecture)"' in dockerfile
    assert "linux-${minio_arch}/archive/minio.${minio_release}" in dockerfile
    assert "sha256sum -c -" in dockerfile
    assert "release/linux-amd64/minio" not in dockerfile
    assert "COPY base_node_tools/package.json" in dockerfile
    assert "npm ci --omit=dev --no-audit --no-fund" in dockerfile
    assert 'ENV TSX_VERSION=4.23.1' in dockerfile
    assert 'ENV TYPESCRIPT_VERSION=7.0.2' in dockerfile
    assert "/usr/local/bin/tsx" in dockerfile
    assert "/usr/local/bin/tsc" in dockerfile


def test_build_image_from_str_reuses_matching_parent_and_recipe() -> None:
    client = MagicMock()
    dockerfile = "FROM slop-code:python3.12\nRUN true\n"
    parent_image_id = "sha256:parent-v1"
    pinned_dockerfile = _pin_first_from_image(dockerfile, parent_image_id)
    current_image = MagicMock()
    current_image.attrs = {
        "Config": {
            "Labels": {
                IMAGE_SPEC_HASH_LABEL: _get_image_spec_hash(
                    pinned_dockerfile,
                    parent_image_id,
                ),
                PARENT_IMAGE_ID_LABEL: parent_image_id,
            }
        }
    }

    with (
        patch(
            "slop_code.execution.docker_runtime.images._find_image",
            return_value=current_image,
        ),
        patch(
            "slop_code.execution.docker_runtime.images._build_image"
        ) as build,
    ):
        result = build_image_from_str(
            client,
            "slop-code:test-agent",
            dockerfile,
            parent_image_id=parent_image_id,
        )

    assert result is current_image
    build.assert_not_called()


def test_build_image_from_str_rebuilds_when_parent_image_changes() -> None:
    client = MagicMock()
    dockerfile = "FROM slop-code:python3.12\nRUN true\n"
    old_parent_id = "sha256:parent-v1"
    new_parent_id = "sha256:parent-v2"
    stale_image = MagicMock()
    stale_image.attrs = {
        "Config": {
            "Labels": {
                IMAGE_SPEC_HASH_LABEL: _get_image_spec_hash(
                    dockerfile,
                    old_parent_id,
                ),
                PARENT_IMAGE_ID_LABEL: old_parent_id,
            }
        }
    }
    rebuilt_image = MagicMock()

    with (
        patch(
            "slop_code.execution.docker_runtime.images._find_image",
            return_value=stale_image,
        ),
        patch(
            "slop_code.execution.docker_runtime.images._make_docker_context",
            return_value=io.BytesIO(),
        ) as make_context,
        patch(
            "slop_code.execution.docker_runtime.images._build_image",
            return_value=rebuilt_image,
        ) as build,
    ):
        result = build_image_from_str(
            client,
            "slop-code:test-agent",
            dockerfile,
            parent_image_id=new_parent_id,
        )

    assert result is rebuilt_image
    rendered_dockerfile = make_context.call_args.args[0]
    assert rendered_dockerfile.splitlines()[0] == f"FROM {new_parent_id}"
    assert (
        f'LABEL {PARENT_IMAGE_ID_LABEL}="{new_parent_id}"'
        in rendered_dockerfile
    )
    pinned_dockerfile = _pin_first_from_image(dockerfile, new_parent_id)
    assert (
        f'LABEL {IMAGE_SPEC_HASH_LABEL}="'
        f'{_get_image_spec_hash(pinned_dockerfile, new_parent_id)}"'
        in rendered_dockerfile
    )
    build.assert_called_once_with(
        "slop-code:test-agent",
        client,
        make_context.return_value,
    )


def test_build_image_from_str_rebuilds_legacy_unlabelled_image() -> None:
    client = MagicMock()
    stale_image = MagicMock()
    stale_image.attrs = {"Config": {"Labels": {}}}
    rebuilt_image = MagicMock()

    with (
        patch(
            "slop_code.execution.docker_runtime.images._find_image",
            return_value=stale_image,
        ),
        patch(
            "slop_code.execution.docker_runtime.images._build_image",
            return_value=rebuilt_image,
        ) as build,
    ):
        result = build_image_from_str(
            client,
            "slop-code:test-agent",
            "FROM slop-code:python3.12\n",
            parent_image_id="sha256:parent-v1",
        )

    assert result is rebuilt_image
    build.assert_called_once()
