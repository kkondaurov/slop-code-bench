from __future__ import annotations

import subprocess
from pathlib import Path

import yaml

EXPECTED_IMAGE_ID = (
    "sha256:f92550022dbc45c417e0c5bfcab706411b7407881ffd2d9b74d4e2049bbce985"
)
EXPECTED_ARCHIVE_SHA256 = (
    "eac1f5965f0c563529861a8b7b62fd2e2c0de6f726ffcdf38bcd8538eb145a44"
)
EXPECTED_RELEASE_TAG = "scbench-v2.1-repro.1"
EXPECTED_RELEASE_ROOT = (
    "https://github.com/kkondaurov/slop-code-bench/releases"
)


def test_scbench_v2_1_base_loader_matches_environment_lock() -> None:
    repository = Path(__file__).parents[2]
    script = repository / "scripts/load_scbench_v2_1_base.sh"
    environment_path = (
        repository
        / "configs"
        / "environments"
        / "docker-python3.12-uv-scb-v2.1.yaml"
    )
    environment = yaml.safe_load(environment_path.read_text())
    manifest = yaml.safe_load(
        (repository / "configs/scbench-v2/manifest.yaml").read_text()
    )
    script_text = script.read_text()

    subprocess.run(  # noqa: S603
        ["/bin/bash", "-n", str(script)],
        check=True,
    )

    docker = environment["docker"]
    assert docker["prebuilt_image"] == EXPECTED_IMAGE_ID
    assert docker["expected_image_id"] == EXPECTED_IMAGE_ID
    assert docker["expected_architecture"] == "arm64"
    prebuilt = manifest["protocol"]["prebuilt_base"]
    assert prebuilt["image_id"] == EXPECTED_IMAGE_ID
    assert prebuilt["architecture"] == "arm64"
    assert prebuilt["bundled_tools"]["node"]["version"] == "22.21.1"
    assert prebuilt["bundled_tools"]["tsx"]["version"] == "4.23.1"
    assert prebuilt["bundled_tools"]["typescript"]["version"] == "7.0.2"
    assert prebuilt["archive"]["sha256"] == EXPECTED_ARCHIVE_SHA256
    assert prebuilt["archive"]["loader"] == (
        "scripts/load_scbench_v2_1_base.sh"
    )
    assert EXPECTED_IMAGE_ID in script_text
    assert EXPECTED_ARCHIVE_SHA256 in script_text
    assert 'EXPECTED_PLATFORM="linux/arm64"' in script_text
    assert f'RELEASE_TAG="{EXPECTED_RELEASE_TAG}"' in script_text
    assert (
        f'RELEASE_URL="{EXPECTED_RELEASE_ROOT}/tag/${{RELEASE_TAG}}"'
        in script_text
    )
    assert (
        f'ARCHIVE_URL="{EXPECTED_RELEASE_ROOT}/download/'
        '${RELEASE_TAG}/${ARCHIVE_NAME}"'
        in script_text
    )
    assert (
        f'CHECKSUM_URL="{EXPECTED_RELEASE_ROOT}/download/'
        '${RELEASE_TAG}/release-assets-v2.1.sha256"'
        in script_text
    )
    assert 'EXPECTED_NODE_VERSION="v22.21.1"' in script_text
    assert 'EXPECTED_TSX_VERSION="tsx v4.23.1"' in script_text
    assert 'EXPECTED_TYPESCRIPT_VERSION="Version 7.0.2"' in script_text
    assert "tsx --eval" in script_text
    assert "--user 1000:1000" in script_text
    assert "zstd --test" in script_text
    assert "| docker load" in script_text
