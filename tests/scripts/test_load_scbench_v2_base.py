from __future__ import annotations

import subprocess
from pathlib import Path

import yaml

EXPECTED_IMAGE_ID = (
    "sha256:d2b862aad2bf40fe80573d0facc462608ce2a2fe76b56927a36050dc02a44f14"
)
EXPECTED_ARCHIVE_SHA256 = (
    "dff52ff24d1d7e7d88525ef403b374d6464d3331c2b3df1ce72e4d56a3ec5df9"
)
EXPECTED_RELEASE_TAG = "scbench-v2-repro.1"
EXPECTED_RELEASE_ROOT = (
    "https://github.com/kkondaurov/slop-code-bench/releases"
)


def test_scbench_v2_base_loader_is_valid_and_matches_environment_lock() -> None:
    repository = Path(__file__).parents[2]
    script = repository / "scripts" / "load_scbench_v2_base.sh"
    environment_path = (
        repository
        / "configs"
        / "environments"
        / "docker-python3.12-uv-scb-v2.yaml"
    )
    environment = yaml.safe_load(environment_path.read_text())
    manifest = yaml.safe_load(
        (
            repository / "configs" / "scbench-v2" / "manifest.yaml"
        ).read_text()
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
    assert prebuilt["bundled_tools"]["minio"] == {
        "release": "RELEASE.2025-09-07T16-13-09Z",
        "architecture": "arm64",
        "sha256": (
            "5c83cd2cf151717ba0243f73e1c7802ff36e272b67144bdd7f1f7d684fd6f03d"
        ),
        "runtime": "linux/arm64",
    }
    assert prebuilt["archive"]["sha256"] == EXPECTED_ARCHIVE_SHA256
    assert prebuilt["archive"]["loader"] == "scripts/load_scbench_v2_base.sh"
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
        '${RELEASE_TAG}/release-assets.sha256"'
        in script_text
    )
    assert 'EXPECTED_MINIO_RELEASE="RELEASE.2025-09-07T16-13-09Z"' in script_text
    assert (
        'EXPECTED_MINIO_SHA256="'
        "5c83cd2cf151717ba0243f73e1c7802ff36e272b67144bdd7f1f7d684fd6f03d"
        '"'
    ) in script_text
    assert "zstd --test" in script_text
    assert "| docker load" in script_text
    assert 'docker run --rm "${EXPECTED_IMAGE_ID}"' in script_text
    assert "linux/arm64" in script_text
