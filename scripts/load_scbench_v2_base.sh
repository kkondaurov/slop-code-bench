#!/usr/bin/env bash

set -euo pipefail

EXPECTED_IMAGE_ID="sha256:d2b862aad2bf40fe80573d0facc462608ce2a2fe76b56927a36050dc02a44f14"
EXPECTED_PLATFORM="linux/arm64"
EXPECTED_ARCHIVE_SHA256="dff52ff24d1d7e7d88525ef403b374d6464d3331c2b3df1ce72e4d56a3ec5df9"
EXPECTED_MINIO_RELEASE="RELEASE.2025-09-07T16-13-09Z"
EXPECTED_MINIO_SHA256="5c83cd2cf151717ba0243f73e1c7802ff36e272b67144bdd7f1f7d684fd6f03d"
ARCHIVE_NAME="slopcodebench-base-scb-v2-linux-arm64-image-d2b862aad2bf.tar.zst"
RELEASE_TAG="scbench-v2-repro.1"
RELEASE_URL="https://github.com/kkondaurov/slop-code-bench/releases/tag/${RELEASE_TAG}"
ARCHIVE_URL="https://github.com/kkondaurov/slop-code-bench/releases/download/${RELEASE_TAG}/${ARCHIVE_NAME}"
CHECKSUM_URL="https://github.com/kkondaurov/slop-code-bench/releases/download/${RELEASE_TAG}/release-assets.sha256"

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
repository_root="$(cd -- "${script_dir}/.." && pwd -P)"
archive="${1:-${repository_root}/outputs/reproducibility-images/${ARCHIVE_NAME}}"

if [[ ! -f "${archive}" ]]; then
  echo "SCBench v2 base archive does not exist: ${archive}" >&2
  echo "Release: ${RELEASE_URL}" >&2
  echo "Archive: ${ARCHIVE_URL}" >&2
  echo "Checksums: ${CHECKSUM_URL}" >&2
  echo "Download the archive or pass its local path as argument 1." >&2
  exit 1
fi

if command -v shasum >/dev/null 2>&1; then
  actual_archive_sha256="$(shasum -a 256 "${archive}" | awk '{print $1}')"
elif command -v sha256sum >/dev/null 2>&1; then
  actual_archive_sha256="$(sha256sum "${archive}" | awk '{print $1}')"
else
  echo "Neither shasum nor sha256sum is available." >&2
  exit 1
fi

if [[ "${actual_archive_sha256}" != "${EXPECTED_ARCHIVE_SHA256}" ]]; then
  echo "Archive SHA256 mismatch." >&2
  echo "Expected: ${EXPECTED_ARCHIVE_SHA256}" >&2
  echo "Actual:   ${actual_archive_sha256}" >&2
  exit 1
fi

zstd --test "${archive}"
zstd --decompress --stdout "${archive}" | docker load

actual_image_id="$(
  docker image inspect --format '{{.Id}}' "${EXPECTED_IMAGE_ID}"
)"
actual_platform="$(
  docker image inspect --format '{{.Os}}/{{.Architecture}}' "${EXPECTED_IMAGE_ID}"
)"

if [[ "${actual_image_id}" != "${EXPECTED_IMAGE_ID}" ]]; then
  echo "Loaded Docker image ID mismatch." >&2
  echo "Expected: ${EXPECTED_IMAGE_ID}" >&2
  echo "Actual:   ${actual_image_id}" >&2
  exit 1
fi

if [[ "${actual_platform}" != "${EXPECTED_PLATFORM}" ]]; then
  echo "Loaded Docker image platform mismatch." >&2
  echo "Expected: ${EXPECTED_PLATFORM}" >&2
  echo "Actual:   ${actual_platform}" >&2
  exit 1
fi

actual_minio_sha256="$(
  docker run --rm "${EXPECTED_IMAGE_ID}" \
    sha256sum /usr/local/bin/minio | awk '{print $1}'
)"
if [[ "${actual_minio_sha256}" != "${EXPECTED_MINIO_SHA256}" ]]; then
  echo "Loaded MinIO SHA256 mismatch." >&2
  echo "Expected: ${EXPECTED_MINIO_SHA256}" >&2
  echo "Actual:   ${actual_minio_sha256}" >&2
  exit 1
fi

minio_version_output="$(
  docker run --rm "${EXPECTED_IMAGE_ID}" /usr/local/bin/minio --version
)"
if ! grep -Fq "minio version ${EXPECTED_MINIO_RELEASE}" \
  <<<"${minio_version_output}"; then
  echo "Loaded MinIO release mismatch." >&2
  echo "${minio_version_output}" >&2
  exit 1
fi
if ! grep -Fq "Runtime: " <<<"${minio_version_output}" \
  || ! grep -Fq "linux/arm64" <<<"${minio_version_output}"; then
  echo "Loaded MinIO runtime is not native linux/arm64." >&2
  echo "${minio_version_output}" >&2
  exit 1
fi

echo "Loaded verified SCBench v2 base ${EXPECTED_IMAGE_ID} (${EXPECTED_PLATFORM})."
