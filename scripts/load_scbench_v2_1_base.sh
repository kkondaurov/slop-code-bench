#!/usr/bin/env bash

set -euo pipefail

EXPECTED_IMAGE_ID="sha256:f92550022dbc45c417e0c5bfcab706411b7407881ffd2d9b74d4e2049bbce985"
EXPECTED_PLATFORM="linux/arm64"
EXPECTED_ARCHIVE_SHA256="eac1f5965f0c563529861a8b7b62fd2e2c0de6f726ffcdf38bcd8538eb145a44"
EXPECTED_MINIO_RELEASE="RELEASE.2025-09-07T16-13-09Z"
EXPECTED_MINIO_SHA256="5c83cd2cf151717ba0243f73e1c7802ff36e272b67144bdd7f1f7d684fd6f03d"
EXPECTED_NODE_VERSION="v22.21.1"
EXPECTED_TSX_VERSION="tsx v4.23.1"
EXPECTED_TYPESCRIPT_VERSION="Version 7.0.2"
ARCHIVE_NAME="slopcodebench-base-scb-v2.1-linux-arm64-image-f92550022dbc.tar.zst"
RELEASE_TAG="scbench-v2.1-repro.1"
RELEASE_URL="https://github.com/kkondaurov/slop-code-bench/releases/tag/${RELEASE_TAG}"
ARCHIVE_URL="https://github.com/kkondaurov/slop-code-bench/releases/download/${RELEASE_TAG}/${ARCHIVE_NAME}"
CHECKSUM_URL="https://github.com/kkondaurov/slop-code-bench/releases/download/${RELEASE_TAG}/release-assets-v2.1.sha256"

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
repository_root="$(cd -- "${script_dir}/.." && pwd -P)"
archive="${1:-${repository_root}/outputs/reproducibility-images/${ARCHIVE_NAME}}"

if [[ ! -f "${archive}" ]]; then
  echo "SCBench v2.1 base archive does not exist: ${archive}" >&2
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

node_version_output="$(
  docker run --rm --user 1000:1000 "${EXPECTED_IMAGE_ID}" node --version
)"
if [[ "${node_version_output}" != "${EXPECTED_NODE_VERSION}" ]]; then
  echo "Loaded Node version mismatch: ${node_version_output}" >&2
  exit 1
fi

tsx_version_output="$(
  docker run --rm --user 1000:1000 "${EXPECTED_IMAGE_ID}" tsx --version
)"
if ! grep -Fxq "${EXPECTED_TSX_VERSION}" <<<"${tsx_version_output}"; then
  echo "Loaded tsx version mismatch." >&2
  echo "${tsx_version_output}" >&2
  exit 1
fi

typescript_version_output="$(
  docker run --rm --user 1000:1000 "${EXPECTED_IMAGE_ID}" tsc --version
)"
if [[ "${typescript_version_output}" != "${EXPECTED_TYPESCRIPT_VERSION}" ]]; then
  echo "Loaded TypeScript version mismatch: ${typescript_version_output}" >&2
  exit 1
fi

tsx_transform_output="$(
  docker run --rm --user 1000:1000 "${EXPECTED_IMAGE_ID}" \
    tsx --eval 'enum Signal { Ready = "ready" }; console.log(Signal.Ready)'
)"
if [[ "${tsx_transform_output}" != "ready" ]]; then
  echo "Loaded tsx transform smoke failed: ${tsx_transform_output}" >&2
  exit 1
fi

echo "Loaded verified SCBench v2.1 base ${EXPECTED_IMAGE_ID} (${EXPECTED_PLATFORM})."
