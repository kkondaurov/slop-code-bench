from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest
import yaml

from slop_code.entrypoints.config.loader import load_run_config

ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "scripts" / "verify_paper_v2.py"
SPEC = importlib.util.spec_from_file_location(
    "verify_paper_v2_module", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _make_catalog(root: Path) -> None:
    alpha = root / "alpha"
    alpha.mkdir(parents=True)
    (alpha / "config.yaml").write_text(
        "checkpoints:\n  checkpoint_1: {}\n  checkpoint_2: {}\n",
        encoding="utf-8",
    )
    (alpha / "checkpoint_1.md").write_text("alpha\n", encoding="utf-8")
    beta = root / "beta"
    beta.mkdir()
    (beta / "config.yaml").write_text(
        "checkpoints:\n  checkpoint_1: {}\n",
        encoding="utf-8",
    )


def test_checked_in_lock_and_manifest_agree() -> None:
    lock = MODULE.load_lock()
    manifest = yaml.safe_load(
        (ROOT / "configs" / "scbench-v2" / "manifest.yaml").read_text(
            encoding="utf-8"
        )
    )

    # Provenance embeds this object verbatim; YAML timestamps must therefore
    # remain quoted strings rather than Python datetime objects.
    json.dumps(manifest, allow_nan=False)

    assert lock["problem_count"] == 36
    assert lock["checkpoint_count"] == 196
    assert sum(lock["problems"].values()) == 196
    assert lock["schema_version"] == 2
    assert lock["hash_algorithm"] == MODULE.HASH_ALGORITHM
    assert manifest["suite_revision"] == "v2.1"
    assert manifest["catalog"]["release"] == lock["source"]["release"]
    assert manifest["catalog"]["commit"] == lock["source"]["commit"]
    environment_path = (
        ROOT
        / "configs"
        / "environments"
        / "docker-python3.12-uv-scb-v2.1.yaml"
    )
    assert (
        manifest["protocol"]["environment_sha256"]
        == MODULE.sha256_file(environment_path)
    )
    prebuilt = manifest["protocol"]["prebuilt_base"]
    assert prebuilt["reference"] == prebuilt["image_id"]
    assert prebuilt["platform"] == "linux/arm64"
    assert prebuilt["publication_status"] == "published"
    runner = manifest["runner"]
    assert runner["repository"] == MODULE.REPRODUCIBILITY_REPOSITORY
    assert runner["upstream_repository"] == (
        MODULE.UPSTREAM_RUNNER_REPOSITORY
    )
    assert runner["base_commit"] == MODULE.RUNNER_BASE_COMMIT
    assert runner["release_tag"] == MODULE.RELEASE_TAG
    assert runner["release_url"] == MODULE.RELEASE_URL
    assert prebuilt["archive"]["download_url"] == (
        MODULE.PREBUILT_ARCHIVE_URL
    )
    assert prebuilt["archive"]["checksum_url"] == (
        MODULE.PREBUILT_CHECKSUM_URL
    )
    assert (
        manifest["profiles"]["gpt-5.5-current-xhigh"]["release_evidence"][
            "registry"
        ]
        == "https://registry.npmjs.org/@openai%2Fcodex/0.146.0"
    )
    assert manifest["catalog"]["content_lock_schema_version"] == 2
    assert manifest["catalog"]["hash_algorithm"] == MODULE.HASH_ALGORITHM
    assert manifest["catalog"]["content_lock_sha256"] == MODULE.sha256_file(
        MODULE.DEFAULT_LOCK_PATH
    )
    assert MODULE.verify_suite_manifest(MODULE.DEFAULT_LOCK_PATH, lock) == []


def test_suite_manifest_verifier_rejects_semantic_count_drift(
    monkeypatch,
) -> None:
    lock = MODULE.load_lock()
    manifest = yaml.safe_load(
        (ROOT / "configs/scbench-v2/manifest.yaml").read_text(
            encoding="utf-8"
        )
    )
    drifted = copy.deepcopy(manifest)
    drifted["diagnostic_subset"]["problems"]["mvvault"] = 99
    monkeypatch.setattr(MODULE.yaml, "safe_load", lambda _: drifted)

    errors = MODULE.verify_suite_manifest(MODULE.DEFAULT_LOCK_PATH, lock)

    assert "suite manifest diagnostic problem counts mismatch" in errors


def test_suite_manifest_verifier_rejects_profile_sri_drift(
    monkeypatch,
) -> None:
    lock = MODULE.load_lock()
    manifest = yaml.safe_load(
        (ROOT / "configs/scbench-v2/manifest.yaml").read_text(
            encoding="utf-8"
        )
    )
    drifted = copy.deepcopy(manifest)
    profile = drifted["profiles"]["gpt-5.5-current-xhigh"]
    profile["release_evidence"]["package_integrity"] = "sha512-drift"
    monkeypatch.setattr(MODULE.yaml, "safe_load", lambda _: drifted)

    errors = MODULE.verify_suite_manifest(MODULE.DEFAULT_LOCK_PATH, lock)

    assert (
        "suite profile gpt-5.5-current-xhigh package integrity mismatch"
        in errors
    )


@pytest.mark.parametrize(
    ("path", "value", "expected_error"),
    [
        (("schema_version",), 99, "suite manifest schema version mismatch"),
        (("suite_revision",), "v2", "suite manifest revision mismatch"),
        (("paper", "arxiv_id"), "wrong", "suite manifest paper arxiv_id mismatch"),
        (
            ("runner", "repository"),
            "https://example.invalid/fork",
            "suite manifest runner repository mismatch",
        ),
        (
            ("runner", "upstream_repository"),
            "https://example.invalid/upstream",
            "suite manifest runner upstream_repository mismatch",
        ),
        (
            ("runner", "base_commit"),
            "0" * 40,
            "suite manifest runner base_commit mismatch",
        ),
        (
            ("runner", "release_tag"),
            "wrong-tag",
            "suite manifest runner release_tag mismatch",
        ),
        (
            ("runner", "release_url"),
            "https://example.invalid/release",
            "suite manifest runner release_url mismatch",
        ),
        (
            ("catalog", "repository"),
            "https://example.invalid/drift",
            "suite manifest catalog repository mismatch",
        ),
        (
            ("protocol", "source_image", "digest"),
            "sha256:" + "0" * 64,
            "suite manifest source image digest mismatch",
        ),
        (
            ("quality_evaluator", "primary", "command"),
            "run-something-else",
            "suite evaluator command mismatch",
        ),
        (
            ("quality_evaluator", "primary", "project"),
            "../outside.toml",
            "suite evaluator project mismatch",
        ),
        (
            ("quality_evaluator", "primary", "lock"),
            "/outside.lock",
            "suite evaluator lock mismatch",
        ),
        (
            ("protocol", "prebuilt_base", "archive", "sha256"),
            "0" * 64,
            "suite archive checksum file content mismatch",
        ),
        (
            ("protocol", "prebuilt_base", "archive", "path"),
            "outputs/reproducibility-images/wrong-name.tar.zst",
            "suite archive checksum file content mismatch",
        ),
        (
            ("protocol", "prebuilt_base", "publication_status"),
            "local-release-candidate",
            "suite manifest prebuilt publication status mismatch",
        ),
        (
            ("protocol", "prebuilt_base", "archive", "download_url"),
            "https://example.invalid/archive",
            "suite prebuilt archive download_url mismatch",
        ),
        (
            ("protocol", "prebuilt_base", "archive", "checksum_url"),
            "https://example.invalid/checksums",
            "suite prebuilt archive checksum_url mismatch",
        ),
        *[
            (
                (
                    "protocol",
                    "prebuilt_base",
                    "bundled_tools",
                    "minio",
                    field,
                ),
                "drift",
                f"suite manifest bundled MinIO {field} mismatch",
            )
            for field in ("release", "architecture", "sha256", "runtime")
        ],
    ],
)
def test_suite_manifest_verifier_binds_published_identity_claims(
    monkeypatch,
    path: tuple[str, ...],
    value: object,
    expected_error: str,
) -> None:
    lock = MODULE.load_lock()
    manifest = yaml.safe_load(
        (ROOT / "configs/scbench-v2/manifest.yaml").read_text(
            encoding="utf-8"
        )
    )
    drifted = copy.deepcopy(manifest)
    target = drifted
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = value
    original_safe_load = MODULE.yaml.safe_load
    calls = 0

    def load_first_document(text: str):
        nonlocal calls
        calls += 1
        if calls == 1:
            return drifted
        return original_safe_load(text)

    monkeypatch.setattr(MODULE.yaml, "safe_load", load_first_document)

    errors = MODULE.verify_suite_manifest(MODULE.DEFAULT_LOCK_PATH, lock)

    assert expected_error in errors


def test_manifest_path_rejects_symlink_component(tmp_path: Path) -> None:
    root = tmp_path / "root"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (outside / "input.toml").write_text("outside\n", encoding="utf-8")
    (root / "linked").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="traverses symlink"):
        MODULE._safe_manifest_path(root, "linked/input.toml")


def test_fixture_catalog_round_trip(tmp_path: Path) -> None:
    catalog_root = tmp_path / "problems"
    _make_catalog(catalog_root)
    lock = MODULE.build_lock(catalog_root)

    assert lock["problem_count"] == 2
    assert lock["checkpoint_count"] == 3
    assert MODULE.verify_catalog(catalog_root, lock) == []


def test_catalog_byte_drift_is_reported(tmp_path: Path) -> None:
    catalog_root = tmp_path / "problems"
    _make_catalog(catalog_root)
    lock = MODULE.build_lock(catalog_root)
    (catalog_root / "alpha" / "checkpoint_1.md").write_text(
        "changed\n", encoding="utf-8"
    )

    errors = MODULE.verify_catalog(catalog_root, lock)

    assert any("tree_sha256 mismatch" in error for error in errors)


def test_catalog_new_file_is_reported(tmp_path: Path) -> None:
    catalog_root = tmp_path / "problems"
    _make_catalog(catalog_root)
    lock = MODULE.build_lock(catalog_root)
    (catalog_root / "alpha" / "new.json").write_text("{}\n")

    errors = MODULE.verify_catalog(catalog_root, lock)

    assert any("file_count mismatch" in error for error in errors)


def test_catalog_ephemera_is_ignored(tmp_path: Path) -> None:
    catalog_root = tmp_path / "problems"
    _make_catalog(catalog_root)
    lock = MODULE.build_lock(catalog_root)
    cache = catalog_root / "alpha" / "__pycache__" / "value.pyc"
    cache.parent.mkdir()
    cache.write_bytes(b"cache")

    assert MODULE.verify_catalog(catalog_root, lock) == []


def test_managed_manifest_commit_drift_is_reported(tmp_path: Path) -> None:
    catalog_root = tmp_path / "problems"
    _make_catalog(catalog_root)
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "version": MODULE.EXPECTED_RELEASE,
                "commit": "0" * 40,
            }
        ),
        encoding="utf-8",
    )

    errors = MODULE.verify_managed_manifest(catalog_root)

    assert any("catalog commit mismatch" in error for error in errors)


def test_missing_managed_manifest_gives_frozen_sync_command(
    tmp_path: Path,
) -> None:
    catalog_root = tmp_path / "problems"
    catalog_root.mkdir()

    errors = MODULE.verify_managed_manifest(catalog_root)

    assert errors == [
        "managed catalog manifest is missing: "
        f"{tmp_path / 'manifest.json'}; "
            "run `UV_NO_CONFIG=1 uv run --frozen slop-code sync v1.0.1`"
    ]


def test_profile_configs_resolve_to_declared_semantics() -> None:
    catalog_problems = list(MODULE.load_lock()["problems"])
    manifest = yaml.safe_load(
        (ROOT / "configs/scbench-v2/manifest.yaml").read_text(
            encoding="utf-8"
        )
    )
    expected_problems = {
        "config": catalog_problems,
        "diagnostic_config": list(manifest["diagnostic_subset"]["problems"]),
        "capability_config": list(manifest["capability_subset"]["problems"]),
    }

    for profile_name, profile in manifest["profiles"].items():
        for config_field, problems in expected_problems.items():
            relative_path = profile.get(config_field)
            if relative_path is None:
                continue
            config = load_run_config(ROOT / relative_path)
            assert config.profile == profile_name
            assert config.agent["version"] == profile["cli_version"]
            assert config.model.provider == profile["provider"]
            assert config.model.name == profile["model"]
            assert config.thinking == profile["reasoning"]
            assert config.problems == problems
