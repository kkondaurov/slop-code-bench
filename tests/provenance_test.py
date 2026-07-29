from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from slop_code import provenance


def _start_running_provenance(
    root: Path,
    run_dir: Path,
    *,
    require_existing: bool = False,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    with (
        patch.object(provenance, "_git_repository_metadata", return_value={}),
        patch.object(provenance, "_host_metadata", return_value={}),
        patch.object(
            provenance,
            "_image_metadata",
            return_value={"available": False},
        ),
        patch.object(
            provenance,
            "_container_tool_versions",
            return_value={"error": "no_image"},
        ),
    ):
        provenance.start_run_provenance(
            repository_root=root,
            run_dir=run_dir,
            profile=None,
            model_provider="test",
            model_name="test",
            agent_type="test",
            agent_version="1",
            thinking=None,
            seed=42,
            problem_names=["problem"],
            catalog_version="test",
            catalog_commit="test",
            num_workers=1,
            evaluate=True,
            environment_name="local",
            source_image_name="",
            base_image_name="",
            agent_image_name="",
            invocation=["slop-code", "run"],
            require_existing=require_existing,
        )


def test_sanitize_invocation_redacts_credentials() -> None:
    arguments = [
        "slop-code",
        "run",
        "agent.env.OPENAI_API_KEY=sk-private",
        "agent.env.CUSTOM=random-private-value",
        "--secret",
        "value",
        "ghp_private",
        "--header=Authorization: Bearer TOPSECRET",
        "model.name=gpt-5.5",
    ]

    sanitized = provenance.sanitize_invocation(arguments)

    assert sanitized == [
        "slop-code",
        "run",
        "agent.env.OPENAI_API_KEY=<redacted>",
        "agent.env.CUSTOM=<redacted>",
        "--secret",
        "<redacted>",
        "<redacted>",
        "--header=<redacted>",
        "model.name=gpt-5.5",
    ]
    assert "private" not in " ".join(sanitized)


def test_remote_url_sanitizer_removes_userinfo_and_query() -> None:
    value = "ssh://user:private@example.test/repo.git?token=private"

    assert provenance._sanitize_remote_url(value) == (
        "ssh://example.test/repo.git"
    )
    assert (
        provenance._sanitize_remote_url(
            "user:private@example.test:org/repo.git"
        )
        == "example.test:org/repo.git"
    )


def test_artifact_checksums_reject_symlinks(
    tmp_path: Path,
) -> None:
    (tmp_path / "result.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "run_agent.log").write_text("live\n", encoding="utf-8")
    (tmp_path / "provenance.json").write_text("{}\n", encoding="utf-8")
    outside = tmp_path.parent / "outside-secret"
    outside.write_text("secret\n", encoding="utf-8")
    (tmp_path / "external-link").symlink_to(outside)

    with pytest.raises(
        provenance.ProvenanceIntegrityError,
        match="unsupported symlink: external-link",
    ):
        provenance.artifact_checksums(tmp_path)


def test_artifact_checksums_exclude_only_top_level_operational_log(
    tmp_path: Path,
) -> None:
    (tmp_path / "provenance.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "run_agent.log").write_text("live\n", encoding="utf-8")
    (tmp_path / "infer.log").write_text("trajectory\n", encoding="utf-8")
    snapshot = tmp_path / "problem/checkpoint_1/snapshot"
    snapshot.mkdir(parents=True)
    (snapshot / "model.log").write_text("evidence\n", encoding="utf-8")
    agent = tmp_path / "problem/checkpoint_1/agent"
    agent.mkdir()
    (agent / "stderr.log").write_text("stderr\n", encoding="utf-8")

    checksums = provenance.artifact_checksums(tmp_path)

    assert "provenance.json" not in checksums
    assert "run_agent.log" not in checksums
    assert "infer.log" in checksums
    assert "problem/checkpoint_1/snapshot/model.log" in checksums
    assert "problem/checkpoint_1/agent/stderr.log" in checksums


def test_artifact_checksums_reject_path_swap_after_descriptor_open(
    tmp_path: Path,
) -> None:
    target = tmp_path / "result.json"
    target.write_text("original\n", encoding="utf-8")
    original_hash = provenance._hash_artifact_descriptor

    def swap_path(
        descriptor: int,
        *,
        relative: Path,
        before: os.stat_result,
    ) -> str:
        held = tmp_path / "held-original"
        target.replace(held)
        target.write_text("replacement\n", encoding="utf-8")
        return original_hash(
            descriptor,
            relative=relative,
            before=before,
        )

    with (
        patch.object(
            provenance,
            "_hash_artifact_descriptor",
            side_effect=swap_path,
        ),
        pytest.raises(
            provenance.ProvenanceIntegrityError,
            match="run artifact changed while hashing",
        ),
    ):
        provenance.artifact_checksums(tmp_path)


def test_missing_or_corrupt_provenance_fails_closed_and_is_preserved(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    with pytest.raises(
        provenance.ProvenanceIntegrityError,
        match="required provenance file is missing",
    ):
        provenance.validate_resumable_provenance(run_dir)
    with pytest.raises(
        provenance.ProvenanceIntegrityError,
        match="required provenance file is missing",
    ):
        provenance.finalize_run_provenance(run_dir, status="completed")

    corrupt = run_dir / "provenance.json"
    original = "{not-json\n"
    corrupt.write_text(original, encoding="utf-8")
    with pytest.raises(
        provenance.ProvenanceIntegrityError,
        match="malformed and was preserved",
    ):
        provenance.validate_resumable_provenance(run_dir)
    with pytest.raises(
        provenance.ProvenanceIntegrityError,
        match="malformed and was preserved",
    ):
        provenance.finalize_run_provenance(run_dir, status="completed")

    assert corrupt.read_text(encoding="utf-8") == original


def test_existing_provenance_cannot_bypass_checksums_as_fresh_run(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    _start_running_provenance(tmp_path, run_dir)
    result = run_dir / "result.json"
    result.write_text("original\n", encoding="utf-8")
    provenance.finalize_run_provenance(run_dir, status="completed")
    result.write_text("tampered\n", encoding="utf-8")

    with pytest.raises(
        provenance.ProvenanceIntegrityError,
        match="run artifacts changed after provenance finalization",
    ):
        _start_running_provenance(
            tmp_path,
            run_dir,
            require_existing=False,
        )

    assert result.read_text(encoding="utf-8") == "tampered\n"


def test_invocation_owns_output_before_preflight_append_crash(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    preflight = run_dir / "scbench_v2_preflight.json"
    _start_running_provenance(tmp_path, run_dir)
    preflight.write_text('{"attempts": [1]}\n', encoding="utf-8")
    provenance.finalize_run_provenance(run_dir, status="completed")
    provenance.validate_resumable_provenance(run_dir)

    # The new invocation begins before its persistent preflight append. A hard
    # crash after the append therefore leaves running provenance, not stale
    # completed checksums.
    _start_running_provenance(
        tmp_path,
        run_dir,
        require_existing=True,
    )
    preflight.write_text('{"attempts": [1, 2]}\n', encoding="utf-8")
    provenance.validate_resumable_provenance(run_dir)
    _start_running_provenance(
        tmp_path,
        run_dir,
        require_existing=True,
    )

    value = json.loads(
        (run_dir / provenance.PROVENANCE_FILENAME).read_text(encoding="utf-8")
    )
    assert value["final_status"] == "running"
    assert len(value["invocations"]) == 3
    assert value["invocations"][1]["status"] == (
        "interrupted_before_next_invocation"
    )


def test_start_and_finalize_write_publishable_metadata(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    run_dir = root / "outputs" / "run"
    run_dir.mkdir(parents=True)
    manifest = root / "configs" / "scbench-v2" / "manifest.yaml"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        "runner:\n"
        "  base_commit: base-sha\n"
        "paper:\n"
        "  arxiv_id: '2603.24755'\n"
        "  version: v2\n",
        encoding="utf-8",
    )
    content_lock = manifest.parent / "content-lock.json"
    content_lock.write_text(
        json.dumps({"tree_sha256": "tree-digest"}),
        encoding="utf-8",
    )
    (root / "pyproject.toml").write_text(
        "[project]\nname = 'test'\n", encoding="utf-8"
    )
    (root / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    (run_dir / "config.yaml").write_text("model: safe\n", encoding="utf-8")
    (run_dir / "environment.yaml").write_text(
        "type: docker\n", encoding="utf-8"
    )
    (run_dir / "problem_catalog.json").write_text(
        '{"version":"v1.0","commit":"catalog-sha"}\n',
        encoding="utf-8",
    )
    (run_dir / "result.json").write_text("{}\n", encoding="utf-8")

    repository = {
        "head": {
            "sha": "head",
            "branch": "scbench-v2",
            "tags": [],
            "dirty": False,
            "diff_sha256": None,
        },
        "fork": {"url": "https://example.test/fork", "sha": "head"},
        "upstream": {
            "url": "https://example.test/upstream",
            "declared_base_sha": "base-sha",
        },
    }
    resumed_repository = json.loads(json.dumps(repository))
    resumed_repository["head"]["sha"] = "resumed-head"
    resumed_repository["fork"]["sha"] = "resumed-head"
    unavailable = {"name": "", "available": False}
    with (
        patch.object(
            provenance,
            "_git_repository_metadata",
            side_effect=[repository, resumed_repository],
        ),
        patch.object(provenance, "_host_metadata", return_value={"os": "test"}),
        patch.object(provenance, "_image_metadata", return_value=unavailable),
        patch.object(
            provenance,
            "_container_tool_versions",
            return_value={"error": "no_image"},
        ),
    ):
        provenance.start_run_provenance(
            repository_root=root,
            run_dir=run_dir,
            profile="paper-v2-reference",
            model_provider="codex_auth",
            model_name="gpt-5.5",
            agent_type="codex",
            agent_version="0.124.0",
            thinking="high",
            seed=42,
            problem_names=["mvvault", "xjq"],
            catalog_version="v1.0",
            catalog_commit="catalog-sha",
            num_workers=1,
            evaluate=True,
            environment_name="python3.12",
            source_image_name="",
            base_image_name="",
            agent_image_name="",
            invocation=["run", "OPENAI_API_KEY=sk-private"],
            preflight={"status": "verified", "catalog": {"status": "verified"}},
        )
        provenance.finalize_run_provenance(run_dir, status="completed")
        provenance.start_run_provenance(
            repository_root=root,
            run_dir=run_dir,
            profile="paper-v2-reference",
            model_provider="codex_auth",
            model_name="gpt-5.5",
            agent_type="codex",
            agent_version="0.124.0",
            thinking="high",
            seed=42,
            problem_names=["mvvault", "xjq"],
            catalog_version="v1.0",
            catalog_commit="catalog-sha",
            executed_problem_names=["xjq"],
            num_workers=1,
            evaluate=True,
            environment_name="python3.12",
            source_image_name="",
            base_image_name="",
            agent_image_name="",
            invocation=["run", "--resume", str(run_dir)],
            preflight={"status": "verified", "catalog": {"status": "verified"}},
            require_existing=True,
        )
        provenance.finalize_run_provenance(run_dir, status="completed")

    value = json.loads(
        (run_dir / "provenance.json").read_text(encoding="utf-8")
    )
    assert value["schema_version"] == 5
    assert value["profile"] == "paper-v2-reference"
    assert value["final_status"] == "completed"
    assert value["inputs"]["suite_content_tree_sha256"] == "tree-digest"
    assert value["run"]["catalog"] == {
        "version": "v1.0",
        "commit": "catalog-sha",
    }
    assert value["invocations"][0]["arguments"] == [
        "run",
        "OPENAI_API_KEY=<redacted>",
    ]
    assert "result.json" in value["artifacts"]["files"]
    assert "provenance.json" not in value["artifacts"]["files"]
    assert "sk-private" not in json.dumps(value)
    assert len(value["invocations"]) == 2
    assert (
        value["invocations"][0]["context"]["repository"]["head"]["sha"]
        == "head"
    )
    assert (
        value["invocations"][1]["context"]["repository"]["head"]["sha"]
        == "resumed-head"
    )
    assert value["invocations"][1]["problems_executed"] == ["xjq"]
    assert value["preflight"]["catalog"]["status"] == "verified"


def test_start_provenance_closes_stale_running_invocation(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    _start_running_provenance(tmp_path, run_dir)
    _start_running_provenance(tmp_path, run_dir)

    saved = json.loads((run_dir / "provenance.json").read_text())
    previous = saved["invocations"][0]
    assert previous["status"] == "interrupted_before_next_invocation"
    assert previous["error_type"] == "UncleanShutdown"
    assert previous["finished_at"] is not None
    assert saved["invocations"][1]["status"] == "running"


def test_finalize_provenance_persists_non_success_details(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    _start_running_provenance(tmp_path, run_dir)
    details = {
        "postprocessing": {
            "status": "incomplete",
            "errors": [{"kind": "missing_checkpoint_reports"}],
        }
    }

    provenance.finalize_run_provenance(
        run_dir,
        status="incomplete_postprocessing",
        details=details,
    )

    saved = json.loads((run_dir / "provenance.json").read_text())
    assert saved["final_status"] == "incomplete_postprocessing"
    assert saved["final_details"] == details
    assert saved["invocations"][-1]["details"] == details


def test_failure_finalization_skips_artifact_hashing(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _start_running_provenance(tmp_path, run_dir)

    with patch.object(
        provenance,
        "artifact_checksums",
        side_effect=AssertionError("failure path must not hash artifacts"),
    ) as checksums:
        provenance.finalize_run_provenance(
            run_dir,
            status="failed",
            error_type="RuntimeError",
            checksum_artifacts=False,
        )

    checksums.assert_not_called()
    saved = json.loads((run_dir / "provenance.json").read_text())
    assert saved["final_status"] == "failed"
    assert saved["artifacts"] == {
        "algorithm": "sha256",
        "complete": False,
        "excluded": ["provenance.json", "run_agent.log"],
        "files": {},
        "reason": "skipped_during_failure_finalization",
        "schema_version": 2,
    }


def test_semantically_empty_invocation_is_not_a_supported_migration(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    original = json.dumps(
        {"schema_version": 5, "invocations": [{}]},
        indent=2,
    )
    path = run_dir / "provenance.json"
    path.write_text(original, encoding="utf-8")

    with pytest.raises(
        provenance.ProvenanceIntegrityError,
        match="semantically invalid and was preserved",
    ):
        provenance.validate_resumable_provenance(run_dir)

    assert path.read_text(encoding="utf-8") == original


def test_unknown_invocation_status_is_semantically_invalid(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    _start_running_provenance(tmp_path, run_dir)
    path = run_dir / "provenance.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["invocations"][-1]["status"] = "plausible-but-invented"
    value["final_status"] = "plausible-but-invented"
    original = json.dumps(value)
    path.write_text(original, encoding="utf-8")

    with pytest.raises(
        provenance.ProvenanceIntegrityError,
        match="invalid status",
    ):
        provenance.validate_resumable_provenance(run_dir)

    assert path.read_text(encoding="utf-8") == original


def test_start_rejects_changed_resume_identity_before_append(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    _start_running_provenance(tmp_path, run_dir)
    provenance.finalize_run_provenance(run_dir, status="completed")
    path = run_dir / "provenance.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["run"]["model"]["name"] = "other-model"
    value["invocations"][-1]["context"]["run"]["model"]["name"] = (
        "other-model"
    )
    original = json.dumps(value)
    path.write_text(original, encoding="utf-8")

    with pytest.raises(
        provenance.ProvenanceIntegrityError,
        match="resume identity does not match",
    ):
        _start_running_provenance(
            tmp_path,
            run_dir,
            require_existing=True,
        )

    assert path.read_text(encoding="utf-8") == original


def test_resume_rejects_changed_complete_artifact_manifest(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    _start_running_provenance(tmp_path, run_dir)
    result = run_dir / "evaluation.json"
    result.write_text('{"passed": true}\n', encoding="utf-8")
    provenance.finalize_run_provenance(run_dir, status="completed")
    provenance.validate_resumable_provenance(run_dir)

    result.write_text('{"passed": false}\n', encoding="utf-8")

    with pytest.raises(
        provenance.ProvenanceIntegrityError,
        match=r"changed=\['evaluation.json'\]",
    ):
        provenance.validate_resumable_provenance(run_dir)


def test_start_rejects_tampered_complete_manifest_before_append(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    _start_running_provenance(tmp_path, run_dir)
    result = run_dir / "result.json"
    result.write_text('{"status": "original"}\n', encoding="utf-8")
    provenance.finalize_run_provenance(run_dir, status="completed")
    original_provenance = (run_dir / "provenance.json").read_text(
        encoding="utf-8"
    )
    result.write_text('{"status": "tampered"}\n', encoding="utf-8")

    with pytest.raises(
        provenance.ProvenanceIntegrityError,
        match=r"changed=\['result.json'\]",
    ):
        _start_running_provenance(
            tmp_path,
            run_dir,
            require_existing=True,
        )

    assert (
        run_dir / "provenance.json"
    ).read_text(encoding="utf-8") == original_provenance


def test_resume_rejects_unexpected_artifact_after_finalization(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    _start_running_provenance(tmp_path, run_dir)
    provenance.finalize_run_provenance(run_dir, status="completed")
    (run_dir / "late.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(
        provenance.ProvenanceIntegrityError,
        match=r"unexpected=\['late.json'\]",
    ):
        provenance.validate_resumable_provenance(run_dir)


def test_running_provenance_without_manifest_remains_resumable(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    _start_running_provenance(tmp_path, run_dir)

    provenance.validate_resumable_provenance(run_dir)


def test_running_provenance_rejects_symlink_before_resume_mutation(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    _start_running_provenance(tmp_path, run_dir)
    outside = tmp_path / "outside.yaml"
    outside.write_text("untouched\n", encoding="utf-8")
    (run_dir / "config.yaml").symlink_to(outside)

    with pytest.raises(
        provenance.ProvenanceIntegrityError,
        match="unsupported symlink: config.yaml",
    ):
        provenance.validate_resumable_provenance(run_dir)

    assert outside.read_text(encoding="utf-8") == "untouched\n"


def test_finalized_snapshot_log_mutation_is_rejected(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _start_running_provenance(tmp_path, run_dir)
    snapshot = run_dir / "problem/checkpoint_1/snapshot"
    snapshot.mkdir(parents=True)
    model_log = snapshot / "model.log"
    model_log.write_text("original\n", encoding="utf-8")
    provenance.finalize_run_provenance(run_dir, status="completed")

    model_log.write_text("tampered\n", encoding="utf-8")

    with pytest.raises(
        provenance.ProvenanceIntegrityError,
        match="run artifacts changed after provenance finalization",
    ):
        provenance.validate_resumable_provenance(run_dir)


def test_schema_four_complete_manifest_is_explicitly_migratable(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    _start_running_provenance(tmp_path, run_dir)
    provenance.finalize_run_provenance(run_dir, status="completed")
    path = run_dir / "provenance.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["schema_version"] = 4
    for artifacts in (
        value["artifacts"],
        value["invocations"][-1]["artifacts"],
    ):
        artifacts.pop("schema_version")
        artifacts.pop("complete")
    path.write_text(json.dumps(value), encoding="utf-8")

    provenance.validate_resumable_provenance(run_dir)


def test_schema_two_complete_manifest_is_explicitly_migratable(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    _start_running_provenance(tmp_path, run_dir)
    provenance.finalize_run_provenance(run_dir, status="completed")
    path = run_dir / "provenance.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["schema_version"] = 2
    value["paper"] = value.pop("suite")
    value.pop("preflight")
    value["run"].pop("catalog")
    invocation = value["invocations"][-1]
    invocation["context"]["paper"] = invocation["context"].pop("suite")
    invocation["context"].pop("preflight")
    invocation["context"]["run"].pop("catalog")
    for artifacts in (value["artifacts"], invocation["artifacts"]):
        artifacts.pop("schema_version")
        artifacts.pop("complete")
    path.write_text(json.dumps(value), encoding="utf-8")

    provenance.validate_resumable_provenance(run_dir)
