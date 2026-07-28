from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from slop_code import provenance


def test_sanitize_invocation_redacts_named_and_token_credentials() -> None:
    arguments = [
        "slop-code",
        "run",
        "agent.env.OPENAI_API_KEY=sk-private",
        "agent.env.CUSTOM=random-private-value",
        "--secret",
        "value",
        "ghp_private",
        "--header=Authorization: Bearer TOPSECRET",
        "model.name=gpt-5.4",
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
        "model.name=gpt-5.4",
    ]
    assert "private" not in " ".join(sanitized)


def test_remote_url_sanitizer_removes_userinfo_and_query() -> None:
    value = "ssh://user:private@example.test/repo.git?token=private"

    assert provenance._sanitize_remote_url(value) == (
        "ssh://example.test/repo.git"
    )
    assert provenance._sanitize_remote_url(
        "user:private@example.test:org/repo.git"
    ) == "example.test:org/repo.git"


def test_artifact_checksums_exclude_logs_provenance_and_symlinks(
    tmp_path: Path,
) -> None:
    (tmp_path / "result.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "run.log").write_text("live\n", encoding="utf-8")
    (tmp_path / "provenance.json").write_text("{}\n", encoding="utf-8")
    outside = tmp_path.parent / "outside-secret"
    outside.write_text("secret\n", encoding="utf-8")
    (tmp_path / "external-link").symlink_to(outside)

    checksums = provenance.artifact_checksums(tmp_path)

    assert set(checksums) == {"result.json"}


def test_start_and_finalize_write_publishable_metadata(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    run_dir = root / "outputs" / "run"
    run_dir.mkdir(parents=True)
    manifest = root / "configs" / "paper-v1" / "manifest.yaml"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        "paper:\n  arxiv_id: '2603.24755'\n  version: v1\n",
        encoding="utf-8",
    )
    content_lock = manifest.parent / "content-lock.json"
    content_lock.write_text(
        json.dumps({"tree_sha256": "tree-digest"}),
        encoding="utf-8",
    )
    (root / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    (run_dir / "config.yaml").write_text("model: safe\n", encoding="utf-8")
    (run_dir / "environment.yaml").write_text(
        "type: docker\n", encoding="utf-8"
    )
    (run_dir / "result.json").write_text("{}\n", encoding="utf-8")

    repository = {
        "head": {
            "sha": "head",
            "branch": "paper-v1",
            "tags": [],
            "dirty": False,
            "diff_sha256": None,
        },
        "fork": {"url": "https://example.test/fork", "sha": "head"},
        "upstream": {
            "url": "https://example.test/upstream",
            "paper_source_sha": "source",
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
            profile="paper-v1",
            model_provider="codex_auth",
            model_name="gpt-5.4",
            agent_type="codex",
            agent_version="0.110.0",
            thinking="high",
            seed=42,
            problem_names=["dag_execution"],
            num_workers=1,
            evaluate=True,
            environment_name="python3.12",
            base_image_name="",
            agent_image_name="",
            invocation=["run", "OPENAI_API_KEY=sk-private"],
        )
        provenance.finalize_run_provenance(run_dir, status="completed")
        provenance.start_run_provenance(
            repository_root=root,
            run_dir=run_dir,
            profile="paper-v1",
            model_provider="codex_auth",
            model_name="gpt-5.4",
            agent_type="codex",
            agent_version="0.110.0",
            thinking="high",
            seed=42,
            problem_names=["dag_execution", "file_backup"],
            executed_problem_names=["file_backup"],
            num_workers=1,
            evaluate=True,
            environment_name="python3.12",
            base_image_name="",
            agent_image_name="",
            invocation=["run", "--resume", str(run_dir)],
        )
        provenance.finalize_run_provenance(run_dir, status="completed")

    value = json.loads(
        (run_dir / "provenance.json").read_text(encoding="utf-8")
    )
    assert value["schema_version"] == 2
    assert value["profile"] == "paper-v1"
    assert value["final_status"] == "completed"
    assert value["inputs"]["paper_content_tree_sha256"] == "tree-digest"
    assert value["invocations"][0]["arguments"] == [
        "run",
        "OPENAI_API_KEY=<redacted>",
    ]
    assert "result.json" in value["artifacts"]["files"]
    assert "provenance.json" not in value["artifacts"]["files"]
    assert "sk-private" not in json.dumps(value)
    assert len(value["invocations"]) == 2
    assert value["invocations"][0]["context"]["repository"]["head"][
        "sha"
    ] == "head"
    assert value["invocations"][1]["context"]["repository"]["head"][
        "sha"
    ] == "resumed-head"
    assert value["invocations"][1]["problems_executed"] == ["file_backup"]
    assert "result.json" in value["invocations"][0]["artifacts"]["files"]
