"""Unit tests for the Claude Code agent configuration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from slop_code.agent_runner.agents.claude_code import ClaudeCodeConfig
from slop_code.agent_runner.agents.claude_code.agent import ClaudeCodeAgent
from slop_code.agent_runner.agents.utils import HOME_PATH
from slop_code.agent_runner.credentials import CredentialType
from slop_code.agent_runner.credentials import ProviderCredential
from slop_code.agent_runner.models import AgentCostLimits
from slop_code.common.llms import APIPricing
from slop_code.execution import DockerConfig
from slop_code.execution import DockerEnvironmentSpec


@pytest.fixture
def mock_cost_limits():
    """Standard cost limits for tests."""
    return AgentCostLimits(
        step_limit=10,
        cost_limit=100.0,
        net_cost_limit=200.0,
    )


@pytest.fixture
def mock_pricing():
    """Standard pricing for tests."""
    return APIPricing(
        input=0.5,
        output=2.0,
        cache_read=0.1,
    )


@pytest.fixture
def mock_credential():
    """Standard credential for tests."""
    return ProviderCredential(
        provider="anthropic",
        value="test-api-key",
        source="ANTHROPIC_API_KEY",
        destination_key="ANTHROPIC_API_KEY",
        credential_type=CredentialType.ENV_VAR,
    )


class FakeRuntime:
    """Minimal runtime stub for testing."""

    def __init__(self) -> None:
        self.cleaned = False

    def cleanup(self) -> None:
        self.cleaned = True


class FakeLogger:
    """Capture debug logs for assertions."""

    def __init__(self) -> None:
        self.debug_calls: list[tuple[str, dict]] = []

    def debug(self, event: str, **kwargs: object) -> None:
        self.debug_calls.append((event, kwargs))


@dataclass
class FakeSession:
    """Fake session for testing."""

    runtime: FakeRuntime
    working_dir: Path
    spec: object | None = None
    last_spawn_env_vars: dict[str, str] | None = None
    last_spawn_mounts: dict[str, dict[str, str] | str] | None = None

    def spawn(self, **_: object) -> FakeRuntime:
        self.last_spawn_env_vars = dict(_.get("env_vars") or {})
        self.last_spawn_mounts = dict(_.get("mounts") or {})
        return self.runtime


class TestClaudeCodeConfig:
    """Tests for ClaudeCodeConfig."""

    def test_version_is_required(self, mock_cost_limits):
        """Version field is required for docker template."""
        with pytest.raises(Exception):  # Pydantic validation error
            ClaudeCodeConfig(
                type="claude_code",
                cost_limits=mock_cost_limits,
                # Missing version
            )

    def test_config_with_version(self, mock_cost_limits):
        """Config can be created with version."""
        config = ClaudeCodeConfig(
            type="claude_code",
            version="2.0.51",
            cost_limits=mock_cost_limits,
        )
        assert config.version == "2.0.51"
        assert config.binary == "claude"

    def test_get_docker_file_renders_version(self, mock_cost_limits):
        """get_docker_file renders version into template."""
        config = ClaudeCodeConfig(
            type="claude_code",
            version="2.0.51",
            cost_limits=mock_cost_limits,
        )
        dockerfile = config.get_docker_file("base-image:latest")
        assert dockerfile is not None
        assert "base-image:latest" in dockerfile
        assert "@anthropic-ai/claude-code@2.0.51" in dockerfile


class TestClaudeCodeAgent:
    """Tests for ClaudeCodeAgent."""

    def test_save_artifacts_copies_claude_traces(
        self,
        tmp_path,
        mock_cost_limits,
        mock_pricing,
        mock_credential,
    ):
        """save_artifacts copies Claude Code trace jsonl files from home."""
        runtime = FakeRuntime()
        spec = DockerEnvironmentSpec(
            name="test",
            docker=DockerConfig(image="test-image"),
        )
        session = FakeSession(
            runtime=runtime,
            working_dir=tmp_path,
            spec=spec,
        )

        agent = ClaudeCodeAgent(
            problem_name="test-problem",
            image="test-image",
            verbose=False,
            cost_limits=mock_cost_limits,
            pricing=mock_pricing,
            credential=mock_credential,
            binary="claude",
            model="claude-test",
            timeout=None,
            settings={},
            env={},
            extra_args=[],
            append_system_prompt=None,
            allowed_tools=[],
            disallowed_tools=[],
            permission_mode=None,
            base_url=None,
            thinking=None,
            max_thinking_tokens=None,
            max_output_tokens=None,
        )

        agent.setup(session)
        assert agent._trace_dir is not None
        trace_dir = agent._trace_dir
        trace_dir.mkdir(parents=True, exist_ok=True)
        trace_file = trace_dir / "trace.jsonl"
        trace_file.write_text('{"type":"system","subtype":"init"}\n')

        output_dir = tmp_path / "artifacts"
        agent.save_artifacts(output_dir)

        saved_trace = output_dir / "trace.jsonl"
        assert saved_trace.exists()
        assert saved_trace.read_text() == trace_file.read_text()

    def test_setup_uses_default_home_for_docker(
        self,
        tmp_path,
        mock_cost_limits,
        mock_pricing,
        mock_credential,
    ):
        """setup keeps HOME at agent home and mounts claude project dir."""
        runtime = FakeRuntime()
        spec = DockerEnvironmentSpec(
            name="test",
            docker=DockerConfig(image="test-image"),
        )
        session = FakeSession(
            runtime=runtime,
            working_dir=tmp_path,
            spec=spec,
        )

        agent = ClaudeCodeAgent(
            problem_name="test-problem",
            image="test-image",
            verbose=False,
            cost_limits=mock_cost_limits,
            pricing=mock_pricing,
            credential=mock_credential,
            binary="claude",
            model="claude-test",
            timeout=None,
            settings={},
            env={},
            extra_args=[],
            append_system_prompt=None,
            allowed_tools=[],
            disallowed_tools=[],
            permission_mode=None,
            base_url=None,
            thinking=None,
            max_thinking_tokens=None,
            max_output_tokens=None,
        )

        agent.setup(session)

        assert session.last_spawn_env_vars is not None
        assert session.last_spawn_env_vars.get("HOME") == HOME_PATH
        assert session.last_spawn_mounts is not None
        assert any(
            isinstance(value, dict)
            and value.get("bind") == f"{HOME_PATH}/.claude/projects"
            for value in session.last_spawn_mounts.values()
        )

        agent_tmp_dir = agent.tmp_dir
        assert agent_tmp_dir.exists()

        agent.cleanup()

        assert runtime.cleaned is True
        assert not agent_tmp_dir.exists()
        assert agent._session is None
        assert agent._environment is None
        assert agent._workspace is None
        assert agent._runtime is None
        assert agent._tmp_dir is None
        assert agent._trace_dir is None
        assert agent._settings_path is None

    def test_save_artifacts_logs_trace_counts(
        self,
        tmp_path,
        mock_cost_limits,
        mock_pricing,
        mock_credential,
    ):
        """_save_claude_traces logs discovered and saved trace counts."""
        agent = ClaudeCodeAgent(
            problem_name="test-problem",
            image="test-image",
            verbose=False,
            cost_limits=mock_cost_limits,
            pricing=mock_pricing,
            credential=mock_credential,
            binary="claude",
            model="claude-test",
            timeout=None,
            settings={},
            env={},
            extra_args=[],
            append_system_prompt=None,
            allowed_tools=[],
            disallowed_tools=[],
            permission_mode=None,
            base_url=None,
            thinking=None,
            max_thinking_tokens=None,
            max_output_tokens=None,
        )
        logger = FakeLogger()
        agent.log = logger

        trace_dir = tmp_path / "claude_projects"
        trace_dir.mkdir(parents=True, exist_ok=True)
        nested = trace_dir / "proj" / "trace.jsonl"
        nested.parent.mkdir(parents=True, exist_ok=True)
        nested.write_text('{"type":"system","subtype":"init"}\n')
        agent._trace_dir = trace_dir

        output_dir = tmp_path / "artifacts"
        agent._save_claude_traces(output_dir)

        assert any(
            event == "agent.claude_code.traces.found"
            and kwargs.get("files") == 1
            for event, kwargs in logger.debug_calls
        )
        assert any(
            event == "agent.claude_code.traces.saved"
            and kwargs.get("saved") == 1
            for event, kwargs in logger.debug_calls
        )

    def test_prepare_mounts_includes_max_output_tokens_in_settings(
        self,
        tmp_path,
        mock_cost_limits,
        mock_pricing,
        mock_credential,
    ):
        """When max_output_tokens is set, it should be written to settings.json."""
        runtime = FakeRuntime()
        spec = DockerEnvironmentSpec(
            name="test",
            docker=DockerConfig(image="test-image"),
        )
        session = FakeSession(
            runtime=runtime,
            working_dir=tmp_path,
            spec=spec,
        )

        agent = ClaudeCodeAgent(
            problem_name="test-problem",
            image="test-image",
            verbose=False,
            cost_limits=mock_cost_limits,
            pricing=mock_pricing,
            credential=mock_credential,
            binary="claude",
            model="claude-test",
            timeout=None,
            settings={"existingSetting": "value"},
            env={},
            extra_args=[],
            append_system_prompt=None,
            allowed_tools=[],
            disallowed_tools=[],
            permission_mode=None,
            base_url=None,
            thinking=None,
            max_thinking_tokens=None,
            max_output_tokens=64000,
        )

        agent.setup(session)

        # Verify the settings file was written with max_output_tokens
        assert agent._settings_path is not None
        settings_content = json.loads(agent._settings_path.read_text())
        assert settings_content["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] == 64000
        # Verify existing settings are preserved
        assert settings_content["existingSetting"] == "value"

    def test_prepare_mounts_excludes_max_output_tokens_when_none(
        self,
        tmp_path,
        mock_cost_limits,
        mock_pricing,
        mock_credential,
    ):
        """When max_output_tokens is None, it should not be in settings.json."""
        runtime = FakeRuntime()
        spec = DockerEnvironmentSpec(
            name="test",
            docker=DockerConfig(image="test-image"),
        )
        session = FakeSession(
            runtime=runtime,
            working_dir=tmp_path,
            spec=spec,
        )

        agent = ClaudeCodeAgent(
            problem_name="test-problem",
            image="test-image",
            verbose=False,
            cost_limits=mock_cost_limits,
            pricing=mock_pricing,
            credential=mock_credential,
            binary="claude",
            model="claude-test",
            timeout=None,
            settings={"existingSetting": "value"},
            env={},
            extra_args=[],
            append_system_prompt=None,
            allowed_tools=[],
            disallowed_tools=[],
            permission_mode=None,
            base_url=None,
            thinking=None,
            max_thinking_tokens=None,
            max_output_tokens=None,
        )

        agent.setup(session)

        # Verify the settings file was written without max_output_tokens
        assert agent._settings_path is not None
        settings_content = json.loads(agent._settings_path.read_text())
        assert "CLAUDE_CODE_MAX_OUTPUT_TOKENS" not in settings_content
        # Verify existing settings are preserved
        assert settings_content["existingSetting"] == "value"

    def test_prepare_mounts_does_not_mutate_original_settings(
        self,
        tmp_path,
        mock_cost_limits,
        mock_pricing,
        mock_credential,
    ):
        """Setting max_output_tokens should not mutate the original settings dict."""
        runtime = FakeRuntime()
        spec = DockerEnvironmentSpec(
            name="test",
            docker=DockerConfig(image="test-image"),
        )
        session = FakeSession(
            runtime=runtime,
            working_dir=tmp_path,
            spec=spec,
        )

        original_settings = {"existingSetting": "value"}
        agent = ClaudeCodeAgent(
            problem_name="test-problem",
            image="test-image",
            verbose=False,
            cost_limits=mock_cost_limits,
            pricing=mock_pricing,
            credential=mock_credential,
            binary="claude",
            model="claude-test",
            timeout=None,
            settings=original_settings,
            env={},
            extra_args=[],
            append_system_prompt=None,
            allowed_tools=[],
            disallowed_tools=[],
            permission_mode=None,
            base_url=None,
            thinking=None,
            max_thinking_tokens=None,
            max_output_tokens=64000,
        )

        agent.setup(session)

        # Original settings should not be mutated
        assert "CLAUDE_CODE_MAX_OUTPUT_TOKENS" not in original_settings
        assert original_settings == {"existingSetting": "value"}


class TestParseLineErrorHandling:
    """Tests for error payload handling in parse_line and _run."""

    def test_parse_line_identifies_successful_result(self):
        """parse_line returns cost and tokens for successful result."""
        payload = {
            "type": "result",
            "total_cost_usd": 0.5,
            "usage": {
                "input_tokens": 100,
                "output_tokens": 50,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
            },
        }
        line = json.dumps(payload)
        cost, tokens, parsed = ClaudeCodeAgent.parse_line(line)
        assert cost == 0.5
        assert tokens is not None
        assert tokens.input == 100
        assert tokens.output == 50
        assert not parsed.get("is_error", False)

    def test_parse_line_identifies_error_result(self):
        """parse_line returns payload with is_error for error results."""
        payload = {
            "type": "result",
            "subtype": "error_during_execution",
            "is_error": True,
            "total_cost_usd": 0,
            "usage": {
                "input_tokens": 0,
                "output_tokens": 0,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
            },
            "errors": ["some error"],
        }
        line = json.dumps(payload)
        cost, tokens, parsed = ClaudeCodeAgent.parse_line(line)
        assert parsed["is_error"] is True

    def test_error_before_success_should_fail_run(
        self,
        tmp_path,
        mock_cost_limits,
        mock_pricing,
        mock_credential,
    ):
        """Error payload before any successful result should mark run as failed."""
        agent = ClaudeCodeAgent(
            problem_name="test-problem",
            image="test-image",
            verbose=False,
            cost_limits=mock_cost_limits,
            pricing=mock_pricing,
            credential=mock_credential,
            binary="claude",
            model="claude-test",
            timeout=None,
            settings={},
            env={},
            extra_args=[],
            append_system_prompt=None,
            allowed_tools=[],
            disallowed_tools=[],
            permission_mode=None,
            base_url=None,
            thinking=None,
            max_thinking_tokens=None,
            max_output_tokens=None,
        )

        # Simulate processing an error payload before any success
        error_payload = {
            "type": "result",
            "is_error": True,
            "total_cost_usd": 0,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        }

        # Process the error - should set _had_error since no success yet
        agent._process_payload_for_error(error_payload)
        assert agent._had_error is True

    def test_success_result_sets_got_successful_result(
        self,
        tmp_path,
        mock_cost_limits,
        mock_pricing,
        mock_credential,
    ):
        """Successful result payload should set _got_successful_result flag."""
        agent = ClaudeCodeAgent(
            problem_name="test-problem",
            image="test-image",
            verbose=False,
            cost_limits=mock_cost_limits,
            pricing=mock_pricing,
            credential=mock_credential,
            binary="claude",
            model="claude-test",
            timeout=None,
            settings={},
            env={},
            extra_args=[],
            append_system_prompt=None,
            allowed_tools=[],
            disallowed_tools=[],
            permission_mode=None,
            base_url=None,
            thinking=None,
            max_thinking_tokens=None,
            max_output_tokens=None,
        )

        # Simulate processing a successful result payload
        success_payload = {
            "type": "result",
            "total_cost_usd": 0.5,
            "usage": {"input_tokens": 100, "output_tokens": 50},
        }

        agent._process_payload_for_error(success_payload)
        assert agent._got_successful_result is True
        assert agent._had_error is False

    def test_error_after_success_should_not_fail_run(
        self,
        tmp_path,
        mock_cost_limits,
        mock_pricing,
        mock_credential,
    ):
        """Error payload after successful result should not mark run as failed.

        This reproduces the bug where:
        1. Task completes successfully (result payload with is_error=False)
        2. Post-completion error occurs (result payload with is_error=True, 403)
        3. Run incorrectly marked as failed

        The agent should NOT set _had_error when error comes after success.
        """
        agent = ClaudeCodeAgent(
            problem_name="test-problem",
            image="test-image",
            verbose=False,
            cost_limits=mock_cost_limits,
            pricing=mock_pricing,
            credential=mock_credential,
            binary="claude",
            model="claude-test",
            timeout=None,
            settings={},
            env={},
            extra_args=[],
            append_system_prompt=None,
            allowed_tools=[],
            disallowed_tools=[],
            permission_mode=None,
            base_url=None,
            thinking=None,
            max_thinking_tokens=None,
            max_output_tokens=None,
        )

        # First: successful result (task completed)
        success_payload = {
            "type": "result",
            "total_cost_usd": 0.5,
            "usage": {"input_tokens": 100, "output_tokens": 50},
        }
        agent._process_payload_for_error(success_payload)

        # Second: error result (post-completion error like 403 telemetry failure)
        error_payload = {
            "type": "result",
            "is_error": True,
            "subtype": "error_during_execution",
            "total_cost_usd": 0,
            "usage": {"input_tokens": 0, "output_tokens": 0},
            "errors": ["AxiosError: Request failed with status code 403"],
        }
        agent._process_payload_for_error(error_payload)

        # _had_error should remain False because we got a successful result first
        assert agent._got_successful_result is True
        assert agent._had_error is False

    def test_reset_clears_got_successful_result(
        self,
        tmp_path,
        mock_cost_limits,
        mock_pricing,
        mock_credential,
    ):
        """reset() should clear _got_successful_result for next checkpoint."""
        agent = ClaudeCodeAgent(
            problem_name="test-problem",
            image="test-image",
            verbose=False,
            cost_limits=mock_cost_limits,
            pricing=mock_pricing,
            credential=mock_credential,
            binary="claude",
            model="claude-test",
            timeout=None,
            settings={},
            env={},
            extra_args=[],
            append_system_prompt=None,
            allowed_tools=[],
            disallowed_tools=[],
            permission_mode=None,
            base_url=None,
            thinking=None,
            max_thinking_tokens=None,
            max_output_tokens=None,
        )

        # Simulate a successful run
        agent._got_successful_result = True

        # Reset for next checkpoint
        agent.reset()

        # Flag should be cleared
        assert agent._got_successful_result is False
