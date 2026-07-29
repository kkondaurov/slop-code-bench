"""Unit tests for the Codex agent."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import cast
from unittest.mock import MagicMock

import pytest
import yaml

from slop_code.agent_runner.agents.codex import CodexAgent
from slop_code.agent_runner.agents.codex import CodexConfig
from slop_code.agent_runner.agents.utils import HOME_PATH
from slop_code.agent_runner.credentials import ProviderCredential
from slop_code.agent_runner.models import AgentCostLimits
from slop_code.common.llms import APIPricing
from slop_code.common.llms import ModelDefinition
from slop_code.common.llms import TokenUsage
from slop_code.execution import DockerConfig
from slop_code.execution import DockerEnvironmentSpec
from slop_code.execution.runtime import RuntimeEvent
from slop_code.execution.runtime import RuntimeResult

VALID_NPM_INTEGRITY = (
    "sha512-"
    "1EVAuPyAQZ8zIVMw3bPJ6a4R8ifLAZ7LGsOyknj5c2he9AFXVRCmWx12WrdZJ25"
    "wcBvOEKt1n1Zx+QAj0EVGbQ=="
)


class FakeRuntime:
    """Minimal runtime stub for testing."""

    def __init__(self) -> None:
        self.events: list[RuntimeEvent] = []
        self.event_batches: list[list[RuntimeEvent]] = []
        self.before_stream: Callable[[int], None] | None = None
        self.stream_calls: list[tuple[tuple, dict]] = []
        self.cleaned = False
        self.last_stream_args: tuple[tuple, dict] | None = None

    def stream(
        self,
        command: str,
        env: dict,
        timeout: float | None,
        stdin: str | list[str] | None = None,
    ) -> Iterable[RuntimeEvent]:
        self.last_stream_args = ((command, env, stdin, timeout), {})
        self.stream_calls.append(self.last_stream_args)
        invocation = len(self.stream_calls) - 1
        if self.before_stream is not None:
            self.before_stream(invocation)
        events = (
            self.event_batches[invocation]
            if self.event_batches
            else self.events
        )
        yield from events

    def cleanup(self) -> None:
        self.cleaned = True


class FakeLogger:
    """Capture debug logs for assertions."""

    def __init__(self) -> None:
        self.debug_calls: list[tuple[str, dict]] = []
        self.info_calls: list[tuple[str, dict]] = []
        self.warning_calls: list[tuple[str, dict]] = []

    def debug(self, event: str, **kwargs: object) -> None:
        self.debug_calls.append((event, kwargs))

    def info(self, event: str, **kwargs: object) -> None:
        self.info_calls.append((event, kwargs))

    def warning(self, event: str, **kwargs: object) -> None:
        self.warning_calls.append((event, kwargs))


def write_codex_trace(
    path: Path,
    *,
    thread_id: str,
    input_tokens: int,
    cached_input_tokens: int,
    output_tokens: int,
    reasoning_output_tokens: int,
    total_cost: float | None = None,
) -> None:
    """Write the correlated subset of a Codex rollout used by telemetry."""
    info: dict[str, object] = {
        "total_token_usage": {
            "input_tokens": input_tokens,
            "cached_input_tokens": cached_input_tokens,
            "output_tokens": output_tokens,
            "reasoning_output_tokens": reasoning_output_tokens,
        }
    }
    if total_cost is not None:
        info["total_cost"] = total_cost
    events = [
        {
            "type": "session_meta",
            "payload": {"id": thread_id},
        },
        {
            "type": "event_msg",
            "payload": {"type": "token_count", "info": info},
        },
    ]
    path.write_text("".join(f"{json.dumps(event)}\n" for event in events))


@dataclass
class FakeDockerSpec:
    """Fake docker spec for testing."""

    workdir: str = "/workspace"
    image: str = "test-image"


@dataclass
class FakeSession:
    """Fake session for testing."""

    runtime: FakeRuntime
    working_dir: Path
    spec: DockerEnvironmentSpec | None = None
    last_spawn_env_vars: dict[str, str] | None = None
    last_spawn_mounts: dict[str, dict[str, str] | str] | None = None

    def spawn(self, **_: object) -> FakeRuntime:
        env_vars = cast("dict[str, str] | None", _.get("env_vars"))
        mounts = cast(
            "dict[str, dict[str, str] | str] | None",
            _.get("mounts"),
        )
        self.last_spawn_env_vars = dict(env_vars or {})
        self.last_spawn_mounts = dict(mounts or {})
        return self.runtime


@pytest.fixture
def mock_pricing():
    """Standard pricing for tests."""
    return APIPricing(
        input=0.5,
        output=2.0,
        cache_read=0.1,
    )


@pytest.fixture
def mock_cost_limits():
    """Standard cost limits for tests."""
    return AgentCostLimits(
        step_limit=10,
        cost_limit=100.0,
        net_cost_limit=200.0,
    )


@pytest.fixture
def mock_model_def(mock_pricing):
    """Standard ModelDefinition for tests."""
    return ModelDefinition(
        internal_name="gpt-4-test",
        provider="openai",
        pricing=mock_pricing,
        provider_slugs={"openai": "gpt-4-test"},
    )


@pytest.fixture
def mock_credential():
    """Standard credential for tests."""
    from slop_code.agent_runner.credentials import CredentialType

    return ProviderCredential(
        provider="openai",
        value="test-api-key",
        source="OPENAI_API_KEY",
        destination_key="OPENAI_API_KEY",
        credential_type=CredentialType.ENV_VAR,
    )


class TestCodexConfig:
    """Tests for CodexConfig."""

    def test_version_is_required(self, mock_cost_limits):
        """Version field is required for docker template."""
        with pytest.raises(Exception):  # Pydantic validation error
            CodexConfig(
                type="codex",
                cost_limits=mock_cost_limits,
                # Missing version
            )

    def test_config_with_version(self, mock_cost_limits):
        """Config can be created with version."""
        config = CodexConfig(
            type="codex",
            version="1.0.0",
            cost_limits=mock_cost_limits,
        )
        assert config.version == "1.0.0"
        assert config.binary == "codex"

    def test_get_docker_file_renders_version(self, mock_cost_limits):
        """get_docker_file renders version into template."""
        config = CodexConfig(
            type="codex",
            version="2.5.0",
            cost_limits=mock_cost_limits,
        )
        dockerfile = config.get_docker_file("base-image:latest")
        assert dockerfile is not None
        assert "base-image:latest" in dockerfile
        assert "@openai/codex@2.5.0" in dockerfile

    def test_get_docker_file_enforces_package_and_platform_integrity(
        self, mock_cost_limits
    ):
        """Frozen profiles verify both the wrapper and native package."""
        config = CodexConfig(
            type="codex",
            version="2.5.0",
            npm_package_integrity=VALID_NPM_INTEGRITY,
            npm_linux_arm64_integrity=VALID_NPM_INTEGRITY,
            npm_linux_x64_integrity=VALID_NPM_INTEGRITY,
            cost_limits=mock_cost_limits,
        )

        dockerfile = config.get_docker_file("base-image:latest")

        assert dockerfile is not None
        assert "npm view" not in dockerfile
        assert "npm pack --silent '@openai/codex@2.5.0'" in dockerfile
        assert "platform_version='2.5.0-linux-arm64'" in dockerfile
        assert "platform_version='2.5.0-linux-x64'" in dockerfile
        assert (
            'npm pack --silent "@openai/codex@${platform_version}"'
            in dockerfile
        )
        assert "createHash('sha512')" in dockerfile
        assert "npm install -g --offline --omit=optional" in dockerfile
        assert 'npm install -g \'@openai/codex@2.5.0\'' not in dockerfile
        assert 'tar -xzf "$platform_tarball"' in dockerfile
        assert VALID_NPM_INTEGRITY in dockerfile

        dockerfile_lines = dockerfile.splitlines()
        start = next(
            index
            for index, line in enumerate(dockerfile_lines)
            if line.startswith("RUN set -eu;")
        )
        shell_lines: list[str] = []
        for index, line in enumerate(dockerfile_lines[start:]):
            shell_lines.append(line.removeprefix("RUN ") if index == 0 else line)
            if not line.endswith("\\"):
                break
        subprocess.run(  # noqa: S603
            ["/bin/sh", "-n", "-c", "\n".join(shell_lines)],
            check=True,
        )

    def test_get_docker_file_rejects_partial_integrity_lock(
        self, mock_cost_limits
    ):
        """A partial integrity lock must not look reproducible."""
        with pytest.raises(ValueError, match="requires package"):
            CodexConfig(
                type="codex",
                version="2.5.0",
                npm_package_integrity=VALID_NPM_INTEGRITY,
                cost_limits=mock_cost_limits,
            )

    def test_config_rejects_malformed_integrity(
        self, mock_cost_limits
    ) -> None:
        """Integrity fields must be canonical SHA-512 SRI values."""
        with pytest.raises(ValueError, match="64-byte SHA-512"):
            CodexConfig(
                type="codex",
                version="2.5.0",
                npm_package_integrity="sha512-d3Jvbmc=",
                npm_linux_arm64_integrity=VALID_NPM_INTEGRITY,
                npm_linux_x64_integrity=VALID_NPM_INTEGRITY,
                cost_limits=mock_cost_limits,
            )

    @pytest.mark.parametrize("version", ["0.124.0", "0.146.0"])
    def test_frozen_profile_config_renders_real_npm_alias_target(
        self,
        version: str,
    ) -> None:
        """Checked-in locks render the alias target that exists in npm."""
        repository_root = Path(__file__).resolve().parents[3]
        config_path = (
            repository_root / "configs" / "agents" / f"codex-{version}.yaml"
        )
        raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        config = CodexConfig.model_validate(raw_config)

        dockerfile = config.get_docker_file("base-image:frozen")

        assert dockerfile is not None
        assert f"platform_version='{version}-linux-arm64'" in dockerfile
        assert f"platform_version='{version}-linux-x64'" in dockerfile
        assert (
            'npm pack --silent "@openai/codex@${platform_version}"'
            in dockerfile
        )


class TestCodexAgent:
    """Tests for CodexAgent."""

    def test_from_config_creates_agent(
        self, mock_cost_limits, mock_model_def, mock_credential
    ):
        """_from_config creates agent from config."""
        config = CodexConfig(
            type="codex",
            version="1.0.0",
            cost_limits=mock_cost_limits,
        )

        agent = CodexAgent._from_config(
            config=config,
            model=mock_model_def,
            credential=mock_credential,
            problem_name="test-problem",
            verbose=False,
            image="test-image",
        )

        assert isinstance(agent, CodexAgent)
        assert agent.binary == "codex"

    def test_from_config_requires_image(
        self, mock_cost_limits, mock_model_def, mock_credential
    ):
        """_from_config requires image."""
        config = CodexConfig(
            type="codex",
            version="1.0.0",
            cost_limits=mock_cost_limits,
        )

        with pytest.raises(ValueError, match="requires an image"):
            CodexAgent._from_config(
                config=config,
                model=mock_model_def,
                credential=mock_credential,
                problem_name="test-problem",
                verbose=False,
                image=None,
            )

    def test_setup_and_cleanup(self, tmp_path, mock_cost_limits, mock_pricing):
        """setup() and cleanup() manage session lifecycle."""
        runtime = FakeRuntime()
        session = FakeSession(runtime=runtime, working_dir=tmp_path)

        agent = CodexAgent(
            problem_name="test-problem",
            verbose=False,
            image="test-image",
            cost_limits=mock_cost_limits,
            pricing=mock_pricing,
            credential=None,
            binary="codex",
            model="gpt-4",
            timeout=60,
            thinking=None,
            max_thinking_tokens=None,
            extra_args=[],
            env={},
        )

        # Before setup, session access should raise
        with pytest.raises(Exception):
            _ = agent.session

        agent.setup(session)

        # After setup, session should be accessible
        assert agent.session == session

        agent.cleanup()

        # After cleanup, session is None
        assert agent._session is None

    def test_reset_clears_state(self, tmp_path, mock_cost_limits, mock_pricing):
        """reset() clears internal state."""
        runtime = FakeRuntime()
        session = FakeSession(runtime=runtime, working_dir=tmp_path)

        agent = CodexAgent(
            problem_name="test-problem",
            verbose=False,
            image="test-image",
            cost_limits=mock_cost_limits,
            pricing=mock_pricing,
            credential=None,
            binary="codex",
            model=None,
            timeout=None,
            thinking=None,
            max_thinking_tokens=None,
            extra_args=[],
            env={},
        )

        agent.setup(session)

        # Set some state
        agent._last_prompt = "some prompt"
        agent._last_command = MagicMock()
        agent._telemetry_invocations = [{"ordinal": 1}]
        agent._telemetry_cost_sources = {"local_repricing"}

        agent.reset()

        assert agent._last_prompt == ""
        assert agent._last_command is None
        assert agent._telemetry_invocations == []
        assert agent._telemetry_cost_sources == set()

    def test_build_command_basic(
        self, tmp_path, mock_cost_limits, mock_pricing
    ):
        """_build_command creates correct base command."""
        agent = CodexAgent(
            problem_name="test-problem",
            verbose=False,
            image="test-image",
            cost_limits=mock_cost_limits,
            pricing=mock_pricing,
            credential=None,
            binary="codex",
            model=None,
            timeout=None,
            thinking=None,
            max_thinking_tokens=None,
            extra_args=[],
            env={},
        )

        command = agent._build_command("do something")

        assert command[0] == "codex"
        assert command[1] == "exec"
        assert "'do something'" in command  # shlex.quote wraps prompt
        assert "--skip-git-repo-check" in command
        assert "--json" in command
        assert "--dangerously-bypass-approvals-and-sandbox" in command

    def test_build_command_with_model(self, mock_cost_limits, mock_pricing):
        """_build_command includes model when specified."""
        agent = CodexAgent(
            problem_name="test-problem",
            verbose=False,
            image="test-image",
            cost_limits=mock_cost_limits,
            pricing=mock_pricing,
            credential=None,
            binary="codex",
            model="gpt-4",
            timeout=None,
            thinking=None,
            max_thinking_tokens=None,
            extra_args=[],
            env={},
        )

        command = agent._build_command("do something")

        assert "--model" in command
        model_idx = command.index("--model")
        assert command[model_idx + 1] == "gpt-4"

    def test_build_command_for_retry_resumes_last_exec_session(
        self, mock_cost_limits, mock_pricing
    ):
        agent = CodexAgent(
            problem_name="test-problem",
            verbose=False,
            image="test-image",
            cost_limits=mock_cost_limits,
            pricing=mock_pricing,
            credential=None,
            binary="codex",
            model=None,
            timeout=None,
            thinking=None,
            max_thinking_tokens=None,
            extra_args=[],
            env={},
        )

        command = agent._build_command("continue", resume=True)

        assert command[:4] == ["codex", "exec", "resume", "--last"]

    def test_build_command_with_thinking(self, mock_cost_limits, mock_pricing):
        """_build_command includes thinking when specified."""
        agent = CodexAgent(
            problem_name="test-problem",
            verbose=False,
            image="test-image",
            cost_limits=mock_cost_limits,
            pricing=mock_pricing,
            credential=None,
            binary="codex",
            model=None,
            timeout=None,
            thinking="high",
            max_thinking_tokens=None,
            extra_args=[],
            env={},
        )

        command = agent._build_command("do something")

        assert "--config" in command
        config_idx = command.index("--config")
        assert 'model_reasoning_effort="high"' in command[config_idx + 1]

    def test_build_command_with_extra_args(
        self, mock_cost_limits, mock_pricing
    ):
        """_build_command appends extra_args."""
        agent = CodexAgent(
            problem_name="test-problem",
            verbose=False,
            image="test-image",
            cost_limits=mock_cost_limits,
            pricing=mock_pricing,
            credential=None,
            binary="codex",
            model=None,
            timeout=None,
            thinking=None,
            max_thinking_tokens=None,
            extra_args=["--custom-flag", "value"],
            env={},
        )

        command = agent._build_command("do something")

        assert "--custom-flag" in command
        assert "value" in command

    def test_save_artifacts_writes_files(
        self, tmp_path, mock_cost_limits, mock_pricing
    ):
        """save_artifacts writes prompt and trajectory files."""
        runtime = FakeRuntime()
        session = FakeSession(runtime=runtime, working_dir=tmp_path)

        agent = CodexAgent(
            problem_name="test-problem",
            verbose=False,
            image="test-image",
            cost_limits=mock_cost_limits,
            pricing=mock_pricing,
            credential=None,
            binary="codex",
            model=None,
            timeout=None,
            thinking=None,
            max_thinking_tokens=None,
            extra_args=[],
            env={},
        )

        agent.setup(session)
        agent._last_prompt = "test prompt"

        output_dir = tmp_path / "artifacts"
        agent.save_artifacts(output_dir)

        prompt_file = output_dir / "prompt.txt"
        assert prompt_file.exists()
        assert prompt_file.read_text() == "test prompt"

    def test_parse_line_does_not_trust_uncorrelated_raw_token_count(
        self, mock_pricing
    ):
        """Raw token_count events are only trusted after trace correlation."""
        payload = {
            "type": "event_msg",
            "payload": {
                "type": "token_count",
                "info": {
                    "total_cost": 1.25,
                    "total_token_usage": {
                        "input_tokens": 100,
                        "output_tokens": 50,
                        "cached_input_tokens": 25,
                        "reasoning_output_tokens": 10,
                    },
                },
            },
        }

        cost, tokens, parsed = CodexAgent.parse_line(
            json.dumps(payload), pricing=mock_pricing
        )

        assert cost is None
        assert tokens is None
        assert parsed == payload

    def test_parse_line_uses_inclusive_stdout_token_semantics(
        self, mock_pricing
    ):
        payload = {
            "type": "turn.completed",
            "usage": {
                "input_tokens": 100,
                "output_tokens": 50,
                "cached_input_tokens": 25,
            },
        }

        cost, tokens, parsed = CodexAgent.parse_line(
            json.dumps(payload), pricing=mock_pricing
        )

        assert tokens is not None
        assert tokens.input == 100
        assert tokens.cache_read == 25
        assert tokens.output == 50
        assert tokens.reasoning == 0
        assert cost == pytest.approx(mock_pricing.get_cost(tokens))
        assert parsed == payload
        assert CodexAgent.TELEMETRY_SEMANTICS_VERSION == 5
        assert "inclusive" in CodexAgent.TELEMETRY_SEMANTICS["input_tokens"]
        assert (
            "per-invocation"
            in (CodexAgent.TELEMETRY_SEMANTICS["invocation_totals"])
        )

    def test_parse_line_preserves_reasoning_and_cache_write(
        self, mock_pricing
    ) -> None:
        payload = {
            "type": "turn.completed",
            "usage": {
                "input_tokens": 100,
                "output_tokens": 50,
                "cached_input_tokens": 25,
                "cache_write_input_tokens": 7,
                "reasoning_output_tokens": 11,
            },
        }

        cost, tokens, parsed = CodexAgent.parse_line(
            json.dumps(payload), pricing=mock_pricing
        )

        assert tokens == TokenUsage(
            input=100,
            output=50,
            cache_read=25,
            cache_write=7,
            reasoning=11,
        )
        assert cost == pytest.approx(mock_pricing.get_cost(tokens))
        assert parsed == payload

    @pytest.mark.parametrize(
        "usage",
        (
            None,
            {},
            "bad",
            {
                "input_tokens": "100",
                "output_tokens": 50,
                "cached_input_tokens": 25,
            },
            {
                "input_tokens": 100,
                "output_tokens": -1,
                "cached_input_tokens": 25,
            },
            {
                "input_tokens": 10,
                "output_tokens": 5,
                "cached_input_tokens": 11,
            },
        ),
    )
    def test_parse_line_rejects_malformed_usage(
        self,
        mock_pricing,
        usage: object,
    ) -> None:
        payload = {"type": "turn.completed", "usage": usage}

        cost, tokens, parsed = CodexAgent.parse_line(
            json.dumps(payload), pricing=mock_pricing
        )

        assert cost is None
        assert tokens is None
        assert parsed == payload

    def test_successful_uncorrelated_resume_is_telemetry_error(
        self,
        tmp_path: Path,
        mock_cost_limits,
        mock_pricing,
    ) -> None:
        payloads = [
            {"type": "thread.started", "thread_id": "thread-a"},
            {"type": "thread.started", "thread_id": "thread-b"},
            {
                "type": "turn.completed",
                "usage": {
                    "input_tokens": 100,
                    "output_tokens": 50,
                    "cached_input_tokens": 25,
                },
            },
        ]
        stdout = "".join(f"{json.dumps(payload)}\n" for payload in payloads)
        runtime = FakeRuntime()
        runtime.events = [
            RuntimeEvent(kind="stdout", text=stdout),
            RuntimeEvent(
                kind="finished",
                result=RuntimeResult(
                    exit_code=0,
                    stdout=stdout,
                    stderr="",
                    setup_stdout="",
                    setup_stderr="",
                    elapsed=0.1,
                    timed_out=False,
                ),
            ),
        ]
        agent = CodexAgent(
            problem_name="test-problem",
            verbose=False,
            image="test-image",
            cost_limits=mock_cost_limits,
            pricing=mock_pricing,
            credential=None,
            binary="codex",
            model=None,
            timeout=None,
            thinking=None,
            max_thinking_tokens=None,
            extra_args=[],
            env={},
        )
        agent.setup(
            FakeSession(
                runtime=runtime,
                working_dir=tmp_path,
                spec=DockerEnvironmentSpec(
                    name="test",
                    docker=DockerConfig(image="test-image"),
                ),
            )
        )

        result = agent._run_invocation("retry", resume=True)

        assert result.had_error is True
        assert result.error_message is not None
        assert "could not be correlated" in result.error_message

    def test_parse_line_ignores_non_object_json(self, mock_pricing):
        assert CodexAgent.parse_line("[]", pricing=mock_pricing) == (
            None,
            None,
            None,
        )

    def test_run_uses_codex_reported_total_cost_when_available(
        self, tmp_path, mock_cost_limits, mock_pricing
    ):
        """Reported token_count totals take precedence over local repricing."""
        thread_id = "thread-reported-cost"
        thread_started = {
            "type": "thread.started",
            "thread_id": thread_id,
        }
        turn_completed = {
            "type": "turn.completed",
            "usage": {
                "input_tokens": 100,
                "output_tokens": 50,
                "cached_input_tokens": 25,
            },
        }
        stdout = f"{json.dumps(thread_started)}\n{json.dumps(turn_completed)}\n"
        runtime = FakeRuntime()
        runtime.events = [
            RuntimeEvent(kind="stdout", text=stdout),
            RuntimeEvent(
                kind="finished",
                result=RuntimeResult(
                    exit_code=0,
                    stdout=stdout,
                    stderr="",
                    setup_stdout="",
                    setup_stderr="",
                    elapsed=0.1,
                    timed_out=False,
                ),
            ),
        ]
        spec = DockerEnvironmentSpec(
            name="test",
            docker=DockerConfig(image="test-image"),
        )
        session = FakeSession(
            runtime=runtime,
            working_dir=tmp_path,
            spec=spec,
        )
        agent = CodexAgent(
            problem_name="test-problem",
            verbose=False,
            image="test-image",
            cost_limits=mock_cost_limits,
            pricing=mock_pricing,
            credential=None,
            binary="codex",
            model=None,
            timeout=None,
            thinking=None,
            max_thinking_tokens=None,
            extra_args=[],
            env={},
        )

        agent.setup(session)
        assert agent._trace_dir is not None
        write_codex_trace(
            agent._trace_dir / "rollout.jsonl",
            thread_id=thread_id,
            input_tokens=100,
            cached_input_tokens=25,
            output_tokens=50,
            reasoning_output_tokens=10,
            total_cost=1.25,
        )
        agent.run("do something")

        assert agent.usage.cost == pytest.approx(1.25)
        assert agent.usage.net_tokens.input == 100
        assert agent.usage.net_tokens.output == 50
        assert agent.usage.net_tokens.cache_read == 25
        assert agent.usage.net_tokens.reasoning == 10

    def test_run_uses_codex_trace_token_count_when_stdout_lacks_reasoning(
        self, tmp_path, mock_cost_limits, mock_pricing
    ):
        """Trace token_count totals fill reasoning missing from stdout."""
        thread_id = "thread-reasoning"
        thread_started = {
            "type": "thread.started",
            "thread_id": thread_id,
        }
        stdout_payload = {
            "type": "turn.completed",
            "usage": {
                "input_tokens": 100,
                "output_tokens": 50,
                "cached_input_tokens": 25,
            },
        }
        stdout = f"{json.dumps(thread_started)}\n{json.dumps(stdout_payload)}\n"
        runtime = FakeRuntime()
        runtime.events = [
            RuntimeEvent(kind="stdout", text=stdout),
            RuntimeEvent(
                kind="finished",
                result=RuntimeResult(
                    exit_code=0,
                    stdout=stdout,
                    stderr="",
                    setup_stdout="",
                    setup_stderr="",
                    elapsed=0.1,
                    timed_out=False,
                ),
            ),
        ]
        spec = DockerEnvironmentSpec(
            name="test",
            docker=DockerConfig(image="test-image"),
        )
        session = FakeSession(
            runtime=runtime,
            working_dir=tmp_path,
            spec=spec,
        )
        agent = CodexAgent(
            problem_name="test-problem",
            verbose=False,
            image="test-image",
            cost_limits=mock_cost_limits,
            pricing=mock_pricing,
            credential=None,
            binary="codex",
            model=None,
            timeout=None,
            thinking=None,
            max_thinking_tokens=None,
            extra_args=[],
            env={},
        )

        agent.setup(session)
        assert agent._trace_dir is not None
        trace_file = agent._trace_dir / "rollout.jsonl"
        write_codex_trace(
            trace_file,
            thread_id=thread_id,
            input_tokens=100,
            cached_input_tokens=25,
            output_tokens=50,
            reasoning_output_tokens=10,
        )

        agent.run("do something")

        assert agent.usage.cost == pytest.approx(
            mock_pricing.get_cost(agent.usage.net_tokens)
        )
        assert agent.usage.net_tokens.input == 100
        assert agent.usage.net_tokens.output == 50
        assert agent.usage.net_tokens.cache_read == 25
        assert agent.usage.net_tokens.reasoning == 10

    def test_run_keeps_final_cumulative_usage_and_correlates_exact_thread(
        self, tmp_path, mock_cost_limits, mock_pricing
    ):
        """Repeated cumulative records are not summed or cross-correlated."""
        thread_id = "thread-final-cumulative"
        stdout_events = [
            {"type": "thread.started", "thread_id": thread_id},
            {
                "type": "turn.completed",
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 4,
                    "cached_input_tokens": 2,
                },
            },
            {
                "type": "turn.completed",
                "usage": {
                    "input_tokens": 30,
                    "output_tokens": 8,
                    "cached_input_tokens": 5,
                },
            },
        ]
        stdout = "".join(f"{json.dumps(event)}\n" for event in stdout_events)
        runtime = FakeRuntime()
        runtime.events = [
            RuntimeEvent(kind="stdout", text=stdout),
            RuntimeEvent(
                kind="finished",
                result=RuntimeResult(
                    exit_code=0,
                    stdout=stdout,
                    stderr="",
                    setup_stdout="",
                    setup_stderr="",
                    elapsed=0.1,
                    timed_out=False,
                ),
            ),
        ]
        session = FakeSession(
            runtime=runtime,
            working_dir=tmp_path,
            spec=DockerEnvironmentSpec(
                name="test",
                docker=DockerConfig(image="test-image"),
            ),
        )
        agent = CodexAgent(
            problem_name="test-problem",
            verbose=False,
            image="test-image",
            cost_limits=mock_cost_limits,
            pricing=mock_pricing,
            credential=None,
            binary="codex",
            model=None,
            timeout=None,
            thinking=None,
            max_thinking_tokens=None,
            extra_args=[],
            env={},
        )

        agent.setup(session)
        assert agent._trace_dir is not None
        write_codex_trace(
            agent._trace_dir / "wrong-thread.jsonl",
            thread_id="other-thread",
            input_tokens=999,
            cached_input_tokens=0,
            output_tokens=999,
            reasoning_output_tokens=0,
            total_cost=99.0,
        )
        write_codex_trace(
            agent._trace_dir / "matching-thread.jsonl",
            thread_id=thread_id,
            input_tokens=30,
            cached_input_tokens=5,
            output_tokens=8,
            reasoning_output_tokens=3,
            total_cost=0.75,
        )

        agent.run("do something")

        assert agent.usage.net_tokens.input == 30
        assert agent.usage.net_tokens.cache_read == 5
        assert agent.usage.net_tokens.output == 8
        assert agent.usage.net_tokens.reasoning == 3
        assert agent.usage.cost == pytest.approx(0.75)

    def test_retry_accounts_only_delta_from_cumulative_thread_totals(
        self, tmp_path, mock_cost_limits, mock_pricing
    ):
        """A cumulative resume must not charge the first invocation twice."""
        thread_id = "thread-cumulative-resume"

        def events(
            *,
            input_tokens: int,
            output_tokens: int,
            cached_input_tokens: int,
            exit_code: int,
        ) -> list[RuntimeEvent]:
            payloads = [
                {"type": "thread.started", "thread_id": thread_id},
                {
                    "type": "turn.completed",
                    "usage": {
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "cached_input_tokens": cached_input_tokens,
                    },
                },
            ]
            stdout = "".join(f"{json.dumps(payload)}\n" for payload in payloads)
            return [
                RuntimeEvent(kind="stdout", text=stdout),
                RuntimeEvent(
                    kind="finished",
                    result=RuntimeResult(
                        exit_code=exit_code,
                        stdout=stdout,
                        stderr="",
                        setup_stdout="",
                        setup_stderr="",
                        elapsed=0.1,
                        timed_out=False,
                    ),
                ),
            ]

        runtime = FakeRuntime()
        runtime.event_batches = [
            events(
                input_tokens=100,
                output_tokens=50,
                cached_input_tokens=25,
                exit_code=1,
            ),
            events(
                input_tokens=180,
                output_tokens=90,
                cached_input_tokens=40,
                exit_code=0,
            ),
        ]
        session = FakeSession(
            runtime=runtime,
            working_dir=tmp_path,
            spec=DockerEnvironmentSpec(
                name="test",
                docker=DockerConfig(image="test-image"),
            ),
        )
        agent = CodexAgent(
            problem_name="test-problem",
            verbose=False,
            image="test-image",
            cost_limits=mock_cost_limits,
            pricing=mock_pricing,
            credential=None,
            binary="codex",
            model=None,
            timeout=None,
            thinking=None,
            max_thinking_tokens=None,
            extra_args=[],
            env={},
        )
        agent.setup(session)
        assert agent._trace_dir is not None
        trace_file = agent._trace_dir / "rollout.jsonl"

        cumulative_trace = [
            {
                "input_tokens": 100,
                "cached_input_tokens": 25,
                "output_tokens": 50,
                "reasoning_output_tokens": 10,
                "total_cost": 1.25,
            },
            {
                "input_tokens": 180,
                "cached_input_tokens": 40,
                "output_tokens": 90,
                "reasoning_output_tokens": 18,
                "total_cost": 2.0,
            },
        ]

        def update_trace(invocation: int) -> None:
            write_codex_trace(
                trace_file,
                thread_id=thread_id,
                **cumulative_trace[invocation],
            )

        runtime.before_stream = update_trace

        result = agent.run_checkpoint("do something")

        assert result.had_error is False
        assert len(runtime.stream_calls) == 2
        assert "exec resume --last" in runtime.stream_calls[1][0][0]
        assert result.usage.net_tokens == TokenUsage(
            input=180,
            output=90,
            cache_read=40,
            reasoning=18,
        )
        assert result.usage.current_tokens == TokenUsage(
            input=80,
            output=40,
            cache_read=15,
            reasoning=8,
        )
        assert result.usage.cost == pytest.approx(2.0)

        artifact_dir = tmp_path / "artifacts"
        agent.save_artifacts(artifact_dir)
        artifact_text = (
            artifact_dir / CodexAgent.TELEMETRY_FILENAME
        ).read_text()
        telemetry = json.loads(artifact_text)

        assert telemetry["schema_version"] == 1
        assert telemetry["telemetry_semantics_version"] == 5
        assert "inclusive" in telemetry["semantics"]["input_tokens"]
        assert "inclusive" in telemetry["semantics"]["output_tokens"]
        assert "per-invocation" in telemetry["semantics"]["invocation_totals"]
        assert telemetry["checkpoint_totals"] == {
            "input_tokens": 180,
            "output_tokens": 90,
            "cached_input_tokens": 40,
            "cache_write_tokens": 0,
            "reasoning_tokens": 18,
            "total_tokens": 270,
            "steps": 0,
            "cost_usd": 2.0,
            "cost_accounting": "reported",
            "reported_cost_used": True,
            "local_repricing_used": False,
        }
        assert telemetry["invocation_count"] == 2
        assert telemetry["invocations_omitted"] == 0
        first, second = telemetry["invocations"]
        assert first["trace_correlation"] == {
            "status": "correlated",
            "reason": None,
            "usage_source": "stdout_and_trace",
        }
        assert first["delta_status"] == "initial_thread_cumulative"
        assert first["accounted_delta"]["input_tokens"] == 100
        assert first["cost_accounting"] == {
            "source": "codex_reported_delta",
            "usd": 1.25,
            "reported_cost_used": True,
            "local_repricing_used": False,
        }
        assert second["resume"] is True
        assert second["thread_reference"] == "stdout"
        assert second["delta_status"] == "cumulative_thread_delta"
        assert second["cumulative_totals"]["input_tokens"] == 180
        assert second["accounted_delta"] == {
            "input_tokens": 80,
            "output_tokens": 40,
            "cached_input_tokens": 15,
            "cache_write_tokens": 0,
            "reasoning_tokens": 8,
            "total_tokens": 120,
        }
        assert second["cost_accounting"]["usd"] == pytest.approx(0.75)
        # Correlation is evidenced without persisting the thread identifier.
        assert thread_id not in artifact_text

    def test_duplicate_matching_rollouts_fall_back_to_stdout_and_local_cost(
        self, tmp_path, mock_cost_limits, mock_pricing
    ):
        """Ambiguous raw traces cannot supply reasoning or reported cost."""
        thread_id = "thread-ambiguous"
        stdout_events = [
            {"type": "thread.started", "thread_id": thread_id},
            {
                "type": "turn.completed",
                "usage": {
                    "input_tokens": 100,
                    "output_tokens": 50,
                    "cached_input_tokens": 25,
                },
            },
        ]
        stdout = "".join(f"{json.dumps(event)}\n" for event in stdout_events)
        runtime = FakeRuntime()
        runtime.events = [
            RuntimeEvent(kind="stdout", text=stdout),
            RuntimeEvent(
                kind="finished",
                result=RuntimeResult(
                    exit_code=0,
                    stdout=stdout,
                    stderr="",
                    setup_stdout="",
                    setup_stderr="",
                    elapsed=0.1,
                    timed_out=False,
                ),
            ),
        ]
        session = FakeSession(
            runtime=runtime,
            working_dir=tmp_path,
            spec=DockerEnvironmentSpec(
                name="test",
                docker=DockerConfig(image="test-image"),
            ),
        )
        agent = CodexAgent(
            problem_name="test-problem",
            verbose=False,
            image="test-image",
            cost_limits=mock_cost_limits,
            pricing=mock_pricing,
            credential=None,
            binary="codex",
            model=None,
            timeout=None,
            thinking=None,
            max_thinking_tokens=None,
            extra_args=[],
            env={},
        )
        logger = FakeLogger()
        agent.log = logger

        agent.setup(session)
        assert agent._trace_dir is not None
        for index in (1, 2):
            write_codex_trace(
                agent._trace_dir / f"duplicate-{index}.jsonl",
                thread_id=thread_id,
                input_tokens=100,
                cached_input_tokens=25,
                output_tokens=50,
                reasoning_output_tokens=10,
                total_cost=1.25,
            )

        agent.run("do something")

        assert agent.usage.net_tokens.input == 100
        assert agent.usage.net_tokens.cache_read == 25
        assert agent.usage.net_tokens.output == 50
        assert agent.usage.net_tokens.reasoning == 0
        assert agent.usage.cost == pytest.approx(
            mock_pricing.get_cost(agent.usage.net_tokens)
        )
        assert any(
            event == "agent.codex.telemetry.trace_fallback"
            and details.get("matches") == 2
            for event, details in logger.warning_calls
        )

        artifact_dir = tmp_path / "fallback-artifacts"
        agent.save_artifacts(artifact_dir)
        telemetry = json.loads(
            (artifact_dir / CodexAgent.TELEMETRY_FILENAME).read_text()
        )
        assert telemetry["checkpoint_totals"]["cost_accounting"] == ("repriced")
        assert telemetry["checkpoint_totals"]["reported_cost_used"] is False
        assert telemetry["checkpoint_totals"]["local_repricing_used"] is True
        invocation = telemetry["invocations"][0]
        assert invocation["trace_correlation"] == {
            "status": "fallback",
            "reason": "raw rollout match count is not one",
            "match_count": 2,
        }
        assert invocation["cost_accounting"]["source"] == "local_repricing"
        assert thread_id not in json.dumps(telemetry)

    def test_success_without_stdout_usage_recovers_from_exact_trace(
        self, tmp_path, mock_cost_limits, mock_pricing
    ):
        """A uniquely correlated rollout can replace omitted stdout usage."""
        thread_id = "thread-trace-only"
        thread_event = {"type": "thread.started", "thread_id": thread_id}
        stdout = f"{json.dumps(thread_event)}\n"
        runtime = FakeRuntime()
        runtime.events = [
            RuntimeEvent(kind="stdout", text=stdout),
            RuntimeEvent(
                kind="finished",
                result=RuntimeResult(
                    exit_code=0,
                    stdout=stdout,
                    stderr="",
                    setup_stdout="",
                    setup_stderr="",
                    elapsed=0.1,
                    timed_out=False,
                ),
            ),
        ]
        session = FakeSession(
            runtime=runtime,
            working_dir=tmp_path,
            spec=DockerEnvironmentSpec(
                name="test",
                docker=DockerConfig(image="test-image"),
            ),
        )
        agent = CodexAgent(
            problem_name="test-problem",
            verbose=False,
            image="test-image",
            cost_limits=mock_cost_limits,
            pricing=mock_pricing,
            credential=None,
            binary="codex",
            model=None,
            timeout=None,
            thinking=None,
            max_thinking_tokens=None,
            extra_args=[],
            env={},
        )
        agent.setup(session)
        assert agent._trace_dir is not None
        write_codex_trace(
            agent._trace_dir / "rollout.jsonl",
            thread_id=thread_id,
            input_tokens=120,
            cached_input_tokens=20,
            output_tokens=45,
            reasoning_output_tokens=15,
            total_cost=1.5,
        )

        agent.run("do something")

        assert agent.usage.net_tokens == TokenUsage(
            input=120,
            output=45,
            cache_read=20,
            reasoning=15,
        )
        assert agent.usage.cost == pytest.approx(1.5)
        artifact_dir = tmp_path / "trace-only-artifacts"
        agent.save_artifacts(artifact_dir)
        telemetry = json.loads(
            (artifact_dir / CodexAgent.TELEMETRY_FILENAME).read_text()
        )
        invocation = telemetry["invocations"][0]
        assert invocation["stdout_usage_present"] is False
        assert invocation["trace_correlation"] == {
            "status": "correlated",
            "reason": None,
            "usage_source": "trace_only",
        }

    def test_success_without_any_trustworthy_usage_is_an_error(
        self, tmp_path, mock_cost_limits, mock_pricing
    ):
        """A successful CLI exit must not be recorded as zero-token success."""
        thread_event = {
            "type": "thread.started",
            "thread_id": "thread-missing-usage",
        }
        stdout = f"{json.dumps(thread_event)}\n"
        runtime = FakeRuntime()
        runtime.events = [
            RuntimeEvent(kind="stdout", text=stdout),
            RuntimeEvent(
                kind="finished",
                result=RuntimeResult(
                    exit_code=0,
                    stdout=stdout,
                    stderr="",
                    setup_stdout="",
                    setup_stderr="",
                    elapsed=0.1,
                    timed_out=False,
                ),
            ),
        ]
        session = FakeSession(
            runtime=runtime,
            working_dir=tmp_path,
            spec=DockerEnvironmentSpec(
                name="test",
                docker=DockerConfig(image="test-image"),
            ),
        )
        agent = CodexAgent(
            problem_name="test-problem",
            verbose=False,
            image="test-image",
            cost_limits=mock_cost_limits.model_copy(update={"max_retries": 0}),
            pricing=mock_pricing,
            credential=None,
            binary="codex",
            model=None,
            timeout=None,
            thinking=None,
            max_thinking_tokens=None,
            extra_args=[],
            env={},
        )
        agent.setup(session)

        result = agent.run_checkpoint("do something")

        assert result.had_error is True
        assert result.error_message is not None
        assert "telemetry integrity failure" in result.error_message
        assert agent.usage.net_tokens == TokenUsage()
        artifact_dir = tmp_path / "missing-usage-artifacts"
        agent.save_artifacts(artifact_dir)
        telemetry = json.loads(
            (artifact_dir / CodexAgent.TELEMETRY_FILENAME).read_text()
        )
        invocation = telemetry["invocations"][0]
        assert invocation["stdout_usage_present"] is False
        assert invocation["trace_correlation"] == {
            "status": "fallback",
            "reason": "raw rollout match count is not one",
            "match_count": 0,
        }

    def test_trace_reconciliation_does_not_apply_old_cache_arithmetic(
        self, tmp_path, mock_cost_limits, mock_pricing
    ):
        """Raw input must equal inclusive stdout input without adding cache."""
        agent = CodexAgent(
            problem_name="test-problem",
            verbose=False,
            image="test-image",
            cost_limits=mock_cost_limits,
            pricing=mock_pricing,
            credential=None,
            binary="codex",
            model=None,
            timeout=None,
            thinking=None,
            max_thinking_tokens=None,
            extra_args=[],
            env={},
        )
        logger = FakeLogger()
        agent.log = logger
        agent._trace_dir = tmp_path
        write_codex_trace(
            tmp_path / "rollout.jsonl",
            thread_id="thread-cache-semantics",
            # This would match only if the old implementation added cache.
            input_tokens=75,
            cached_input_tokens=25,
            output_tokens=50,
            reasoning_output_tokens=10,
            total_cost=1.25,
        )

        reconciled = agent._reconcile_trace_usage(
            thread_id="thread-cache-semantics",
            stdout_tokens=TokenUsage(
                input=100,
                cache_read=25,
                output=50,
            ),
        )

        assert reconciled == (None, None, None)
        assert any(
            details.get("reason") == "raw rollout usage does not match stdout"
            for _, details in logger.warning_calls
        )

    def test_save_artifacts_copies_codex_traces(
        self, tmp_path, mock_cost_limits, mock_pricing
    ):
        """save_artifacts copies Codex trace jsonl files from home."""
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

        agent = CodexAgent(
            problem_name="test-problem",
            verbose=False,
            image="test-image",
            cost_limits=mock_cost_limits,
            pricing=mock_pricing,
            credential=None,
            binary="codex",
            model=None,
            timeout=None,
            thinking=None,
            max_thinking_tokens=None,
            extra_args=[],
            env={},
        )

        agent.setup(session)
        assert agent._trace_dir is not None
        trace_dir = agent._trace_dir / "traces"
        trace_dir.mkdir(parents=True, exist_ok=True)
        trace_file = trace_dir / "trace.jsonl"
        trace_file.write_text('{"type":"turn.started"}\n')

        output_dir = tmp_path / "artifacts"
        agent.save_artifacts(output_dir)

        saved_trace = output_dir / "trace.jsonl"
        assert saved_trace.exists()
        assert saved_trace.read_text() == trace_file.read_text()

    def test_setup_uses_default_home_for_docker(
        self, tmp_path, mock_cost_limits, mock_pricing
    ):
        """setup keeps HOME at agent home and mounts codex dir."""
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

        agent = CodexAgent(
            problem_name="test-problem",
            verbose=False,
            image="test-image",
            cost_limits=mock_cost_limits,
            pricing=mock_pricing,
            credential=None,
            binary="codex",
            model=None,
            timeout=None,
            thinking=None,
            max_thinking_tokens=None,
            extra_args=[],
            env={},
        )

        agent.setup(session)

        assert session.last_spawn_env_vars is not None
        assert session.last_spawn_env_vars.get("HOME") == HOME_PATH
        assert session.last_spawn_mounts is not None
        assert any(
            isinstance(value, dict)
            and value.get("bind") == f"{HOME_PATH}/.codex"
            for value in session.last_spawn_mounts.values()
        )

    def test_save_artifacts_logs_trace_counts(
        self, tmp_path, mock_cost_limits, mock_pricing
    ):
        """_save_codex_traces logs discovered and saved trace counts."""
        agent = CodexAgent(
            problem_name="test-problem",
            verbose=False,
            image="test-image",
            cost_limits=mock_cost_limits,
            pricing=mock_pricing,
            credential=None,
            binary="codex",
            model=None,
            timeout=None,
            thinking=None,
            max_thinking_tokens=None,
            extra_args=[],
            env={},
        )
        logger = FakeLogger()
        agent.log = logger

        trace_dir = tmp_path / "codex_traces"
        trace_dir.mkdir(parents=True, exist_ok=True)
        (trace_dir / "trace.jsonl").write_text('{"type":"turn.started"}\n')
        agent._trace_dir = trace_dir

        output_dir = tmp_path / "artifacts"
        agent._save_codex_traces(output_dir)

        assert any(
            event == "agent.codex.traces.found" and kwargs.get("files") == 1
            for event, kwargs in logger.debug_calls
        )
        assert any(
            event == "agent.codex.traces.saved" and kwargs.get("saved") == 1
            for event, kwargs in logger.debug_calls
        )

    def test_save_artifacts_only_copies_new_codex_trace_files_between_checkpoints(
        self, tmp_path, mock_cost_limits, mock_pricing
    ):
        """Later checkpoints should only save newly created Codex trace files."""
        agent = CodexAgent(
            problem_name="test-problem",
            verbose=False,
            image="test-image",
            cost_limits=mock_cost_limits,
            pricing=mock_pricing,
            credential=None,
            binary="codex",
            model=None,
            timeout=None,
            thinking=None,
            max_thinking_tokens=None,
            extra_args=[],
            env={},
        )

        trace_dir = tmp_path / "codex_traces"
        first_trace = trace_dir / "trace1.jsonl"
        trace_dir.mkdir(parents=True, exist_ok=True)
        first_trace.write_text('{"type":"turn.started"}\n')
        agent._trace_dir = trace_dir

        first_output = tmp_path / "checkpoint_1"
        agent._save_codex_traces(first_output)
        assert (first_output / "trace1.jsonl").exists()

        agent.reset()

        second_trace = trace_dir / "trace2.jsonl"
        second_trace.write_text('{"type":"turn.completed"}\n')

        second_output = tmp_path / "checkpoint_2"
        agent._save_codex_traces(second_output)

        assert not (second_output / "trace1.jsonl").exists()
        saved_new_trace = second_output / "trace2.jsonl"
        assert saved_new_trace.exists()
        assert saved_new_trace.read_text() == second_trace.read_text()

    def test_build_command_with_thinking_disabled(
        self, mock_cost_limits, mock_pricing
    ):
        """_build_command sets model_max_output_tokens=0 when thinking is disabled."""
        agent = CodexAgent(
            problem_name="test-problem",
            verbose=False,
            image="test-image",
            cost_limits=mock_cost_limits,
            pricing=mock_pricing,
            credential=None,
            binary="codex",
            model=None,
            timeout=None,
            thinking="disabled",
            max_thinking_tokens=None,
            extra_args=[],
            env={},
        )

        command = agent._build_command("do something")

        assert "--config" in command
        config_idx = command.index("--config")
        assert "model_max_output_tokens=0" in command[config_idx + 1]
        # model_reasoning_effort should NOT be present
        command_str = " ".join(command)
        assert "model_reasoning_effort" not in command_str

    def test_build_command_with_max_thinking_tokens(
        self, mock_cost_limits, mock_pricing
    ):
        """_build_command includes max_thinking_tokens when specified."""
        agent = CodexAgent(
            problem_name="test-problem",
            verbose=False,
            image="test-image",
            cost_limits=mock_cost_limits,
            pricing=mock_pricing,
            credential=None,
            binary="codex",
            model=None,
            timeout=None,
            thinking=None,
            max_thinking_tokens=8192,
            extra_args=[],
            env={},
        )

        command = agent._build_command("do something")

        assert "--config" in command
        config_idx = command.index("--config")
        assert "model_max_output_tokens=8192" in command[config_idx + 1]
