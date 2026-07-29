"Codex agent implementation."

from __future__ import annotations

import base64
import binascii
import functools
import json
import shlex
import shutil
import tempfile
import typing as tp
from dataclasses import dataclass
from pathlib import Path

from jinja2 import Template
from pydantic import Field
from pydantic import field_validator
from pydantic import model_validator

from slop_code.agent_runner.agent import RETRY_PROMPT
from slop_code.agent_runner.agent import Agent
from slop_code.agent_runner.agent import AgentConfigBase
from slop_code.agent_runner.agents.cli_utils import AgentCommandResult
from slop_code.agent_runner.agents.cli_utils import stream_cli_command
from slop_code.agent_runner.agents.utils import HOME_PATH
from slop_code.agent_runner.agents.utils import copy_jsonl_files
from slop_code.agent_runner.agents.utils import find_jsonl_files
from slop_code.agent_runner.credentials import CredentialType
from slop_code.agent_runner.credentials import ProviderCredential
from slop_code.agent_runner.models import AgentCostLimits
from slop_code.agent_runner.models import AgentError
from slop_code.agent_runner.registry import register_agent
from slop_code.common.llms import APIPricing
from slop_code.common.llms import ModelDefinition
from slop_code.common.llms import ThinkingPreset
from slop_code.common.llms import TokenUsage
from slop_code.common.temp import temporary_directory
from slop_code.execution import DockerEnvironmentSpec
from slop_code.execution import EnvironmentSpec
from slop_code.execution import Session
from slop_code.execution import StreamingRuntime
from slop_code.logging import get_logger

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class _CodexCumulativeTelemetry:
    """Last trustworthy cumulative totals observed for one Codex thread."""

    tokens: TokenUsage
    reasoning_tokens: int | None
    reported_cost: float | None


class CodexConfig(AgentConfigBase):
    """Configuration for ``CodexAgent`` instances."""

    type: tp.Literal["codex"] = "codex"
    version: str
    binary: str = "codex"
    docker_template: Path = Path(__file__).parent / "docker.j2"
    extra_args: list[str] = Field(
        default_factory=list,
        description="Additional arguments appended to the CLI invocation.",
    )
    env: dict[str, str] = Field(
        default_factory=dict,
        description="Environment variable overrides applied to the invocation.",
    )
    timeout: int | None = Field(
        default=None,
        description="Optional timeout (in seconds) for the CLI invocation.",
    )
    npm_package_integrity: str | None = Field(
        default=None,
        description="Expected npm dist.integrity for @openai/codex.",
    )
    npm_linux_arm64_integrity: str | None = Field(
        default=None,
        description=(
            "Expected npm integrity for the @openai/codex Linux arm64 "
            "alias target."
        ),
    )
    npm_linux_x64_integrity: str | None = Field(
        default=None,
        description=(
            "Expected npm integrity for the @openai/codex Linux x64 "
            "alias target."
        ),
    )

    @field_validator(
        "npm_package_integrity",
        "npm_linux_arm64_integrity",
        "npm_linux_x64_integrity",
    )
    @classmethod
    def _validate_npm_integrity(cls, value: str | None) -> str | None:
        """Require one canonical SHA-512 Subresource Integrity value."""
        if value is None:
            return None
        prefix = "sha512-"
        if not value.startswith(prefix):
            raise ValueError("Codex npm integrity must start with 'sha512-'")
        try:
            digest = base64.b64decode(
                value.removeprefix(prefix),
                validate=True,
            )
        except (binascii.Error, ValueError) as exc:
            raise ValueError(
                "Codex npm integrity must contain valid base64"
            ) from exc
        if len(digest) != 64:
            raise ValueError(
                "Codex npm integrity must contain a 64-byte SHA-512 digest"
            )
        return value

    @model_validator(mode="after")
    def _validate_complete_npm_integrity_lock(self) -> tp.Self:
        """Prevent a partial platform lock from appearing reproducible."""
        integrity_values = (
            self.npm_package_integrity,
            self.npm_linux_arm64_integrity,
            self.npm_linux_x64_integrity,
        )
        if any(integrity_values) and not all(integrity_values):
            raise ValueError(
                "Codex npm integrity enforcement requires package, "
                "linux-arm64, and linux-x64 integrity values"
            )
        return self

    def get_docker_file(self, base_image: str) -> str | None:
        """Render the Docker template with version."""
        if self.docker_template is None:
            return None
        integrity_values = (
            self.npm_package_integrity,
            self.npm_linux_arm64_integrity,
            self.npm_linux_x64_integrity,
        )
        if any(integrity_values) and not all(integrity_values):
            raise ValueError(
                "Codex npm integrity enforcement requires package, "
                "linux-arm64, and linux-x64 integrity values"
            )
        template = self.docker_template.read_text()
        return Template(template).render(
            base_image=base_image,
            version=self.version,
            npm_package_integrity=self.npm_package_integrity,
            npm_linux_arm64_integrity=self.npm_linux_arm64_integrity,
            npm_linux_x64_integrity=self.npm_linux_x64_integrity,
        )


class CodexAgent(Agent):
    """Agent implementation built on top of the Codex CLI executor."""

    PROMPT_FILENAME = "prompt.txt"
    STDOUT_FILENAME = "stdout.jsonl"
    STDERR_FILENAME = "stderr.log"
    TELEMETRY_FILENAME = "telemetry.json"
    TELEMETRY_ARTIFACT_SCHEMA_VERSION = 1
    TELEMETRY_MAX_INVOCATIONS = 64
    TELEMETRY_SEMANTICS_VERSION = 5
    TELEMETRY_SEMANTICS: tp.ClassVar[dict[str, str]] = {
        "input_tokens": "input tokens inclusive of cached input",
        "cached_input_tokens": "cached subset of input tokens",
        "output_tokens": "output tokens inclusive of reasoning",
        "reasoning_tokens": "reasoning subset of output tokens",
        "invocation_totals": (
            "per-invocation deltas from cumulative Codex thread totals"
        ),
    }

    def __init__(
        self,
        problem_name: str,
        verbose: bool,  # noqa: FBT001
        image: str,
        # From base config
        cost_limits: AgentCostLimits,
        pricing: APIPricing | None,
        credential: ProviderCredential | None,
        # Codex specific
        binary: str,
        model: str,
        timeout: int | None,
        thinking: ThinkingPreset | None,
        max_thinking_tokens: int | None,
        extra_args: list[str],
        env: dict[str, str],
    ) -> None:
        super().__init__(
            agent_name="codex",
            problem_name=problem_name,
            cost_limits=cost_limits,
            pricing=pricing,
            verbose=verbose,
        )

        # Store all config values as instance attributes
        self.credential = credential
        self.binary = binary
        self.model = model
        self.timeout = timeout
        self.thinking = thinking
        self.max_thinking_tokens = max_thinking_tokens
        self.extra_args = extra_args
        self.env = env

        self._image = image
        self._session: Session | None = None

        self._environment: EnvironmentSpec | None = None
        self._runtime: StreamingRuntime | None = None
        self._trace_tmp: tempfile.TemporaryDirectory | None = None
        self._trace_dir: Path | None = None
        self._saved_trace_paths: set[Path] = set()
        self._telemetry_baselines: dict[str, _CodexCumulativeTelemetry] = {}
        self._active_telemetry_thread_id: str | None = None
        self._telemetry_invocations: list[dict[str, tp.Any]] = []
        self._telemetry_invocations_omitted = 0
        self._telemetry_cost_sources: set[str] = set()
        self._pending_telemetry_evidence: dict[str, tp.Any] | None = None
        self._current_trace_evidence: dict[str, tp.Any] = {
            "status": "not_attempted"
        }

        # Get auth file from credential if it's a file credential
        self._auth_file: Path | None = None
        if (
            self.credential is not None
            and self.credential.credential_type == CredentialType.FILE
        ):
            candidate = Path(self.credential.source)
            self._auth_file = candidate if candidate.exists() else None

        self._last_prompt: str = ""
        self._last_command: AgentCommandResult | None = None

    @classmethod
    def _from_config(
        cls,
        config: AgentConfigBase,
        model: ModelDefinition,
        credential: ProviderCredential,
        problem_name: str,
        verbose: bool,  # noqa: FBT001
        image: str | None,
        thinking_preset: ThinkingPreset | None = None,
        thinking_max_tokens: int | None = None,
    ) -> Agent:
        """Create a CodexAgent from a CodexConfig."""
        if not isinstance(config, CodexConfig):
            raise TypeError(
                f"Expected CodexConfig, got {type(config).__name__}"
            )
        if image is None:
            raise ValueError("CodexAgent requires an image")

        # Get model slug for API calls
        model_slug = model.get_model_slug(credential.provider)

        # Resolve thinking: CLI/config override > model default
        thinking: ThinkingPreset | None = thinking_preset
        max_thinking_tokens: int | None = thinking_max_tokens
        if thinking is None and max_thinking_tokens is None:
            thinking, max_thinking_tokens = model.get_thinking_config("codex")

        return cls(
            problem_name=problem_name,
            verbose=verbose,
            image=image,
            cost_limits=config.cost_limits,
            pricing=model.pricing,
            credential=credential,
            binary=config.binary,
            model=model_slug,
            timeout=config.timeout,
            thinking=thinking,
            max_thinking_tokens=max_thinking_tokens,
            extra_args=config.extra_args,
            env=config.env,
        )

    @staticmethod
    def parse_line(
        line: str,
        pricing: APIPricing | None = None,
    ) -> tuple[float | None, TokenUsage | None, dict | None]:
        """Parse a single JSONL line from Codex output.

        Returns (cost, tokens, payload) matching Claude's pattern.
        """
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            return None, None, None
        if not isinstance(payload, dict):
            return None, None, None

        if payload.get("type") != "turn.completed":
            return None, None, payload

        usage = payload.get("usage")
        if not isinstance(usage, dict):
            log.warning(
                "agent.codex.telemetry.stdout_usage_invalid",
                usage_type=type(usage).__name__,
            )
            return None, None, payload

        def token_count(key: str, *, required: bool) -> int | None:
            if key not in usage:
                return None if required else 0
            value = usage[key]
            if type(value) is not int or value < 0:
                return None
            return value

        input_tokens = token_count("input_tokens", required=True)
        output_tokens = token_count("output_tokens", required=True)
        cache_read_tokens = token_count("cached_input_tokens", required=True)
        cache_write_tokens = token_count(
            "cache_write_input_tokens",
            required=False,
        )
        reasoning_tokens = token_count(
            "reasoning_output_tokens",
            required=False,
        )
        counts = (
            input_tokens,
            output_tokens,
            cache_read_tokens,
            cache_write_tokens,
            reasoning_tokens,
        )
        if any(value is None for value in counts):
            log.warning(
                "agent.codex.telemetry.stdout_usage_invalid",
                reason="missing or invalid token count",
            )
            return None, None, payload
        input_tokens = tp.cast("int", input_tokens)
        output_tokens = tp.cast("int", output_tokens)
        cache_read_tokens = tp.cast("int", cache_read_tokens)
        cache_write_tokens = tp.cast("int", cache_write_tokens)
        reasoning_tokens = tp.cast("int", reasoning_tokens)
        if (
            cache_read_tokens > input_tokens
            or reasoning_tokens > output_tokens
        ):
            log.warning(
                "agent.codex.telemetry.stdout_usage_invalid",
                reason="token subsets exceed inclusive totals",
            )
            return None, None, payload

        tokens = TokenUsage(
            # Codex JSON reports input_tokens inclusive of the cached subset.
            # APIPricing.get_cost() subtracts cache_read before applying the
            # full input rate, so these fields must remain inclusive here.
            input=input_tokens,
            output=output_tokens,
            cache_read=cache_read_tokens,
            cache_write=cache_write_tokens,
            reasoning=reasoning_tokens,
        )

        cost = pricing.get_cost(tokens) if pricing else 0.0
        return (cost, tokens, payload)

    @property
    def session(self) -> Session:
        if self._session is None:
            raise AgentError("CodexAgent has not been set up with a session")
        return self._session

    @property
    def spec(self) -> EnvironmentSpec:
        if self._environment is None:
            raise AgentError("CodexAgent has not been set up with a session")
        return self._environment

    @property
    def runtime(self) -> StreamingRuntime:
        if self._runtime is None:
            raise AgentError("CodexAgent has not been set up with a runtime")
        return self._runtime

    def setup(
        self,
        session: Session,
    ) -> None:
        self._session = session
        self._environment = session.spec
        self._saved_trace_paths = set()
        self._reset_telemetry_state()
        mounts: dict[str, dict[str, str] | str] = {}
        if isinstance(session.spec, DockerEnvironmentSpec):
            self._trace_tmp = temporary_directory()
            self._trace_dir = Path(self._trace_tmp.name)
            self._trace_dir.mkdir(parents=True, exist_ok=True)
            self._trace_dir.chmod(0o777)
            if self._auth_file is not None:
                shutil.copy2(self._auth_file, self._trace_dir / "auth.json")
            mounts[str(self._trace_dir)] = {
                "bind": f"{HOME_PATH}/.codex",
                "mode": "rw",
            }
        self._runtime = session.spawn(
            mounts=mounts,
            env_vars={
                "HOME": HOME_PATH,
            },
            image=self._image,
            user="agent",
            disable_setup=True,
        )

    def run(self, task: str) -> None:
        self._last_prompt = task
        self._last_command = None

        log_kwargs: dict[str, tp.Any] = {
            "workspace": str(self.session.working_dir),
            "prompt_chars": len(task),
            "environment": self.session.spec.type,
            "extra_args": self.extra_args,
        }
        if isinstance(self.session.spec, DockerEnvironmentSpec):
            log_kwargs["image"] = self.session.spec.docker.image
        self.log.info("agent.codex.start", **log_kwargs)

        command_result = self._run_invocation(task)
        self._last_command = command_result

        self._sync_usage(command_result.usage_totals)

        runtime_result = command_result.result
        if runtime_result is None:
            message = "Codex process failed to start"
            self.log.error(
                "agent.codex.start_failed",
                error_message=message,
                agent_message=command_result.error_message,
                stdout=command_result.stdout,
                stderr=command_result.stderr,
            )
            raise AgentError(message)
        if runtime_result.timed_out:
            message = (
                f"Codex process timed out after {self.timeout}s."
                if self.timeout is not None
                else "Codex process timed out."
            )
            self.log.error(
                "agent.codex.timeout",
                error_message=message,
                timeout=self.timeout,
            )
            raise AgentError(message)

        if runtime_result.exit_code != 0:
            message = f"Codex process failed with exit code {runtime_result.exit_code}"
            if runtime_result.stderr:
                message = f"{message}\n--- Stderr ---\n{runtime_result.stderr.strip()}"
            self.log.error(
                "agent.codex.exit",
                error_message=message,
                exit_code=runtime_result.exit_code,
            )
            raise AgentError(message)
        if command_result.had_error:
            message = command_result.error_message or (
                "Codex invocation failed telemetry integrity checks"
            )
            self.log.error(
                "agent.codex.telemetry.integrity_failure",
                error_message=message,
            )
            raise AgentError(message)

    def retry(self) -> None:
        self._last_prompt = RETRY_PROMPT
        self._last_command = None

        self.log.info(
            "agent.codex.retry",
            workspace=str(self.session.working_dir),
            environment=self.session.spec.type,
        )

        command_result = self._run_invocation(RETRY_PROMPT, resume=True)
        self._last_command = command_result

        self._sync_usage(command_result.usage_totals)

        runtime_result = command_result.result
        if runtime_result is None:
            message = "Codex retry process failed to start"
            self.log.error(
                "agent.codex.retry.start_failed",
                error_message=message,
                agent_message=command_result.error_message,
            )
            raise AgentError(message)
        if runtime_result.timed_out:
            message = (
                f"Codex retry process timed out after {self.timeout}s."
                if self.timeout is not None
                else "Codex retry process timed out."
            )
            self.log.error(
                "agent.codex.retry.timeout",
                error_message=message,
                timeout=self.timeout,
            )
            raise AgentError(message)

        if runtime_result.exit_code != 0:
            message = (
                "Codex retry process failed with exit code "
                f"{runtime_result.exit_code}"
            )
            if runtime_result.stderr:
                message = f"{message}\n--- Stderr ---\n{runtime_result.stderr.strip()}"
            self.log.error(
                "agent.codex.retry.exit",
                error_message=message,
                exit_code=runtime_result.exit_code,
            )
            raise AgentError(message)
        if command_result.had_error:
            message = command_result.error_message or (
                "Codex retry invocation failed telemetry integrity checks"
            )
            self.log.error(
                "agent.codex.retry.telemetry.integrity_failure",
                error_message=message,
            )
            raise AgentError(message)

    def _run_invocation(
        self,
        task: str,
        *,
        resume: bool = False,
    ) -> AgentCommandResult:
        """Execute a Codex CLI invocation and return results."""
        command, env_overrides = self._prepare_runtime_execution(
            task,
            resume=resume,
        )

        if self._session is None:
            raise AgentError("CodexAgent has not been set up with a session")
        command_str = " ".join(command)

        # Use partial to bind pricing to parse_line
        parser = tp.cast(
            "tp.Callable[[str], tuple[float | None, TokenUsage | None, dict]]",
            functools.partial(self.parse_line, pricing=self.pricing),
        )

        cumulative_tokens: TokenUsage | None = None
        step_count = 0
        runtime_result = None
        thread_id: str | None = None
        conflicting_thread_ids = False
        invalid_stdout_usage = False
        self._current_trace_evidence = {"status": "not_attempted"}

        for item in stream_cli_command(
            runtime=self.runtime,
            command=command_str,
            parser=parser,
            env=env_overrides,
            timeout=(float(self.timeout) if self.timeout is not None else None),
        ):
            # Final item is RuntimeResult
            if not isinstance(item, tuple):
                runtime_result = item
                break

            _cost, tokens, payload = item
            self.log.debug("Received item", item=item, verbose=True)
            if tokens is not None:
                # Codex emits cumulative usage. Keep the final record instead
                # of summing repeated cumulative snapshots.
                cumulative_tokens = tokens

            # Count steps from turn.started and item.completed events
            if payload is not None:
                event_type = payload.get("type")
                if event_type == "turn.completed" and tokens is None:
                    invalid_stdout_usage = True
                if event_type == "thread.started":
                    candidate = payload.get("thread_id")
                    if isinstance(candidate, str) and candidate:
                        if thread_id is None and not conflicting_thread_ids:
                            thread_id = candidate
                        elif candidate != thread_id:
                            conflicting_thread_ids = True
                            thread_id = None
                if event_type in ("turn.started", "item.completed"):
                    step_count += 1
                    self.usage.steps += 1

        stdout = runtime_result.stdout if runtime_result else ""
        stderr = runtime_result.stderr if runtime_result else ""
        effective_thread_id = thread_id
        thread_reference = "stdout" if thread_id is not None else "missing"
        if (
            not conflicting_thread_ids
            and effective_thread_id is None
            and resume
        ):
            effective_thread_id = self._active_telemetry_thread_id
            if effective_thread_id is not None:
                thread_reference = "prior_resume"
        if conflicting_thread_ids:
            thread_reference = "conflicting"

        stdout_usage_present = cumulative_tokens is not None
        trace_tokens: TokenUsage | None = None
        trace_cost: float | None = None
        reasoning_tokens: int | None = (
            cumulative_tokens.reasoning
            if cumulative_tokens is not None
            else None
        )
        if conflicting_thread_ids:
            self._warn_trace_fallback("conflicting stdout thread ids")
        else:
            trace_reasoning_tokens: int | None
            (
                trace_tokens,
                trace_cost,
                trace_reasoning_tokens,
            ) = self._reconcile_trace_usage(
                thread_id=effective_thread_id,
                stdout_tokens=cumulative_tokens,
            )
            if trace_reasoning_tokens is not None:
                reasoning_tokens = trace_reasoning_tokens
        if cumulative_tokens is None and trace_tokens is not None:
            cumulative_tokens = trace_tokens

        telemetry_error: str | None = None
        if (
            cumulative_tokens is None
            and runtime_result is not None
            and runtime_result.exit_code == 0
            and not runtime_result.timed_out
        ):
            telemetry_error = (
                "Codex telemetry integrity failure: the successful process "
                + (
                    "emitted malformed stdout usage"
                    if invalid_stdout_usage
                    else "emitted no stdout usage"
                )
                + " and no exactly matched raw rollout usage was available"
            )

        cumulative = _CodexCumulativeTelemetry(
            tokens=cumulative_tokens or TokenUsage(),
            reasoning_tokens=reasoning_tokens,
            reported_cost=trace_cost,
        )
        (
            final_tokens,
            invocation_cost,
            delta_status,
        ) = self._invocation_telemetry_delta(
            thread_id=(None if conflicting_thread_ids else effective_thread_id),
            cumulative=cumulative,
            resume=resume,
        )
        if (
            telemetry_error is None
            and delta_status == "resume_delta_unavailable_zero"
            and runtime_result is not None
            and runtime_result.exit_code == 0
            and not runtime_result.timed_out
        ):
            telemetry_error = (
                "Codex telemetry integrity failure: a successful resume "
                "could not be correlated to a cumulative telemetry baseline"
            )
        has_reported_cost = invocation_cost is not None
        reported_cost_micros = (
            int(round(invocation_cost * 1_000_000))
            if invocation_cost is not None
            else 0
        )
        self._record_invocation_telemetry(
            resume=resume,
            thread_reference=thread_reference,
            stdout_usage_present=stdout_usage_present,
            cumulative=cumulative,
            delta=final_tokens,
            delta_status=delta_status,
        )

        return AgentCommandResult(
            result=runtime_result,
            steps=[],
            usage_totals={
                "input_tokens": final_tokens.input,
                "output_tokens": final_tokens.output,
                "cached_input_tokens": final_tokens.cache_read,
                "cache_write_input_tokens": final_tokens.cache_write,
                "reasoning_tokens": final_tokens.reasoning,
                "total_tokens": final_tokens.total,
                "steps": step_count,
                "reported_cost_present": int(has_reported_cost),
                "reported_cost_micros": reported_cost_micros,
                "telemetry_semantics_version": (
                    self.TELEMETRY_SEMANTICS_VERSION
                ),
            },
            stdout=stdout,
            stderr=stderr,
            had_error=telemetry_error is not None,
            error_message=telemetry_error,
        )

    def _warn_trace_fallback(
        self,
        reason: str,
        **details: object,
    ) -> tuple[None, None, None]:
        self.log.warning(
            "agent.codex.telemetry.trace_fallback",
            reason=reason,
            **details,
        )
        evidence: dict[str, tp.Any] = {
            "status": "fallback",
            "reason": reason,
        }
        matches = details.get("matches")
        if type(matches) is int:
            evidence["match_count"] = matches
        self._current_trace_evidence = evidence
        return None, None, None

    @staticmethod
    def _telemetry_token_totals(
        tokens: TokenUsage,
        *,
        reasoning_tokens: int | None,
    ) -> dict[str, int | None]:
        return {
            "input_tokens": tokens.input,
            "output_tokens": tokens.output,
            "cached_input_tokens": tokens.cache_read,
            "cache_write_tokens": tokens.cache_write,
            "reasoning_tokens": reasoning_tokens,
            "total_tokens": tokens.total,
        }

    def _record_invocation_telemetry(
        self,
        *,
        resume: bool,
        thread_reference: str,
        stdout_usage_present: bool,
        cumulative: _CodexCumulativeTelemetry,
        delta: TokenUsage,
        delta_status: str,
    ) -> None:
        """Retain bounded, credential-free evidence until artifact saving."""
        evidence: dict[str, tp.Any] = {
            "ordinal": (
                len(self._telemetry_invocations)
                + self._telemetry_invocations_omitted
                + 1
            ),
            "resume": resume,
            "thread_reference": thread_reference,
            "stdout_usage_present": stdout_usage_present,
            "trace_correlation": dict(self._current_trace_evidence),
            "cumulative_totals": self._telemetry_token_totals(
                cumulative.tokens,
                reasoning_tokens=cumulative.reasoning_tokens,
            ),
            "reported_cumulative_cost_usd": cumulative.reported_cost,
            "delta_status": delta_status,
            "accounted_delta": self._telemetry_token_totals(
                delta,
                reasoning_tokens=delta.reasoning,
            ),
            "cost_accounting": {"source": "pending", "usd": None},
        }
        if len(self._telemetry_invocations) < self.TELEMETRY_MAX_INVOCATIONS:
            self._telemetry_invocations.append(evidence)
            self._pending_telemetry_evidence = evidence
        else:
            self._telemetry_invocations_omitted += 1
            self._pending_telemetry_evidence = None

    @staticmethod
    def _token_delta(
        current: TokenUsage,
        previous: TokenUsage,
    ) -> TokenUsage | None:
        """Return a component-wise cumulative delta, or None on regression."""
        fields = ("input", "output", "cache_read", "cache_write")
        if any(
            getattr(current, field) < getattr(previous, field)
            for field in fields
        ):
            return None
        return TokenUsage(
            input=current.input - previous.input,
            output=current.output - previous.output,
            cache_read=current.cache_read - previous.cache_read,
            cache_write=current.cache_write - previous.cache_write,
        )

    def _invocation_telemetry_delta(
        self,
        *,
        thread_id: str | None,
        cumulative: _CodexCumulativeTelemetry,
        resume: bool,
    ) -> tuple[TokenUsage, float | None, str]:
        """Convert cumulative thread telemetry to one invocation's delta."""
        if thread_id is None:
            if resume:
                self.log.warning(
                    "agent.codex.telemetry.resume_delta_unavailable",
                    reason="no trustworthy thread id",
                )
                # A resume record is cumulative. Adding it wholesale would
                # certainly double-count the prior invocation.
                return TokenUsage(), None, "resume_delta_unavailable_zero"
            return (
                cumulative.tokens.model_copy(
                    update={"reasoning": cumulative.reasoning_tokens or 0}
                ),
                cumulative.reported_cost,
                "uncorrelated_initial_cumulative",
            )

        previous = self._telemetry_baselines.get(thread_id)
        if previous is None:
            delta_tokens = cumulative.tokens.model_copy(
                update={"reasoning": cumulative.reasoning_tokens or 0}
            )
            delta_cost = cumulative.reported_cost
            delta_status = (
                "new_resume_thread_cumulative"
                if resume
                else "initial_thread_cumulative"
            )
        else:
            delta_tokens = self._token_delta(
                cumulative.tokens,
                previous.tokens,
            )
            if delta_tokens is None:
                self.log.warning(
                    "agent.codex.telemetry.cumulative_regression",
                    thread_id=thread_id,
                    previous=previous.tokens.model_dump(),
                    current=cumulative.tokens.model_dump(),
                )
                # Treat a regressing record as a new cumulative epoch. This
                # avoids negative accounting while preserving its usage.
                delta_tokens = cumulative.tokens.model_copy(
                    update={"reasoning": cumulative.reasoning_tokens or 0}
                )
                delta_cost = cumulative.reported_cost
                delta_status = "cumulative_regression_new_epoch"
            else:
                previous_reasoning = previous.reasoning_tokens
                current_reasoning = cumulative.reasoning_tokens
                if current_reasoning is None:
                    reasoning_delta = 0
                elif previous_reasoning is None:
                    # Earlier reasoning was unavailable and counted as zero;
                    # catch up once a correlated cumulative trace appears.
                    reasoning_delta = current_reasoning
                elif current_reasoning >= previous_reasoning:
                    reasoning_delta = current_reasoning - previous_reasoning
                else:
                    self.log.warning(
                        "agent.codex.telemetry.reasoning_regression",
                        thread_id=thread_id,
                        previous=previous_reasoning,
                        current=current_reasoning,
                    )
                    reasoning_delta = 0
                delta_tokens = delta_tokens.model_copy(
                    update={"reasoning": reasoning_delta}
                )

                previous_cost = previous.reported_cost
                current_cost = cumulative.reported_cost
                if (
                    current_cost is not None
                    and previous_cost is not None
                    and current_cost >= previous_cost
                ):
                    delta_cost = current_cost - previous_cost
                else:
                    if (
                        current_cost is not None
                        and previous_cost is not None
                        and current_cost < previous_cost
                    ):
                        self.log.warning(
                            "agent.codex.telemetry.cost_regression",
                            thread_id=thread_id,
                            previous=previous_cost,
                            current=current_cost,
                        )
                    # Reprice only this token delta if the reported-cost chain
                    # is incomplete; never add a cumulative cost twice.
                    delta_cost = None
                delta_status = "cumulative_thread_delta"

        baseline_reasoning = cumulative.reasoning_tokens
        if baseline_reasoning is None and previous is not None:
            baseline_reasoning = previous.reasoning_tokens
        self._telemetry_baselines[thread_id] = _CodexCumulativeTelemetry(
            tokens=cumulative.tokens,
            reasoning_tokens=baseline_reasoning,
            reported_cost=cumulative.reported_cost,
        )
        self._active_telemetry_thread_id = thread_id
        return delta_tokens, delta_cost, delta_status

    def _reset_telemetry_baselines(self) -> None:
        self._telemetry_baselines = {}
        self._active_telemetry_thread_id = None

    def _reset_telemetry_state(self) -> None:
        self._reset_telemetry_baselines()
        self._telemetry_invocations = []
        self._telemetry_invocations_omitted = 0
        self._telemetry_cost_sources = set()
        self._pending_telemetry_evidence = None
        self._current_trace_evidence = {"status": "not_attempted"}

    @staticmethod
    def _read_rollout_token_record(
        path: Path,
    ) -> tuple[str | None, object | None, str | None]:
        """Read a rollout's session id and final non-null token record."""
        session_id: str | None = None
        final_info: object | None = None
        malformed: str | None = None

        try:
            with path.open(encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        malformed = f"invalid JSON at line {line_number}"
                        continue
                    if not isinstance(event, dict):
                        malformed = f"non-object JSON at line {line_number}"
                        continue

                    payload = event.get("payload")
                    if not isinstance(payload, dict):
                        continue
                    if event.get("type") == "session_meta":
                        candidate = payload.get("id")
                        if isinstance(candidate, str) and candidate:
                            if (
                                session_id is not None
                                and candidate != session_id
                            ):
                                malformed = "conflicting session ids"
                            session_id = candidate
                        continue
                    if (
                        event.get("type") == "event_msg"
                        and payload.get("type") == "token_count"
                        and payload.get("info") is not None
                    ):
                        # Records are cumulative and may be duplicated. Keeping
                        # the final non-null record is deliberate.
                        final_info = payload.get("info")
        except (OSError, UnicodeError) as exc:
            return None, None, f"cannot read rollout: {type(exc).__name__}"

        return session_id, final_info, malformed

    @staticmethod
    def _strict_token_count(value: object) -> int | None:
        if type(value) is not int or value < 0:
            return None
        return value

    def _reconcile_trace_usage(
        self,
        thread_id: str | None,
        stdout_tokens: TokenUsage | None,
    ) -> tuple[TokenUsage | None, float | None, int | None]:
        """Correlate raw trace telemetry to one Codex stdout thread."""
        try:
            if not thread_id:
                return self._warn_trace_fallback("missing stdout thread id")
            if self._trace_dir is None:
                return self._warn_trace_fallback(
                    "raw rollout directory unavailable",
                    thread_id=thread_id,
                )

            candidates: list[tuple[Path, object | None, str | None]] = []
            for path in self._new_trace_files():
                session_id, final_info, malformed = (
                    self._read_rollout_token_record(path)
                )
                if session_id == thread_id:
                    candidates.append((path, final_info, malformed))

            if len(candidates) != 1:
                return self._warn_trace_fallback(
                    "raw rollout match count is not one",
                    thread_id=thread_id,
                    matches=len(candidates),
                )

            path, final_info, malformed = candidates[0]
            if malformed is not None:
                return self._warn_trace_fallback(
                    "malformed raw rollout",
                    thread_id=thread_id,
                    rollout=path.name,
                    detail=malformed,
                )
            if not isinstance(final_info, dict):
                return self._warn_trace_fallback(
                    "missing final non-null token count",
                    thread_id=thread_id,
                    rollout=path.name,
                )

            raw_usage = final_info.get("total_token_usage")
            if not isinstance(raw_usage, dict):
                return self._warn_trace_fallback(
                    "malformed cumulative token count",
                    thread_id=thread_id,
                    rollout=path.name,
                )

            actual = {
                key: self._strict_token_count(raw_usage.get(key))
                for key in (
                    "input_tokens",
                    "cached_input_tokens",
                    "output_tokens",
                    "reasoning_output_tokens",
                )
            }
            actual["cache_write_input_tokens"] = self._strict_token_count(
                raw_usage.get("cache_write_input_tokens", 0)
            )
            if any(value is None for value in actual.values()):
                return self._warn_trace_fallback(
                    "malformed cumulative token values",
                    thread_id=thread_id,
                    rollout=path.name,
                )

            # Both stdout and rollout input counts are inclusive of the
            # cached subset. Never add or subtract cache when comparing them.
            comparable_actual = {
                key: actual[key]
                for key in (
                    "input_tokens",
                    "cached_input_tokens",
                    "cache_write_input_tokens",
                    "output_tokens",
                )
            }
            if stdout_tokens is not None:
                expected = {
                    "input_tokens": stdout_tokens.input,
                    "cached_input_tokens": stdout_tokens.cache_read,
                    "cache_write_input_tokens": stdout_tokens.cache_write,
                    "output_tokens": stdout_tokens.output,
                }
                if comparable_actual != expected:
                    return self._warn_trace_fallback(
                        "raw rollout usage does not match stdout",
                        thread_id=thread_id,
                        rollout=path.name,
                        expected=expected,
                        actual=comparable_actual,
                    )

            trace_tokens = TokenUsage(
                input=tp.cast("int", actual["input_tokens"]),
                output=tp.cast("int", actual["output_tokens"]),
                cache_read=tp.cast("int", actual["cached_input_tokens"]),
                cache_write=tp.cast(
                    "int", actual["cache_write_input_tokens"]
                ),
            )

            reasoning_tokens = tp.cast("int", actual["reasoning_output_tokens"])
            if (
                trace_tokens.cache_read > trace_tokens.input
                or reasoning_tokens > trace_tokens.output
            ):
                return self._warn_trace_fallback(
                    "token subsets exceed inclusive totals",
                    thread_id=thread_id,
                    rollout=path.name,
                    cache_read=trace_tokens.cache_read,
                    input=trace_tokens.input,
                    reasoning=reasoning_tokens,
                    output=trace_tokens.output,
                )

            raw_cost = final_info.get("total_cost")
            if raw_cost is None:
                raw_cost = final_info.get("cost_usd")
            if raw_cost is None:
                cost = None
            elif isinstance(raw_cost, bool) or not isinstance(
                raw_cost, int | float
            ):
                return self._warn_trace_fallback(
                    "malformed cumulative cost",
                    thread_id=thread_id,
                    rollout=path.name,
                )
            elif raw_cost < 0:
                return self._warn_trace_fallback(
                    "negative cumulative cost",
                    thread_id=thread_id,
                    rollout=path.name,
                )
            else:
                cost = float(raw_cost)
            self._current_trace_evidence = {
                "status": "correlated",
                "reason": None,
                "usage_source": (
                    "stdout_and_trace"
                    if stdout_tokens is not None
                    else "trace_only"
                ),
            }
            return trace_tokens, cost, reasoning_tokens
        except Exception as exc:  # noqa: BLE001
            return self._warn_trace_fallback(
                "unexpected raw rollout telemetry error",
                thread_id=thread_id,
                error_type=type(exc).__name__,
            )

    def _new_trace_files(self) -> list[Path]:
        if self._trace_dir is None:
            return []
        return sorted(
            path
            for path in find_jsonl_files(self._trace_dir)
            if path.relative_to(self._trace_dir) not in self._saved_trace_paths
        )

    def _sync_usage(self, totals: dict[str, int]) -> None:
        totals = totals or {}
        input_tokens = int(totals.get("input_tokens") or 0)
        output_tokens = int(totals.get("output_tokens") or 0)
        cache_read_tokens = int(totals.get("cached_input_tokens") or 0)
        cache_write_tokens = int(
            totals.get("cache_write_input_tokens") or 0
        )
        reasoning_tokens = int(totals.get("reasoning_tokens") or 0)
        tokens = TokenUsage(
            input=input_tokens,
            output=output_tokens,
            cache_read=cache_read_tokens,
            cache_write=cache_write_tokens,
            reasoning=reasoning_tokens,
        )
        if int(totals.get("reported_cost_present") or 0):
            cost = (
                float(int(totals.get("reported_cost_micros") or 0)) / 1_000_000
            )
            cost_source = "codex_reported_delta"
        else:
            cost = self.pricing.get_cost(tokens) if self.pricing else 0.0
            cost_source = (
                "local_repricing" if self.pricing else "unavailable_zero"
            )
        self._telemetry_cost_sources.add(cost_source)
        if self._pending_telemetry_evidence is not None:
            self._pending_telemetry_evidence["cost_accounting"] = {
                "source": cost_source,
                "usd": cost,
                "reported_cost_used": (cost_source == "codex_reported_delta"),
                "local_repricing_used": cost_source == "local_repricing",
            }
        self._pending_telemetry_evidence = None
        # Update tokens and cost without incrementing steps (already done during streaming)
        self.usage.cost += cost
        self.usage.net_tokens += tokens
        self.usage.current_tokens = tokens

        if self.cost_limits.is_above_limits(
            self.usage,
            prior_cost=self.prior_cost,
        ):
            raise AgentError("CodexAgent exceeded configured usage limits")

    def _prepare_runtime_execution(
        self,
        task: str,
        *,
        resume: bool = False,
    ) -> tuple[list[str], dict[str, str]]:
        """Prepare command and environment overrides for runtime execution."""
        env_overrides = {key: str(value) for key, value in self.env.items()}

        # Set credential in environment if it's an env var credential
        if (
            self.credential is not None
            and self.credential.credential_type == CredentialType.ENV_VAR
        ):
            env_overrides[self.credential.destination_key] = (
                self.credential.value
            )
        command = self._build_command(task, resume=resume)

        return command, env_overrides

    def _build_command(
        self,
        prompt: str,
        *,
        resume: bool = False,
    ) -> list[str]:
        command = [self.binary, "exec"]
        if resume:
            command.extend(["resume", "--last"])
        command.extend(
            [
                shlex.quote(prompt),
                "--skip-git-repo-check",
                "--json",
                "--dangerously-bypass-approvals-and-sandbox",
            ]
        )
        if self.model:
            command.extend(["--model", self.model])
            if self.model == "gpt-5.2-codex":
                command.extend(["--config", "model_verbosity='medium'"])

        # Handle thinking configuration
        if self.thinking in {"disabled", "none"}:
            # Disabled: omit model_reasoning_effort, set output tokens to 0
            command.extend(["--config", "model_max_output_tokens=0"])
        elif self.thinking:
            # Preset (low/medium/high): set reasoning effort
            command.extend(
                [
                    "--config",
                    f'model_reasoning_effort="{self.thinking}"',
                ]
            )

        elif self.max_thinking_tokens is not None:
            # Explicit token limit
            command.extend(
                [
                    "--config",
                    f"model_max_output_tokens={self.max_thinking_tokens}",
                ]
            )

        command.extend(self.extra_args)
        return command

    @classmethod
    def _write_artifacts(
        cls,
        output_dir: Path,
        stdout_text: str,
        stderr_text: str,
    ) -> None:
        (output_dir / cls.STDOUT_FILENAME).write_text(stdout_text)
        (output_dir / cls.STDERR_FILENAME).write_text(stderr_text)

    def reset(self) -> None:
        self._last_prompt = ""
        self._last_command = None
        self._reset_telemetry_state()

    def save_artifacts(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        if self._last_prompt:
            (path / self.PROMPT_FILENAME).write_text(self._last_prompt)

        stdout_text = ""
        stderr_text = ""
        if self._last_command is not None:
            stdout_text = self._last_command.stdout or ""
            stderr_text = self._last_command.stderr or ""

        self._write_artifacts(path, stdout_text, stderr_text)
        self._write_telemetry_artifact(path)
        self._save_codex_traces(path)

    def _write_telemetry_artifact(self, output_dir: Path) -> None:
        sources = sorted(self._telemetry_cost_sources)
        if not sources:
            accounting_mode = "none"
        elif sources == ["codex_reported_delta"]:
            accounting_mode = "reported"
        elif sources == ["local_repricing"]:
            accounting_mode = "repriced"
        elif sources == ["unavailable_zero"]:
            accounting_mode = "unavailable"
        else:
            accounting_mode = "mixed"

        totals = self._telemetry_token_totals(
            self.usage.net_tokens,
            reasoning_tokens=self.usage.net_tokens.reasoning,
        )
        artifact = {
            "schema_version": self.TELEMETRY_ARTIFACT_SCHEMA_VERSION,
            "telemetry_semantics_version": self.TELEMETRY_SEMANTICS_VERSION,
            "semantics": dict(self.TELEMETRY_SEMANTICS),
            "checkpoint_totals": {
                **totals,
                "steps": self.usage.steps,
                "cost_usd": self.usage.cost,
                "cost_accounting": accounting_mode,
                "reported_cost_used": (
                    "codex_reported_delta" in self._telemetry_cost_sources
                ),
                "local_repricing_used": (
                    "local_repricing" in self._telemetry_cost_sources
                ),
            },
            "invocation_count": (
                len(self._telemetry_invocations)
                + self._telemetry_invocations_omitted
            ),
            "invocations_omitted": self._telemetry_invocations_omitted,
            "invocations": self._telemetry_invocations,
        }
        (output_dir / self.TELEMETRY_FILENAME).write_text(
            json.dumps(artifact, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _save_codex_traces(self, output_dir: Path) -> None:
        if self._trace_dir is None:
            self.log.debug("agent.codex.traces.skipped", reason="no_trace_dir")
            return
        jsonl_files = find_jsonl_files(self._trace_dir)
        new_jsonl_files = [
            path
            for path in jsonl_files
            if path.relative_to(self._trace_dir) not in self._saved_trace_paths
        ]
        self.log.debug(
            "agent.codex.traces.found",
            trace_dir=str(self._trace_dir),
            files=len(jsonl_files),
        )
        copied = copy_jsonl_files(new_jsonl_files, output_dir)
        for path in new_jsonl_files:
            self._saved_trace_paths.add(path.relative_to(self._trace_dir))
        self.log.debug(
            "agent.codex.traces.saved",
            output_dir=str(output_dir),
            saved=len(copied),
        )

    def cleanup(self) -> None:
        """Clean up resources held by the Codex agent."""
        if self._runtime is not None:
            self._runtime.cleanup()
            self._runtime = None
        self._session = None
        self._environment = None
        if self._trace_tmp is not None:
            self._trace_tmp.cleanup()
            self._trace_tmp = None
        self._trace_dir = None
        self._saved_trace_paths = set()
        self._reset_telemetry_state()
        self.log.debug("agent.codex.cleanup")


# Register this agent type with the agent registry
register_agent("codex", CodexAgent)
