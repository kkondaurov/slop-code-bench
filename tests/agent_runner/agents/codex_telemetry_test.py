"""Codex CLI telemetry normalization and rollout reconciliation tests."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from slop_code.agent_runner.agents.codex import CodexAgent
from slop_code.agent_runner.models import AgentCostLimits
from slop_code.common.llms import APIPricing
from slop_code.common.llms import TokenUsage
from slop_code.execution.runtime import RuntimeEvent
from slop_code.execution.runtime import RuntimeResult


class FakeRuntime:
    def __init__(self, events: list[RuntimeEvent]) -> None:
        self.events = events

    def stream(
        self,
        command: str,
        env: dict[str, str],
        timeout: float | None,
    ) -> Iterable[RuntimeEvent]:
        yield from self.events

    def cleanup(self) -> None:
        pass


@dataclass
class FakeSession:
    runtime: FakeRuntime
    working_dir: Path
    spec: object | None = None

    def spawn(self, **_: object) -> FakeRuntime:
        return self.runtime


def _pricing() -> APIPricing:
    return APIPricing(input=0.5, output=2.0, cache_read=0.1)


def _make_agent(tmp_path: Path, runtime: FakeRuntime) -> CodexAgent:
    agent = CodexAgent(
        problem_name="telemetry-test",
        verbose=False,
        image="test-image",
        cost_limits=AgentCostLimits(
            step_limit=0,
            cost_limit=0,
            net_cost_limit=0,
        ),
        pricing=_pricing(),
        credential=None,
        binary="codex",
        model="gpt-test",
        timeout=60,
        thinking="high",
        max_thinking_tokens=None,
        extra_args=[],
        env={},
    )
    agent.setup(FakeSession(runtime=runtime, working_dir=tmp_path))
    agent.log = MagicMock()
    return agent


def _token_info(
    *,
    input_tokens: object = 100,
    cached_input_tokens: object = 80,
    output_tokens: object = 30,
    reasoning_output_tokens: object = 7,
) -> dict[str, object]:
    return {
        "total_token_usage": {
            "input_tokens": input_tokens,
            "cached_input_tokens": cached_input_tokens,
            "output_tokens": output_tokens,
            "reasoning_output_tokens": reasoning_output_tokens,
            "total_tokens": 130,
        },
        "last_token_usage": None,
    }


def _write_rollout(
    path: Path,
    thread_id: str,
    infos: list[object | None],
) -> None:
    events = [
        {
            "type": "session_meta",
            "payload": {"id": thread_id},
        },
        *[
            {
                "type": "event_msg",
                "payload": {"type": "token_count", "info": info},
            }
            for info in infos
        ],
    ]
    path.write_text(
        "".join(f"{json.dumps(event)}\n" for event in events),
        encoding="utf-8",
    )


def test_parse_line_separates_cached_input_and_prices_once() -> None:
    cost, tokens, payload = CodexAgent.parse_line(
        json.dumps(
            {
                "type": "turn.completed",
                "usage": {
                    "input_tokens": 100,
                    "cached_input_tokens": 80,
                    "output_tokens": 30,
                },
            }
        ),
        pricing=_pricing(),
    )

    assert payload is not None
    assert tokens == TokenUsage(
        input=20,
        cache_read=80,
        output=30,
        reasoning=0,
    )
    assert cost == pytest.approx(_pricing().get_cost(tokens))


def test_parse_line_clamps_negative_uncached_input_to_zero() -> None:
    _, tokens, _ = CodexAgent.parse_line(
        json.dumps(
            {
                "type": "turn.completed",
                "usage": {
                    "input_tokens": 50,
                    "cached_input_tokens": 80,
                    "output_tokens": 10,
                },
            }
        )
    )

    assert tokens is not None
    assert tokens.input == 0
    assert tokens.cache_read == 80


@pytest.mark.parametrize("line", ["null", "[]", '"scalar"', "42"])
def test_parse_line_ignores_non_object_json(line: str) -> None:
    assert CodexAgent.parse_line(line) == (None, None, None)


def test_invocation_uses_final_cumulative_records_and_syncs_reasoning_once(
    tmp_path: Path,
) -> None:
    thread_id = "thread-good"
    stdout_events = [
        {"type": "thread.started", "thread_id": thread_id},
        {"type": "turn.started"},
        {
            "type": "turn.completed",
            "usage": {
                "input_tokens": 90,
                "cached_input_tokens": 72,
                "output_tokens": 20,
            },
        },
        {
            "type": "turn.completed",
            "usage": {
                "input_tokens": 100,
                "cached_input_tokens": 80,
                "output_tokens": 30,
            },
        },
    ]
    stdout = "".join(f"{json.dumps(event)}\n" for event in stdout_events)
    runtime_result = RuntimeResult(
        exit_code=0,
        stdout=stdout,
        stderr="",
        setup_stdout="",
        setup_stderr="",
        elapsed=1.0,
        timed_out=False,
    )
    runtime = FakeRuntime(
        [
            RuntimeEvent(kind="stdout", text=stdout),
            RuntimeEvent(kind="finished", result=runtime_result),
        ]
    )
    agent = _make_agent(tmp_path, runtime)
    agent._trace_dir = tmp_path

    _write_rollout(
        tmp_path / "wrong-thread.jsonl",
        "thread-other",
        [_token_info(reasoning_output_tokens=29)],
    )
    final_info = _token_info(reasoning_output_tokens=7)
    _write_rollout(
        tmp_path / "matching-thread.jsonl",
        thread_id,
        [
            None,
            _token_info(
                input_tokens=90,
                cached_input_tokens=72,
                output_tokens=20,
                reasoning_output_tokens=5,
            ),
            final_info,
            final_info,
            None,
        ],
    )

    command_result = agent._run_invocation("solve it")

    assert command_result.usage_totals == {
        "input_tokens": 20,
        "output_tokens": 30,
        "cached_input_tokens": 80,
        "reasoning_output_tokens": 7,
        "total_tokens": 130,
        "steps": 1,
        "telemetry_semantics_version": 1,
    }
    assert CodexAgent.TELEMETRY_SEMANTICS == {
        "input_tokens": "uncached input tokens",
        "cached_input_tokens": (
            "cached subset of Codex's inclusive raw input tokens"
        ),
        "output_tokens": "output tokens inclusive of reasoning",
        "reasoning_output_tokens": "reasoning subset of output tokens",
    }

    agent._sync_usage(command_result.usage_totals)

    expected = TokenUsage(
        input=20,
        cache_read=80,
        output=30,
        reasoning=7,
    )
    assert agent.usage.net_tokens == expected
    assert agent.usage.current_tokens == expected
    assert agent.usage.cost == pytest.approx(_pricing().get_cost(expected))


@pytest.mark.parametrize("failure", ["absent", "malformed", "mismatch"])
def test_reasoning_reconciliation_warns_and_falls_back(
    tmp_path: Path,
    failure: str,
) -> None:
    agent = _make_agent(tmp_path, FakeRuntime([]))
    agent._trace_dir = tmp_path
    thread_id = "thread-fallback"

    if failure == "malformed":
        path = tmp_path / "malformed.jsonl"
        path.write_text(
            json.dumps(
                {"type": "session_meta", "payload": {"id": thread_id}}
            )
            + "\n{not-json}\n",
            encoding="utf-8",
        )
    elif failure == "mismatch":
        _write_rollout(
            tmp_path / "mismatch.jsonl",
            thread_id,
            [_token_info(input_tokens=101)],
        )

    reasoning = agent._reconcile_reasoning_tokens(
        thread_id=thread_id,
        stdout_tokens=TokenUsage(input=20, cache_read=80, output=30),
    )

    assert reasoning == 0
    agent.log.warning.assert_called_once()
    assert (
        agent.log.warning.call_args.args[0]
        == "agent.codex.telemetry.reasoning_fallback"
    )
