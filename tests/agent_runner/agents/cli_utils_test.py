"""Fail-closed tests for shared CLI stream handling."""

from __future__ import annotations

from collections.abc import Iterable
from typing import cast

from slop_code.agent_runner.agents.cli_utils import stream_cli_command
from slop_code.execution.protocols import StreamingRuntime
from slop_code.execution.runtime import RuntimeEvent
from slop_code.execution.runtime import RuntimeResult


class _RuntimeWithoutTerminalEvent:
    def stream(self, **_: object) -> Iterable[RuntimeEvent]:
        yield RuntimeEvent(kind="stdout", text='{"type":"message"}\n')


def test_stream_cli_command_never_fabricates_success_without_terminal_event(
) -> None:
    runtime = cast("StreamingRuntime", _RuntimeWithoutTerminalEvent())

    items = list(
        stream_cli_command(
            runtime=runtime,
            command="agent",
            parser=lambda line: (None, None, {"line": line}),
        )
    )

    assert items[0] == (None, None, {"line": '{"type":"message"}'})
    result = items[-1]
    assert isinstance(result, RuntimeResult)
    assert result.exit_code == -1
    assert "without a terminal finished event" in result.stderr
