from __future__ import annotations

import queue
import threading
from collections.abc import Iterator

from slop_code.execution import stream_processor
from slop_code.execution.runtime import RuntimeEvent
from slop_code.execution.runtime import RuntimeResult


class DelayedFinalThread:
    """Enqueue final output only when process_stream joins the pump."""

    def __init__(
        self,
        event_queue: queue.Queue[tuple[str, str | None]],
    ) -> None:
        self._event_queue = event_queue
        self._joined = False

    def join(self) -> None:
        if self._joined:
            return
        self._joined = True
        self._event_queue.put(("stdout", "late stdout"))
        self._event_queue.put(("stderr", "late stderr"))
        self._event_queue.put(("finished", None))


def consume_stream(
    events: Iterator[RuntimeEvent],
) -> tuple[list[RuntimeEvent], RuntimeResult]:
    collected = []
    while True:
        try:
            collected.append(next(events))
        except StopIteration as stop:
            assert isinstance(stop.value, RuntimeResult)
            return collected, stop.value


def test_process_stream_drains_output_enqueued_during_final_join(
    monkeypatch,
) -> None:
    def start_delayed_pump(
        stream,
        event_queue,
        stop_event: threading.Event,
    ) -> DelayedFinalThread:
        del stream, stop_event
        return DelayedFinalThread(event_queue)

    monkeypatch.setattr(
        stream_processor,
        "start_stream_pump",
        start_delayed_pump,
    )

    events, result = consume_stream(
        stream_processor.process_stream(
            iter(()),
            timeout=1,
            poll_fn=lambda: 0,
        )
    )

    assert [(event.kind, event.text) for event in events] == [
        ("stdout", "late stdout"),
        ("stderr", "late stderr"),
    ]
    assert result.stdout == "late stdout"
    assert result.stderr == "late stderr"
    assert result.exit_code == 0
