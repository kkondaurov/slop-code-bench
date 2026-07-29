"""Tests for runtime stream processing helpers."""

from __future__ import annotations

import queue
import threading
from collections.abc import Iterator

import pytest

from slop_code.execution import stream_processor
from slop_code.execution.runtime import RuntimeEvent
from slop_code.execution.runtime import RuntimeResult
from slop_code.execution.stream_processor import ensure_string


def test_ensure_string_preserves_text_around_invalid_utf8_bytes() -> None:
    decoded = ensure_string(b'{"type":"message_update","data":"ok"}\xff\n')

    assert '{"type":"message_update","data":"ok"}' in decoded
    assert decoded.endswith("\n")


class DelayedFinalThread:
    """Enqueue final output only when ``process_stream`` joins the pump."""

    def __init__(
        self,
        event_queue: queue.Queue[tuple[str, str | None]],
    ) -> None:
        self._event_queue = event_queue
        self._joined = False

    def join(self, timeout: float | None = None) -> None:
        del timeout
        if self._joined:
            return
        self._joined = True
        self._event_queue.put(("stdout", "late stdout"))
        self._event_queue.put(("stderr", "late stderr"))
        self._event_queue.put(("finished", None))

    def is_alive(self) -> bool:
        return False


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


def test_process_stream_propagates_iterator_error_without_hanging() -> None:
    def broken_stream() -> Iterator[tuple[str, str]]:
        yield "before failure", ""
        raise RuntimeError("stream iterator exploded")

    events = stream_processor.process_stream(
        broken_stream(),
        timeout=1,
        poll_fn=lambda: None,
    )

    first_event = next(events)
    assert (first_event.kind, first_event.text) == (
        "stdout",
        "before failure",
    )
    with pytest.raises(RuntimeError, match="stream iterator exploded"):
        next(events)


def test_stream_eof_waits_for_delayed_success_status() -> None:
    """EOF may arrive just before the process reports its zero exit code."""
    poll_results = iter((None, None, None, 0))

    _events, result = consume_stream(
        stream_processor.process_stream(
            iter(()),
            timeout=1,
            poll_fn=lambda: next(poll_results),
        )
    )

    assert result.exit_code == 0
    assert result.timed_out is False


class StalledThread:
    def __init__(self) -> None:
        self.join_timeouts: list[float | None] = []

    def join(self, timeout: float | None = None) -> None:
        self.join_timeouts.append(timeout)

    def is_alive(self) -> bool:
        return True


def test_process_stream_uses_bounded_join_for_stalled_pump(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stalled = StalledThread()

    def start_stalled_pump(
        stream,
        event_queue,
        stop_event: threading.Event,
    ) -> StalledThread:
        del stream, event_queue, stop_event
        return stalled

    monkeypatch.setattr(
        stream_processor,
        "start_stream_pump",
        start_stalled_pump,
    )

    with pytest.raises(RuntimeError, match="Stream pump did not stop"):
        consume_stream(
            stream_processor.process_stream(
                iter(()),
                timeout=1,
                poll_fn=lambda: 0,
            )
        )

    assert stalled.join_timeouts == [
        stream_processor.STREAM_PUMP_JOIN_TIMEOUT,
        stream_processor.STREAM_PUMP_JOIN_TIMEOUT,
    ]


def test_process_stream_polls_process_with_bounded_queue_wait(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    timeouts: list[float | None] = []

    class EmptyQueue:
        def get(self, timeout: float | None = None):
            timeouts.append(timeout)
            raise queue.Empty

        def get_nowait(self):
            raise queue.Empty

    class FinishedThread:
        def join(self, timeout: float | None = None) -> None:
            del timeout

        def is_alive(self) -> bool:
            return False

    monkeypatch.setattr(stream_processor.queue, "Queue", EmptyQueue)
    monkeypatch.setattr(
        stream_processor,
        "start_stream_pump",
        lambda *_: FinishedThread(),
    )
    poll_results = iter((None, 0))

    _events, result = consume_stream(
        stream_processor.process_stream(
            iter(()),
            timeout=100,
            poll_fn=lambda: next(poll_results),
        )
    )

    assert result.exit_code == 0
    assert timeouts
    assert timeouts[0] is not None
    assert timeouts[0] <= stream_processor.STREAM_EXIT_POLL_INTERVAL
