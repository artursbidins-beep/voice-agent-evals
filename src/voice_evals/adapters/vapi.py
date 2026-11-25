"""Map a Vapi call object onto :class:`CallRecord`.

What this adapter can and cannot give you, stated plainly because it changes
what the report is allowed to claim:

* **Turns** come from ``artifact.messages``. Timestamps have appeared both as
  seconds from the start of the call and as Unix milliseconds depending on the
  payload, so the adapter detects which it is rather than trusting one.
* **Tool calls** come from the ``tool_calls`` messages in the same array.
* **Latency** is only ``response_total``, the gap between the caller finishing
  and the agent starting to speak. A platform call object does not expose when
  the transcriber finalised, when the first model token arrived or when speech
  synthesis returned its first byte. Those four stages need the pipeline's own
  event stream, so the four stage budget stays empty here and the report shows
  it as missing instead of inventing it.

That last point is the honest limit of scoring a hosted platform from the
outside, and it is the reason the model has an ``events`` list at all.
"""

from __future__ import annotations

import json
from typing import Any, Iterable

from voice_evals.models import CallRecord, Event, ToolCall, Turn

#: Anything larger than this is a Unix timestamp in milliseconds, not an
#: offset. Ten million seconds is 115 days, longer than any phone call.
_EPOCH_GUARD_MS = 10_000_000

_CALLER_ROLES = {"user", "customer", "human"}
_AGENT_ROLES = {"assistant", "bot", "agent"}
_TOOL_ROLES = {"tool_calls", "tool-calls", "function_call"}


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _seconds_to_ms(value: Any) -> int | None:
    """``secondsFromStart`` is an offset in seconds, such as 3.2."""

    number = _number(value)
    return None if number is None else int(round(number * 1000))


def _milliseconds(value: Any) -> int | None:
    """``duration`` is already in milliseconds."""

    number = _number(value)
    return None if number is None else int(round(number))


def _stamp_to_ms(value: Any) -> int | None:
    """``time`` and ``endTime`` have appeared as Unix ms and as seconds."""

    number = _number(value)
    if number is None:
        return None
    if number > _EPOCH_GUARD_MS:
        return int(number)
    return int(round(number * 1000))


def _message_window(message: dict[str, Any], origin_ms: int) -> tuple[int, int]:
    start = _seconds_to_ms(message.get("secondsFromStart"))
    if start is None:
        start = _stamp_to_ms(message.get("time"))
        if start is not None and start > _EPOCH_GUARD_MS:
            start -= origin_ms
    start = max(0, start or 0)

    end = _stamp_to_ms(message.get("endTime"))
    if end is not None and end > _EPOCH_GUARD_MS:
        end -= origin_ms
    if end is None:
        duration = _milliseconds(message.get("duration"))
        end = start + duration if duration is not None else start
    return start, max(start, end)


def _origin_ms(messages: Iterable[dict[str, Any]]) -> int:
    """The call start, needed when timestamps are absolute."""

    for message in messages:
        stamp = _stamp_to_ms(message.get("time"))
        if stamp is not None and stamp > _EPOCH_GUARD_MS:
            return stamp
    return 0


def _tool_calls_of(message: dict[str, Any]) -> list[dict[str, Any]]:
    raw = message.get("toolCalls") or message.get("tool_calls") or []
    return [call for call in raw if isinstance(call, dict)]


def _arguments_of(call: dict[str, Any]) -> dict[str, Any]:
    function = call.get("function") or call
    arguments = function.get("arguments", {})
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return {"_raw": arguments}
    return arguments if isinstance(arguments, dict) else {"_value": arguments}


def call_record_from_vapi(
    payload: dict[str, Any],
    scenario_id: str,
    language: str = "en",
) -> CallRecord:
    """Build a record from the object returned by the Get Call endpoint."""

    artifact = payload.get("artifact") or {}
    messages = [m for m in (artifact.get("messages") or []) if isinstance(m, dict)]
    origin = _origin_ms(messages)

    turns: list[Turn] = []
    tool_calls: list[ToolCall] = []

    for message in messages:
        role = str(message.get("role", "")).lower()
        start, end = _message_window(message, origin)

        if role in _TOOL_ROLES or _tool_calls_of(message):
            for call in _tool_calls_of(message):
                function = call.get("function") or {}
                name = function.get("name") or call.get("name") or "unknown"
                tool_calls.append(
                    ToolCall(name=str(name), arguments=_arguments_of(call), at_ms=start)
                )
            continue

        text = str(message.get("message") or message.get("content") or "").strip()
        if not text:
            continue
        if role in _CALLER_ROLES:
            turns.append(Turn(role="caller", text=text, start_ms=start, end_ms=end))
        elif role in _AGENT_ROLES:
            turns.append(Turn(role="agent", text=text, start_ms=start, end_ms=end))
        # system and tool result messages are not spoken, so they are not turns

    events = _response_events(turns)

    return CallRecord(
        call_id=str(payload.get("id", "")),
        scenario_id=scenario_id,
        language=language,
        turns=turns,
        events=events,
        tool_calls=tool_calls,
        metadata={
            "provider": "vapi",
            "assistant_id": payload.get("assistantId", ""),
            "ended_reason": payload.get("endedReason", ""),
            "stage_breakdown": "unavailable from the call object",
        },
    )


def _response_events(turns: list[Turn]) -> list[Event]:
    """Derive the one stage a call object can honestly support.

    Each caller turn followed by an agent turn gives a ``user_speech_end`` and
    an ``agent_audio_start``, which is enough for the total response time and
    nothing more.
    """

    events: list[Event] = []
    index = 0
    for position, turn in enumerate(turns[:-1]):
        following = turns[position + 1]
        if turn.role != "caller" or following.role != "agent":
            continue
        index += 1
        events.append(Event(type="user_speech_end", at_ms=turn.end_ms, turn=index))
        events.append(Event(type="agent_audio_start", at_ms=following.start_ms, turn=index))
    return events
