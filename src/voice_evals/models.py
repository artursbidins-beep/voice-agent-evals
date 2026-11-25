"""Provider neutral data model.

Every runner, live or offline, produces a :class:`CallRecord`. The metrics only
ever see this shape, so supporting a second platform means writing one adapter
and changing nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Literal

Role = Literal["caller", "agent"]

#: Timestamps a voice pipeline has to emit before a per stage latency budget is
#: possible. The names are deliberately platform neutral.
EVENT_TYPES = (
    "user_speech_end",    # the endpointer decided the caller stopped talking
    "asr_final",          # the final transcript of that utterance is available
    "llm_first_token",    # first token of the model response
    "tts_first_byte",     # first audio byte back from speech synthesis
    "agent_audio_start",  # the caller actually starts hearing the answer
)


@dataclass(frozen=True)
class Turn:
    """One spoken utterance and the window of wall clock time it occupied."""

    role: Role
    text: str
    start_ms: int
    end_ms: int

    @property
    def duration_ms(self) -> int:
        return max(0, self.end_ms - self.start_ms)


@dataclass(frozen=True)
class Event:
    """A pipeline timestamp belonging to one exchange."""

    type: str
    at_ms: int
    turn: int = 0

    def __post_init__(self) -> None:
        if self.type not in EVENT_TYPES:
            allowed = ", ".join(EVENT_TYPES)
            raise ValueError(f"unknown event type {self.type!r}, expected one of {allowed}")


@dataclass(frozen=True)
class ToolCall:
    """A function the agent asked the backend to run."""

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    at_ms: int = 0


@dataclass(frozen=True)
class ExpectedToolCall:
    """What a scenario requires. Only the listed arguments are compared."""

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    required: bool = True


@dataclass
class CallRecord:
    """One finished conversation, normalised."""

    call_id: str
    scenario_id: str
    language: str = "en"
    turns: list[Turn] = field(default_factory=list)
    events: list[Event] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    reference_transcript: dict[int, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def caller_turns(self) -> list[Turn]:
        return [t for t in self.turns if t.role == "caller"]

    def agent_turns(self) -> list[Turn]:
        return [t for t in self.turns if t.role == "agent"]

    def events_of(self, event_type: str) -> list[Event]:
        return [e for e in self.events if e.type == event_type]

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CallRecord":
        reference = payload.get("reference_transcript") or {}
        return cls(
            call_id=payload["call_id"],
            scenario_id=payload["scenario_id"],
            language=payload.get("language", "en"),
            turns=[Turn(**t) for t in payload.get("turns", [])],
            events=[Event(**e) for e in payload.get("events", [])],
            tool_calls=[ToolCall(**c) for c in payload.get("tool_calls", [])],
            reference_transcript={int(k): v for k, v in reference.items()},
            metadata=payload.get("metadata", {}),
        )


@dataclass
class Scenario:
    """One test case: what the caller does, and what has to come out of it."""

    id: str
    title: str
    language: str = "en"
    persona: str = ""
    script: list[str] = field(default_factory=list)
    expected_tool_calls: list[ExpectedToolCall] = field(default_factory=list)
    must_say_any: list[str] = field(default_factory=list)
    must_not_say: list[str] = field(default_factory=list)
    asks_if_human: bool = False
    tags: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Scenario":
        expect = payload.get("expect") or {}
        return cls(
            id=payload["id"],
            title=payload.get("title", payload["id"]),
            language=payload.get("language", "en"),
            persona=payload.get("persona", ""),
            script=list(payload.get("script") or []),
            expected_tool_calls=[
                ExpectedToolCall(
                    name=call["name"],
                    arguments=call.get("arguments") or {},
                    required=call.get("required", True),
                )
                for call in (expect.get("tool_calls") or [])
            ],
            must_say_any=list(expect.get("must_say_any") or []),
            must_not_say=list(expect.get("must_not_say") or []),
            asks_if_human=bool(expect.get("asks_if_human", False)),
            tags=list(payload.get("tags") or []),
        )


def turns_in_order(turns: Iterable[Turn]) -> list[Turn]:
    return sorted(turns, key=lambda turn: (turn.start_ms, turn.end_ms))
