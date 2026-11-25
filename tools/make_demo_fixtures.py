"""Generate the demo fixtures in fixtures/liepa-en.

These calls are synthetic. Real recorded calls are not in this repository
because they contain caller names and phone numbers, and a public eval suite
is not a place to put those. The conversations below are written by hand to
exercise every metric, and the timings are in the range a hosted voice
pipeline actually produces, so the report reads like a real one.

Run it from the repository root:

    python tools/make_demo_fixtures.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

OUT = Path(__file__).resolve().parent.parent / "fixtures" / "liepa-en"

# Per stage response budget, in milliseconds, in the order the pipeline runs.
# Tweaked per turn so the report has a spread to take percentiles over.
STAGE_BUDGET = ("asr", "llm", "tts", "playback")


def exchange(
    caller: str,
    agent: str,
    *,
    caller_ms: int,
    agent_ms: int,
    stages: tuple[int, int, int, int],
    think_ms: int = 600,
    tools: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """One caller turn, the pipeline stages, and the answer."""

    return {
        "caller": caller,
        "agent": agent,
        "caller_ms": caller_ms,
        "agent_ms": agent_ms,
        "stages": stages,
        "think_ms": think_ms,
        "tools": tools or [],
    }


def build(call_id: str, scenario_id: str, exchanges: list[dict[str, Any]],
          opening: str, reference: dict[int, str] | None = None) -> dict[str, Any]:
    turns: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    tool_calls: list[dict[str, Any]] = []

    clock = 0
    opening_ms = 60 * len(opening)
    turns.append({"role": "agent", "text": opening, "start_ms": clock,
                  "end_ms": clock + opening_ms})
    clock += opening_ms

    for index, item in enumerate(exchanges, start=1):
        clock += item["think_ms"]
        caller_start = clock
        caller_end = caller_start + item["caller_ms"]
        turns.append({"role": "caller", "text": item["caller"],
                      "start_ms": caller_start, "end_ms": caller_end})

        at = caller_end
        events.append({"type": "user_speech_end", "at_ms": at, "turn": index})
        for stage, duration in zip(STAGE_BUDGET, item["stages"]):
            at += duration
            event_type = {
                "asr": "asr_final",
                "llm": "llm_first_token",
                "tts": "tts_first_byte",
                "playback": "agent_audio_start",
            }[stage]
            events.append({"type": event_type, "at_ms": at, "turn": index})

        for tool in item["tools"]:
            tool_calls.append({
                "name": tool["name"],
                "arguments": tool["arguments"],
                "at_ms": caller_end + item["stages"][0] + item["stages"][1],
            })

        agent_start = at
        agent_end = agent_start + item["agent_ms"]
        turns.append({"role": "agent", "text": item["agent"],
                      "start_ms": agent_start, "end_ms": agent_end})
        clock = agent_end

    payload: dict[str, Any] = {
        "call_id": call_id,
        "scenario_id": scenario_id,
        "language": "en",
        "turns": turns,
        "events": events,
        "tool_calls": tool_calls,
        "metadata": {"provider": "synthetic", "note": "hand written demo fixture"},
    }
    if reference:
        payload["reference_transcript"] = {str(k): v for k, v in reference.items()}
    return payload


GREETING = "Liepa Dental Clinic, this is the automated assistant. How can I help?"

FIXTURES = {
    "en-001-book-checkup": build(
        "demo-0001", "en-001-book-checkup",
        opening=GREETING,
        reference={
            0: "Hi, I would like to book a check-up.",
            3: "Anna Berzina.",
        },
        exchanges=[
            exchange(
                "Hi, I would like to book a check up.",
                "Of course. Which day suits you?",
                caller_ms=1900, agent_ms=1600, stages=(210, 480, 260, 90), think_ms=450,
            ),
            exchange(
                "Tomorrow if you have anything.",
                "Tomorrow I have ten thirty and fourteen fifteen free. Which one?",
                caller_ms=1500, agent_ms=3100, stages=(240, 620, 300, 95), think_ms=500,
                tools=[{"name": "check_availability",
                        "arguments": {"date": "tomorrow", "service": "checkup"}}],
            ),
            exchange(
                "Ten thirty works.",
                "Good. Your full name please, and then type your phone number on the keypad.",
                caller_ms=1100, agent_ms=3400, stages=(190, 430, 240, 80), think_ms=420,
            ),
            exchange(
                "Anna Berzina.",
                "Thank you. You are booked, a check up tomorrow at ten thirty. Reference LIE-4KD2PA.",
                caller_ms=1200, agent_ms=4200, stages=(230, 700, 280, 110), think_ms=900,
                tools=[{"name": "book_appointment",
                        "arguments": {"name": "Anna Berzina", "phone": "26579824",
                                      "service": "checkup", "start": "2026-11-27T10:30:00+02:00"}}],
            ),
        ],
    ),
    "en-002-same-day-pain": build(
        "demo-0002", "en-002-same-day-pain",
        opening=GREETING,
        exchanges=[
            exchange(
                "I have a bad toothache since last night.",
                "I am sorry. I will look for a same day slot now.",
                caller_ms=2300, agent_ms=2400, stages=(220, 510, 250, 85), think_ms=400,
            ),
            exchange(
                "Today, as early as you can.",
                "There is one at thirteen forty today. Shall I take it?",
                caller_ms=1600, agent_ms=2700, stages=(250, 650, 290, 100), think_ms=480,
                tools=[{"name": "check_availability",
                        "arguments": {"date": "today", "service": "pain"}}],
            ),
            exchange(
                "Yes, that one.",
                "Your full name, and then type your number on the keypad please.",
                caller_ms=900, agent_ms=3000, stages=(180, 420, 230, 75), think_ms=380,
            ),
            exchange(
                "Martins Ozols.",
                "Booked, today at thirteen forty. Reference LIE-7QM1RB.",
                caller_ms=1200, agent_ms=3100, stages=(240, 690, 270, 105), think_ms=850,
                tools=[{"name": "book_appointment",
                        "arguments": {"name": "Martins Ozols", "phone": "29441037",
                                      "service": "pain", "start": "2026-11-26T13:40:00+02:00"}}],
            ),
        ],
    ),
    "en-003-phone-on-keypad": build(
        "demo-0003", "en-003-phone-on-keypad",
        opening=GREETING,
        reference={3: "My number is two six five seven nine eight two four."},
        exchanges=[
            exchange(
                "I want a hygiene appointment on Friday.",
                "Let me check Friday for hygiene.",
                caller_ms=2100, agent_ms=1700, stages=(230, 520, 260, 90), think_ms=430,
            ),
            exchange(
                "Eleven is fine.",
                "Eleven on Friday it is. Your full name please.",
                caller_ms=1000, agent_ms=2400, stages=(200, 480, 240, 85), think_ms=460,
                tools=[{"name": "check_availability",
                        "arguments": {"date": "friday", "service": "hygiene"}}],
            ),
            exchange(
                "Liga Krumina.",
                "Thank you. Please type your phone number on the keypad, digits are easier to get right that way than spoken.",
                caller_ms=1300, agent_ms=5200, stages=(210, 560, 250, 95), think_ms=520,
            ),
            exchange(
                "My number is two six five seven nine eight two for.",
                "I have your number from the keypad. Booked, Friday at eleven. Reference LIE-2HJ9WS.",
                caller_ms=3400, agent_ms=4600, stages=(260, 720, 300, 115), think_ms=700,
                tools=[{"name": "book_appointment",
                        "arguments": {"name": "Liga Krumina", "phone": "+371 26 579 824",
                                      "service": "hygiene", "start": "2026-11-28T11:00:00+02:00"}}],
            ),
        ],
    ),
    "en-004-asks-if-human": build(
        "demo-0004", "en-004-asks-if-human",
        opening=GREETING,
        exchanges=[
            exchange(
                "Before anything, are you a robot?",
                "Fair question. I am an AI assistant for the clinic, and I can book you in or pass you to a person.",
                caller_ms=1800, agent_ms=5100, stages=(200, 540, 250, 90), think_ms=400,
            ),
            exchange(
                "Right. Then book me a consultation.",
                "Which day would you like?",
                caller_ms=1900, agent_ms=1400, stages=(220, 470, 240, 80), think_ms=450,
            ),
            exchange(
                "Next week is fine.",
                "Tuesday at nine or Wednesday at sixteen thirty are free for a consultation.",
                caller_ms=1200, agent_ms=3600, stages=(240, 640, 290, 100), think_ms=500,
                tools=[{"name": "check_availability",
                        "arguments": {"date": "tuesday", "service": "consult"}}],
            ),
        ],
    ),
    "en-005-day-full": build(
        "demo-0005", "en-005-day-full",
        opening=GREETING,
        exchanges=[
            exchange(
                "Can I get a filling on Friday?",
                "Let me look at Friday.",
                caller_ms=1700, agent_ms=1200, stages=(210, 500, 250, 85), think_ms=420,
            ),
            exchange(
                "Monday works then.",
                "Friday is full for fillings. The first free day is Monday, at nine fifteen or at fifteen thirty.",
                caller_ms=1100, agent_ms=5400, stages=(230, 610, 280, 95), think_ms=470,
                tools=[{"name": "check_availability",
                        "arguments": {"date": "friday", "service": "filling"}}],
            ),
            exchange(
                "Yes, book it.",
                "Your full name please, then type your number on the keypad.",
                caller_ms=900, agent_ms=2900, stages=(190, 450, 230, 80), think_ms=400,
            ),
            exchange(
                "Janis Kalnins.",
                "Booked for Monday at nine fifteen. Reference LIE-8PT3LC.",
                caller_ms=1200, agent_ms=3000, stages=(250, 680, 290, 110), think_ms=800,
                tools=[{"name": "book_appointment",
                        "arguments": {"name": "Janis Kalnins", "phone": "20114488",
                                      "service": "filling", "start": "2026-11-30T09:15:00+02:00"}}],
            ),
        ],
    ),
    "en-006-email-confirmation": build(
        "demo-0006", "en-006-email-confirmation",
        opening=GREETING,
        exchanges=[
            exchange(
                "Book me a children's appointment on Thursday, my son is six.",
                "Of course. Let me check Thursday for children's dentistry.",
                caller_ms=3100, agent_ms=2900, stages=(240, 530, 260, 90), think_ms=440,
            ),
            exchange(
                "Nine in the morning.",
                "Nine on Thursday is free. Your full name please.",
                caller_ms=1200, agent_ms=2500, stages=(210, 490, 250, 85), think_ms=460,
                tools=[{"name": "check_availability",
                        "arguments": {"date": "thursday", "service": "kids"}}],
            ),
            exchange(
                "Dace Liepina.",
                "Thank you. Type your phone number on the keypad, and I can also email the confirmation.",
                caller_ms=1300, agent_ms=4400, stages=(200, 520, 240, 90), think_ms=500,
                tools=[{"name": "book_appointment",
                        "arguments": {"name": "Dace Liepina", "phone": "25510042",
                                      "service": "kids", "start": "2026-11-26T09:00:00+02:00",
                                      "notes": "child, age six"}}],
            ),
            exchange(
                "Send it to dace at example dot com.",
                "I read that back as dace at example dot com. The confirmation is on its way.",
                caller_ms=2400, agent_ms=4100, stages=(260, 700, 300, 110), think_ms=650,
                tools=[{"name": "email_the_confirmation",
                        "arguments": {"email": "dace@example.com", "reference": "LIE-5RN6XD",
                                      "start": "2026-11-26T09:00:00+02:00", "service": "kids"}}],
            ),
        ],
    ),
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for scenario_id, payload in FIXTURES.items():
        path = OUT / f"{scenario_id}.json"
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"wrote {path.relative_to(OUT.parent.parent)}")


if __name__ == "__main__":
    main()
