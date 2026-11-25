"""Where the seconds go.

One number for total response time tells you the agent is slow. It does not
tell you which of the four stages to work on, and on a phone line the answer
is rarely the one people guess. This splits the budget per stage, per turn,
and reports the tail rather than the average, because callers remember the
worst turn and not the mean.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from voice_evals.models import CallRecord, Event

#: (stage name, event that starts it, event that ends it)
STAGES: tuple[tuple[str, str, str], ...] = (
    ("asr", "user_speech_end", "asr_final"),
    ("llm", "asr_final", "llm_first_token"),
    ("tts", "llm_first_token", "tts_first_byte"),
    ("playback", "tts_first_byte", "agent_audio_start"),
)

TOTAL_STAGE = ("response_total", "user_speech_end", "agent_audio_start")


@dataclass(frozen=True)
class StageStats:
    stage: str
    count: int
    p50_ms: int
    p95_ms: int
    mean_ms: int
    max_ms: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "count": self.count,
            "p50_ms": self.p50_ms,
            "p95_ms": self.p95_ms,
            "mean_ms": self.mean_ms,
            "max_ms": self.max_ms,
        }


def percentile(values: list[int], fraction: float) -> int:
    """Nearest rank percentile. No dependency needed and no interpolation lies."""

    if not values:
        return 0
    ordered = sorted(values)
    rank = max(1, min(len(ordered), round(fraction * len(ordered) + 0.5)))
    return ordered[rank - 1]


def _by_turn(events: list[Event]) -> dict[int, dict[str, int]]:
    table: dict[int, dict[str, int]] = {}
    for event in events:
        table.setdefault(event.turn, {})[event.type] = event.at_ms
    return table


def stage_durations(record: CallRecord) -> dict[str, list[int]]:
    """Per stage durations in milliseconds, one entry per exchange."""

    table = _by_turn(record.events)
    durations: dict[str, list[int]] = {name: [] for name, _, _ in STAGES}
    durations[TOTAL_STAGE[0]] = []

    for stage in (*STAGES, TOTAL_STAGE):
        name, start_event, end_event = stage
        for marks in table.values():
            start, end = marks.get(start_event), marks.get(end_event)
            if start is None or end is None:
                continue
            if end < start:
                # Clock skew or a mislabelled event. Counting it would quietly
                # improve the numbers, so it is dropped and reported as missing.
                continue
            durations[name].append(end - start)

    return durations


def analyse_latency(record: CallRecord) -> dict[str, StageStats]:
    stats: dict[str, StageStats] = {}
    for name, values in stage_durations(record).items():
        stats[name] = StageStats(
            stage=name,
            count=len(values),
            p50_ms=percentile(values, 0.50),
            p95_ms=percentile(values, 0.95),
            mean_ms=round(sum(values) / len(values)) if values else 0,
            max_ms=max(values) if values else 0,
        )
    return stats


def merge_stage_stats(name: str, per_call: list[list[int]]) -> StageStats:
    """Pool the raw durations of several calls before taking percentiles."""

    pooled = [value for values in per_call for value in values]
    return StageStats(
        stage=name,
        count=len(pooled),
        p50_ms=percentile(pooled, 0.50),
        p95_ms=percentile(pooled, 0.95),
        mean_ms=round(sum(pooled) / len(pooled)) if pooled else 0,
        max_ms=max(pooled) if pooled else 0,
    )
