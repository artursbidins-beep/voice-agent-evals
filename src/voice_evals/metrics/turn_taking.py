"""Talking over the caller, ignoring an interruption, and dead air.

These three are what people mean when they say an agent "feels wrong" on the
phone, and none of them show up in a transcript. They only exist in the
timestamps, which is why this module works on turn windows and not on text.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from voice_evals.models import Turn, turns_in_order


@dataclass
class Overlap:
    """Both parties speaking at once."""

    start_ms: int
    end_ms: int
    interrupted: str  # which role was already speaking

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms


@dataclass
class DeadAir:
    after_role: str
    start_ms: int
    end_ms: int

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms


@dataclass
class TurnTakingResult:
    talkovers: list[Overlap] = field(default_factory=list)
    barge_ins: list[Overlap] = field(default_factory=list)
    barge_ins_ignored: int = 0
    dead_air: list[DeadAir] = field(default_factory=list)
    gaps_ms: list[int] = field(default_factory=list)

    @property
    def longest_gap_ms(self) -> int:
        return max(self.gaps_ms) if self.gaps_ms else 0

    @property
    def median_gap_ms(self) -> int:
        if not self.gaps_ms:
            return 0
        ordered = sorted(self.gaps_ms)
        middle = len(ordered) // 2
        if len(ordered) % 2:
            return ordered[middle]
        return (ordered[middle - 1] + ordered[middle]) // 2

    def as_dict(self) -> dict[str, Any]:
        return {
            "talkovers": len(self.talkovers),
            "talkover_ms_total": sum(o.duration_ms for o in self.talkovers),
            "barge_ins": len(self.barge_ins),
            "barge_ins_ignored": self.barge_ins_ignored,
            "dead_air_events": len(self.dead_air),
            "longest_gap_ms": self.longest_gap_ms,
            "median_gap_ms": self.median_gap_ms,
        }


def analyse_turn_taking(
    turns: list[Turn],
    *,
    overlap_min_ms: int = 300,
    dead_air_ms: int = 2500,
    barge_in_yield_ms: int = 700,
) -> TurnTakingResult:
    """Walk the timeline once and classify every overlap and every silence.

    ``overlap_min_ms`` exists because a 60 ms overlap is two people being
    human, not an agent steamrolling a caller. ``barge_in_yield_ms`` is how
    long the agent may keep talking after the caller cuts in before it counts
    as ignoring the interruption.
    """

    ordered = turns_in_order(turns)
    result = TurnTakingResult()

    for index, turn in enumerate(ordered):
        for other in ordered[index + 1 :]:
            if other.start_ms >= turn.end_ms:
                break
            if other.role == turn.role:
                continue

            overlap = Overlap(
                start_ms=other.start_ms,
                end_ms=min(turn.end_ms, other.end_ms),
                interrupted=turn.role,
            )
            if overlap.duration_ms < overlap_min_ms:
                continue

            if turn.role == "caller":
                # The agent started while the caller was still speaking.
                result.talkovers.append(overlap)
            else:
                # The caller cut in. Did the agent stop reasonably quickly?
                result.barge_ins.append(overlap)
                if turn.end_ms - other.start_ms > barge_in_yield_ms:
                    result.barge_ins_ignored += 1

    previous = None
    for turn in ordered:
        if previous is not None and turn.role != previous.role:
            gap = turn.start_ms - previous.end_ms
            if gap >= 0:
                result.gaps_ms.append(gap)
                if gap > dead_air_ms:
                    result.dead_air.append(
                        DeadAir(
                            after_role=previous.role,
                            start_ms=previous.end_ms,
                            end_ms=turn.start_ms,
                        )
                    )
        previous = turn

    return result
