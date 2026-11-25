"""Did the agent call the right functions, with the right arguments, in order.

This is the metric that catches the expensive failures. A voice agent that
sounds perfect and books the wrong slot is worse than one that sounds clumsy
and books the right one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from voice_evals.models import ExpectedToolCall, ToolCall

_NON_DIGITS = re.compile(r"\D")
_WHITESPACE = re.compile(r"\s+")


@dataclass
class ArgumentMismatch:
    tool: str
    argument: str
    expected: Any
    actual: Any


@dataclass
class ToolCallResult:
    matched: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    unexpected: list[str] = field(default_factory=list)
    argument_mismatches: list[ArgumentMismatch] = field(default_factory=list)
    out_of_order: int = 0

    @property
    def precision(self) -> float:
        produced = len(self.matched) + len(self.unexpected)
        return len(self.matched) / produced if produced else 1.0

    @property
    def recall(self) -> float:
        required = len(self.matched) + len(self.missing)
        return len(self.matched) / required if required else 1.0

    @property
    def f1(self) -> float:
        precision, recall = self.precision, self.recall
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    @property
    def passed(self) -> bool:
        return not self.missing and not self.argument_mismatches and self.out_of_order == 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "matched": self.matched,
            "missing": self.missing,
            "unexpected": self.unexpected,
            "out_of_order": self.out_of_order,
            "argument_mismatches": [
                {
                    "tool": m.tool,
                    "argument": m.argument,
                    "expected": m.expected,
                    "actual": m.actual,
                }
                for m in self.argument_mismatches
            ],
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "passed": self.passed,
        }


def _looks_like_phone(value: str) -> bool:
    digits = _NON_DIGITS.sub("", value)
    return len(digits) >= 7 and len(digits) >= len(value) - 6


def normalise_argument(value: Any) -> Any:
    """Compare what a human would compare.

    A phone number is the digits, not the spaces and the plus sign. A date is
    the date, whether or not the model padded the month. Everything else is
    compared casefolded with its whitespace collapsed.
    """

    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, dict):
        return {k: normalise_argument(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [normalise_argument(v) for v in value]

    text = _WHITESPACE.sub(" ", str(value).strip())

    # Dates and clock times first: "2026-01-07" is eight digits and would
    # otherwise be mistaken for a phone number.
    iso_date = re.fullmatch(r"(\d{4})-(\d{1,2})-(\d{1,2})", text)
    if iso_date:
        year, month, day = (int(part) for part in iso_date.groups())
        return f"{year:04d}-{month:02d}-{day:02d}"

    clock = re.fullmatch(r"(\d{1,2})[:.](\d{2})", text)
    if clock:
        hour, minute = (int(part) for part in clock.groups())
        return f"{hour:02d}:{minute:02d}"

    if _looks_like_phone(text):
        # Compare the national significant number. "+371 26 579 824" and
        # "26579824" are the same phone, and which form the model produced is
        # a formatting question, not a correctness one. Eight digits is the
        # length of a Latvian subscriber number; longer numbers keep their
        # last eight, which is specific enough inside one clinic's book.
        digits = _NON_DIGITS.sub("", text).lstrip("0")
        return digits[-8:] if len(digits) >= 8 else digits

    return text.casefold()


def compare_tool_calls(
    expected: list[ExpectedToolCall],
    actual: list[ToolCall],
) -> ToolCallResult:
    """Match expected calls to actual ones, left to right.

    An actual call may only match an expected one that has not been matched
    yet, so a repeated call does not satisfy two expectations at once.
    """

    result = ToolCallResult()
    remaining = list(enumerate(actual))
    matched_positions: list[int] = []

    for expectation in expected:
        hit = None
        for index, (position, call) in enumerate(remaining):
            if call.name == expectation.name:
                hit = (index, position, call)
                break

        if hit is None:
            if expectation.required:
                result.missing.append(expectation.name)
            continue

        index, position, call = hit
        remaining.pop(index)
        result.matched.append(expectation.name)
        matched_positions.append(position)

        for key, want in expectation.arguments.items():
            got = call.arguments.get(key)
            if normalise_argument(got) != normalise_argument(want):
                result.argument_mismatches.append(
                    ArgumentMismatch(tool=call.name, argument=key, expected=want, actual=got)
                )

    result.unexpected = [call.name for _, call in remaining]
    result.out_of_order = _inversions(matched_positions)
    return result


def _inversions(positions: list[int]) -> int:
    """How many pairs of matched calls came back in the wrong order."""

    count = 0
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            if positions[i] > positions[j]:
                count += 1
    return count
