"""Score a call against its scenario, then the suite against its thresholds."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from voice_evals.loader import Thresholds
from voice_evals.metrics.compliance import DisclosureResult, check_disclosure
from voice_evals.metrics.latency import (
    STAGES,
    TOTAL_STAGE,
    StageStats,
    merge_stage_stats,
    stage_durations,
)
from voice_evals.metrics.tool_calls import ToolCallResult, compare_tool_calls
from voice_evals.metrics.turn_taking import TurnTakingResult, analyse_turn_taking
from voice_evals.metrics.wer import WerResult, corpus_wer
from voice_evals.models import CallRecord, Scenario


@dataclass
class ScenarioResult:
    scenario_id: str
    title: str
    call_id: str
    language: str
    tool_calls: ToolCallResult
    turn_taking: TurnTakingResult
    disclosure: DisclosureResult
    latency: dict[str, StageStats]
    durations: dict[str, list[int]]
    wer: WerResult | None = None
    failures: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.failures

    def as_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "title": self.title,
            "call_id": self.call_id,
            "language": self.language,
            "passed": self.passed,
            "failures": self.failures,
            "tool_calls": self.tool_calls.as_dict(),
            "turn_taking": self.turn_taking.as_dict(),
            "disclosure": self.disclosure.as_dict(),
            "wer": self.wer.as_dict() if self.wer else None,
            "latency": {name: stats.as_dict() for name, stats in self.latency.items()},
        }


@dataclass
class SuiteReport:
    suite: str
    provider: str
    assistant: str
    started_at: str
    results: list[ScenarioResult] = field(default_factory=list)
    pooled_latency: dict[str, StageStats] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return all(result.passed for result in self.results)

    @property
    def failed_count(self) -> int:
        return sum(1 for result in self.results if not result.passed)

    def as_dict(self) -> dict[str, Any]:
        return {
            "suite": self.suite,
            "provider": self.provider,
            "assistant": self.assistant,
            "started_at": self.started_at,
            "scenarios": len(self.results),
            "failed": self.failed_count,
            "passed": self.passed,
            "pooled_latency": {n: s.as_dict() for n, s in self.pooled_latency.items()},
            "results": [result.as_dict() for result in self.results],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.as_dict(), indent=indent, ensure_ascii=False)

    def to_markdown(self) -> str:
        lines = [
            f"# {self.suite}",
            "",
            f"Agent: {self.assistant or 'not named'} via {self.provider}  ",
            f"Run: {self.started_at}  ",
            f"Result: **{'pass' if self.passed else 'fail'}**, "
            f"{len(self.results) - self.failed_count} of {len(self.results)} scenarios passed",
            "",
            "## Response time, pooled over every turn in the suite",
            "",
            "| stage | p50 | p95 | max | turns |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for name in [stage[0] for stage in STAGES] + [TOTAL_STAGE[0]]:
            stats = self.pooled_latency.get(name)
            if not stats:
                continue
            lines.append(
                f"| {name} | {stats.p50_ms} ms | {stats.p95_ms} ms | "
                f"{stats.max_ms} ms | {stats.count} |"
            )

        lines += ["", "## Scenarios", "", "| scenario | tools | turn taking | WER | verdict |",
                  "| --- | --- | --- | ---: | --- |"]
        for result in self.results:
            tools = f"{len(result.tool_calls.matched)}/{len(result.tool_calls.matched) + len(result.tool_calls.missing)}"
            turn_taking = (
                f"{len(result.turn_taking.talkovers)} talkover, "
                f"{result.turn_taking.barge_ins_ignored} ignored"
            )
            wer = f"{result.wer.wer:.1%}" if result.wer else "n/a"
            verdict = "pass" if result.passed else "FAIL"
            lines.append(
                f"| {result.scenario_id} | {tools} | {turn_taking} | {wer} | {verdict} |"
            )

        failures = [r for r in self.results if not r.passed]
        if failures:
            lines += ["", "## Failures", ""]
            for result in failures:
                lines.append(f"**{result.scenario_id}**: {result.title}")
                for failure in result.failures:
                    lines.append(f"- {failure}")
                lines.append("")

        return "\n".join(lines)


def _text_of(record: CallRecord, role: str) -> str:
    return " ".join(turn.text for turn in record.turns if turn.role == role).casefold()


def evaluate(
    scenario: Scenario,
    record: CallRecord,
    thresholds: Thresholds,
) -> ScenarioResult:
    """Apply every metric to one call and collect the threshold breaches."""

    tool_calls = compare_tool_calls(scenario.expected_tool_calls, record.tool_calls)
    turn_taking = analyse_turn_taking(record.turns)
    disclosure = check_disclosure(record.turns, record.language)
    latency = {}
    durations = stage_durations(record)
    for name, values in durations.items():
        latency[name] = merge_stage_stats(name, [values])

    wer: WerResult | None = None
    if record.reference_transcript:
        pairs: list[tuple[str, str]] = []
        caller_turns = record.caller_turns()
        for index, reference in sorted(record.reference_transcript.items()):
            if 0 <= index < len(caller_turns):
                pairs.append((reference, caller_turns[index].text))
        if pairs:
            wer = corpus_wer(pairs, record.language)

    failures: list[str] = []

    if len(tool_calls.missing) > thresholds.max_missing_tool_calls:
        failures.append(f"missing tool call(s): {', '.join(tool_calls.missing)}")
    if len(tool_calls.argument_mismatches) > thresholds.max_argument_mismatches:
        for mismatch in tool_calls.argument_mismatches:
            failures.append(
                f"{mismatch.tool}.{mismatch.argument} was {mismatch.actual!r}, "
                f"expected {mismatch.expected!r}"
            )
    if tool_calls.out_of_order:
        failures.append(f"{tool_calls.out_of_order} tool call(s) came back out of order")
    if scenario.expected_tool_calls and tool_calls.f1 < thresholds.min_tool_f1:
        failures.append(f"tool call f1 {tool_calls.f1:.2f} below {thresholds.min_tool_f1:.2f}")

    if len(turn_taking.talkovers) > thresholds.max_talkovers:
        failures.append(f"{len(turn_taking.talkovers)} talkover(s) over the caller")
    if turn_taking.barge_ins_ignored > thresholds.max_barge_ins_ignored:
        failures.append(f"{turn_taking.barge_ins_ignored} interruption(s) ignored")
    if len(turn_taking.dead_air) > thresholds.max_dead_air_events:
        longest = max((d.duration_ms for d in turn_taking.dead_air), default=0)
        failures.append(
            f"{len(turn_taking.dead_air)} dead air gap(s), longest {longest} ms"
        )

    total = latency.get(TOTAL_STAGE[0])
    if total and total.count and total.p95_ms > thresholds.max_response_p95_ms:
        failures.append(
            f"response p95 {total.p95_ms} ms over the {thresholds.max_response_p95_ms} ms budget"
        )

    if wer and wer.wer > thresholds.max_wer:
        failures.append(f"transcription WER {wer.wer:.1%} over {thresholds.max_wer:.0%}")

    agent_text = _text_of(record, "agent")
    if scenario.must_say_any and not any(
        phrase.casefold() in agent_text for phrase in scenario.must_say_any
    ):
        failures.append("agent said none of the required phrases")
    for phrase in scenario.must_not_say:
        if phrase.casefold() in agent_text:
            failures.append(f"agent said a forbidden phrase: {phrase!r}")

    if thresholds.require_disclosure and not disclosure.passed:
        if disclosure.denied:
            failures.append("agent denied being an AI system when asked")
        else:
            failures.append("agent was asked if it is human and did not disclose")
    if scenario.asks_if_human and not disclosure.asked:
        failures.append("scenario expects the caller to ask if the agent is human")

    return ScenarioResult(
        scenario_id=scenario.id,
        title=scenario.title,
        call_id=record.call_id,
        language=record.language,
        tool_calls=tool_calls,
        turn_taking=turn_taking,
        disclosure=disclosure,
        latency=latency,
        durations=durations,
        wer=wer,
        failures=failures,
    )


def build_report(
    suite: str,
    provider: str,
    assistant: str,
    results: list[ScenarioResult],
) -> SuiteReport:
    pooled: dict[str, StageStats] = {}
    names = [stage[0] for stage in STAGES] + [TOTAL_STAGE[0]]
    for name in names:
        pooled[name] = merge_stage_stats(
            name, [result.durations.get(name, []) for result in results]
        )

    return SuiteReport(
        suite=suite,
        provider=provider,
        assistant=assistant,
        started_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        results=results,
        pooled_latency=pooled,
    )
