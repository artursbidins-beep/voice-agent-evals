"""The suite itself is a test: the demo fixtures have to stay green.

The second half of this file mutates a fixture in a temporary directory and
checks the harness notices. A regression suite that cannot fail is decoration.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from voice_evals.loader import SuiteConfig, load_scenarios
from voice_evals.report import build_report, evaluate
from voice_evals.runners.offline import OfflineRunner

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "config" / "liepa-en.yaml"


def run_suite(fixtures_dir: Path):
    config = SuiteConfig.load(CONFIG)
    scenarios = load_scenarios(config.scenarios_dir)
    runner = OfflineRunner(fixtures_dir)
    results = [evaluate(s, runner.run(s), config.thresholds) for s in scenarios]
    return build_report(config.name, "offline", config.assistant_name, results)


def test_every_scenario_has_a_fixture():
    config = SuiteConfig.load(CONFIG)
    for scenario in load_scenarios(config.scenarios_dir):
        assert (config.fixtures_dir / f"{scenario.id}.json").is_file(), scenario.id


def test_the_demo_suite_passes():
    report = run_suite(SuiteConfig.load(CONFIG).fixtures_dir)
    assert report.passed, [r.failures for r in report.results if not r.passed]
    assert len(report.results) == 6


def test_the_report_renders_markdown_with_the_pooled_budget():
    report = run_suite(SuiteConfig.load(CONFIG).fixtures_dir)
    markdown = report.to_markdown()
    assert "response_total" in markdown
    assert "| scenario |" in markdown
    json.loads(report.to_json())


@pytest.fixture()
def mutable_fixtures(tmp_path: Path) -> Path:
    target = tmp_path / "liepa-en"
    shutil.copytree(SuiteConfig.load(CONFIG).fixtures_dir, target)
    return target


def mutate(directory: Path, scenario_id: str, change) -> None:
    path = directory / f"{scenario_id}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    change(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")


def failures_for(report, scenario_id: str) -> list[str]:
    return next(r.failures for r in report.results if r.scenario_id == scenario_id)


def test_a_dropped_booking_call_fails_the_suite(mutable_fixtures: Path):
    mutate(
        mutable_fixtures,
        "en-001-book-checkup",
        lambda p: p.__setitem__(
            "tool_calls", [c for c in p["tool_calls"] if c["name"] != "book_appointment"]
        ),
    )
    report = run_suite(mutable_fixtures)
    assert not report.passed
    assert "missing tool call(s): book_appointment" in failures_for(
        report, "en-001-book-checkup"
    )


def test_a_slow_turn_breaks_the_latency_budget(mutable_fixtures: Path):
    def slow(payload):
        for event in payload["events"]:
            if event["turn"] >= 2 and event["type"] != "user_speech_end":
                event["at_ms"] += 6000

    mutate(mutable_fixtures, "en-002-same-day-pain", slow)
    report = run_suite(mutable_fixtures)
    assert any("response p95" in f for f in failures_for(report, "en-002-same-day-pain"))


def test_an_agent_that_denies_being_an_ai_fails(mutable_fixtures: Path):
    def deny(payload):
        for turn in payload["turns"]:
            if turn["role"] == "agent" and "AI assistant" in turn["text"]:
                turn["text"] = "No, I am a real person here at the clinic."

    mutate(mutable_fixtures, "en-004-asks-if-human", deny)
    report = run_suite(mutable_fixtures)
    assert any(
        "denied being an AI system" in failure
        for failure in failures_for(report, "en-004-asks-if-human")
    )


def test_talking_over_the_caller_fails(mutable_fixtures: Path):
    def overlap(payload):
        turns = payload["turns"]
        for index, turn in enumerate(turns):
            if turn["role"] == "caller" and index + 1 < len(turns):
                turns[index + 1]["start_ms"] = turn["start_ms"] + 200

    mutate(mutable_fixtures, "en-005-day-full", overlap)
    report = run_suite(mutable_fixtures)
    assert any("talkover" in f for f in failures_for(report, "en-005-day-full"))


def test_the_wrong_service_in_the_booking_is_caught(mutable_fixtures: Path):
    def wrong_service(payload):
        for call in payload["tool_calls"]:
            if call["name"] == "book_appointment":
                call["arguments"]["service"] = "hygiene"

    mutate(mutable_fixtures, "en-002-same-day-pain", wrong_service)
    report = run_suite(mutable_fixtures)
    assert any(
        "book_appointment.service" in f for f in failures_for(report, "en-002-same-day-pain")
    )
