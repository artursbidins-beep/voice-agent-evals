"""Replay recorded calls.

Every live call costs money and takes a minute of wall clock time, so the
suite that runs on every push replays calls that were recorded earlier. The
metrics cannot tell the difference, which is the point: a change to the
scoring code is caught immediately, for free, and deterministically.
"""

from __future__ import annotations

import json
from pathlib import Path

from voice_evals.models import CallRecord, Scenario
from voice_evals.runners.base import Runner, RunnerError


class OfflineRunner(Runner):
    name = "offline"

    def __init__(self, fixtures_dir: str | Path) -> None:
        self.fixtures_dir = Path(fixtures_dir)

    def run(self, scenario: Scenario) -> CallRecord:
        path = self.fixtures_dir / f"{scenario.id}.json"
        if not path.is_file():
            raise RunnerError(
                f"no recorded call for scenario {scenario.id!r} at {path}. "
                "Run the live suite once with --save-fixtures to create it."
            )

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise RunnerError(f"{path} is not valid JSON: {error}") from error

        payload.setdefault("scenario_id", scenario.id)
        payload.setdefault("language", scenario.language)
        return CallRecord.from_dict(payload)
