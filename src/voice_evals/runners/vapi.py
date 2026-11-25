"""Fetch finished Vapi calls and score them.

Placing the call is deliberately not this runner's job yet. Scoring recorded
production traffic is the cheaper half of the problem and the half that finds
real defects, because the calls already happened and nobody had to pay for a
synthetic one.

A scenario is tied to a call by a ``call_id`` in the scenario file or by a
``calls.yaml`` next to the fixtures that maps scenario ids to call ids.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import yaml

from voice_evals.adapters.vapi import call_record_from_vapi
from voice_evals.models import CallRecord, Scenario
from voice_evals.runners.base import Runner, RunnerError

API_ROOT = "https://api.vapi.ai"


class VapiRunner(Runner):
    name = "vapi"

    def __init__(
        self,
        assistant_id: str = "",
        save_to: str | Path | None = None,
        api_key: str | None = None,
        calls_file: str | Path | None = None,
    ) -> None:
        self.assistant_id = assistant_id
        self.save_to = Path(save_to) if save_to else None
        self.api_key = api_key or os.environ.get("VAPI_API_KEY", "")
        self._calls = self._load_call_map(calls_file)

        if not self.api_key:
            raise RunnerError(
                "VAPI_API_KEY is not set. The offline provider needs no key: "
                "voice-evals run <config> --provider offline"
            )

    @staticmethod
    def _load_call_map(calls_file: str | Path | None) -> dict[str, str]:
        if not calls_file:
            return {}
        path = Path(calls_file)
        if not path.is_file():
            return {}
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return {str(k): str(v) for k, v in payload.items()}

    def _call_id_for(self, scenario: Scenario) -> str:
        call_id = self._calls.get(scenario.id)
        if not call_id:
            raise RunnerError(
                f"scenario {scenario.id!r} has no call id. Add it to calls.yaml."
            )
        return call_id

    def fetch_call(self, call_id: str) -> dict[str, Any]:
        request = urllib.request.Request(
            f"{API_ROOT}/call/{call_id}",
            headers={"Authorization": f"Bearer {self.api_key}"},
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            raise RunnerError(f"Vapi returned {error.code} for call {call_id}") from error
        except urllib.error.URLError as error:
            raise RunnerError(f"could not reach Vapi: {error.reason}") from error

    def run(self, scenario: Scenario) -> CallRecord:
        payload = self.fetch_call(self._call_id_for(scenario))
        record = call_record_from_vapi(payload, scenario.id, scenario.language)

        if not record.turns:
            raise RunnerError(
                f"call {record.call_id} has no messages in its artifact. "
                "Recording and transcripts have to be on for the assistant."
            )

        if self.save_to:
            self.save_to.mkdir(parents=True, exist_ok=True)
            fixture = {
                "call_id": record.call_id,
                "scenario_id": record.scenario_id,
                "language": record.language,
                "turns": [vars(turn) for turn in record.turns],
                "events": [vars(event) for event in record.events],
                "tool_calls": [vars(call) for call in record.tool_calls],
                "metadata": record.metadata,
            }
            target = self.save_to / f"{scenario.id}.json"
            target.write_text(
                json.dumps(fixture, indent=2, ensure_ascii=False), encoding="utf-8"
            )

        return record
