"""Load the suite configuration and the scenario files."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from voice_evals.models import Scenario


@dataclass
class Thresholds:
    """What counts as a regression. Every one of these fails the build."""

    max_wer: float = 0.20
    min_tool_f1: float = 0.95
    max_missing_tool_calls: int = 0
    max_argument_mismatches: int = 0
    max_talkovers: int = 1
    max_barge_ins_ignored: int = 0
    max_dead_air_events: int = 0
    max_response_p95_ms: int = 3000
    require_disclosure: bool = True

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "Thresholds":
        payload = payload or {}
        known = {f: payload[f] for f in cls.__dataclass_fields__ if f in payload}
        unknown = set(payload) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown threshold(s): {', '.join(sorted(unknown))}")
        return cls(**known)


@dataclass
class SuiteConfig:
    name: str
    language: str = "en"
    description: str = ""
    provider: str = "offline"
    assistant_id: str = ""
    assistant_name: str = ""
    scenarios_dir: Path = Path("scenarios")
    fixtures_dir: Path = Path("fixtures")
    thresholds: Thresholds = field(default_factory=Thresholds)

    @classmethod
    def load(cls, path: str | Path) -> "SuiteConfig":
        path = Path(path)
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        target = payload.get("target") or {}
        root = path.parent.parent if path.parent.name == "config" else path.parent

        return cls(
            name=payload.get("name", path.stem),
            language=payload.get("language", "en"),
            description=payload.get("description", ""),
            provider=target.get("provider", "offline"),
            assistant_id=target.get("assistant_id", ""),
            assistant_name=target.get("assistant_name", ""),
            scenarios_dir=root / payload.get("scenarios", "scenarios"),
            fixtures_dir=root / payload.get("fixtures", "fixtures"),
            thresholds=Thresholds.from_dict(payload.get("thresholds")),
        )


def load_scenarios(directory: str | Path) -> list[Scenario]:
    """Read every scenario file in a directory, sorted by id."""

    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"no scenario directory at {directory}")

    scenarios: list[Scenario] = []
    seen: set[str] = set()

    for path in sorted(directory.glob("*.y*ml")):
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not payload:
            continue
        scenario = Scenario.from_dict(payload)
        if scenario.id in seen:
            raise ValueError(f"duplicate scenario id {scenario.id!r} in {path}")
        seen.add(scenario.id)
        scenarios.append(scenario)

    if not scenarios:
        raise FileNotFoundError(f"no scenarios found in {directory}")
    return scenarios
