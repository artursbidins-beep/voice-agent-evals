"""The one interface a platform has to satisfy."""

from __future__ import annotations

from abc import ABC, abstractmethod

from voice_evals.models import CallRecord, Scenario


class Runner(ABC):
    """Run one scenario against an agent and return a normalised record."""

    name: str = "runner"

    @abstractmethod
    def run(self, scenario: Scenario) -> CallRecord:
        """Execute the scenario. Raise :class:`RunnerError` if it cannot."""

    def close(self) -> None:
        """Release anything the runner opened. Safe to call twice."""


class RunnerError(RuntimeError):
    """The call could not be made or could not be read back."""
