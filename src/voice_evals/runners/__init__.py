"""Runners turn a scenario into a :class:`CallRecord`."""

from voice_evals.runners.base import Runner
from voice_evals.runners.offline import OfflineRunner

__all__ = ["OfflineRunner", "Runner", "get_runner"]


def get_runner(provider: str, **kwargs: object) -> Runner:
    """Pick a runner by name.

    ``offline`` replays recorded calls and needs no credentials, which is what
    continuous integration runs on every push. ``vapi`` places real calls and
    costs money, so it runs on a schedule.
    """

    if provider in ("offline", "fixtures"):
        return OfflineRunner(**kwargs)  # type: ignore[arg-type]
    if provider == "vapi":
        from voice_evals.runners.vapi import VapiRunner

        return VapiRunner(**kwargs)  # type: ignore[arg-type]
    raise ValueError(f"unknown provider {provider!r}")
