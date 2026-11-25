"""Metrics. Each module takes the normalised model and returns a dataclass."""

from voice_evals.metrics.compliance import DisclosureResult, check_disclosure
from voice_evals.metrics.latency import StageStats, analyse_latency
from voice_evals.metrics.tool_calls import ToolCallResult, compare_tool_calls
from voice_evals.metrics.turn_taking import TurnTakingResult, analyse_turn_taking
from voice_evals.metrics.wer import WerResult, normalise, word_error_rate

__all__ = [
    "DisclosureResult",
    "StageStats",
    "ToolCallResult",
    "TurnTakingResult",
    "WerResult",
    "analyse_latency",
    "analyse_turn_taking",
    "check_disclosure",
    "compare_tool_calls",
    "normalise",
    "word_error_rate",
]
