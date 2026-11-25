from voice_evals.metrics.latency import analyse_latency, percentile, stage_durations
from voice_evals.models import CallRecord, Event


def record(*events):
    return CallRecord(call_id="c", scenario_id="s", events=list(events))


def exchange(turn, start, asr, llm, tts, playback):
    at = start
    out = [Event("user_speech_end", at, turn)]
    for name, duration in (
        ("asr_final", asr),
        ("llm_first_token", llm),
        ("tts_first_byte", tts),
        ("agent_audio_start", playback),
    ):
        at += duration
        out.append(Event(name, at, turn))
    return out


def test_each_stage_is_measured_separately():
    stats = analyse_latency(record(*exchange(1, 0, 200, 500, 250, 80)))
    assert stats["asr"].p50_ms == 200
    assert stats["llm"].p50_ms == 500
    assert stats["tts"].p50_ms == 250
    assert stats["playback"].p50_ms == 80
    assert stats["response_total"].p50_ms == 1030


def test_percentiles_use_the_tail_not_the_mean():
    events = []
    for index in range(1, 20):
        events += exchange(index, index * 10_000, 200, 400, 200, 50)
    events += exchange(20, 300_000, 200, 4000, 200, 50)  # one very slow turn
    stats = analyse_latency(record(*events))
    assert stats["response_total"].p50_ms == 850
    assert stats["response_total"].p95_ms == 4450
    assert stats["response_total"].count == 20


def test_an_incomplete_exchange_is_skipped_not_guessed():
    events = exchange(1, 0, 200, 400, 200, 50)
    events.append(Event("user_speech_end", 50_000, 2))  # turn 2 never answered
    durations = stage_durations(record(*events))
    assert len(durations["response_total"]) == 1


def test_negative_durations_are_dropped_rather_than_improving_the_numbers():
    events = [
        Event("user_speech_end", 5_000, 1),
        Event("agent_audio_start", 4_000, 1),  # clock skew
    ]
    assert stage_durations(record(*events))["response_total"] == []


def test_percentile_of_nothing_is_zero():
    assert percentile([], 0.95) == 0
