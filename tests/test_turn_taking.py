from voice_evals.metrics.turn_taking import analyse_turn_taking
from voice_evals.models import Turn


def turn(role, start, end, text="..."):
    return Turn(role=role, text=text, start_ms=start, end_ms=end)


def test_clean_alternating_conversation_has_nothing_to_report():
    result = analyse_turn_taking([
        turn("agent", 0, 1500),
        turn("caller", 2000, 3500),
        turn("agent", 4000, 5500),
    ])
    assert result.talkovers == []
    assert result.barge_ins == []
    assert result.dead_air == []


def test_agent_speaking_over_the_caller_is_a_talkover():
    result = analyse_turn_taking([
        turn("caller", 0, 3000),
        turn("agent", 1500, 4000),
    ])
    assert len(result.talkovers) == 1
    assert result.talkovers[0].duration_ms == 1500
    assert result.talkovers[0].interrupted == "caller"


def test_a_very_short_overlap_is_two_people_being_human():
    result = analyse_turn_taking([
        turn("caller", 0, 3000),
        turn("agent", 2900, 4000),
    ])
    assert result.talkovers == []


def test_agent_that_yields_quickly_to_a_barge_in_passes():
    result = analyse_turn_taking([
        turn("agent", 0, 2400),
        turn("caller", 2000, 3500),
    ])
    assert len(result.barge_ins) == 1
    assert result.barge_ins_ignored == 0


def test_agent_that_keeps_talking_through_a_barge_in_fails():
    result = analyse_turn_taking([
        turn("agent", 0, 6000),
        turn("caller", 2000, 3500),
    ])
    assert result.barge_ins_ignored == 1


def test_a_long_silence_before_the_answer_is_dead_air():
    result = analyse_turn_taking([
        turn("caller", 0, 1000),
        turn("agent", 5000, 6000),
    ])
    assert len(result.dead_air) == 1
    assert result.dead_air[0].duration_ms == 4000
    assert result.longest_gap_ms == 4000


def test_median_gap_ignores_the_one_bad_turn():
    result = analyse_turn_taking([
        turn("caller", 0, 1000),
        turn("agent", 1800, 2500),
        turn("caller", 3300, 4000),
        turn("agent", 4800, 5500),
        turn("caller", 6300, 7000),
        turn("agent", 11000, 12000),
    ])
    assert result.median_gap_ms == 800
    assert result.longest_gap_ms == 4000
