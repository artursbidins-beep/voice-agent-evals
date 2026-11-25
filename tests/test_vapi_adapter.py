from voice_evals.adapters.vapi import call_record_from_vapi


def payload(messages):
    return {"id": "call-1", "assistantId": "a-1", "artifact": {"messages": messages}}


def test_seconds_from_start_are_read_as_an_offset():
    record = call_record_from_vapi(
        payload([
            {"role": "assistant", "message": "Hello", "secondsFromStart": 0.5, "duration": 1200},
            {"role": "user", "message": "Hi", "secondsFromStart": 3.2, "duration": 800},
        ]),
        "s-1",
    )
    assert [t.start_ms for t in record.turns] == [500, 3200]
    assert record.turns[0].end_ms == 1700


def test_unix_millisecond_timestamps_are_rebased_on_the_first_message():
    start = 1_763_300_000_000
    record = call_record_from_vapi(
        payload([
            {"role": "assistant", "message": "Hello", "time": start, "endTime": start + 1200},
            {"role": "user", "message": "Hi", "time": start + 3200, "endTime": start + 4000},
        ]),
        "s-1",
    )
    assert [t.start_ms for t in record.turns] == [0, 3200]
    assert record.turns[1].end_ms == 4000


def test_tool_calls_are_pulled_out_and_arguments_parsed_from_json():
    record = call_record_from_vapi(
        payload([
            {"role": "user", "message": "Tomorrow please", "secondsFromStart": 1.0},
            {
                "role": "tool_calls",
                "secondsFromStart": 2.0,
                "toolCalls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "check_availability",
                            "arguments": '{"date": "tomorrow", "service": "checkup"}',
                        },
                    }
                ],
            },
        ]),
        "s-1",
    )
    assert len(record.tool_calls) == 1
    call = record.tool_calls[0]
    assert call.name == "check_availability"
    assert call.arguments == {"date": "tomorrow", "service": "checkup"}
    assert len(record.turns) == 1  # a tool call is not something anyone heard


def test_unparsable_arguments_are_kept_rather_than_dropped():
    record = call_record_from_vapi(
        payload([
            {
                "role": "tool_calls",
                "secondsFromStart": 1.0,
                "toolCalls": [{"function": {"name": "book", "arguments": "{not json"}}],
            }
        ]),
        "s-1",
    )
    assert record.tool_calls[0].arguments == {"_raw": "{not json"}


def test_response_events_pair_each_caller_turn_with_the_answer():
    record = call_record_from_vapi(
        payload([
            {"role": "user", "message": "Hi", "secondsFromStart": 1.0, "duration": 1000},
            {"role": "assistant", "message": "Hello", "secondsFromStart": 2.4, "duration": 900},
        ]),
        "s-1",
    )
    kinds = {event.type for event in record.events}
    assert kinds == {"user_speech_end", "agent_audio_start"}
    starts = {event.type: event.at_ms for event in record.events}
    assert starts["agent_audio_start"] - starts["user_speech_end"] == 400


def test_the_stage_breakdown_is_declared_unavailable_not_invented():
    record = call_record_from_vapi(payload([]), "s-1")
    assert record.metadata["stage_breakdown"] == "unavailable from the call object"
    assert record.events == []


def test_system_messages_are_not_turns():
    record = call_record_from_vapi(
        payload([
            {"role": "system", "message": "You are a receptionist", "secondsFromStart": 0},
            {"role": "assistant", "message": "Hello", "secondsFromStart": 0.5},
        ]),
        "s-1",
    )
    assert len(record.turns) == 1
