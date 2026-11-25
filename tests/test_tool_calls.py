from voice_evals.metrics.tool_calls import compare_tool_calls, normalise_argument
from voice_evals.models import ExpectedToolCall, ToolCall


def test_happy_path():
    result = compare_tool_calls(
        [ExpectedToolCall("check_availability"), ExpectedToolCall("book_appointment")],
        [ToolCall("check_availability"), ToolCall("book_appointment")],
    )
    assert result.passed
    assert result.f1 == 1.0


def test_missing_call_is_caught():
    result = compare_tool_calls(
        [ExpectedToolCall("check_availability"), ExpectedToolCall("book_appointment")],
        [ToolCall("check_availability")],
    )
    assert result.missing == ["book_appointment"]
    assert not result.passed
    assert result.recall == 0.5


def test_booking_without_checking_first_is_out_of_order():
    result = compare_tool_calls(
        [ExpectedToolCall("check_availability"), ExpectedToolCall("book_appointment")],
        [ToolCall("book_appointment"), ToolCall("check_availability")],
    )
    assert result.out_of_order == 1
    assert not result.passed


def test_extra_call_lowers_precision_but_is_not_fatal():
    result = compare_tool_calls(
        [ExpectedToolCall("book_appointment")],
        [ToolCall("book_appointment"), ToolCall("send_sms")],
    )
    assert result.unexpected == ["send_sms"]
    assert result.precision == 0.5
    assert result.passed


def test_wrong_argument_is_reported_with_both_values():
    result = compare_tool_calls(
        [ExpectedToolCall("book_appointment", {"service": "checkup"})],
        [ToolCall("book_appointment", {"service": "filling"})],
    )
    assert not result.passed
    mismatch = result.argument_mismatches[0]
    assert (mismatch.argument, mismatch.expected, mismatch.actual) == (
        "service", "checkup", "filling",
    )


def test_arguments_not_named_in_the_expectation_are_ignored():
    result = compare_tool_calls(
        [ExpectedToolCall("book_appointment", {"service": "checkup"})],
        [ToolCall("book_appointment", {"service": "checkup", "notes": "nervous"})],
    )
    assert result.passed


def test_a_repeated_call_does_not_satisfy_two_expectations():
    result = compare_tool_calls(
        [ExpectedToolCall("check_availability"), ExpectedToolCall("check_availability")],
        [ToolCall("check_availability")],
    )
    assert result.missing == ["check_availability"]


def test_phone_numbers_compare_by_national_number():
    assert normalise_argument("+371 26 579 824") == normalise_argument("26579824")
    assert normalise_argument("+371 26 579 824") != normalise_argument("26579825")


def test_dates_and_clock_times_are_padded_before_comparing():
    assert normalise_argument("2026-1-7") == normalise_argument("2026-01-07")
    assert normalise_argument("9:15") == normalise_argument("09:15")


def test_optional_calls_are_not_required():
    result = compare_tool_calls(
        [ExpectedToolCall("email_the_confirmation", required=False)],
        [],
    )
    assert result.missing == []
    assert result.passed
