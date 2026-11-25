# voice-agent-evals

A voice agent that sounds fine and books the wrong slot is worse than one that
sounds clumsy and books the right one. Transcripts do not show that, and
neither does listening to a few calls after a prompt change.

This is a regression suite for production voice agents. Scenarios are data,
the metrics are the four things that actually break on a phone line, and the
run exits non zero when one of them gets worse, so it belongs in a build
rather than on a dashboard nobody opens.

```
$ voice-evals run config/liepa-en.yaml

Result: pass, 6 of 6 scenarios passed

| stage          |   p50 |   p95 |   max | turns |
| -------------- | ----: | ----: | ----: | ----: |
| asr            |  220  |  260  |  260  |    23 |
| llm            |  530  |  700  |  720  |    23 |
| tts            |  260  |  300  |  300  |    23 |
| playback       |   90  |  110  |  115  |    23 |
| response_total | 1100  | 1370  | 1395  |    23 |
```

## What it measures

| metric | what it catches | why it is in here |
| --- | --- | --- |
| **Tool calls** | a missing call, a call with the wrong arguments, a booking made before availability was checked | the expensive failures, and the only ones a caller cannot hear |
| **Turn taking** | the agent talking over the caller, ignoring an interruption, dead air before an answer | what people mean when they say an agent "feels wrong", and it exists only in the timestamps |
| **Latency per stage** | which of transcription, model, speech synthesis or playback ate the budget | one total response number tells you it is slow, not what to fix |
| **Transcription** | word error rate, with a normaliser that knows Latvian and English | a small language line fails here first, and raw WER punishes "15" against "fifteen" for no reason |
| **AI disclosure** | the agent denying it is an AI, or dodging the question | easy to lose in a prompt rewrite, and the EU AI Act transparency rules apply from 2 August 2026 |

Thresholds live in the suite config. Any breach fails the run and is named in
the report:

```
**en-001-book-checkup**: Caller books a routine check-up for tomorrow
- missing tool call(s): book_appointment
- response p95 7370 ms over the 3000 ms budget
```

## Try it

```bash
pip install -e ".[dev]"
pytest                                   # 54 tests over the metrics themselves
voice-evals run config/liepa-en.yaml     # the recorded suite, no credentials
```

## A scenario

Scenarios are YAML, so adding one is a file and not a code change.

```yaml
id: en-003-phone-on-keypad
title: The phone number is taken on the keypad, not by ear
language: en
script:
  - "I want a hygiene appointment on Friday."
  - "Eleven is fine."
  - "My number is two six five seven nine eight two four."
expect:
  tool_calls:
    - name: check_availability
      arguments: { service: hygiene }
    - name: book_appointment
      arguments: { service: hygiene, phone: "26579824" }
  must_say_any: ["keypad", "type it"]
```

Arguments not named in the expectation are ignored, so a scenario stays
readable. Phone numbers compare by their national number, dates and clock
times are padded before comparing, and spoken numbers are mapped to digits
before scoring a transcript. Each of those is a line of judgement, each one is
documented where it happens, and each one has a test.

## Running it against a real agent

Two providers. `offline` replays recorded calls, needs no credentials and runs
on every push. `vapi` reads finished calls through the platform API and scores
those, which is the half of the problem that finds real defects, because the
calls already happened and nobody paid for a synthetic one.

```bash
export VAPI_API_KEY=...
voice-evals run config/liepa-en.yaml --provider vapi --save-fixtures
```

`--save-fixtures` writes each scored call into `fixtures/`, so tomorrow's
offline run replays today's production traffic.

## What it does not do

Stated plainly, because an eval suite that overstates itself is worse than none.

- **The four stage latency split needs the pipeline's own events.** A hosted
  platform's call object knows when the caller stopped and when the agent
  started, and nothing in between. The adapter fills `response_total` and
  marks the rest as unavailable rather than inventing it. The full split comes
  from a server event stream, which is the next thing going in here.
- **The fixtures in this repository are synthetic.** Real recorded calls carry
  caller names and phone numbers and do not belong in a public repository.
  They are written by hand in `tools/make_demo_fixtures.py` to exercise every
  metric, with timings in the range a real pipeline produces.
- **Liepa Dental Clinic is a fictional clinic** used as a demo target. The
  scenarios are modelled on a real reception line, the clinic is not.
- **The disclosure check is a regression test, not legal advice.** It checks
  one behaviour. Passing it is not compliance with anything.

## Layout

```
src/voice_evals/
  models.py          the normalised call record every metric reads
  loader.py          suite config and scenario files
  report.py          scoring, thresholds, JSON and markdown output
  cli.py             voice-evals run / show
  metrics/           tool_calls, turn_taking, latency, wer, compliance
  adapters/vapi.py   platform payload to call record, pure function
  runners/           offline replay, vapi fetch
scenarios/en/        six scenarios for the demo clinic line
fixtures/liepa-en/   the recorded calls the offline suite replays
tools/               the fixture generator
```

The adapter is a pure function on purpose: no network, no credentials, no
clock. That is what makes it testable, and an adapter that cannot be tested is
one that rots silently when the platform renames a field.

## Next

- The server event stream adapter, for the real four stage split.
- A Latvian suite against a Latvian line, where the WER normaliser earns its keep.
- More scenarios: wrong language mid call, two callers asking for the same
  slot, a caller who changes their mind after the booking.

MIT licensed.
