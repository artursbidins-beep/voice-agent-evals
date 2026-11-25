"""Word error rate, with a text normaliser that knows Latvian and English.

Raw WER punishes a transcript for writing "15" where the reference writes
"piecpadsmit", which tells you nothing about whether the agent understood the
caller. Normalising before scoring is standard practice in speech recognition,
so this module reports both numbers and the report shows both.

Diacritics are never stripped. On a Latvian line the difference between "gaļa"
and "gala" is the whole point of measuring.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_PUNCTUATION = re.compile(r"[^\w\s]", re.UNICODE)
_WHITESPACE = re.compile(r"\s+", re.UNICODE)

#: Spoken forms that mean the same number. Only the range a receptionist
#: actually says out loud: times, dates, ages, prices under a hundred.
NUMBER_WORDS: dict[str, dict[str, str]] = {
    "lv": {
        "nulle": "0", "viens": "1", "divi": "2", "tris": "3", "trīs": "3",
        "cetri": "4", "četri": "4", "pieci": "5", "sesi": "6", "seši": "6",
        "septini": "7", "septiņi": "7", "astoni": "8", "astoņi": "8",
        "devini": "9", "deviņi": "9", "desmit": "10", "vienpadsmit": "11",
        "divpadsmit": "12", "trispadsmit": "13", "trīspadsmit": "13",
        "cetrpadsmit": "14", "četrpadsmit": "14", "piecpadsmit": "15",
        "sespadsmit": "16", "sešpadsmit": "16", "septinpadsmit": "17",
        "septiņpadsmit": "17", "astonpadsmit": "18", "astoņpadsmit": "18",
        "devinpadsmit": "19", "deviņpadsmit": "19", "divdesmit": "20",
        "trisdesmit": "30", "trīsdesmit": "30", "cetrdesmit": "40",
        "četrdesmit": "40", "piecdesmit": "50",
    },
    "en": {
        "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
        "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
        "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
        "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
        "eighteen": "18", "nineteen": "19", "twenty": "20", "thirty": "30",
        "forty": "40", "fifty": "50",
    },
}

#: Words a transcript may add or drop without changing the meaning.
FILLER_WORDS: dict[str, set[str]] = {
    "lv": {"nu", "ee", "mmm", "tatad", "tātad"},
    "en": {"uh", "um", "erm", "mm"},
}


@dataclass(frozen=True)
class WerResult:
    """The error rate and the three kinds of error behind it."""

    wer: float
    substitutions: int
    deletions: int
    insertions: int
    reference_words: int

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    def as_dict(self) -> dict[str, float | int]:
        return {
            "wer": round(self.wer, 4),
            "substitutions": self.substitutions,
            "deletions": self.deletions,
            "insertions": self.insertions,
            "reference_words": self.reference_words,
        }


def normalise(
    text: str,
    language: str = "en",
    *,
    drop_fillers: bool = True,
    map_numbers: bool = True,
) -> list[str]:
    """Casefold, drop punctuation, map spoken numbers to digits, return tokens.

    With both flags off this is the raw tokenisation, which is what the raw
    WER number is computed from.
    """

    cleaned = _PUNCTUATION.sub(" ", text.casefold())
    tokens = [token for token in _WHITESPACE.split(cleaned) if token]

    numbers = NUMBER_WORDS.get(language, {}) if map_numbers else {}
    fillers = FILLER_WORDS.get(language, set()) if drop_fillers else set()

    out: list[str] = []
    for token in tokens:
        if token in fillers:
            continue
        out.append(numbers.get(token, token))
    return out


def _edit_counts(reference: list[str], hypothesis: list[str]) -> tuple[int, int, int]:
    """Levenshtein distance that keeps the three operation counts apart."""

    rows, cols = len(reference) + 1, len(hypothesis) + 1
    # cost[i][j] plus the counts that produced it
    cost = [[0] * cols for _ in range(rows)]
    ops = [[(0, 0, 0)] * cols for _ in range(rows)]  # (sub, del, ins)

    for i in range(1, rows):
        cost[i][0] = i
        ops[i][0] = (0, i, 0)
    for j in range(1, cols):
        cost[0][j] = j
        ops[0][j] = (0, 0, j)

    for i in range(1, rows):
        for j in range(1, cols):
            if reference[i - 1] == hypothesis[j - 1]:
                cost[i][j] = cost[i - 1][j - 1]
                ops[i][j] = ops[i - 1][j - 1]
                continue

            substitute = cost[i - 1][j - 1] + 1
            delete = cost[i - 1][j] + 1
            insert = cost[i][j - 1] + 1
            best = min(substitute, delete, insert)
            cost[i][j] = best

            if best == substitute:
                sub, dele, ins = ops[i - 1][j - 1]
                ops[i][j] = (sub + 1, dele, ins)
            elif best == delete:
                sub, dele, ins = ops[i - 1][j]
                ops[i][j] = (sub, dele + 1, ins)
            else:
                sub, dele, ins = ops[i][j - 1]
                ops[i][j] = (sub, dele, ins + 1)

    return ops[-1][-1]


def word_error_rate(
    reference: str,
    hypothesis: str,
    language: str = "en",
    *,
    normalised: bool = True,
) -> WerResult:
    """Word error rate of one utterance.

    With ``normalised=False`` the text is only casefolded and stripped of
    punctuation, which is the number to quote when someone asks for raw WER.
    """

    ref_tokens = normalise(
        reference, language, drop_fillers=normalised, map_numbers=normalised
    )
    hyp_tokens = normalise(
        hypothesis, language, drop_fillers=normalised, map_numbers=normalised
    )

    substitutions, deletions, insertions = _edit_counts(ref_tokens, hyp_tokens)
    reference_words = len(ref_tokens)

    if reference_words == 0:
        rate = 0.0 if not hyp_tokens else 1.0
    else:
        rate = (substitutions + deletions + insertions) / reference_words

    return WerResult(
        wer=rate,
        substitutions=substitutions,
        deletions=deletions,
        insertions=insertions,
        reference_words=reference_words,
    )


def corpus_wer(pairs: list[tuple[str, str]], language: str = "en") -> WerResult:
    """WER over a whole call: errors and reference words summed, then divided.

    Averaging per utterance WER would let one three word sentence weigh as much
    as a forty word one.
    """

    substitutions = deletions = insertions = reference_words = 0
    for reference, hypothesis in pairs:
        result = word_error_rate(reference, hypothesis, language)
        substitutions += result.substitutions
        deletions += result.deletions
        insertions += result.insertions
        reference_words += result.reference_words

    errors = substitutions + deletions + insertions
    rate = errors / reference_words if reference_words else 0.0
    return WerResult(rate, substitutions, deletions, insertions, reference_words)
