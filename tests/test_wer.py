from voice_evals.metrics.wer import corpus_wer, normalise, word_error_rate


def test_perfect_match_is_zero():
    assert word_error_rate("book me a check up", "book me a check up").wer == 0.0


def test_counts_each_operation_separately():
    result = word_error_rate("one two three four", "one three four five")
    assert (result.substitutions, result.deletions, result.insertions) == (0, 1, 1)
    assert result.reference_words == 4
    assert result.wer == 0.5


def test_spoken_numbers_are_not_errors_in_latvian():
    spoken = "pierakstu uz piecpadsmit un trisdesmit"
    digits = "pierakstu uz 15 un 30"
    assert word_error_rate(spoken, digits, "lv").wer == 0.0


def test_latvian_diacritics_are_never_folded_away():
    # gala and gaļa are different words, and a transcriber that loses the
    # diacritic has made a real error.
    assert word_error_rate("gaļa", "gala", "lv").wer == 1.0


def test_raw_mode_counts_the_number_format_as_an_error():
    raw = word_error_rate("piecpadsmit", "15", "lv", normalised=False)
    assert raw.wer == 1.0


def test_fillers_are_dropped_when_normalising():
    assert word_error_rate("nu labi", "labi", "lv").wer == 0.0


def test_corpus_wer_weighs_by_length_not_by_utterance():
    pairs = [
        ("yes", "no"),                                   # 1 word, 1 error
        ("i would like to book a hygiene appointment",
         "i would like to book a hygiene appointment"),  # 7 words, 0 errors
    ]
    result = corpus_wer(pairs)
    assert result.reference_words == 9
    assert round(result.wer, 3) == 0.111


def test_empty_reference_with_output_is_a_full_error():
    assert word_error_rate("", "hello").wer == 1.0


def test_normalise_strips_punctuation_and_case():
    assert normalise("Hello, World!") == ["hello", "world"]
