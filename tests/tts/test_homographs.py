import os

import pytest

from flasktts.tts.homographs import HomographDisambiguator, HomographMarkup

MODEL_DIR = os.getenv("HOMOGRAPH_MODEL_DIR")


def fake_predict(targets):
    return [f"{t.homograph.upper()}@{t.sentence[t.start : t.end]}" for t in targets]


@pytest.fixture
def markup():
    return HomographMarkup(["lead", "read", "row", "upset"])


def test_wraps_each_occurrence_keeping_its_spelling(markup):
    text = "Lead pipes. We read it; they read it."

    assert markup.annotate(text, fake_predict) == (
        "[Lead](/LEAD@Lead/) pipes. We [read](/READ@read/) it; they [read](/READ@read/) it."
    )


def test_targets_carry_their_own_sentence(markup):
    text = 'He said "Stop." Then we read the lead story! Done.'

    targets = markup.targets(text)

    assert [(t.homograph, t.sentence) for t in targets] == [
        ("read", "Then we read the lead story! "),
        ("lead", "Then we read the lead story! "),
    ]
    for t in targets:
        assert text[t.offset + t.start : t.offset + t.end] == t.sentence[t.start : t.end]


def test_skips_compounds_contractions_and_longer_words(markup):
    text = "Lead-based paint, re-read notes, rowboats, the row's end, and reader."

    assert markup.targets(text) == []


def test_skips_existing_markup(markup):
    text = "Already [read](/ɹˈɛd/) and now read."

    assert markup.annotate(text, fake_predict) == "Already [read](/ɹˈɛd/) and now [read](/READ@read/)."


def test_none_prediction_leaves_word_to_misaki(markup):
    text = "We read the lead."

    assert markup.annotate(text, lambda targets: [None, "lˈɛd"]) == "We read the [lead](/lˈɛd/)."


def test_text_without_homographs_is_unchanged(markup):
    assert markup.annotate("Nothing to see here.", fake_predict) == "Nothing to see here."


@pytest.mark.skipif(not MODEL_DIR, reason="HOMOGRAPH_MODEL_DIR not set")
@pytest.mark.parametrize(
    "text, expected",
    [
        ("Every Sunday we read the long reads that piled up during the week and argue about them.", "[read](/ɹˈid/)"),
        ("Yesterday we read the paper and wrote about it.", "[read](/ɹˈɛd/)"),
        ("Engineers found traces of lead in the solder joints.", "[lead](/lˈɛd/)"),
        ("She took the lead on the migration.", "[lead](/lˈid/)"),
        ("Users were upset that the update removed offline mode.", "[upset](/ˌʌpsˈɛt/)"),
        ("The ruling was a major upset for the incumbent.", "[upset](/ˈʌpsˌɛt/)"),
        ("They had a blazing row over the budget.", "[row](/ɹˈW/)"),
    ],
)
def test_model_disambiguates(text, expected):
    assert expected in disambiguator().annotate(text)


@pytest.mark.skipif(not MODEL_DIR, reason="HOMOGRAPH_MODEL_DIR not set")
def test_target_past_max_length_is_left_alone():
    text = "word " * 400 + "and then we read it."

    assert disambiguator().annotate(text) == text


_disambiguator = None


def disambiguator():
    global _disambiguator
    if _disambiguator is None:
        _disambiguator = HomographDisambiguator(MODEL_DIR)
    return _disambiguator
