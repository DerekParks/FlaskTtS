import random
from pathlib import Path

import pytest

from flasktts.text.quote_eval import evaluate, load_cases
from flasktts.text.quotes import Segment, quote_spans, split_quotes

CASES = load_cases(Path(__file__).with_name("quote_cases.jsonl"))

# Just under what the rules achieve on the gold set, so a regression fails
MIN_F1 = 0.98
MIN_LENIENT_F1 = 0.985
MAX_SCARE_FP_RATE = 0.04

MARKS = "\"“”‘’'«»—>"


def quotes(text):
    return [s.text for s in split_quotes(text) if s.kind == "quote"]


def assert_covers(text, segments):
    assert "".join(s.text for s in segments) == text
    cursor = 0
    for segment in segments:
        assert segment.start == cursor
        assert segment.end == segment.start + len(segment.text)
        assert text[segment.start : segment.end] == segment.text
        assert segment.text
        cursor = segment.end
    assert cursor == len(text)


# --- segments ----------------------------------------------------------------------


def test_marks_belong_to_narration():
    text = 'He said, "Rates stay high." Then he left.'

    assert split_quotes(text) == [
        Segment("narration", 'He said, "', 0, 10),
        Segment("quote", "Rates stay high.", 10, 26),
        Segment("narration", '" Then he left.', 26, 41),
    ]


def test_text_without_quotes_is_one_narration_segment():
    assert split_quotes("Yields rose.") == [Segment("narration", "Yields rose.", 0, 12)]
    assert split_quotes("") == []


def test_quote_spans_match_segments():
    text = '"We are buyers here," said one allocator. "We are sellers," said another.'

    assert [text[s:e] for s, e in quote_spans(text)] == quotes(text)


# --- marks ---------------------------------------------------------------------------


def test_straight_and_curly_double_quotes():
    assert quotes('"We will not cut this year," the governor said.') == ["We will not cut this year,"]
    assert quotes("“We will not cut this year,” the governor said.") == ["We will not cut this year,"]


def test_balanced_single_quotes():
    assert quotes("‘We are not a bank,’ the founder said.") == ["We are not a bank,"]
    assert quotes("The note said: 'Sell everything that is not nailed down.'") == [
        "Sell everything that is not nailed down."
    ]


@pytest.mark.parametrize(
    "text",
    [
        "Don't expect the Fed's minutes to move anyone's view.",
        "It was rock 'n' roll economics in the '90s.",
        "’Tis the season for outlooks, and the ’80s are back.",
        "The investors' letter and the founders' reply crossed in the mail.",
        "He's got 'em where he wants 'em, 'cause nobody sells 'til March.",
    ],
)
def test_apostrophes_are_not_quotes(text):
    assert quotes(text) == []


@pytest.mark.parametrize(
    "text",
    [
        'At 6\'2" he was the tallest person on the desk.',
        'She is 5\' 10" and played volleyball.',
        'We ordered a 12" pizza and a 16" one.',
        "The bezel is 2″ wide.",
    ],
)
def test_measurements_are_not_quotes(text):
    assert quotes(text) == []


def test_measurement_inside_a_quote_does_not_end_it():
    assert quotes('The rep insisted, "Nobody buys the 13" anymore."') == ['Nobody buys the 13" anymore.']


def test_guillemets():
    assert quotes("The minister was blunt: « We will not bail out the speculators. »") == [
        "We will not bail out the speculators."
    ]


# --- speech or not -------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        'The so-called "soft landing" looks like a landing.',
        'Management keeps saying "transitory."',
        'Crypto lenders promised "risk-free yield." You know how that went.',
        'The startup says its AI is "agentic," whatever that means.',
        'Who still says "synergy" with a straight face?',
        'The "buy now, pay later" lenders want patience.',
    ],
)
def test_scare_quotes_and_short_terms_stay_narration(text):
    assert quotes(text) == []


@pytest.mark.parametrize(
    "text",
    [
        'This week in "The Bond Market": yields and spreads.',
        'Her new book, "The Long Squeeze," is out next month.',
        'Yesterday we ran "Is the Bond Rally Over?" and got angry email.',
        'A post headlined "the end of cheap money is not the end of the world" went around.',
    ],
)
def test_titles_and_headlines_stay_narration(text):
    assert quotes(text) == []


def test_four_words_is_speech():
    assert quotes('The chair said inflation was "not yet on a sustainable path" toward target.') == [
        "not yet on a sustainable path"
    ]


def test_sentence_punctuation_is_speech():
    assert quotes('Her reply was short: "Not now."') == ["Not now."]
    assert quotes('Her answer was immediate. "Absolutely."') == ["Absolutely."]


@pytest.mark.parametrize(
    "text, expected",
    [
        ('"No," said the chief executive.', ["No,"]),
        ('Asked about layoffs, he replied, "Yes."', ["Yes."]),
        ('"Fine," she texted back.', ["Fine,"]),
        ('"Rates," she said, "are high."', ["Rates,", "are high."]),
        ('According to one banker, "nobody wants this."', ["nobody wants this."]),
    ],
)
def test_attribution_makes_short_quotes_speech(text, expected):
    assert quotes(text) == expected


def test_clause_after_a_quote_is_not_attribution():
    assert quotes('He used the word "transitory," which is what everyone said that year.') == []


def test_all_caps_speech_and_shouted_terms():
    assert quotes('"WE ARE SO BACK," the founder posted.') == ["WE ARE SO BACK,"]
    assert quotes('Crypto Twitter is back to yelling "HODL" at sellers.') == []


def test_quote_followed_by_footnote_marker():
    assert quotes("“This is a liquidity problem, not a solvency problem.”² It did not age well.") == [
        "This is a liquidity problem, not a solvency problem."
    ]


def test_code_is_not_speech():
    assert quotes('Run "pip install --upgrade ledgerkit" to get the fix.') == []


# --- structure -----------------------------------------------------------------------


def test_nested_quote_stays_inside_the_outer_one():
    assert quotes("“The board told me, ‘Fix it or leave,’ so I fixed it,” the CEO said.") == [
        "The board told me, ‘Fix it or leave,’ so I fixed it,"
    ]
    assert quotes("‘The regulator called it “a serious lapse”, and we accept that,’ the bank said.") == [
        "The regulator called it “a serious lapse”, and we accept that,"
    ]


def test_multi_paragraph_quote_reopens_each_paragraph():
    text = "“We assumed money would be free forever.\n\n“That is gone.\n\n“What replaces it is the question.”\n\nEnd."

    assert quotes(text) == [
        "We assumed money would be free forever.",
        "That is gone.",
        "What replaces it is the question.",
    ]


def test_multi_paragraph_chain_that_never_closes_is_narration():
    text = '"This quote never closes\n\n"and neither does this one\n\nThen plain narration.'

    assert quotes(text) == []


def test_stray_mark_does_not_flip_the_rest_of_the_document():
    text = 'Revenue grew "strongly.\n\nCosts rose. "We are investing for growth," the CFO said.\n\nThe stock fell.'

    assert quotes(text) == ["We are investing for growth,"]


def test_unbalanced_marks_fall_back_to_narration():
    assert quotes('The ticker "XYZ jumped twelve percent on the news, the biggest move since the IPO.') == []
    assert quotes('Shares fell 4% after the call." Analysts had expected a beat.') == []


def test_hard_wrapped_quote_is_rejoined():
    assert quotes('The chair: "We are not\nannouncing anything today."') == ["We are not\nannouncing anything today."]
    assert quotes('"We will keep buying\nBitcoin at any price," the company said.') == [
        "We will keep buying\nBitcoin at any price,"
    ]


def test_email_block_quotes_one_segment_per_line():
    text = "Dana wrote:\n> Agreed on the pricing.\n>> Should we hold the launch?\n>\n\nLet us hold it."

    assert quotes(text) == ["Agreed on the pricing.", "Should we hold the launch?"]


def test_indented_block_quote_but_not_indented_list():
    assert quotes("The prospectus:\n\n    The Company may never be profitable.\n\nWow.") == [
        "The Company may never be profitable."
    ]
    assert quotes("Watch:\n    - the jobs report\n    - the auction") == []


def test_em_dash_dialogue_but_not_signatures():
    assert quotes("— We are not selling the division, and we never were.\n\nThat was the CEO.") == [
        "We are not selling the division, and we never were."
    ]
    assert quotes("Thanks for reading.\n\n— Sam") == []


# --- properties and thresholds ----------------------------------------------------------


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_segments_reassemble_gold_cases(case):
    assert_covers(case["text"], split_quotes(case["text"]))


def test_segments_reassemble_random_strings():
    rng = random.Random(1234)
    alphabet = MARKS + "abc XYZ 12.,;:!?…()[]\n\n\t    é🚀"
    words = ["said", "the", "Fed", "6'2\"", "don't", "'90s", "“Rates", "high,”", "\n\n“", "> ", "    "]
    for _ in range(400):
        if rng.random() < 0.5:
            text = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 120)))
        else:
            text = " ".join(rng.choice(words) for _ in range(rng.randint(0, 40)))
        assert_covers(text, split_quotes(text))


def test_gold_set_thresholds():
    report = evaluate(CASES)

    assert len(CASES) >= 150
    assert report.total.f1 >= MIN_F1
    assert report.total.lenient_f1 >= MIN_LENIENT_F1
    assert report.scare_fp_rate <= MAX_SCARE_FP_RATE
