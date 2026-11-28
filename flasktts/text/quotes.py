"""Find quoted speech in newsletter text, so it can later be read in a second voice.

`split_quotes(text)` cuts text into narration and quote segments that concatenate back
to the input exactly. Quotation marks belong to the narration: a quote segment holds
only the quoted content, without the marks or the whitespace just inside them. For
block quotes the marker (`>` or the indentation) is narration and the rest of the line
is the quote.

Standard library only, so it runs (and is tested) without the TTS models.

Rules, in the order they are applied:

1. Paragraphs. Text is cut into lines. A run of non-blank lines is one paragraph only
   where a line break looks like a hard wrap: the line does not end in sentence
   punctuation and the next starts with a lowercase letter, or the line leaves a quote
   open and the next line's first mark closes it. Otherwise every line is its own
   paragraph, which matches inbox2pod's one-line-per-paragraph output. Inline quote
   marks are only paired inside one paragraph (rule 6 is the exception).
2. Block quotes. A line starting with `>` (email quoting, nested `>>` included) or
   indented at least BLOCK_INDENT columns deeper than the document's shallowest line is
   a quote, one segment per line. Indented list items are not.
3. Marks. Straight `"` and curly `“ ”` are double quotes, `« »` guillemets, and `'`,
   `‘ ’` single quotes. A straight mark opens when it follows whitespace, an opening
   bracket or a dash and precedes a non-space; it closes when it follows a non-space
   and precedes a non-alphanumeric. Curly marks prefer their own direction.
4. Apostrophes and primes are not quote marks: a mark between two letters (don't,
   Derek's), after a digit (6'2", 5' 10"), the elisions 'n', 'em, 'tis, 'twas, 'cause,
   'til, 'bout and '90s, and a curly `’` before a word (’tis). A straight `"` after a
   digit (12" pizza) is weak: it only closes a quote that no ordinary closer can, and a
   single `'`/`’` after an s (investors' money) is weak in the same way.
5. Pairing. Marks of one family pair left to right; a closer with nothing open is
   dropped, and an opener followed by another opener is abandoned. Unpaired marks are
   narration, so one stray mark never reaches past its paragraph.
6. Multi-paragraph quotes. A double quote left open at the end of a paragraph continues
   only if the next paragraph starts with a reopening mark, the English convention;
   the chain must close eventually or it all falls back to narration. Each paragraph's
   share is its own quote segment.
7. Nesting. A pair inside another pair stays inside the outer quote segment (single in
   double, or double in single for British style). Crossing pairs keep the double one.
8. Speech or not. A paired span becomes a quote segment when it reads like speech or
   quoted prose. Only CONTEXT_CHARS either side of the marks are looked at. In order:
   - an attribution next to it is speech: a speech verb and up to
     ATTRIBUTION_SUBJECT_WORDS words, then `,` or `:`, before it (`she told reporters,
     "...`, `according to the filing, "...`); or a subject and verb after a quote that
     ends in punctuation (`"...," he wrote`), unless the subject reads as a clause
     (`which is what he said`), the verb is followed by `that` or `:` (it introduces
     something else), or the verb has no comma and the quote is one word (`said "buy"`);
   - a title cue just before it (titled, dubbed, so-called, the phrase) makes it a name;
   - code (flags, braces, `=`, `_`, backticks) is not speech;
   - a Title Case span of 2 to MAX_TITLE_WORDS words is a title, headline or name (all
     caps does not count as Title Case);
   - a span used as a modifier after a determiner or possessive (`a "buy now, pay
     later" product`) is a term;
   - MIN_SPEECH_WORDS words or more is speech;
   - a whole sentence on its own, starting with a capital and ending in sentence
     punctuation after a paragraph start, a sentence end or a colon, is speech at any
     length (`immediate. "Absolutely."`);
   - MIN_PUNCTUATED_WORDS words or more is speech after a colon (`Her reply: "lol no"`),
     or when it starts with a capital and ends in sentence punctuation. A lowercase
     `"risk-free yield."` is a scare quote with American punctuation, not speech.
   Everything else (scare quotes, short terms) stays narration.
9. Dash dialogue. A paragraph opening with an em dash or horizontal bar is a quote up
   to the next dash, if that part has MIN_SPEECH_WORDS words and ends in punctuation.
"""

import re
from bisect import bisect_right
from dataclasses import dataclass
from typing import List, Literal, NamedTuple, Optional, Tuple

MIN_SPEECH_WORDS = 4
MIN_PUNCTUATED_WORDS = 2
MAX_TITLE_WORDS = 8
ATTRIBUTION_SUBJECT_WORDS = 4
# Characters of context either side of a quote that the speech heuristics look at
CONTEXT_CHARS = 120
BLOCK_INDENT = 4

FAMILIES = {
    '"': "double",
    "“": "double",
    "”": "double",
    "«": "guillemet",
    "»": "guillemet",
    "'": "single",
    "‘": "single",
    "’": "single",
}
# Families whose quotes may run across paragraphs (rule 6)
CHAINING_FAMILIES = ("double", "guillemet")
OPENERS = "“«‘"
CLOSERS = "”»’"
OPEN_LEFT = "([{<—–-―/:;…\"'“‘«"
DASHES = "—―"

SPEECH_VERBS = (
    "said", "says", "say", "told", "tells", "wrote", "writes", "asked", "asks", "argued", "argues",
    "added", "adds", "noted", "notes", "explained", "explains", "replied", "replies", "responded",
    "answered", "tweeted", "posted", "quipped", "joked", "jokes", "insisted", "insists", "warned",
    "warns", "claimed", "claims", "conceded", "concedes", "admitted", "admits", "stated", "states",
    "announced", "observed", "remarked", "continued", "recalled", "recalls", "suggested",
    "suggests", "cautioned", "predicted", "declared", "shouted", "yelled", "whispered", "laughed",
    "sighed", "wondered", "wonders", "complained", "complains", "boasted", "acknowledged",
    "emailed", "texted", "began", "begins", "commented", "mused", "concluded", "countered", "counters", "put it",
    "puts it", "according to", "pointed out", "points out", "went on", "goes on",
    "in (?:his|her|their|its|our|my) (?:own )?words", "in the words of", "reads", "read",
)
VERB = r"(?:" + "|".join(sorted(SPEECH_VERBS, key=len, reverse=True)) + r")"
WORD = r"[\w’'.\-]+"
ATTRIBUTION_BEFORE = re.compile(rf"\b{VERB}(?:\s+{WORD}){{0,{ATTRIBUTION_SUBJECT_WORDS}}}\s*[,:]\s*$", re.IGNORECASE)
# `she said "sell everything"`: a bare verb right before the mark, trusted only for MIN_PUNCTUATED_WORDS+
ATTRIBUTION_BARE = re.compile(rf"\b{VERB}\s*$", re.IGNORECASE)
ATTRIBUTION_AFTER = re.compile(
    rf"^\s*[,—–-]?\s*((?:{WORD}\s+){{0,{ATTRIBUTION_SUBJECT_WORDS}}}?){VERB}\b(?!\s*:)(?!\s+that\b)", re.IGNORECASE
)
# Words that turn "<quote> which is what he said" into a clause rather than an attribution
NOT_SUBJECT = {"which", "what", "who", "that", "is", "was", "are", "were", "when", "where", "because",
               "and", "but", "if", "so", "than", "as", "to", "not", "never"}
TITLE_CUE = re.compile(
    r"\b(?:titled|entitled|headlined|headline|named|dubbed|nicknamed|subtitled|tagline|slogan|motto|"
    r"hashtag|label(?:ed|led)?|so-called|the phrase|the term|the word|subject line|captioned)"
    r"(?:\s+(?:is|was))?\s*[:,]?\s*$",
    re.IGNORECASE,
)
DETERMINER = re.compile(r"(?:\b(?:the|a|an|this|that|these|those|its|their|our|his|her|your|my)|\w['’]s)\s*$",
                        re.IGNORECASE)
SMALL_WORDS = {"a", "an", "the", "and", "or", "but", "nor", "of", "in", "on", "at", "to", "for", "by",
               "with", "from", "as", "vs", "vs.", "via", "per", "into", "onto", "up", "&"}
ABBREVIATIONS = {"inc", "corp", "co", "ltd", "jr", "sr", "vs", "mr", "mrs", "ms", "dr", "st",
                 "bros", "llc", "plc", "u.s", "u.k", "e.g", "i.e"}
ELISION = re.compile(r"(?:n['’‘](?!\w)|em\b|tis\b|twas\b|cause\b|cuz\b|til\b|bout\b|nuff\b|\d\d?s?\b)",
                     re.IGNORECASE)
BLOCK_MARKER = re.compile(r"[ \t]*(?:>[ \t]?)+")
LIST_ITEM = re.compile(r"(?:[-*+•▪◦]|\d+[.)]|[a-z][.)])\s")
SENTENCE_BREAK = re.compile(r"[.!?…]\s+[\"'“‘(]?[A-Z]")
# Commands, flags, markup and identifiers: `"pip install --upgrade x"` is not speech
CODE_LIKE = re.compile(r"(?:^|\s)--?[a-z]|[{}<>=_|\\`]|\(\)")
TERMINAL_PUNCT = re.compile(r"(?:[!?…—–]|\.\.\.)[\"'”’)\]]*$")


@dataclass
class Segment:
    kind: Literal["narration", "quote"]
    text: str
    start: int
    end: int


class Mark(NamedTuple):
    pos: int
    family: str
    can_open: bool
    can_close: bool
    weak: bool


class Span(NamedTuple):
    """An opener at `open` paired with a closer at `close`; `close` is the paragraph end for a chained piece."""

    open: int
    close: int
    family: str
    forced: bool


def split_quotes(text: str) -> List[Segment]:
    """Narration and quote segments, in order, covering `text` exactly."""
    segments, cursor = [], 0
    for start, end in quote_spans(text):
        if start > cursor:
            segments.append(Segment("narration", text[cursor:start], cursor, start))
        segments.append(Segment("quote", text[start:end], start, end))
        cursor = end
    if cursor < len(text):
        segments.append(Segment("narration", text[cursor:], cursor, len(text)))
    return segments


def quote_spans(text: str) -> List[Tuple[int, int]]:
    """(start, end) of each quote's content, sorted and non-overlapping."""
    lines = _lines(text)
    spans = _block_quote_spans(text, lines)
    blocked = {i for i, _, _ in spans}
    paragraphs = _paragraphs(text, lines, blocked)
    found = [(s, e) for _, s, e in spans]
    found += _inline_spans(text, paragraphs)
    return _clean(found, len(text))


# ---------------------------------------------------------------------------
# Lines, block quotes and paragraphs
# ---------------------------------------------------------------------------


def _lines(text: str) -> List[Tuple[int, int]]:
    lines, start = [], 0
    for line in text.split("\n"):
        lines.append((start, start + len(line)))
        start += len(line) + 1
    return lines


def _indent(line: str) -> int:
    return len(line.expandtabs(4)) - len(line.expandtabs(4).lstrip(" "))


def _block_quote_spans(text: str, lines: List[Tuple[int, int]]) -> List[Tuple[int, int, int]]:
    """(line index, start, end) of each `>` or indented block-quote line's content."""
    found = []
    indents = [_indent(text[s:e]) for s, e in lines if text[s:e].strip()]
    base = min(indents, default=0)
    for i, (s, e) in enumerate(lines):
        line = text[s:e]
        if not line.strip():
            continue
        marker = BLOCK_MARKER.match(line)
        if marker and ">" in marker.group():
            content_start = s + marker.end()
        elif _indent(line) >= base + BLOCK_INDENT and not LIST_ITEM.match(line.lstrip()):
            content_start = s + len(line) - len(line.lstrip())
        else:
            continue
        content = text[content_start:e]
        stripped = content.strip()
        if any(ch.isalnum() for ch in stripped):
            lead = len(content) - len(content.lstrip())
            found.append((i, content_start + lead, content_start + lead + len(stripped)))
        elif marker and ">" in marker.group():
            found.append((i, e, e))  # an empty `>` line is still part of the block, not prose
    return found


def _paragraphs(text: str, lines: List[Tuple[int, int]], blocked: set) -> List[Tuple[int, int]]:
    """Paragraph (start, end) ranges over the lines that are not block quotes."""
    paragraphs = []
    current = None
    prev_index = None
    for i, (s, e) in enumerate(lines):
        line = text[s:e]
        if i in blocked or not line.strip():
            current = None
            continue
        stripped = line.strip()
        if current is not None and prev_index == i - 1 and _soft_wrap(text[current[0] : current[1]], stripped):
            current[1] = e
        else:
            current = [s, e]
            paragraphs.append(current)
        prev_index = i
    return [(s, e) for s, e in paragraphs]


def _soft_wrap(previous: str, line: str) -> bool:
    previous = previous.rstrip()
    return bool(previous) and previous[-1] not in ".!?:…\"”’'" and line[:1].islower()


# ---------------------------------------------------------------------------
# Inline marks
# ---------------------------------------------------------------------------


def _marks(text: str, start: int, end: int) -> List[Mark]:
    marks = []
    for i in range(start, end):
        ch = text[i]
        family = FAMILIES.get(ch)
        if family is None:
            continue
        prev = text[i - 1] if i > start else ""
        nxt = text[i + 1] if i + 1 < end else ""
        left_boundary = prev == "" or prev.isspace() or prev in OPEN_LEFT
        right_boundary = nxt == "" or not nxt.isalnum()
        opens = left_boundary and nxt != "" and not nxt.isspace()
        closes = prev != "" and not prev.isspace() and right_boundary
        weak = False
        if ch in OPENERS:
            opens = nxt != "" and (left_boundary or not prev.isalnum())
            closes = closes and not opens
        elif ch in CLOSERS:
            closes = prev != "" and (not prev.isspace() or ch == "»")  # « spaced guillemets »
            opens = opens and not closes and ch != "’"
        if family == "single":
            if prev.isalnum() and nxt.isalnum():
                continue  # don't, Derek's
            if prev.isdigit():
                continue  # 6'2", 5' 10"
            if opens and ELISION.match(text, i + 1, end):
                continue  # 'n', 'em, 'tis, '90s
            if ch == "’" and left_boundary and nxt.isalnum():
                continue  # ’tis
            if prev == "n" and i >= 2 and text[i - 2] in "'‘’":
                continue  # the second mark of 'n'
            weak = prev in "sS" and closes
        elif family == "double" and ch == '"' and prev.isdigit():
            weak = True  # 12" pizza
        if opens or closes:
            marks.append(Mark(i, family, opens, closes, weak))
    return marks


def _pair(marks: List[Mark]) -> Tuple[List[Tuple[int, int]], Optional[int]]:
    """Pairs of (opener, closer) positions, and an opener left open at the paragraph's end."""
    strong = [m for m in marks if not m.weak]
    pairs, dangling, open_at = [], [], None
    for mark in strong:
        if open_at is not None and mark.can_close:
            pairs.append((open_at, mark.pos))
            open_at = None
        elif mark.can_open:
            if open_at is not None:
                dangling.append(open_at)
            open_at = mark.pos
    if open_at is not None:
        dangling.append(open_at)

    weak = [m.pos for m in marks if m.weak and m.can_close]
    if weak and dangling:
        # Each abandoned opener may take the first weak closer before the next opener; the windows are disjoint
        openers = [m.pos for m in strong if m.can_open]
        still_open = []
        for d in dangling:
            k = bisect_right(openers, d)
            limit = openers[k] if k < len(openers) else float("inf")
            w = bisect_right(weak, d)
            if w < len(weak) and weak[w] < limit:
                pairs.append((d, weak[w]))
            else:
                still_open.append(d)
        dangling = still_open
    pairs.sort()
    last = max(dangling, default=None)
    if last is not None and any(o > last for o, _ in pairs):
        last = None
    return pairs, last


def _inline_spans(text: str, paragraphs: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    paragraphs = _join_wrapped_quotes(text, paragraphs)
    by_paragraph = []
    for ps, pe in paragraphs:
        marks = _marks(text, ps, pe)
        result = {}
        for family in ("double", "guillemet", "single"):
            result[family] = _pair([m for m in marks if m.family == family])
        by_paragraph.append(result)

    chained = _chains(text, paragraphs, by_paragraph)

    found = []
    for (ps, pe), result in zip(paragraphs, by_paragraph):
        candidates = [Span(o, c, f, (o, c) in chained) for f, (pairs, _) in result.items() for o, c in pairs]
        candidates += [Span(o, pe, "double", True) for o, c in chained if ps <= o < pe and c == pe]
        top = _top_level(candidates)
        for span in top:
            cs, ce = _content(text, span.open + 1, span.close)
            if ce <= cs:
                continue
            if span.forced or _is_speech(text, span, cs, ce, ps, pe):
                found.append((cs, ce))
        if not top:
            dash = _dash_dialogue(text, ps, pe)
            if dash:
                found.append(dash)
    return found


def _join_wrapped_quotes(text: str, paragraphs: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Rejoin a quote hard-wrapped before a capital: a line left open, the next line opening with its closer."""
    joined = []
    for ps, pe in paragraphs:
        if joined:
            qs, qe = joined[-1]
            if text[qe:ps] == "\n" and text[qs:qe].rstrip()[-1:] not in ".!?:…" and _closes_wrap(text, qs, qe, ps, pe):
                joined[-1] = (qs, pe)
                continue
        joined.append((ps, pe))
    return joined


def _closes_wrap(text: str, qs: int, qe: int, ps: int, pe: int) -> bool:
    for family in CHAINING_FAMILIES:
        _, dangling = _pair([m for m in _marks(text, qs, qe) if m.family == family])
        first = next((m for m in _marks(text, ps, pe) if m.family == family), None)
        if dangling is not None and first is not None and first.can_close and not first.can_open:
            return True
    return False


def _chains(text, paragraphs, by_paragraph) -> set:
    """(open, close) pairs and (open, paragraph end) pieces belonging to multi-paragraph quotes."""
    chained = set()
    failed_until = {family: -1 for family in CHAINING_FAMILIES}
    for i, result in enumerate(by_paragraph):
        for family in CHAINING_FAMILIES:
            _, dangling = result[family]
            if dangling is None or (dangling, paragraphs[i][1]) in chained or i < failed_until[family]:
                continue  # a chain starting inside a failed one would fail at the same place
            pieces, closed, j = [(dangling, paragraphs[i][1])], False, i + 1
            while j < len(paragraphs) and not text[paragraphs[j - 1][1] : paragraphs[j][0]].strip():
                ps, pe = paragraphs[j]
                lead = ps + len(text[ps:pe]) - len(text[ps:pe].lstrip())
                if FAMILIES.get(text[lead]) != family:
                    break
                next_pairs, next_dangling = by_paragraph[j][family]
                if next_pairs and next_pairs[0][0] == lead:
                    chained.update(pieces)
                    chained.add(next_pairs[0])
                    closed = True
                    break
                if next_dangling != lead:
                    break
                pieces.append((lead, pe))
                j += 1
            if not closed:
                failed_until[family] = j
    return chained


def _top_level(candidates: List[Span]) -> List[Span]:
    """Outermost spans: doubles win crossings, singles only count when they wrap or avoid doubles."""
    kept: List[Span] = []
    for span in sorted((c for c in candidates if c.family != "single"), key=lambda c: (c.open, -c.close)):
        if not kept or span.open > kept[-1].close:
            kept.append(span)
    opens = [k.open for k in kept]
    for span in sorted((c for c in candidates if c.family == "single"), key=lambda c: (c.open, -c.close)):
        first = bisect_right(opens, span.open)
        last = bisect_right(opens, span.close)
        before_clear = first == 0 or kept[first - 1].close < span.open
        inside_clear = all(k.close < span.close for k in kept[first:last])
        if before_clear and inside_clear:
            kept[first:last] = [span]
            opens[first:last] = [span.open]
    return kept


def _content(text: str, start: int, end: int) -> Tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


# ---------------------------------------------------------------------------
# Speech heuristics
# ---------------------------------------------------------------------------


def _words(content: str) -> List[str]:
    return [t for t in content.split() if any(ch.isalnum() for ch in t)]


def _is_speech(text: str, span: Span, cs: int, ce: int, ps: int, pe: int) -> bool:
    content = text[cs:ce]
    words = _words(content)
    if not words:
        return False
    before = text[max(ps, span.open - CONTEXT_CHARS) : span.open]
    after = text[span.close + 1 : min(pe, span.close + 1 + CONTEXT_CHARS)] if span.close < pe else ""
    if _attributed(content, before, after) or (len(words) >= MIN_PUNCTUATED_WORDS and ATTRIBUTION_BARE.search(before)):
        return True
    if TITLE_CUE.search(before) or CODE_LIKE.search(content):
        return False
    if _is_title(words) or _is_modifier(content, before, after):
        return False
    if len(words) >= MIN_SPEECH_WORDS:
        return True
    # A whole sentence of its own (`immediate. "Absolutely."`, `said "NO."`) is speech at any length
    last = before.rstrip()[-1:]
    standalone = not text[ps : span.open].strip() if not last else last in ".!?:" or ATTRIBUTION_BARE.search(before)
    if standalone and _starts_sentence(content) and _sentence_punctuation(content):
        return True
    if len(words) < MIN_PUNCTUATED_WORDS:
        return False
    if last == ":":
        return True  # `Her reply: "lol no"`
    return _starts_sentence(content) and _sentence_punctuation(content)


def _starts_sentence(content: str) -> bool:
    first = next((ch for ch in content if ch.isalnum()), "")
    return first.isupper() or first.isdigit()


def _attributed(content: str, before: str, after: str) -> bool:
    if ATTRIBUTION_BEFORE.search(before):
        return True
    match = ATTRIBUTION_AFTER.match(after)
    if not match:
        return False
    subject = match.group(1).split()
    if any(w.lower() in NOT_SUBJECT for w in subject):
        return False
    return bool(re.search(r"[,.!?…—–-]$", content)) or after.lstrip()[:1] in ",—–"


def _is_title(words: List[str]) -> bool:
    if len(words) < 2 or len(words) > MAX_TITLE_WORDS:
        return False
    letters = "".join(ch for w in words for ch in w if ch.isalpha())
    if not letters or letters.isupper():
        return False  # all caps is shouting, not a title
    for k, word in enumerate(words):
        bare = word.strip("\"'“”‘’()[],.:;!?")
        if k and bare.lower() in SMALL_WORDS:
            continue
        first = next((ch for ch in bare if ch.isalnum()), "")
        if first and not (first.isupper() or first.isdigit()):
            return False
    return True


def _is_modifier(content: str, before: str, after: str) -> bool:
    """`a "buy now, pay later" product`: a quoted term used as an adjective."""
    if TERMINAL_PUNCT.search(content) or content.endswith("."):
        return False
    return bool(DETERMINER.search(before)) and bool(re.match(r"\s+[a-z]", after))


def _sentence_punctuation(content: str) -> bool:
    if TERMINAL_PUNCT.search(content) or SENTENCE_BREAK.search(content):
        return True
    trimmed = content.rstrip("\"'”’)]")
    if trimmed.endswith("."):
        last = trimmed.split()[-1].rstrip(".").lower()
        return not (last in ABBREVIATIONS or re.fullmatch(r"(?:[a-z]\.)*[a-z]", last) and len(last) <= 3)
    return False


def _dash_dialogue(text: str, ps: int, pe: int) -> Optional[Tuple[int, int]]:
    para = text[ps:pe]
    lead = len(para) - len(para.lstrip())
    if lead >= len(para) or para[lead] not in DASHES:
        return None
    start = ps + lead + 1
    stop = pe
    for i in range(start, pe):
        if text[i] in DASHES + "–":
            stop = i
            break
    cs, ce = _content(text, start, stop)
    content = text[cs:ce]
    if len(_words(content)) >= MIN_SPEECH_WORDS and re.search(r"[.!?…,]$", content):
        return cs, ce
    return None


def _clean(spans: List[Tuple[int, int]], length: int) -> List[Tuple[int, int]]:
    """Sorted, in-bounds, non-empty, non-overlapping spans."""
    result = []
    for start, end in sorted(set(spans)):
        if 0 <= start < end <= length and (not result or start >= result[-1][1]):
            result.append((start, end))
    return result
