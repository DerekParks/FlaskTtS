"""Score the quote detector against a gold set, and spot-check it on real newsletters.

    python -m flasktts.text.quote_eval [cases.jsonl]
    python -m flasktts.text.quote_eval --real ~/projects/inbox2pod/tests/fixtures

Each gold case is a JSON line `{"id", "text", "quotes", "tags"}` where `quotes` lists the
exact content of every quote segment, in order, without the quotation marks (one entry
per paragraph of a multi-paragraph quote and per line of a block quote).

Metrics: span-level precision/recall/F1 on exact content, a lenient character-overlap F1,
the same per tag, and the scare-quote false-positive rate: the share of cases tagged
`scare` where anything that is not gold speech came out as a quote.
"""

import argparse
import email
import json
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from email import policy
from html.parser import HTMLParser
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from flasktts.text.quotes import quote_spans

DEFAULT_CASES = Path(__file__).resolve().parents[2] / "tests" / "text" / "quote_cases.jsonl"
INBOX2POD_VENV_PYTHON = Path.home() / "projects" / "inbox2pod" / ".venv" / "bin" / "python"
QUOTE_CHARS = "\"“”«»‘’'"
SCARE_TAG = "scare"


@dataclass
class Tally:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    chars_tp: int = 0
    chars_pred: int = 0
    chars_gold: int = 0
    cases: int = 0
    passed: int = 0

    def add(self, other: "Tally") -> None:
        for name in vars(self):
            setattr(self, name, getattr(self, name) + getattr(other, name))

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 1.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 1.0

    @property
    def f1(self) -> float:
        return _f1(self.precision, self.recall)

    @property
    def lenient_f1(self) -> float:
        p = self.chars_tp / self.chars_pred if self.chars_pred else 1.0
        r = self.chars_tp / self.chars_gold if self.chars_gold else 1.0
        return _f1(p, r)


@dataclass
class Report:
    total: Tally = field(default_factory=Tally)
    by_tag: Dict[str, Tally] = field(default_factory=lambda: defaultdict(Tally))
    scare_cases: int = 0
    scare_false_positives: int = 0
    failures: List[Tuple[dict, List[str]]] = field(default_factory=list)

    @property
    def scare_fp_rate(self) -> float:
        return self.scare_false_positives / self.scare_cases if self.scare_cases else 0.0


def _f1(p: float, r: float) -> float:
    return 2 * p * r / (p + r) if p + r else 0.0


def load_cases(path: Path) -> List[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def locate(text: str, quotes: List[str]) -> List[Tuple[int, int]]:
    """Gold quote strings as (start, end), preferring occurrences right after a mark or line start."""
    spans, cursor = [], 0
    for quote in quotes:
        at, fallback = text.find(quote, cursor), None
        while at >= 0:
            fallback = at if fallback is None else fallback
            before = text[:at].rstrip(" \t")
            if not before or before[-1] in QUOTE_CHARS + ">\n—―(\t":
                break
            at = text.find(quote, at + 1)
        start = at if at >= 0 else fallback
        if start is None:
            continue
        spans.append((start, start + len(quote)))
        cursor = start + len(quote)
    return spans


def score_case(case: dict) -> Tuple[Tally, List[str]]:
    text, gold = case["text"], case["quotes"]
    predicted_spans = quote_spans(text)
    predicted = [text[s:e] for s, e in predicted_spans]
    overlap = Counter(predicted) & Counter(gold)
    tally = Tally(cases=1)
    tally.tp = sum(overlap.values())
    tally.fp = len(predicted) - tally.tp
    tally.fn = len(gold) - tally.tp
    gold_chars = {i for s, e in locate(text, gold) for i in range(s, e)}
    pred_chars = {i for s, e in predicted_spans for i in range(s, e)}
    tally.chars_tp, tally.chars_pred, tally.chars_gold = len(gold_chars & pred_chars), len(pred_chars), len(gold_chars)
    tally.passed = int(predicted == gold)
    return tally, predicted


def evaluate(cases: List[dict]) -> Report:
    report = Report()
    for case in cases:
        tally, predicted = score_case(case)
        report.total.add(tally)
        for tag in case.get("tags", []):
            report.by_tag[tag].add(tally)
        if SCARE_TAG in case.get("tags", []):
            report.scare_cases += 1
            report.scare_false_positives += tally.fp > 0
        if not tally.passed:
            report.failures.append((case, predicted))
    return report


def print_report(report: Report) -> None:
    t = report.total
    print(f"cases: {t.cases}  passed: {t.passed}  gold spans: {t.tp + t.fn}  predicted spans: {t.tp + t.fp}")
    print(f"span exact   P {t.precision:.3f}  R {t.recall:.3f}  F1 {t.f1:.3f}")
    print(f"char lenient F1 {t.lenient_f1:.3f}")
    print(f"scare-quote false-positive rate: {report.scare_false_positives}/{report.scare_cases} = {report.scare_fp_rate:.3f}")
    print()
    print(f"{'tag':<20} {'cases':>5} {'pass':>5} {'P':>6} {'R':>6} {'F1':>6} {'lenF1':>6}")
    for tag in sorted(report.by_tag):
        g = report.by_tag[tag]
        print(f"{tag:<20} {g.cases:>5} {g.passed:>5} {g.precision:>6.3f} {g.recall:>6.3f} {g.f1:>6.3f} {g.lenient_f1:>6.3f}")
    print()
    print(f"failures: {len(report.failures)}")
    for case, predicted in report.failures:
        print(f"--- {case['id']}  [{', '.join(case.get('tags', []))}]")
        print(f"    text:     {case['text']!r}")
        print(f"    expected: {case['quotes']!r}")
        print(f"    got:      {predicted!r}")


# ---------------------------------------------------------------------------
# --real: spot-check on .eml files
# ---------------------------------------------------------------------------

_INBOX2POD_SCRIPT = """
import json, sys
sys.path.insert(0, sys.argv[1])
from inbox2pod.extract import extract_email
out = {}
for path in sys.argv[2:]:
    with open(path, "rb") as f:
        out[path] = extract_email(f.read()).text
print(json.dumps(out))
"""


class _TextExtractor(HTMLParser):
    BLOCKS = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "blockquote", "table"}
    SKIP = {"script", "style", "head", "title"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts, self.skipping = [], 0

    def handle_starttag(self, tag, attrs):
        self.skipping += tag in self.SKIP
        if tag in self.BLOCKS:
            self.parts.append("\n\n")

    def handle_endtag(self, tag):
        self.skipping -= tag in self.SKIP and self.skipping > 0
        if tag in self.BLOCKS:
            self.parts.append("\n\n")

    def handle_data(self, data):
        if not self.skipping:
            self.parts.append(" ".join(data.split()) if data.strip() else "")

    def text(self) -> str:
        paragraphs = [" ".join(p.split()) for p in "".join(self.parts).split("\n\n")]
        return "\n\n".join(p for p in paragraphs if p)


def stdlib_extract(raw: bytes) -> str:
    message = email.message_from_bytes(raw, policy=policy.default)
    html_part = message.get_body(preferencelist=("html",))
    if html_part is not None:
        parser = _TextExtractor()
        parser.feed(html_part.get_content())
        return parser.text()
    plain = message.get_body(preferencelist=("plain",))
    return plain.get_content() if plain is not None else ""


def extract_texts(paths: List[Path]) -> Tuple[Dict[str, str], str]:
    """Plain text per .eml, via inbox2pod's extractor when its venv imports cleanly."""
    project = paths[0].resolve().parents[2] if paths else None
    python = project / ".venv" / "bin" / "python" if project else None
    if python is None or not python.exists():
        python = INBOX2POD_VENV_PYTHON
    if project is not None and python.exists():
        try:
            result = subprocess.run(
                [str(python), "-I", "-c", _INBOX2POD_SCRIPT, str(project), *map(str, paths)],
                capture_output=True,
                text=True,
                timeout=300,
                check=True,
            )
            return json.loads(result.stdout), "inbox2pod extract_email"
        except (subprocess.SubprocessError, ValueError, OSError) as e:
            print(f"inbox2pod extraction failed ({e}); using the stdlib HTML pass", file=sys.stderr)
    return {str(p): stdlib_extract(p.read_bytes()) for p in paths}, "stdlib HTML-to-text"


def spot_check(directory: Path) -> None:
    paths = sorted(directory.expanduser().glob("*.eml"))
    if not paths:
        print(f"no .eml files in {directory}")
        return
    texts, extractor = extract_texts(paths)
    print(f"extracted {len(texts)} files with {extractor}")
    for path in paths:
        text = texts.get(str(path), "")
        spans = quote_spans(text)
        print(f"\n===== {path.name}: {len(spans)} quotes")
        for s, e in spans:
            print(f"  QUOTE  {text[s:e]!r}")
        voiced = {i for s, e in spans for i in range(s, e)}
        for line in _lines_with_unvoiced_marks(text, voiced):
            print(f"  marks left in narration: {line!r}")


def _lines_with_unvoiced_marks(text: str, voiced: set) -> List[str]:
    found, start = [], 0
    for line in text.split("\n"):
        end = start + len(line)
        marks = [i for i in range(start, end) if text[i] in "\"“”«»‘"]
        if any(i - 1 not in voiced and i + 1 not in voiced for i in marks):
            found.append(line if len(line) < 160 else line[:157] + "...")
        start = end + 1
    return found


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cases", nargs="?", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--real", type=Path, metavar="DIR", help="also print detected quotes for the .eml files in DIR")
    args = parser.parse_args(argv)
    print_report(evaluate(load_cases(args.cases)))
    if args.real:
        print()
        spot_check(args.real)
    return 0


if __name__ == "__main__":
    sys.exit(main())
