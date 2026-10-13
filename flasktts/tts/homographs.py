import json
import re
from pathlib import Path
from typing import Callable, Iterable, List, NamedTuple, Optional

import torch
from safetensors.torch import load_file
from torch import nn
from transformers import AutoConfig, AutoModel, AutoTokenizer

SENTENCE_END = re.compile(r"(?<=[.!?…])[\"'”’)\]]*\s+")
MISAKI_MARKUP = re.compile(r"\[[^\]]*\]\([^)]*\)")
BATCH_SIZE = 64


class Target(NamedTuple):
    """A homograph occurrence: `start`/`end` index `sentence`, `offset` is the sentence's start in the full text."""

    homograph: str
    sentence: str
    start: int
    end: int
    offset: int


class HomographMarkup:
    """Finds homographs in text and wraps them in misaki inline phonemes, e.g. `we [read](/ɹˈid/) it`."""

    def __init__(self, words: Iterable[str]):
        alternatives = "|".join(map(re.escape, sorted(words, key=len, reverse=True)))
        self.pattern = re.compile(rf"(?<![\w'’-])(?:{alternatives})(?![\w'’-])", re.IGNORECASE)

    def targets(self, text: str) -> List[Target]:
        """Homograph occurrences outside existing markup, with the sentence each sits in."""
        markup = [m.span() for m in MISAKI_MARKUP.finditer(text)]
        boundaries = [0] + [m.end() for m in SENTENCE_END.finditer(text)] + [len(text)]
        found = []
        for match in self.pattern.finditer(text):
            if any(start <= match.start() < end for start, end in markup):
                continue
            sentence_start = max(b for b in boundaries if b <= match.start())
            sentence_end = min(b for b in boundaries if b > match.start())
            found.append(
                Target(
                    match.group().lower(),
                    text[sentence_start:sentence_end],
                    match.start() - sentence_start,
                    match.end() - sentence_start,
                    sentence_start,
                )
            )
        return found

    def annotate(self, text: str, predict: Callable[[List[Target]], List[Optional[str]]]) -> str:
        """`predict` returns phonemes per target, or None to leave that occurrence to misaki."""
        found = self.targets(text)
        if not found:
            return text
        pieces, cursor = [], 0
        for target, phonemes in zip(found, predict(found)):
            if phonemes is None:
                continue
            start, end = target.offset + target.start, target.offset + target.end
            pieces.append(f"{text[cursor:start]}[{text[start:end]}](/{phonemes}/)")
            cursor = end
        pieces.append(text[cursor:])
        return "".join(pieces)


class HomographModel(nn.Module):
    def __init__(self, config, num_classes: int):
        super().__init__()
        self.encoder = AutoModel.from_config(config)
        self.head = nn.Sequential(nn.Dropout(0.1), nn.Linear(config.hidden_size, num_classes))

    def forward(self, input_ids, attention_mask, span_mask, candidate_mask):
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        weights = span_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * weights).sum(1) / weights.sum(1).clamp(min=1)
        logits = self.head(pooled).float()
        return logits.masked_fill(~candidate_mask, float("-inf"))


class HomographDisambiguator:
    """Picks each homograph's pronunciation from its sentence with a model exported by the homographs repo."""

    def __init__(self, model_dir: str, device: Optional[str] = None):
        model_dir = Path(model_dir)
        meta = json.loads((model_dir / "homographs.json").read_text())
        self.max_length = meta["max_length"]
        self.classes = [tuple(c) for c in meta["classes"]]
        self.candidates = {}
        for i, (word, _) in enumerate(self.classes):
            self.candidates.setdefault(word, []).append(i)
        self.markup = HomographMarkup(self.candidates)

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = HomographModel(AutoConfig.from_pretrained(model_dir), len(self.classes))
        self.model.load_state_dict(load_file(model_dir / "model.safetensors"))
        self.model.to(self.device, self.dtype).eval()

    @torch.no_grad()
    def predict(self, targets: List[Target]) -> List[Optional[str]]:
        """Phonemes per target; None where the target falls past the tokenizer's max length."""
        phonemes = []
        for i in range(0, len(targets), BATCH_SIZE):
            batch = targets[i : i + BATCH_SIZE]
            enc = self.tokenizer(
                [t.sentence for t in batch],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_offsets_mapping=True,
                return_tensors="pt",
            )
            offsets = enc.pop("offset_mapping")
            starts = torch.tensor([t.start for t in batch]).unsqueeze(1)
            ends = torch.tensor([t.end for t in batch]).unsqueeze(1)
            span_mask = (offsets[..., 1] > offsets[..., 0]) & (offsets[..., 0] < ends) & (offsets[..., 1] > starts)
            candidate_mask = torch.zeros(len(batch), len(self.classes), dtype=torch.bool)
            for row, t in enumerate(batch):
                candidate_mask[row, self.candidates[t.homograph]] = True
            logits = self.model(
                enc["input_ids"].to(self.device),
                enc["attention_mask"].to(self.device),
                span_mask.to(self.device),
                candidate_mask.to(self.device),
            )
            visible = span_mask.any(1).tolist()
            phonemes.extend(self.classes[c][1] if v else None for c, v in zip(logits.argmax(-1).tolist(), visible))
        return phonemes

    def annotate(self, text: str) -> str:
        return self.markup.annotate(text, self.predict)
