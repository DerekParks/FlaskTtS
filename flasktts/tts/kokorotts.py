import os
import re
import threading
from contextlib import contextmanager
from numbers import Number
from typing import Dict, Iterator, Optional

import numpy as np
import soundfile as sf
import torch
from kokoro import KModel, KPipeline
from misaki import en

from flasktts.config import Config
from flasktts.tts.homographs import HomographDisambiguator

REPO_ID = "hexgrad/Kokoro-82M"
SAMPLE_RATE = 24000
DEFAULT_VOICE = "af_heart"
PARAGRAPH_PAUSE_SEC = 0.3
WARM_UP_TEXT = "Ready to read."
ENGLISH_LANG_CODES = ("a", "b")
AMERICAN_LANG_CODE = "a"

# misaki's lexicon has the biblical name Dives (DYE-veez) for every case of "dives"
LEXICON_OVERRIDES = {
    "dives": "dˈIvz",
}


def apply_lexicon_overrides(g2p: en.G2P) -> None:
    g2p.lexicon.golds.update(en.Lexicon.grow_dictionary(LEXICON_OVERRIDES))


class ModelLock:
    """Mutex where interactive holders go ahead of waiting batch holders."""

    def __init__(self):
        self._condition = threading.Condition()
        self._held = False
        self._interactive_waiting = 0

    @property
    def held(self) -> bool:
        return self._held

    @contextmanager
    def __call__(self, interactive: bool = False):
        with self._condition:
            self._interactive_waiting += interactive
            try:
                while self._held or (not interactive and self._interactive_waiting):
                    self._condition.wait()
            finally:
                self._interactive_waiting -= interactive
            self._held = True
        try:
            yield
        finally:
            with self._condition:
                self._held = False
                self._condition.notify_all()


class KokoroTTSHighlander:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = KokoroTTS(Config.TTS_WORKDIR)
        return cls._instance


class KokoroTTS:
    def __init__(
        self,
        output_dir: str,
        device: Optional[str] = None,
        fp16: Optional[bool] = None,
        homograph_model_dir: Optional[str] = None,
    ):
        """
        Args:
            output_dir (str): Output directory to save the generated audio files
            device (str, optional): Device to use for inference. Defaults to None for auto-detect.
            fp16 (bool, optional): Run inference under fp16 autocast. Only applies on CUDA.
                Defaults to Config.KOKORO_FP16.
            homograph_model_dir (str, optional): Exported homograph model, applied to American English voices.
                Defaults to Config.HOMOGRAPH_MODEL_DIR; disabled when neither is set.
        """

        if device is None:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif (
                False and torch.backends.mps.is_available()
            ):  # MPS is not supported check https://github.com/pytorch/pytorch/issues/77764
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        self.output_dir = output_dir
        self.fp16 = (Config.KOKORO_FP16 if fp16 is None else fp16) and (
            torch.device(device).type == "cuda"
        )
        self.model = KModel(repo_id=REPO_ID).to(device).eval()
        self.pipelines: Dict[str, KPipeline] = {}
        self.lock = ModelLock()
        homograph_model_dir = homograph_model_dir or Config.HOMOGRAPH_MODEL_DIR
        self.homographs = HomographDisambiguator(homograph_model_dir, device) if homograph_model_dir else None

    def _pipeline_for(self, voice: str) -> KPipeline:
        """Pipeline whose G2P matches the voice's language prefix, e.g. 'b' for bf_emma."""
        lang_code = voice[0]
        if lang_code not in self.pipelines:
            pipeline = KPipeline(lang_code=lang_code, repo_id=REPO_ID, model=self.model)
            if lang_code in ENGLISH_LANG_CODES:
                apply_lexicon_overrides(pipeline.g2p)
            self.pipelines[lang_code] = pipeline
        return self.pipelines[lang_code]

    def stream(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: Number = 1,
        interactive: bool = False,
    ) -> Iterator[np.ndarray]:
        """
        Synthesize text to speech chunk by chunk, with a pause between paragraphs

        The model lock is held only while computing each chunk, so concurrent streams take turns,
        interactive streams ahead of batch ones.

        Args:
            text (str): Text to synthesize, paragraphs separated by newlines
            voice (str, optional): Voice to use for synthesis. Defaults to DEFAULT_VOICE.
            speed (Number, optional): Speed of speech. Defaults to 1.
            interactive (bool, optional): Someone is waiting to hear this, e.g. a voice assistant reply.

        Yields:
            np.ndarray: float32 mono audio at SAMPLE_RATE
        """
        voice = voice or DEFAULT_VOICE
        with self.lock(interactive):
            pipeline = self._pipeline_for(voice)
        homographs = self.homographs if voice[0] == AMERICAN_LANG_CODE else None
        pause = np.zeros(int(PARAGRAPH_PAUSE_SEC * SAMPLE_RATE), dtype=np.float32)

        emitted = False
        for paragraph in re.split(r"\n+", text.strip()):
            if not paragraph.strip():
                continue
            with self.lock(interactive):
                if homographs:
                    paragraph = homographs.annotate(paragraph)
                results = pipeline(paragraph, voice=voice, speed=speed, split_pattern=None)
            for i, audio in enumerate(self._locked_chunks(results, interactive)):
                if emitted and i == 0:
                    yield pause
                emitted = True
                yield audio

    def _locked_chunks(self, results: Iterator, interactive: bool) -> Iterator[np.ndarray]:
        """Advances a pipeline's lazy generator one chunk at a time under the model lock."""
        while True:
            with self.lock(interactive), torch.autocast("cuda", dtype=torch.float16, enabled=self.fp16):
                result = next(results, None)
                if result is None:
                    return
                if result.audio is None:
                    continue
                audio = result.audio.float().cpu().numpy()
            yield audio

    def warm_up(self, voices) -> None:
        """Builds each voice's pipeline and runs it once, homograph model included, so the first real request skips loading."""
        for voice in voices:
            for _ in self.stream(WARM_UP_TEXT, voice):
                pass

    def synth_text(
        self, text: str, uuid: str, voice: Optional[str] = None, speed: Number = 1
    ) -> str:
        """
        Synthesize text to speech

        Args:
            text (str): Text to synthesize, paragraphs separated by newlines
            uuid (str): Unique identifier for the job
            voice (str, optional): Voice to use for synthesis. Defaults to DEFAULT_VOICE.
            speed (Number, optional): Speed of speech. Defaults to 1.

        Returns:
            str: Path to the generated WAV file
        """
        segments = list(self.stream(text, voice, speed))
        if not segments:
            raise ValueError(f"Kokoro produced no audio for {uuid}")

        output_path = os.path.join(self.output_dir, f"{uuid}.wav")
        sf.write(output_path, np.concatenate(segments), SAMPLE_RATE)
        print(f"TTS completed for {uuid}, output saved to {output_path}")

        return output_path


if __name__ == "__main__":
    text = """
    The sky above the port was the color of television, tuned to a dead channel.
    "It's not like I'm using," Case heard someone say, as he shouldered his way through the crowd around the door of the Chat. "It's like my body's developed this massive drug deficiency."
    It was a Sprawl voice and a Sprawl joke. The Chatsubo was a bar for professional expatriates; you could drink there for a week and never hear two words in Japanese.

    These were to have an enormous impact, not only because they were associated with Constantine, but also because, as in so many other areas, the decisions taken by Constantine (or in his name) were to have great significance for centuries to come. One of the main issues was the shape that Christian churches were to take, since there was not, apparently, a tradition of monumental church buildings when Constantine decided to help the Christian church build a series of truly spectacular structures. The main form that these churches took was that of the basilica, a multipurpose rectangular structure, based ultimately on the earlier Greek stoa, which could be found in most of the great cities of the empire. Christianity, unlike classical polytheism, needed a large interior space for the celebration of its religious services, and the basilica aptly filled that need. We naturally do not know the degree to which the emperor was involved in the design of new churches, but it is tempting to connect this with the secular basilica that Constantine completed in the Roman forum (the so-called Basilica of Maxentius) and the one he probably built in Trier, in connection with his residence in the city at a time when he was still caesar.

    [Kokoro](/kˈOkəɹO/) is an open-weight TTS model with 82 million parameters. Despite its lightweight architecture, it delivers comparable quality to larger models while being significantly faster and more cost-efficient. With Apache-licensed weights, [Kokoro](/kˈOkəɹO/) can be deployed anywhere from production environments to personal projects.
    """

    os.makedirs("test_output", exist_ok=True)
    KokoroTTS("test_output").synth_text(text, "test-kokoro", voice=DEFAULT_VOICE)
