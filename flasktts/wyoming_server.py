"""Wyoming protocol text-to-speech server backed by the shared Kokoro instance."""

import asyncio
import logging
from functools import partial
from pathlib import Path
from typing import Iterator, List, Optional

import numpy as np
from huggingface_hub import snapshot_download
from sentence_stream import SentenceBoundaryDetector
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.error import Error
from wyoming.event import Event
from wyoming.info import Attribution, Describe, Info, TtsProgram, TtsVoice
from wyoming.server import AsyncEventHandler, AsyncServer
from wyoming.tts import (
    Synthesize,
    SynthesizeChunk,
    SynthesizeStart,
    SynthesizeStop,
    SynthesizeStopped,
    SynthesizeVoice,
)

from flasktts.tts.kokorotts import DEFAULT_VOICE, REPO_ID, SAMPLE_RATE, KokoroTTS

_LOGGER = logging.getLogger(__name__)

SAMPLE_WIDTH = 2
CHANNELS = 1
SAMPLES_PER_CHUNK = 1024
LANGUAGES = {"a": "en_US", "b": "en_GB"}
LANGUAGE_DEFAULT_VOICES = {"en_US": DEFAULT_VOICE, "en_GB": "bf_emma"}
ATTRIBUTION = Attribution(name="hexgrad", url="https://huggingface.co/hexgrad/Kokoro-82M")


def kokoro_voices() -> List[str]:
    """Kokoro voice names whose language pipeline FlaskTtS supports, e.g. af_heart, bf_emma.

    Downloads the voices, falling back to the Hugging Face cache when the Hub is unreachable.
    """
    snapshot = Path(snapshot_download(REPO_ID, allow_patterns=["voices/*.pt"]))
    return sorted(path.stem for path in (snapshot / "voices").glob("*.pt") if path.stem[0] in LANGUAGES)


def build_info(voices: List[str]) -> Info:
    return Info(
        tts=[
            TtsProgram(
                name="kokoro",
                description="Kokoro 82M via FlaskTtS",
                attribution=ATTRIBUTION,
                installed=True,
                version=None,
                voices=[
                    TtsVoice(
                        name=name,
                        description=name,
                        attribution=ATTRIBUTION,
                        installed=True,
                        version=None,
                        languages=[LANGUAGES[name[0]]],
                    )
                    for name in voices
                ],
                supports_synthesize_streaming=True,
            )
        ]
    )


def to_pcm16(audio: np.ndarray) -> bytes:
    return (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2").tobytes()


class KokoroEventHandler(AsyncEventHandler):
    """One client connection. Streams audio sentence by sentence as Kokoro produces it.

    Streaming requests arrive as synthesize-start, synthesize-chunk..., an optional synthesize
    (sent for older servers, ignored here), and synthesize-stop.
    """

    def __init__(self, info: Info, tts: KokoroTTS, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.info = info
        self.tts = tts
        self.voices = {v.name for v in info.tts[0].voices}
        self.stream_voice: Optional[str] = None
        self.sentences: Optional[SentenceBoundaryDetector] = None

    async def handle_event(self, event: Event) -> bool:
        if Describe.is_type(event.type):
            await self.write_event(self.info.event())
            return True

        try:
            if Synthesize.is_type(event.type):
                if self.sentences is None:
                    synthesize = Synthesize.from_event(event)
                    await self.speak(synthesize.text, self.resolve_voice(synthesize.voice))
            elif SynthesizeStart.is_type(event.type):
                self.stream_voice = self.resolve_voice(SynthesizeStart.from_event(event).voice)
                self.sentences = SentenceBoundaryDetector()
            elif SynthesizeChunk.is_type(event.type):
                for sentence in self.sentences.add_chunk(SynthesizeChunk.from_event(event).text):
                    await self.speak(sentence, self.stream_voice)
            elif SynthesizeStop.is_type(event.type):
                rest = self.sentences.finish()
                if rest.strip():
                    await self.speak(rest, self.stream_voice)
                self.sentences = None
                await self.write_event(SynthesizeStopped().event())
        except Exception as err:
            await self.write_event(Error(text=str(err), code=err.__class__.__name__).event())
            raise

        return True

    def resolve_voice(self, voice: Optional[SynthesizeVoice]) -> str:
        if voice is None or (voice.name is None and voice.language is None):
            return DEFAULT_VOICE
        if voice.name is not None:
            if voice.name not in self.voices:
                raise ValueError(f"Unknown voice {voice.name!r}")
            return voice.name
        if voice.language not in LANGUAGE_DEFAULT_VOICES:
            raise ValueError(f"Unsupported language {voice.language!r}")
        return LANGUAGE_DEFAULT_VOICES[voice.language]

    async def speak(self, text: str, voice: str) -> None:
        """audio-start, audio-chunk per SAMPLES_PER_CHUNK as each Kokoro chunk finishes, audio-stop."""
        await self.write_event(AudioStart(rate=SAMPLE_RATE, width=SAMPLE_WIDTH, channels=CHANNELS).event())
        chunks: Iterator[np.ndarray] = self.tts.stream(text, voice, interactive=True)
        while (audio := await asyncio.to_thread(next, chunks, None)) is not None:
            pcm = to_pcm16(audio)
            step = SAMPLES_PER_CHUNK * SAMPLE_WIDTH * CHANNELS
            for offset in range(0, len(pcm), step):
                await self.write_event(
                    AudioChunk(
                        audio=pcm[offset : offset + step],
                        rate=SAMPLE_RATE,
                        width=SAMPLE_WIDTH,
                        channels=CHANNELS,
                    ).event()
                )
        await self.write_event(AudioStop().event())


async def serve(uri: str, tts: KokoroTTS, voices: List[str]) -> None:
    info = build_info(voices)
    _LOGGER.info("Wyoming TTS listening on %s with %d voices", uri, len(voices))
    await AsyncServer.from_uri(uri).run(partial(KokoroEventHandler, info, tts))
