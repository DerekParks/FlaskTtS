import asyncio
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.error import Error
from wyoming.info import Describe, Info
from wyoming.tts import (
    Synthesize,
    SynthesizeChunk,
    SynthesizeStart,
    SynthesizeStop,
    SynthesizeStopped,
    SynthesizeVoice,
)

from flasktts.wyoming_server import (
    SAMPLES_PER_CHUNK,
    KokoroEventHandler,
    build_info,
    kokoro_voices,
    to_pcm16,
)

VOICES = ["af_heart", "am_adam", "bf_emma"]


class FakeKokoro:
    """Yields one chunk per sentence-ish piece of text, 1.5 audio chunks long."""

    def __init__(self):
        self.calls = []

    def stream(self, text, voice, interactive=False):
        assert interactive
        self.calls.append((text, voice))
        yield np.full(SAMPLES_PER_CHUNK + SAMPLES_PER_CHUNK // 2, 0.5, dtype=np.float32)


def make_handler():
    tts = FakeKokoro()
    handler = KokoroEventHandler(build_info(VOICES), tts, MagicMock(), MagicMock())
    events = []

    async def capture(event):
        events.append(event)

    handler.write_event = capture
    return handler, tts, events


def send(handler, *eventables):
    async def run():
        for e in eventables:
            await handler.handle_event(e.event())

    asyncio.run(run())


def types(events):
    return [e.type for e in events]


def test_describe_advertises_streaming_and_voices():
    handler, _, events = make_handler()

    send(handler, Describe())

    tts = Info.from_event(events[0]).tts[0]
    assert tts.supports_synthesize_streaming
    assert {v.name: v.languages for v in tts.voices} == {
        "af_heart": ["en_US"],
        "am_adam": ["en_US"],
        "bf_emma": ["en_GB"],
    }


def test_synthesize_streams_one_audio_segment():
    handler, tts, events = make_handler()

    send(handler, Synthesize(text="Hello there. How are you?", voice=SynthesizeVoice(name="am_adam")))

    assert tts.calls == [("Hello there. How are you?", "am_adam")]
    assert types(events) == ["audio-start", "audio-chunk", "audio-chunk", "audio-stop"]
    start = AudioStart.from_event(events[0])
    assert (start.rate, start.width, start.channels) == (24000, 2, 1)
    assert [len(AudioChunk.from_event(e).audio) for e in events[1:3]] == [SAMPLES_PER_CHUNK * 2, SAMPLES_PER_CHUNK]


def test_text_stream_speaks_each_sentence_as_it_completes():
    handler, tts, events = make_handler()

    send(
        handler,
        SynthesizeStart(voice=SynthesizeVoice(name="bf_emma")),
        SynthesizeChunk(text="Hello there. How "),
        SynthesizeChunk(text="are you today? Bye"),
        Synthesize(text="Hello there. How are you today? Bye", voice=SynthesizeVoice(name="bf_emma")),
        SynthesizeStop(),
    )

    assert tts.calls == [("Hello there.", "bf_emma"), ("How are you today?", "bf_emma"), ("Bye", "bf_emma")]
    per_sentence = ["audio-start", "audio-chunk", "audio-chunk", "audio-stop"]
    assert types(events) == per_sentence * 3 + [SynthesizeStopped().event().type]


def test_synthesize_after_a_stream_is_spoken_again():
    handler, tts, _ = make_handler()

    send(handler, SynthesizeStart(), SynthesizeChunk(text="One."), SynthesizeStop(), Synthesize(text="Two."))

    assert [text for text, _ in tts.calls] == ["One.", "Two."]


@pytest.mark.parametrize(
    "voice, expected",
    [
        (None, "af_heart"),
        (SynthesizeVoice(), "af_heart"),
        (SynthesizeVoice(language="en_GB"), "bf_emma"),
        (SynthesizeVoice(name="am_adam"), "am_adam"),
    ],
)
def test_voice_resolution(voice, expected):
    handler, tts, _ = make_handler()

    send(handler, Synthesize(text="Hi.", voice=voice))

    assert tts.calls == [("Hi.", expected)]


@pytest.mark.parametrize("voice", [SynthesizeVoice(name="jf_alpha"), SynthesizeVoice(language="fr_FR")])
def test_unknown_voice_reports_error(voice):
    handler, tts, events = make_handler()

    with pytest.raises(ValueError):
        send(handler, Synthesize(text="Hi.", voice=voice))

    assert tts.calls == []
    assert Error.from_event(events[-1]).code == "ValueError"


def test_to_pcm16_clips_and_scales():
    pcm = np.frombuffer(to_pcm16(np.array([0.0, 0.5, -1.5, 2.0], dtype=np.float32)), dtype="<i2")

    assert pcm.tolist() == [0, 16383, -32767, 32767]


def test_kokoro_voices_keeps_supported_languages():
    files = ["config.json", "voices/af_heart.pt", "voices/bm_george.pt", "voices/jf_alpha.pt", "voices/zf_xiaoyi.pt"]

    with patch("flasktts.wyoming_server.list_repo_files", return_value=files):
        assert kokoro_voices() == ["af_heart", "bm_george"]
