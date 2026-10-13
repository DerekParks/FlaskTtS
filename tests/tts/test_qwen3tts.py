from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf
import torch

pytest.importorskip("qwen_tts")

from flasktts.tts.qwen3tts import Qwen3TTS

SAMPLE_RATE = 24000


def fake_generate(max_batch=None):
    """Each chunk's waveform is filled with its chunk number, e.g. "3." -> 3/32."""
    calls = []

    def generate(text, voice_clone_prompt):
        calls.append(list(text))
        assert len(voice_clone_prompt) == len(text)
        if max_batch is not None and len(text) > max_batch:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory")
        return [np.full(10, int(t.rstrip(".")) / 32, dtype=np.float32) for t in text], SAMPLE_RATE

    return MagicMock(side_effect=generate), calls


@pytest.fixture
def qwen(tmp_path):
    with patch("flasktts.tts.qwen3tts.Qwen3TTSModel") as model_cls:
        model_cls.from_pretrained.return_value.create_voice_clone_prompt.return_value = ["prompt"]
        yield Qwen3TTS(str(tmp_path), device="cpu", speech_rate=1.0, batch_size=4)


def chunks(n):
    return " ".join(f"{i}." for i in range(1, n + 1))


def synth(qwen, n):
    with patch.object(Qwen3TTS, "_chunk_text", side_effect=lambda text: text.split()):
        audio, _ = sf.read(qwen.synth_text(chunks(n), "job"))
    return audio


def test_chunks_generated_in_batches_in_order(qwen):
    qwen.model.generate_voice_clone, calls = fake_generate()

    audio = synth(qwen, 10)

    assert [len(c) for c in calls] == [4, 4, 2]
    expected = np.concatenate([np.full(10, i / 32) for i in range(1, 11)])
    np.testing.assert_allclose(audio, expected, atol=1e-4)


def test_out_of_memory_halves_batch_size_and_keeps_it(qwen):
    qwen.model.generate_voice_clone, calls = fake_generate(max_batch=2)

    audio = synth(qwen, 7)

    assert [len(c) for c in calls] == [4, 2, 2, 2, 1]
    assert qwen.batch_size == 2
    assert len(audio) == 70


def test_out_of_memory_on_a_single_chunk_raises(qwen):
    qwen.model.generate_voice_clone, _ = fake_generate(max_batch=0)

    with pytest.raises(torch.cuda.OutOfMemoryError):
        synth(qwen, 3)


def test_chunk_text_packs_sentences_up_to_target():
    text = "One two. Three four. Five six."

    assert Qwen3TTS._chunk_text(text, target_chars=20) == ["One two. Three four.", "Five six."]
