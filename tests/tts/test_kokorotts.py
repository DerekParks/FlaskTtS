from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf
import torch
from misaki import en

from flasktts.config import Config
from flasktts.tts.kokorotts import (
    DEFAULT_VOICE,
    PARAGRAPH_PAUSE_SEC,
    SAMPLE_RATE,
    KokoroTTS,
    apply_lexicon_overrides,
)

SAMPLES_PER_CHUNK = 100


def fake_pipeline(lang_code, repo_id, model):
    """Yields two chunks per paragraph, each filled with level int(paragraph) / 32."""

    def run(text, voice, speed, split_pattern):
        level = int(text) / 32
        for _ in range(2):
            yield SimpleNamespace(audio=torch.full((SAMPLES_PER_CHUNK,), level))

    return MagicMock(side_effect=run)


@pytest.fixture
def tts(tmp_path):
    with (
        patch.object(Config, "HOMOGRAPH_MODEL_DIR", None),
        patch("flasktts.tts.kokorotts.KModel") as kmodel,
        patch(
            "flasktts.tts.kokorotts.KPipeline", side_effect=fake_pipeline
        ) as kpipeline,
    ):
        instance = KokoroTTS(str(tmp_path), device="cpu")
        instance.kpipeline = kpipeline
        instance.kmodel = kmodel
        yield instance


def test_paragraphs_concatenated_in_order_with_pauses(tts):
    paragraphs = range(1, 13)
    text = "\n\n".join(str(p) for p in paragraphs)

    wav_path = tts.synth_text(text, "job", voice="af_heart")

    audio, sample_rate = sf.read(wav_path)
    pause = np.zeros(int(PARAGRAPH_PAUSE_SEC * SAMPLE_RATE))
    expected = []
    for p in paragraphs:
        if expected:
            expected.append(pause)
        expected.append(np.full(2 * SAMPLES_PER_CHUNK, p / 32))
    assert sample_rate == SAMPLE_RATE
    np.testing.assert_allclose(audio, np.concatenate(expected), atol=1e-4)


def test_blank_lines_do_not_add_pauses(tts):
    wav_path = tts.synth_text("\n  1\n   \n\n2\n", "job", voice="af_heart")

    audio, _ = sf.read(wav_path)
    pause_samples = int(PARAGRAPH_PAUSE_SEC * SAMPLE_RATE)
    assert len(audio) == 4 * SAMPLES_PER_CHUNK + pause_samples


def test_missing_voice_uses_default(tts):
    tts.synth_text("1", "job", voice=None)

    pipeline = tts.pipelines[DEFAULT_VOICE[0]]
    assert pipeline.call_args.kwargs["voice"] == DEFAULT_VOICE


def test_pipeline_per_language_shares_one_model(tts):
    tts.synth_text("1", "job-a", voice="af_heart")
    tts.synth_text("1", "job-b", voice="bf_emma")
    tts.synth_text("1", "job-a2", voice="am_adam")

    lang_codes = [c.kwargs["lang_code"] for c in tts.kpipeline.call_args_list]
    models = {id(c.kwargs["model"]) for c in tts.kpipeline.call_args_list}
    assert lang_codes == ["a", "b"]
    assert models == {id(tts.model)}
    tts.kmodel.assert_called_once()


def test_no_audio_raises(tts):
    with pytest.raises(ValueError):
        tts.synth_text("\n\n", "job", voice="af_heart")


@pytest.mark.parametrize("british", [False, True])
def test_dives_pronounced_as_verb(british):
    g2p = en.G2P(trf=False, british=british, fallback=None, unk="")
    apply_lexicon_overrides(g2p)

    phonemes, _ = g2p("Deep dives. Dives deep. DIVES.")

    assert phonemes.count("dˈIvz") == 3
    assert "iz" not in phonemes


def test_lexicon_overrides_only_on_english_pipelines(tts):
    tts.synth_text("1", "job-a", voice="af_heart")
    tts.synth_text("1", "job-j", voice="jf_alpha")

    tts.pipelines["a"].g2p.lexicon.golds.update.assert_called_once()
    tts.pipelines["j"].g2p.lexicon.golds.update.assert_not_called()


@pytest.mark.parametrize(
    "device, fp16, expected",
    [("cuda", True, True), ("cuda", False, False), ("cpu", True, False)],
)
def test_fp16_only_on_cuda(tmp_path, device, fp16, expected):
    with patch("flasktts.tts.kokorotts.KModel"):
        assert KokoroTTS(str(tmp_path), device=device, fp16=fp16).fp16 is expected


def test_inference_runs_inside_autocast(tts):
    tts.fp16 = True
    states = []
    autocast = MagicMock()
    autocast.return_value.__enter__.side_effect = lambda: states.append("enter")
    autocast.return_value.__exit__.side_effect = lambda *a: states.append("exit")
    tts._pipeline_for("af_heart").side_effect = lambda text, **k: (
        states.append("synth") or iter([SimpleNamespace(audio=torch.ones(10))])
    )

    with patch("flasktts.tts.kokorotts.torch.autocast", autocast):
        tts.synth_text("1", "job", voice="af_heart")

    assert autocast.call_args.kwargs == {"dtype": torch.float16, "enabled": True}
    assert states == ["enter", "synth", "exit"]


def test_half_precision_audio_written_as_float(tts):
    tts._pipeline_for("af_heart").side_effect = lambda text, **k: iter(
        [SimpleNamespace(audio=torch.full((SAMPLES_PER_CHUNK,), 0.25, dtype=torch.float16))]
    )

    audio, _ = sf.read(tts.synth_text("1", "job", voice="af_heart"))

    np.testing.assert_allclose(audio, 0.25, atol=1e-4)


def test_homographs_annotate_only_american_voices(tts):
    tts.homographs = MagicMock()
    tts.homographs.annotate.side_effect = lambda text: text

    tts.synth_text("1\n\n2", "job-a", voice="af_heart")
    tts.synth_text("1", "job-b", voice="bf_emma")

    assert [c.args[0] for c in tts.homographs.annotate.call_args_list] == ["1", "2"]


def test_annotated_paragraph_reaches_pipeline(tts):
    tts.homographs = MagicMock()
    tts.homographs.annotate.side_effect = lambda text: f"{int(text) + 1}"

    tts.synth_text("1", "job", voice="af_heart")

    assert tts.pipelines["a"].call_args.args[0] == "2"


def test_homographs_disabled_without_model_dir(tts):
    assert tts.homographs is None
