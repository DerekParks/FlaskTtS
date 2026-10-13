import os

import ffmpeg
import numpy as np
import soundfile as sf

from flasktts.tasks.ffmpeg import convert_wav_to_mp3


def test_convert_keeps_sample_rate_and_sets_bitrate(tmp_path):
    wav_path = str(tmp_path / "job.wav")
    t = np.linspace(0, 2, 2 * 24000, endpoint=False)
    sf.write(wav_path, 0.3 * np.sin(2 * np.pi * 440 * t), 24000)

    mp3_path = convert_wav_to_mp3(wav_path)

    stream = ffmpeg.probe(mp3_path)["streams"][0]
    assert not os.path.exists(wav_path)
    assert stream["sample_rate"] == "24000"
    assert stream["channels"] == 1
    assert stream["bit_rate"] == "96000"
