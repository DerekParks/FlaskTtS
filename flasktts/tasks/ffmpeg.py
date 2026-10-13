import os

import ffmpeg


def convert_wav_to_mp3(wav_path):
    """Convert WAV audio file to MP3 and delete the source WAV.

    Runs the equivalent of: ffmpeg -i wav_path -ac 1 out_path"""

    out_path = os.path.splitext(wav_path)[0] + ".mp3"

    ffmpeg.input(wav_path).output(out_path, ac=1).run()
    os.remove(wav_path)
    return out_path

