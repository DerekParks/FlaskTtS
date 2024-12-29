import os

import ffmpeg


def convert_wav_to_mp3(wav_path):
    """Convert WAV audio fie to MP3

    Runs the equivalent of: ffmpeg -i wav_path -ac 1 -ar 22050 -o out_path"""

    out_path = os.path.splitext(wav_path)[0] + ".mp3"

    ffmpeg.input(wav_path).output(out_path, ac=1, ar=22050).run()
    return out_path
