import os

import ffmpeg

MP3_BITRATE = "96k"


def convert_wav_to_mp3(wav_path):
    """Convert WAV audio file to MP3 and delete the source WAV.

    Runs the equivalent of: ffmpeg -i wav_path -ac 1 -b:a MP3_BITRATE out_path"""

    out_path = os.path.splitext(wav_path)[0] + ".mp3"

    ffmpeg.input(wav_path).output(out_path, ac=1, audio_bitrate=MP3_BITRATE).run()
    os.remove(wav_path)
    return out_path

