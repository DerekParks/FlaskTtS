import glob
import os

import ffmpeg


def convert_wav_to_mp3(wav_path):
    """Convert WAV audio fie to MP3

    Runs the equivalent of: ffmpeg -i wav_path -ac 1 -ar 22050 -o out_path"""

    out_path = os.path.splitext(wav_path)[0] + ".mp3"

    ffmpeg.input(wav_path).output(out_path, ac=1, ar=22050).run()
    return out_path


def convert_wav_dir_to_mp3(wav_dir):
    """Convert a directory of WAVs in order to MP3s

    Runs the equivalent of: ffmpeg -f concat -safe 0 -i files.txt -c:a libmp3lame output.mp3"""

    with open("temp_files.txt", "w") as f:
        for wav_file in sorted(glob.glob(os.path.join(wav_dir, "*.wav"))):
            f.write(f"file '{os.path.abspath(wav_file)}'\n")

    out_path = f"{wav_dir}.mp3"

    ffmpeg.input("temp_files.txt", f="concat", safe=0).output(
        out_path, ac=1, ar=22050
    ).run()

    os.remove("temp_files.txt")
    return out_path
