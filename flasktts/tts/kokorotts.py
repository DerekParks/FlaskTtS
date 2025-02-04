import os
import time
from numbers import Number
from typing import Optional

import soundfile as sf
import torch
from kokoro import KPipeline

from flasktts.config import Config


class KokoroTTSHighlander:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = KokoroTTS(Config.TTS_WORKDIR)
        return cls._instance


class KokoroTTS:
    def __init__(
        self, output_dir: str, lang_code: str = "a", device: Optional[str] = None
    ):
        """
        Args:
            output_dir (str): Output directory to save the generated audio files
            lang_code (str):
                 🇺🇸 'a' => American English,
                 🇬🇧 'b' => British English
                 🇯🇵 'j' => Japanese: pip install misaki[ja]
                🇨🇳 'z' => Mandarin Chinese: pip install misaki[zh]
            device (str, optional): Device to use for inference. Defaults to None for auto-detect.
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
        self.pipeline = KPipeline(lang_code=lang_code, device=device)

    def synth_text(self, text: str, uuid: str, voice: str, speed: Number = 1) -> str:
        """
        Synthesize text to speech

        Args:
            text (str): Text to synthesize
            uuid (str): Unique identifier for the job
            voice (str): Voice to use for synthesis
            speed (Number, optional): Speed of speech. Defaults to 1.

        Returns:
            str: Output path where the generated audio files
        """
        output_path = os.path.join(self.output_dir, uuid)
        if os.path.exists(output_path):
            print(
                f"Output path {output_path} already exists, deleting in 3 seconds Ctrl+C to cancel"
            )
            time.sleep(3)
            for file in os.listdir(output_path):
                os.remove(os.path.join(output_path, file))
            os.rmdir(output_path)
        os.makedirs(output_path)

        generator = self.pipeline(
            text,
            voice=voice,
            speed=speed,
            split_pattern=r"\n+",
        )

        for i, (_, _, audio) in enumerate(generator):
            print(i)
            sf.write(
                os.path.join(output_path, f"{int(time.time())}_{i}.wav"), audio, 24000
            )

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

    KokoroTTS("test_output", lang_code="a").synth_text(
        text, "test-kokoro", voice="af_heart"
    )
