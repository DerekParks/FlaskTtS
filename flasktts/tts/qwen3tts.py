import os
from typing import Optional

import numpy as np
import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel

from flasktts.config import Config

# Default reference voice: Kokoro af_heart cloned via Qwen3 Base model
DEFAULT_REF_AUDIO = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "voices",
    "kokoro-af-heart.wav",
)
DEFAULT_REF_TEXT = (
    "The sky above the port was the color of television, tuned to a dead channel. "
    "It is not like I am using, Case heard someone say."
)


class Qwen3TTSHighlander:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = Qwen3TTS(Config.TTS_WORKDIR)
        return cls._instance


class Qwen3TTS:
    DEFAULT_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"

    def __init__(
        self,
        output_dir: str,
        model_id: Optional[str] = None,
        device: Optional[str] = None,
        ref_audio: Optional[str] = None,
        ref_text: Optional[str] = None,
    ):
        """
        Initialize the Qwen3 TTS model with voice cloning from a reference audio.

        The voice clone prompt is pre-computed at startup so inference is a clean
        forward pass — text in, audio out. This makes future offload to accelerators
        like Hailo-8 (via ONNX/HEF) straightforward.

        Args:
            output_dir (str): Output directory for generated audio files
            model_id (str, optional): HuggingFace model ID. Defaults to 0.6B Base.
            device (str, optional): Device for inference. Defaults to auto-detect.
            ref_audio (str, optional): Path to reference audio WAV for voice cloning.
            ref_text (str, optional): Transcript of the reference audio.
        """
        self.device = self._select_device(device)
        self.output_dir = output_dir
        self.model_id = model_id or self.DEFAULT_MODEL_ID

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        print(f"Starting Qwen3-TTS: {self.device} (torch {torch.__version__})")
        print(f"Loading model: {self.model_id}")

        self.model = Qwen3TTSModel.from_pretrained(
            self.model_id,
            device_map=str(self.device),
            dtype="auto",
        )

        # Pre-compute voice clone prompt at startup
        ref_audio = ref_audio or DEFAULT_REF_AUDIO
        ref_text = ref_text or DEFAULT_REF_TEXT
        print(f"Pre-computing voice clone prompt from: {ref_audio}")
        self.voice_prompt = self.model.create_voice_clone_prompt(
            ref_audio=ref_audio,
            ref_text=ref_text,
        )
        print("Voice clone prompt ready")

    @staticmethod
    def _select_device(device: Optional[str] = None) -> torch.device:
        """Select the best available compute device.

        Device selection is extracted as a static method to make it easy to
        extend with additional backends (e.g. Hailo-8 via hailo_platform)
        without touching inference logic.

        Note: MPS is currently disabled for Qwen3-TTS due to bf16 matmul
        incompatibilities in PyTorch MPS backend.
        """
        if device is not None:
            return torch.device(device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        # MPS crashes on Qwen3-TTS bf16 matmul ops — skip it
        return torch.device("cpu")

    def synth_text(self, text: str, uuid: str) -> str:
        """Synthesize text to speech using the pre-computed cloned voice.

        Args:
            text (str): Text to synthesize
            uuid (str): Unique identifier for the job

        Returns:
            str: Output path of the generated WAV file
        """
        output_path = os.path.join(self.output_dir, f"{uuid}.wav")

        # Generate audio using the cached voice clone prompt
        audio_segments, sample_rate = self.model.generate_voice_clone(
            text=text,
            voice_clone_prompt=self.voice_prompt,
        )

        combined_audio = np.concatenate(audio_segments)
        sf.write(output_path, combined_audio, sample_rate)

        print(f"Qwen3-TTS completed for {uuid}, output saved to {output_path}")
        return output_path

    def cleanup(self, task_id=None):
        """Remove generated files from the output directory."""
        if task_id is not None:
            for file in os.listdir(self.output_dir):
                if task_id in file:
                    os.remove(os.path.join(self.output_dir, file))
        else:
            for file in os.listdir(self.output_dir):
                os.remove(os.path.join(self.output_dir, file))


if __name__ == "__main__":
    text = (
        "The sky above the port was the color of television, tuned to a dead channel. "
        "It's not like I'm using, Case heard someone say, as he shouldered his way "
        "through the crowd around the door of the Chat."
    )
    tts = Qwen3TTS("test_output")
    tts.synth_text(text, "test-qwen3-clone")
