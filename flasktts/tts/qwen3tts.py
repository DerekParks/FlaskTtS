import gc
import os
import re
import time
from typing import List, Optional, Tuple

import librosa
import numpy as np
import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel

from flasktts.config import Config

# Target length per chunk in characters. Qwen3-TTS works best on short-ish
# segments; long inputs cause very slow generation and higher memory.
CHUNK_TARGET_CHARS = 500

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
        speech_rate: Optional[float] = None,
        batch_size: Optional[int] = None,
    ):
        """
        Initialize the Qwen3 TTS model with voice cloning from a reference audio.

        The voice clone prompt is pre-computed at startup so inference is a clean
        forward pass — text in, audio out.

        Args:
            output_dir (str): Output directory for generated audio files
            model_id (str, optional): HuggingFace model ID. Defaults to 0.6B Base.
            device (str, optional): Device for inference. Defaults to auto-detect.
            ref_audio (str, optional): Path to reference audio WAV for voice cloning.
            ref_text (str, optional): Transcript of the reference audio.
            speech_rate (float, optional): Pitch-preserving time stretch applied to
                the output audio. >1.0 = faster, <1.0 = slower. Defaults to
                Config.QWEN3_SPEECH_RATE.
            batch_size (int, optional): Chunks generated together. Defaults to
                Config.QWEN3_BATCH_SIZE.
        """
        self.device = self._select_device(device)
        self.output_dir = output_dir
        self.model_id = model_id or self.DEFAULT_MODEL_ID
        self.speech_rate = (
            speech_rate if speech_rate is not None else Config.QWEN3_SPEECH_RATE
        )
        self.batch_size = batch_size or Config.QWEN3_BATCH_SIZE

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        print(f"Starting Qwen3-TTS: {self.device} (torch {torch.__version__})")
        print(f"Speech rate: {self.speech_rate}")
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

        Honors the TTS_DEVICE env var as an override (e.g. "cpu", "cuda",
        "cuda:0") for hosts where auto-detection picks the wrong backend —
        for example an older GPU whose compute capability isn't supported
        by the installed PyTorch build.

        Note: MPS is disabled for Qwen3-TTS due to bf16 matmul
        incompatibilities in the PyTorch MPS backend.
        """
        if device is not None:
            return torch.device(device)
        env_device = os.getenv("TTS_DEVICE")
        if env_device:
            return torch.device(env_device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        # MPS crashes on Qwen3-TTS bf16 matmul ops — skip it
        return torch.device("cpu")

    @staticmethod
    def _chunk_text(text: str, target_chars: int = CHUNK_TARGET_CHARS) -> List[str]:
        """Split text into chunks at sentence boundaries, close to target_chars each.

        Long inputs cause very slow generation and memory pressure. Splitting at
        sentence boundaries keeps prosody natural while bounding per-call work.
        """
        text = text.strip()
        if not text:
            return []

        # Split into sentences (keeping the punctuation)
        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks: List[str] = []
        current = ""
        for sentence in sentences:
            if not sentence:
                continue
            if not current:
                current = sentence
            elif len(current) + 1 + len(sentence) <= target_chars:
                current = f"{current} {sentence}"
            else:
                chunks.append(current)
                current = sentence
        if current:
            chunks.append(current)
        return chunks

    def _generate(self, chunks: List[str]) -> Tuple[List[np.ndarray], int]:
        """One waveform per chunk, generated self.batch_size chunks at a time.

        A batch that runs out of CUDA memory is retried at half the size, and the smaller size
        is kept for later batches.
        """
        segments: List[np.ndarray] = []
        sample_rate = None
        i = 0
        while i < len(chunks):
            batch = chunks[i : i + self.batch_size]
            batch_start = time.perf_counter()
            try:
                wavs, sample_rate = self.model.generate_voice_clone(
                    text=batch,
                    voice_clone_prompt=self.voice_prompt * len(batch),
                )
            except torch.cuda.OutOfMemoryError:
                if len(batch) == 1:
                    raise
                self.batch_size = len(batch) // 2
                print(f"  CUDA out of memory at batch size {len(batch)}, retrying at {self.batch_size}")
                continue
            finally:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            segments.extend(wavs)
            i += len(batch)

            audio_len = sum(len(w) for w in wavs) / sample_rate
            elapsed = time.perf_counter() - batch_start
            print(
                f"  chunks {i}/{len(chunks)}: {sum(len(c) for c in batch)} chars -> "
                f"{audio_len:.1f}s audio in {elapsed:.1f}s (RTF {elapsed / audio_len:.2f})"
            )
        return segments, sample_rate

    def synth_text(self, text: str, uuid: str) -> str:
        """Synthesize text to speech using the pre-computed cloned voice.

        The input is split into sentence-aligned chunks, generated in batches of self.batch_size.

        Args:
            text (str): Text to synthesize
            uuid (str): Unique identifier for the job

        Returns:
            str: Output path of the generated WAV file
        """
        output_path = os.path.join(self.output_dir, f"{uuid}.wav")

        chunks = self._chunk_text(text)
        if not chunks:
            raise ValueError("Empty text passed to synth_text")

        print(
            f"Qwen3-TTS {uuid}: generating {len(chunks)} chunk(s) "
            f"({sum(len(c) for c in chunks)} chars) in batches of {self.batch_size}"
        )

        t0 = time.perf_counter()
        all_segments, sample_rate = self._generate(chunks)
        combined_audio = np.concatenate(all_segments)

        if self.speech_rate != 1.0:
            combined_audio = librosa.effects.time_stretch(
                combined_audio, rate=self.speech_rate
            )

        sf.write(output_path, combined_audio, sample_rate)

        total = time.perf_counter() - t0
        total_audio = len(combined_audio) / sample_rate
        print(
            f"Qwen3-TTS {uuid} complete: {total_audio:.1f}s audio in "
            f"{total:.1f}s (RTF {total / total_audio:.2f}) -> {output_path}"
        )
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
