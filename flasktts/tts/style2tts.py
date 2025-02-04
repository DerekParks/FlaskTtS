import base64
import os
import random
import re
import sys
from itertools import chain
from typing import Optional

import librosa
import numpy as np
import phonemizer
import torch
import torchaudio
import yaml
from nltk.tokenize import word_tokenize
from scipy.io.wavfile import write
from styletts2.models import *
from styletts2.Modules.diffusion.sampler import (
    ADPM2Sampler,
    DiffusionSampler,
    KarrasSchedule,
)
from styletts2.text_utils import TextCleaner
from styletts2.utils import *
from styletts2.Utils.PLBERT.util import load_plbert

from flasktts.config import Config


class Style2TTSHighlander:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = Style2TTS(Config.TTS_WORKDIR)
        return cls._instance


class Style2TTS:
    def __init__(self, output_dir: str, device: Optional[str] = None):
        """
        Initialize the Style2TTS model

        Args:
            output_dir (str): Output directory for generated audio files
            device (str, optional): Device to use for inference. Defaults to None for auto-detect.
        """
        if device is not None:
            self.device = device
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        print(f"Starting TTS: {self.device} {torch.__version__}")
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.textclenaer = TextCleaner()
        self.seed_init()
        self.load_models()
        # comply with the model license
        self.preroll = base64.b64decode(
            b"VGhlIGZvbGxvd2luZyBpcyBiZWluZyByZWFkIGJ5IGFuIEFJIHZvaWNlLg=="
        ).decode("utf-8")

        self.s_prev = None
        self.sample_rate = 24000

    def seed_init(self, seed=0):
        """Reset all seeds for reproducibility"""

        torch.manual_seed(0)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        random.seed(0)
        np.random.seed(0)

    def load_models(self):
        """Load all models and parameters"""
        self.to_mel = torchaudio.transforms.MelSpectrogram(
            n_mels=80, n_fft=2048, win_length=1200, hop_length=300
        )
        self.mean, self.std = -4, 4

        self.global_phonemizer = phonemizer.backend.EspeakBackend(
            language="en-us", preserve_punctuation=True, with_stress=True
        )

        config = yaml.safe_load(open("Models/LJSpeech/config.yml"))

        # load pretrained ASR model
        ASR_config = config.get("ASR_config", False)
        ASR_path = config.get("ASR_path", False)
        text_aligner = load_ASR_models(ASR_path, ASR_config)

        # load pretrained F0 model
        F0_path = config.get("F0_path", False)
        pitch_extractor = load_F0_models(F0_path)

        # load BERT model
        BERT_path = config.get("PLBERT_dir", False)
        plbert = load_plbert(BERT_path)

        self.model = build_model(
            recursive_munch(config["model_params"]),
            text_aligner,
            pitch_extractor,
            plbert,
        )
        _ = [self.model[key].eval() for key in self.model]
        _ = [self.model[key].to(self.device) for key in self.model]

        params_whole = torch.load(
            "Models/LJSpeech/epoch_2nd_00100.pth", map_location="cpu"
        )
        params = params_whole["net"]

        for key in self.model:
            if key in params:
                print("%s loaded" % key)
                try:
                    self.model[key].load_state_dict(params[key])
                except:
                    from collections import OrderedDict

                    state_dict = params[key]
                    new_state_dict = OrderedDict()
                    for k, v in state_dict.items():
                        name = k[7:]  # remove `module.`
                        new_state_dict[name] = v
                    # load params
                    self.model[key].load_state_dict(new_state_dict, strict=False)
        #             except:
        #                 _load(params[key], model[key])
        _ = [self.model[key].eval() for key in self.model]

        self.sampler = DiffusionSampler(
            self.model.diffusion.diffusion,
            sampler=ADPM2Sampler(),
            sigma_schedule=KarrasSchedule(
                sigma_min=0.0001, sigma_max=3.0, rho=9.0
            ),  # empirical parameters
            clamp=False,
        )

    def length_to_mask(self, lengths):
        mask = (
            torch.arange(lengths.max())
            .unsqueeze(0)
            .expand(lengths.shape[0], -1)
            .type_as(lengths)
        )
        mask = torch.gt(mask + 1, lengths.unsqueeze(1))
        return mask

    def preprocess(wave):
        wave_tensor = torch.from_numpy(wave).float()
        mel_tensor = self.to_mel(wave_tensor)
        mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - self.mean) / self.std
        return mel_tensor

    def compute_style(ref_dicts):
        reference_embeddings = {}
        for key, path in ref_dicts.items():
            wave, sr = librosa.load(path, sr=self.sample_rate)
            audio, index = librosa.effects.trim(wave, top_db=30)
            if sr != self.sample_rate:
                audio = librosa.resample(audio, sr, self.sample_rate)
            mel_tensor = preprocess(audio).to(device)

            with torch.no_grad():
                ref = self.model.style_encoder(mel_tensor.unsqueeze(1))
            reference_embeddings[key] = (ref.squeeze(1), audio)

        return reference_embeddings

    def long_form_inference(
        self, text, s_prev, noise, alpha=0.7, diffusion_steps=10, embedding_scale=1.5
    ):
        """Long-form inference"""
        text = text.strip()
        text = text.replace('"', "")
        ps = self.global_phonemizer.phonemize([text])
        ps = word_tokenize(ps[0])
        ps = " ".join(ps)

        tokens = self.textclenaer(ps)

        tokens.insert(0, 0)

        # Limit the number of tokens
        if len(tokens) > 512 - 1:
            tokens = tokens[:512]

        tokens = torch.LongTensor(tokens).to(self.device).unsqueeze(0)

        with torch.no_grad():
            input_lengths = torch.LongTensor([tokens.shape[-1]]).to(tokens.device)
            text_mask = self.length_to_mask(input_lengths).to(tokens.device)

            t_en = self.model.text_encoder(tokens, input_lengths, text_mask)

            bert_dur = self.model.bert(tokens, attention_mask=(~text_mask).int())
            d_en = self.model.bert_encoder(bert_dur).transpose(-1, -2)

            s_pred = self.sampler(
                noise,
                embedding=bert_dur[0].unsqueeze(0),
                num_steps=diffusion_steps,
                embedding_scale=embedding_scale,
            ).squeeze(0)

            if s_prev is not None:
                # convex combination of previous and current style
                s_pred = alpha * s_prev + (1 - alpha) * s_pred

            s = s_pred[:, 128:]
            ref = s_pred[:, :128]

            d = self.model.predictor.text_encoder(d_en, s, input_lengths, text_mask)

            x, _ = self.model.predictor.lstm(d)
            duration = self.model.predictor.duration_proj(x)
            duration = torch.sigmoid(duration).sum(axis=-1)
            pred_dur = torch.round(duration.squeeze()).clamp(min=1)

            pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().data))
            c_frame = 0
            for i in range(pred_aln_trg.size(0)):
                pred_aln_trg[i, c_frame : c_frame + int(pred_dur[i].data)] = 1
                c_frame += int(pred_dur[i].data)

            # encode prosody
            en = d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(self.device)
            F0_pred, N_pred = self.model.predictor.F0Ntrain(en, s)
            out = self.model.decoder(
                (t_en @ pred_aln_trg.unsqueeze(0).to(self.device)),
                F0_pred,
                N_pred,
                ref.squeeze().unsqueeze(0),
            )

        return out.squeeze().cpu().numpy(), s_pred

    def tts_line(self, line, max_batch_size=500):
        sentences = re.split("[?.,!]", line)  # simple split by comma

        for text in sentences:
            if text.strip() == "":
                continue
            text += "."  # add it back
            noise = torch.randn(1, 1, 256).to(self.device)

            sys.stdout.flush()
            wav, self.s_prev = self.long_form_inference(text, self.s_prev, noise)
            yield wav

    def synth_text(self, text: str, uuid: str) -> str:
        """Synthesize text to speech

        Args:
            text (str): Text to synthesize
            uuid (str): Unique identifier for the job

        Returns:
            str: Output path where the generated audio files
        """

        output_path = os.path.join(self.output_dir, f"{uuid}.wav")
        all_wavs = []

        for i, line in enumerate(chain([self.preroll], text.splitlines())):
            for wav in self.tts_line(line, i):
                all_wavs.append(wav)

        # Concatenate all the wavs into a single array
        combined_wav = np.concatenate(all_wavs)
        combined_wav = np.array(
            combined_wav * 32767, dtype=np.int16
        )  # Convert to 16-bit PCM

        # Write the combined wav to the output file
        write(output_path, self.sample_rate, combined_wav)

        print(f"TTS completed for {uuid}, output saved to {output_path}")

        return output_path

    def cleanup(self, task_id=None):
        """Remove all files from the output directory"""
        if task_id is not None:
            for file in os.listdir(self.output_dir):
                if task_id in file:
                    os.remove(os.path.join(self.output_dir, file))
        else:
            for file in os.listdir(self.output_dir):
                os.remove(os.path.join(self.output_dir, file))


if __name__ == "__main__":
    tts = Style2TTS("test_output")
    text = "This is a test of the TTS system. It should work well. Let's see how it goes. I hope it works. I really do."
    tts.synth_text(text, "test")
