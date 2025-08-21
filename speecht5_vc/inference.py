import io
import wave
from typing import Optional
import numpy as np
import torch
from transformers import SpeechT5Processor, SpeechT5ForSpeechToSpeech, SpeechT5HifiGan
from .embeddings import build_speaker_embedder

def _to_device(device: Optional[str]) -> str:
    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"

def _float_to_wav_bytes(signal: np.ndarray, sample_rate: int = 16000) -> bytes:
    """Convert float32 [-1,1] to 16-bit PCM WAV bytes."""
    sig = np.clip(signal, -1.0, 1.0)
    sig_i16 = (sig * 32767.0).astype(np.int16)
    bio = io.BytesIO()
    with wave.open(bio, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(sig_i16.tobytes())
    return bio.getvalue()

class VCInference:
    def __init__(self, model_path: str = "microsoft/speecht5_vc", device: Optional[str] = None):
        self.device = _to_device(device)
        self.processor = SpeechT5Processor.from_pretrained(model_path)
        self.model = SpeechT5ForSpeechToSpeech.from_pretrained(model_path).to(self.device).eval()
        self.vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(self.device).eval()
        self.embedder = build_speaker_embedder(self.device)

    @torch.inference_mode()
    def convert(
        self,
        src_wave: np.ndarray,
        ref_wave: Optional[np.ndarray] = None,
        sample_rate: int = 16000,
        return_wav_bytes: bool = False,
    ):
        proc = self.processor(
            audio=src_wave,
            audio_target=src_wave,  # not used for inference but required by processor signature
            sampling_rate=sample_rate,
            return_attention_mask=False,
        )
        inp = torch.tensor(proc["input_values"][0], dtype=torch.float32, device=self.device).unsqueeze(0)

        # Speaker embedding from ref (or from src if not provided)
        ref = ref_wave if ref_wave is not None else src_wave
        spk = torch.tensor(self.embedder(ref), dtype=torch.float32, device=self.device).unsqueeze(0)

        speech = self.model.generate_speech(inp, spk, vocoder=self.vocoder)
        out = speech.detach().cpu().numpy()

        if return_wav_bytes:
            return _float_to_wav_bytes(out, sample_rate)
        return out
