# SageMaker Inference Toolkit entry points:
#   model_fn, input_fn, predict_fn, output_fn
import base64
import io
import json
import wave
import numpy as np
import torch
from transformers import SpeechT5Processor, SpeechT5ForSpeechToSpeech, SpeechT5HifiGan
from speecht5_vc.embeddings import build_speaker_embedder

_SAMPLE_RATE = 16000

def _bytes_to_wave_float(b: bytes) -> np.ndarray:
    with wave.open(io.BytesIO(b), "rb") as wf:
        assert wf.getnchannels() == 1, "Only mono WAV supported"
        assert wf.getframerate() == _SAMPLE_RATE, f"Expected {_SAMPLE_RATE} Hz"
        frames = wf.readframes(wf.getnframes())
        data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32767.0
        return data

def _float_to_wav_bytes(signal: np.ndarray, sample_rate: int = _SAMPLE_RATE) -> bytes:
    sig = np.clip(signal, -1.0, 1.0)
    sig_i16 = (sig * 32767.0).astype(np.int16)
    bio = io.BytesIO()
    with wave.open(bio, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(sig_i16.tobytes())
    return bio.getvalue()

def model_fn(model_dir):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = SpeechT5Processor.from_pretrained(model_dir)
    model = SpeechT5ForSpeechToSpeech.from_pretrained(model_dir).to(device).eval()
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(device).eval()
    embedder = build_speaker_embedder(device)
    return {"processor": processor, "model": model, "vocoder": vocoder, "embedder": embedder, "device": device}

def input_fn(request_body, request_content_type="application/json"):
    if request_content_type != "application/json":
        raise ValueError("Only application/json supported")
    payload = json.loads(request_body)
    # Expect base64 wav(s): {"src_wav_b64": "...", "ref_wav_b64": "..."}
    src_b = base64.b64decode(payload["src_wav_b64"])
    src = _bytes_to_wave_float(src_b)
    ref = None
    if "ref_wav_b64" in payload and payload["ref_wav_b64"]:
        ref = _bytes_to_wave_float(base64.b64decode(payload["ref_wav_b64"]))
    return {"src": src, "ref": ref}

@torch.inference_mode()
def predict_fn(inputs, model_artifacts):
    src = inputs["src"]
    ref = inputs["ref"] if inputs["ref"] is not None else src
    processor = model_artifacts["processor"]
    model = model_artifacts["model"]
    vocoder = model_artifacts["vocoder"]
    embedder = model_artifacts["embedder"]
    device = model_artifacts["device"]

    proc = processor(audio=src, audio_target=src, sampling_rate=_SAMPLE_RATE, return_attention_mask=False)
    inp = torch.tensor(proc["input_values"][0], dtype=torch.float32, device=device).unsqueeze(0)
    spk = torch.tensor(embedder(ref), dtype=torch.float32, device=device).unsqueeze(0)
    out = model.generate_speech(inp, spk, vocoder=vocoder).cpu().numpy()
    wav_b = _float_to_wav_bytes(out, _SAMPLE_RATE)
    return {"wav_b64": base64.b64encode(wav_b).decode("utf-8")}

def output_fn(prediction, accept="application/json"):
    return json.dumps(prediction), accept
