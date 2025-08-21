import os
import tempfile
import numpy as np
import torch
import torch.nn.functional as F

def build_speaker_embedder(device: str, model_id: str = "speechbrain/spkrec-xvect-voxceleb"):
    """
    Lazy-load SpeechBrain x-vector encoder.
    Returns: callable (waveform: np.ndarray[T]) -> np.ndarray[512]
    """
    try:
        from speechbrain.inference import EncoderClassifier
        from speechbrain.utils.fetching import LocalStrategy
    except Exception as e:
        raise ImportError(
            "SpeechBrain is required to compute x-vectors on-the-fly. "
            "Install it or run with precomputed xvectors.\n"
            "  pip install -U speechbrain"
        ) from e

    cache_dir = os.path.join(tempfile.gettempdir(), model_id.replace("/", "_"))
    os.makedirs(cache_dir, exist_ok=True)

    speaker_model = EncoderClassifier.from_hparams(
        source=model_id,
        run_opts={"device": device},
        savedir=cache_dir,
        local_strategy=LocalStrategy.COPY,  # avoid symlinks on Windows
    )

    def create_speaker_embedding(waveform: np.ndarray) -> np.ndarray:
        wav = torch.as_tensor(waveform, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.inference_mode():
            emb = speaker_model.encode_batch(wav).squeeze()
            emb = F.normalize(emb, dim=0)
        return emb.detach().cpu().numpy().astype(np.float32)

    return create_speaker_embedding
