from typing import Any, Dict, List, Optional
import numpy as np
import torch
from datasets import Audio
from transformers import SpeechT5Processor, SpeechT5ForSpeechToSpeech
from .data import cmu_key

def make_prepare_fn(
    processor: SpeechT5Processor,
    embed_fn=None,
    xvec_map: Optional[Dict[str, np.ndarray]] = None,
    avg_xvec_by_spk: Optional[Dict[str, np.ndarray]] = None,
    xvector_mode: str = "average",
):
    assert xvector_mode in ("average", "utterance")

    def _prepare(batch):
        src = batch["src_wav"]
        tgt = batch["tgt_wav"]
        ex = processor(
            audio=src["array"],
            audio_target=tgt["array"],
            sampling_rate=src["sampling_rate"],
            return_attention_mask=False,
        )
        ex["input_values"] = ex["input_values"][0]
        ex["labels"] = ex["labels"][0]

        spk_emb = None
        if xvec_map is not None:
            spk = batch.get("tgt_spk", None)
            uttid = batch.get("uttid", None)
            if xvector_mode == "average" and avg_xvec_by_spk is not None and spk in avg_xvec_by_spk:
                spk_emb = avg_xvec_by_spk[spk]
            elif xvector_mode == "utterance" and uttid is not None and spk is not None:
                key = cmu_key(spk, uttid)
                spk_emb = xvec_map.get(key, None)

        if spk_emb is None and embed_fn is not None:
            ref = batch.get("spk_ref_wav", tgt)
            spk_emb = embed_fn(ref["array"])

        if spk_emb is None:
            raise RuntimeError("Could not obtain speaker embedding (x-vector).")

        ex["speaker_embeddings"] = spk_emb
        return ex

    return _prepare


class VCDataCollatorWithPadding:
    def __init__(self, processor: Any, model: SpeechT5ForSpeechToSpeech):
        self.processor = processor
        self.model = model

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        in_feats = [{"input_values": f["input_values"]} for f in features]
        lab_feats = [{"input_values": f["labels"]} for f in features]
        spk_feats = [f["speaker_embeddings"] for f in features]

        batch = self.processor.pad(
            input_values=in_feats,
            labels=lab_feats,
            return_tensors="pt",
        )

        # Mask padded label frames with -100
        batch["labels"] = batch["labels"].masked_fill(
            batch["decoder_attention_mask"].unsqueeze(-1).ne(1), -100
        )

        # Align to reduction factor
        red = getattr(self.model.config, "reduction_factor", 1)
        if red > 1:
            target_lengths = torch.tensor([len(f["input_values"]) for f in lab_feats], dtype=torch.long)
            target_lengths = target_lengths - (target_lengths % red)
            max_len = int(target_lengths.max().item())
            batch["labels"] = batch["labels"][:, :max_len]
            batch["decoder_attention_mask"] = batch["decoder_attention_mask"][:, :max_len]

        del batch["decoder_attention_mask"]
        batch["speaker_embeddings"] = torch.tensor(np.stack(spk_feats), dtype=torch.float32)
        return batch
