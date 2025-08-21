import torch
import torch.nn.functional as F
from transformers import Seq2SeqTrainer

class VCTrainer(Seq2SeqTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Let the model try to compute loss (if it supports it)
        outputs = model(**inputs)

        # 1) Robustly fetch loss if present (no KeyError)
        loss = getattr(outputs, "loss", None)
        if loss is None and isinstance(outputs, dict):
            loss = outputs.get("loss", None)

        # 2) Fallback: masked L1 on spectrogram vs labels
        if loss is None:
            preds = getattr(outputs, "spectrogram", None)
            if preds is None and isinstance(outputs, dict):
                preds = outputs.get("spectrogram", None)
            if preds is None:
                raise RuntimeError("Model outputs have no 'spectrogram' to compute fallback loss.")

            labels = inputs["labels"]
            # Align time and mel dims defensively
            T = min(preds.size(1), labels.size(1))
            C = min(preds.size(-1), labels.size(-1))
            preds = preds[:, :T, :C]
            labels = labels[:, :T, :C]

            # Prefer decoder_attention_mask; also respect -100 sentinels
            dam = inputs.get("decoder_attention_mask", None)
            if dam is not None:
                # dam: [B, T] -> [B, T, 1] -> broadcast to [B, T, C]
                valid = dam[:, :T].unsqueeze(-1).bool().expand(-1, -1, C)
            else:
                valid = torch.ones_like(labels, dtype=torch.bool)

            valid = valid & labels.ne(-100)

            if valid.any():
                loss = F.l1_loss(preds[valid], labels[valid])
            else:
                # No valid targets? fall back to unmasked L1 to avoid NaNs
                loss = F.l1_loss(preds, labels)

        return (loss, outputs) if return_outputs else loss
