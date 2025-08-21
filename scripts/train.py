import argparse
import os
import torch
from transformers import (
    SpeechT5Processor,
    SpeechT5ForSpeechToSpeech,
    Seq2SeqTrainingArguments,
    set_seed,
)

from speecht5_vc.config import DataConfig, TrainConfig
from speecht5_vc.data import read_manifest, load_cmu_arctic_pairs, maybe_load_cmu_xvectors
from speecht5_vc.embeddings import build_speaker_embedder
from speecht5_vc.prepare import make_prepare_fn, VCDataCollatorWithPadding
from speecht5_vc.trainer import VCTrainer

def parse_args():
    p = argparse.ArgumentParser()
    # Data
    p.add_argument("--dataset_name", type=str, default="cmu_arctic")
    p.add_argument("--src_spk", type=str, default="awb")
    p.add_argument("--tgt_spk", type=str, default="clb")
    p.add_argument("--val_ratio", type=float, default=0.05)
    p.add_argument("--max_train_pairs", type=int, default=None)
    p.add_argument("--train_csv", type=str, default=None)
    p.add_argument("--eval_csv", type=str, default=None)
    p.add_argument("--use_precomputed_xvectors", action="store_true", default=True)
    p.add_argument("--xvector_mode", type=str, default="average", choices=["average", "utterance"])

    # Train
    p.add_argument("--output_dir", type=str, default="outputs/speecht5_vc_ft")
    p.add_argument("--max_steps", type=int, default=2000)
    p.add_argument("--per_device_train_batch_size", type=int, default=2)
    p.add_argument("--per_device_eval_batch_size", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--warmup_steps", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_every", type=int, default=1000)
    p.add_argument("--eval_every", type=int, default=1000)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--bf16", action="store_true")
    return p.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_vc")
    model = SpeechT5ForSpeechToSpeech.from_pretrained("microsoft/speecht5_vc")
    model.config.use_cache = False

    # Data
    if args.dataset_name.lower() == "cmu_arctic":
        train_ds, eval_ds = load_cmu_arctic_pairs(
            src_spk=args.src_spk, tgt_spk=args.tgt_spk,
            val_ratio=args.val_ratio, seed=args.seed,
            max_train_pairs=args.max_train_pairs,
        )
    else:
        if not args.train_csv:
            raise ValueError("Provide --train_csv or set --dataset_name cmu_arctic.")
        train_ds = read_manifest(args.train_csv)
        eval_ds = read_manifest(args.eval_csv) if args.eval_csv else train_ds.train_test_split(test_size=0.05, seed=args.seed)["test"]

    # Speaker embeddings
    xvec_map, avg_by_spk = (None, None)
    embed_fn = None
    if args.use_precomputed_xvectors and args.dataset_name.lower() == "cmu_arctic":
        xvec_map, avg_by_spk = maybe_load_cmu_xvectors(average_by_speaker=True)
        if xvec_map is None:
            print("[WARN] Could not load precomputed x-vectors; falling back to SpeechBrain at runtime.")
    if xvec_map is None:
        embed_fn = build_speaker_embedder(device=device)

    # Map
    prepare_fn = make_prepare_fn(
        processor,
        embed_fn=embed_fn,
        xvec_map=xvec_map,
        avg_xvec_by_spk=avg_by_spk,
        xvector_mode=args.xvector_mode,
    )
    train_proc = train_ds.map(prepare_fn, remove_columns=train_ds.column_names)
    eval_proc  = eval_ds.map(prepare_fn,  remove_columns=eval_ds.column_names)

    data_collator = VCDataCollatorWithPadding(processor=processor, model=model)

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=False,
        fp16=args.fp16,
        bf16=args.bf16,
        eval_strategy="steps",
        eval_steps=args.eval_every,
        save_steps=args.save_every,
        logging_steps=25,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        greater_is_better=False,
        label_names=["labels"],
        remove_unused_columns=False,
        save_total_limit=2,
        logging_dir="logs/training-logs"
    )

    trainer = VCTrainer(
        args=training_args,
        model=model,
        train_dataset=train_proc,
        eval_dataset=eval_proc,
        data_collator=data_collator,
        tokenizer=processor,
    )

    trainer.train()
    trainer.save_model(args.output_dir)

    # Optional smoke test
    try:
        from transformers import SpeechT5HifiGan
        ex = eval_proc[0]
        vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
        with torch.no_grad():
            inp = torch.tensor(ex["input_values"], dtype=torch.float32).unsqueeze(0).to(device)
            spk = torch.tensor(ex["speaker_embeddings"], dtype=torch.float32).unsqueeze(0).to(device)
            model.to(device).eval()
            speech = model.generate_speech(inp, spk, vocoder=vocoder)
        os.makedirs(args.output_dir, exist_ok=True)
        import soundfile as sf
        sf.write(os.path.join(args.output_dir, "sample_converted.wav"), speech.cpu().numpy(), 16000)
        print(f"[OK] Wrote {os.path.join(args.output_dir, 'sample_converted.wav')}")
    except Exception as e:
        print(f"[WARN] Inference smoke test skipped: {e}")

if __name__ == "__main__":
    main()
