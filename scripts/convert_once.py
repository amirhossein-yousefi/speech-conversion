import argparse
import soundfile as sf
from datasets import Audio
from transformers import SpeechT5Processor, SpeechT5ForSpeechToSpeech, SpeechT5HifiGan
import torch
from speecht5_vc.embeddings import build_speaker_embedder

def read_wav(path, sr=16000):
    audio = Audio(sampling_rate=sr)
    ex = audio.decode_example({"path": path})
    return ex["array"], ex["sampling_rate"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True, help="Path or HF id")
    ap.add_argument("--src", type=str, required=True)
    ap.add_argument("--ref", type=str, required=True)
    ap.add_argument("--out", type=str, default="converted.wav")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = SpeechT5Processor.from_pretrained(args.checkpoint)
    model = SpeechT5ForSpeechToSpeech.from_pretrained(args.checkpoint).to(device).eval()
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(device).eval()
    embed = build_speaker_embedder(device)

    src, sr = read_wav(args.src, 16000)
    ref, _  = read_wav(args.ref, 16000)

    proc = processor(audio=src, audio_target=src, sampling_rate=sr, return_attention_mask=False)
    inp = torch.tensor(proc["input_values"][0], dtype=torch.float32, device=device).unsqueeze(0)
    spk = torch.tensor(embed(ref), dtype=torch.float32, device=device).unsqueeze(0)

    with torch.no_grad():
        out = model.generate_speech(inp, spk, vocoder=vocoder).cpu().numpy()
    sf.write(args.out, out, sr)
    print(f"Wrote {args.out}")

if __name__ == "__main__":
    main()
