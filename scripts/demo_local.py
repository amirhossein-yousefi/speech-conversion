"""
Local demo:
- Loads a (fine-tuned) checkpoint directory OR the base 'microsoft/speecht5_vc'
- Converts a source wav into target voice using a reference wav
Usage:
  python scripts/demo_local.py --checkpoint outputs/speecht5_vc_ft --src path/to/src.wav --ref path/to/ref.wav --out demo.wav
"""
import argparse
import soundfile as sf
from datasets import Audio
from speecht5_vc.inference import VCInference

def read_wav(path, sr=16000):
    audio = Audio(sampling_rate=sr)
    ex = audio.decode_example({"path": path})
    return ex["array"], ex["sampling_rate"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, default="microsoft/speecht5_vc")
    ap.add_argument("--src", type=str, required=True)
    ap.add_argument("--ref", type=str, required=True)
    ap.add_argument("--out", type=str, default="converted.wav")
    args = ap.parse_args()

    src, sr = read_wav(args.src, 16000)
    ref, _  = read_wav(args.ref, 16000)

    vc = VCInference(model_path=args.checkpoint)
    out = vc.convert(src_wave=src, ref_wave=ref, sample_rate=sr, return_wav_bytes=False)
    sf.write(args.out, out, sr)
    print(f"Wrote {args.out}")

if __name__ == "__main__":
    main()
