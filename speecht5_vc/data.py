from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict, Audio, concatenate_datasets

# ---------- CSV manifest ----------
def read_manifest(csv_path: str) -> Dataset:
    """CSV columns: src_wav, tgt_wav [, spk_ref_wav]."""
    df = pd.read_csv(csv_path)
    cols = {c.lower(): c for c in df.columns}
    required = {"src_wav", "tgt_wav"}
    missing = required - set(cols)
    if missing:
        raise ValueError(f"CSV must contain {sorted(required)}; found {list(df.columns)}")

    df = df.rename(columns={cols["src_wav"]: "src_wav", cols["tgt_wav"]: "tgt_wav"})
    if "spk_ref_wav" in cols:
        df = df.rename(columns={cols["spk_ref_wav"]: "spk_ref_wav"})
    else:
        df["spk_ref_wav"] = df["tgt_wav"]

    ds = Dataset.from_pandas(df, preserve_index=False)
    for col in ["src_wav", "tgt_wav", "spk_ref_wav"]:
        ds = ds.cast_column(col, Audio(sampling_rate=16000))
    return ds

# ---------- CMU ARCTIC helpers ----------
def _safe_concat_all(dsd: Union[Dataset, DatasetDict]) -> Dataset:
    if isinstance(dsd, Dataset):
        return dsd
    if "default" in dsd:
        return dsd["default"]
    return concatenate_datasets([dsd[k] for k in dsd.keys()])

def _add_uttid_column(ds: Dataset) -> Dataset:
    def get_uttid(example):
        fname = example["file"]
        example["uttid"] = fname.split("-")[-1].split(".")[0]  # a0001
        return example
    return ds.map(get_uttid)

def cmu_key(speaker: str, uttid: str) -> str:
    return f"cmu_us_{speaker}_arctic-wav-arctic_{uttid}"

def load_cmu_arctic_pairs(
    src_spk: str,
    tgt_spk: str,
    val_ratio: float,
    seed: int,
    max_train_pairs: Optional[int] = None,
) -> Tuple[Dataset, Dataset]:
    dsd = load_dataset("MikhailT/cmu-arctic")  # speaker, file, text, audio
    ds_all = _safe_concat_all(dsd)
    ds_all = ds_all.cast_column("audio", Audio(sampling_rate=16000))
    ds_all = _add_uttid_column(ds_all)

    src = ds_all.filter(lambda e: e["speaker"] == src_spk)
    tgt = ds_all.filter(lambda e: e["speaker"] == tgt_spk)

    src_ids, tgt_ids = set(src["uttid"]), set(tgt["uttid"])
    commons = sorted(src_ids & tgt_ids)
    if len(commons) == 0:
        raise ValueError(f"No parallel overlap between speakers {src_spk} and {tgt_spk}.")

    rng = np.random.default_rng(seed)
    commons = list(commons)
    rng.shuffle(commons)

    n_val = max(1, int(len(commons) * val_ratio))
    val_ids = set(commons[:n_val])
    train_ids = commons[n_val:]
    if max_train_pairs is not None:
        train_ids = train_ids[:max_train_pairs]

    src_index = {u: a for u, a in zip(src["uttid"], src["audio"])}
    tgt_index = {u: a for u, a in zip(tgt["uttid"], tgt["audio"])}

    def make_rows(utt_ids: List[str]):
        rows = []
        for u in utt_ids:
            rows.append({
                "src_wav": src_index[u],
                "tgt_wav": tgt_index[u],
                "spk_ref_wav": tgt_index[u],
                "tgt_spk": tgt_spk,
                "uttid": u,
            })
        return rows

    train_ds = Dataset.from_list(make_rows(train_ids))
    eval_ds  = Dataset.from_list(make_rows(sorted(val_ids)))
    return train_ds, eval_ds

# ---------- Precomputed x-vectors ----------
def maybe_load_cmu_xvectors(average_by_speaker: bool = True):
    """
    Returns:
      - map_key_to_xvec: dict[str, np.ndarray] where key matches cmu_key(speaker, uttid)
      - optional avg_by_spk: dict[str, np.ndarray] if average_by_speaker
    """
    try:
        xds = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
        keys = xds["filename"]
        vecs = xds["xvector"]
        map_key_to_xvec = {k: np.asarray(v, dtype=np.float32) for k, v in zip(keys, vecs)}

        avg_by_spk = None
        if average_by_speaker:
            buckets = {}
            for k, v in map_key_to_xvec.items():
                spk = k.split("_")[2]
                buckets.setdefault(spk, []).append(v)
            avg_by_spk = {spk: np.mean(np.stack(vs, axis=0), axis=0) for spk, vs in buckets.items()}
        return map_key_to_xvec, avg_by_spk
    except Exception:
        return None, None
