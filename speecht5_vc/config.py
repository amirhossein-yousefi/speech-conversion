from dataclasses import dataclass
from typing import Optional

@dataclass
class DataConfig:
    # Option A: built-in CMU ARCTIC
    dataset_name: str = "cmu_arctic"
    src_spk: str = "awb"
    tgt_spk: str = "clb"
    val_ratio: float = 0.05
    max_train_pairs: Optional[int] = None

    # Option B: CSV manifests
    train_csv: Optional[str] = None
    eval_csv: Optional[str] = None

    # Speaker embeddings
    use_precomputed_xvectors: bool = True
    xvector_mode: str = "average"  # "average" | "utterance"

@dataclass
class TrainConfig:
    output_dir: str = "outputs/speecht5_vc_ft"
    max_steps: int = 2000
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 1
    grad_accum: int = 8
    lr: float = 1e-5
    warmup_steps: int = 500
    seed: int = 42
    save_every: int = 1000
    eval_every: int = 1000
    fp16: bool = False
    bf16: bool = False

@dataclass
class InferenceConfig:
    device: Optional[str] = None  # "cuda" | "cpu" | None -> auto
    sample_rate: int = 16000
