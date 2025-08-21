import sagemaker
from sagemaker.huggingface import HuggingFace, HuggingFaceModel

session = sagemaker.Session()
role = "arn:aws:iam::<ACCOUNT_ID>:role/<SageMakerRole>"   # TODO

# --- Train ---
hub = {
    "HF_TASK": "custom",
}
hyperparameters = {
    "dataset_name": "cmu_arctic",
    "src_spk": "awb",
    "tgt_spk": "clb",
    "val_ratio": 0.05,
    "max_train_pairs": 200,      # small for a quick run
    "use_precomputed_xvectors": True,
    "xvector_mode": "average",
    "max_steps": 500,
    "per_device_train_batch_size": 2,
    "per_device_eval_batch_size": 1,
    "grad_accum": 4,
    "lr": 1e-5,
    "warmup_steps": 50,
    "eval_every": 100,
    "save_every": 250,
    "seed": 42,
}

estimator = HuggingFace(
    entry_point="train.py",
    source_dir="sagemaker",
    instance_type="ml.g5.2xlarge",
    instance_count=1,
    role=role,
    transformers_version="4.42",
    pytorch_version="2.3",
    py_version="py310",
    hyperparameters=hyperparameters,
    environment=hub,
)

estimator.fit()

# --- Deploy ---
model = HuggingFaceModel(
    model_data=estimator.model_data,
    role=role,
    transformers_version="4.42",
    pytorch_version="2.3",
    py_version="py310",
    entry_point="inference.py",
    source_dir="sagemaker",
    # Install extra deps for hosting if needed:
    requirements="requirements.txt",
)
predictor = model.deploy(
    initial_instance_count=1,
    instance_type="ml.g5.xlarge",
)

# --- Invoke ---
import base64, json
def read_wav_b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

payload = {
    "src_wav_b64": read_wav_b64("path/to/source.wav"),
    "ref_wav_b64": read_wav_b64("path/to/target_ref.wav"),
}
result = predictor.predict(payload)
wav_b64 = json.loads(result)["wav_b64"]
with open("converted_sagemaker.wav", "wb") as f:
    f.write(base64.b64decode(wav_b64))

# predictor.delete_endpoint()  # clean up when done
