import argparse
import os
import torch
from scripts.train import main as local_train_main

# We re-use scripts/train.py's CLI so SageMaker can pass hyperparameters directly.
# This file just adapts output_dir to SM_MODEL_DIR if not provided.

if __name__ == "__main__":
    # SageMaker passes args to entry point; we'll read them via scripts/train.py
    # Ensure default output_dir honors SageMaker's SM_MODEL_DIR
    sm_model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
    os.makedirs(sm_model_dir, exist_ok=True)

    # Let scripts/train.py parse args and run training.
    # Users can still set --output_dir explicitly via hyperparameters; if not set,
    # scripts/train.py default will be used, but we prefer SM_MODEL_DIR:
    # To enforce SM default, we can prepend it when not provided:
    import sys
    if "--output_dir" not in sys.argv:
        sys.argv += ["--output_dir", sm_model_dir]

    local_train_main()
