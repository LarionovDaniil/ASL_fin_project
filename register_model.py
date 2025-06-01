from pathlib import Path

import mlflow
import mlflow.pyfunc

from src.mlflow_wrapper import SignModelWrapper

mlflow.set_tracking_uri("http://127.0.0.1:8080")
mlflow.set_experiment("sign-gesture")

with mlflow.start_run(run_name="inference-serving"):
    mlflow.pyfunc.log_model(
        artifact_path="sign_model",
        python_model=SignModelWrapper(),
        artifacts={
            "checkpoint_path": str(Path("checkpoints/best-checkpoint.ckpt").resolve())
        },
    )
