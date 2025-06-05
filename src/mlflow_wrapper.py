import mlflow.pyfunc
import pandas as pd
import torch
from mlflow.models.signature import infer_signature
from PIL import Image
from torchvision import transforms

from src.models.classifier import SignClassifier


class SignModelWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.model = SignClassifier.load_from_checkpoint(
            context.artifacts["checkpoint_path"],
            model_name="tf_efficientnetv2_s",
            num_classes=20,
            lr=1e-3,
            pretrained=False,
        )
        self.model.eval()
        self.device = torch.device(
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
        self.model.to(self.device)

        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]
        )

    def predict(self, context, model_input):
        decoding_dict = {
            0: "Number 1",
            1: "Number 2",
            2: "Number 3",
            3: "Number 4",
            4: "Number 5",
            5: "Letter A",
            6: "Letter B",
            7: "Letter C",
            8: "Letter D",
            9: "Letter F",
            10: "Letter G",
            11: "Letter H",
            12: "Letter I",
            13: "Letter J",
            14: "Letter K",
            15: "Letter L",
            16: "Letter R",
            17: "Letter S",
            18: "Letter W",
            19: "Letter Y",
        }
        image_path = model_input.iloc[0, 0]
        image = Image.open(image_path).convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(tensor)
            pred = output.argmax(dim=1).item()

        return decoding_dict[pred]


if __name__ == "main":
    checkpoint_path = "checkpoints/best-checkpoint.ckpt"

    example_df = pd.DataFrame(
        {
            "image_path": [
                "data_storage/images/images/test/0ab99c38c46c01f05cdb8f63690a9a4e.jpg"
            ]
        }
    )

    mlflow.pyfunc.save_model(
        path="mlruns_model/sign_model",
        python_model=SignModelWrapper(),
        artifacts={"checkpoint_path": checkpoint_path},
        signature=infer_signature(example_df),
        input_example=example_df,
    )

    print("Модель сохранена в формате MLflow PyFunc и готова к деплою.")
