import mlflow.pyfunc
import torch
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
        # model_input — путь к изображению или PIL.Image
        image = Image.open(model_input.iloc[0]).convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(tensor)
            pred = output.argmax(dim=1).item()

        return [pred]
