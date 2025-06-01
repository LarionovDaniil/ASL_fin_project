from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
from PIL import Image
from torchvision import transforms

from src.models.classifier import SignClassifier


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
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

    image_dir = Path(cfg.infer.image_dir)
    checkpoint_path = Path(cfg.infer.checkpoint_path)

    if not image_dir.exists():
        raise FileNotFoundError(f"Папка не найдена: {image_dir}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Чекпойнт не найден: {checkpoint_path}")

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    model = SignClassifier.load_from_checkpoint(
        checkpoint_path,
        model_name=cfg.model.model_name,
        num_classes=cfg.model.num_classes,
        lr=cfg.train.learning_rate,
        pretrained=False,
    )
    model.eval()

    # Прогон по всем картинкам
    image_paths = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
    if not image_paths:
        print("В папке нет изображений .jpg или .png")
        return

    for img_path in image_paths:
        image = Image.open(img_path).convert("RGB")
        input_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            output = model(input_tensor)
            pred_idx = output.argmax(dim=1).item()
            class_name = decoding_dict[pred_idx]

        print(f"{img_path.name}: {class_name} (index: {pred_idx})")


if __name__ == "__main__":
    main()
