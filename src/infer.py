from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
from PIL import Image
from torchvision import transforms

from src.models.classifier import SignClassifier


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # Расшифровка классов
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

    # Проверка путей
    image_path = Path(cfg.infer.image_path).resolve()
    checkpoint_path = Path(cfg.infer.checkpoint_path).resolve()

    if not image_path.exists():
        raise FileNotFoundError(f"Изображение не найдено: {image_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Чекпойнт не найден: {checkpoint_path}")

    # Трансформации
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    # Подготовка изображения
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)

    # Загрузка модели
    model = SignClassifier.load_from_checkpoint(
        checkpoint_path,
        model_name=cfg.model.model_name,
        num_classes=cfg.model.num_classes,
        lr=cfg.train.learning_rate,
        pretrained=False,
    )
    model.eval()

    # Отправляем модель и данные на один и тот же девайс
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    input_tensor = input_tensor.to(device)

    # Предсказание
    with torch.no_grad():
        output = model(input_tensor)
        pred_idx = output.argmax(dim=1).item()
        class_name = decoding_dict[pred_idx]

    print(f"Предсказано: {class_name} (index: {pred_idx})")


if __name__ == "__main__":
    main()
