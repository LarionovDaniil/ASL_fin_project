import os

import hydra
import torch
from omegaconf import DictConfig

from src.models.classifier import SignClassifier


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # Загружаем модель из чекпоинта
    model = SignClassifier.load_from_checkpoint(
        cfg.checkpoint_path,
        model_name=cfg.model.model_name,
        num_classes=cfg.model.num_classes,
        lr=cfg.learning_rate,
        pretrained=False,
    )
    model.eval()

    # Dummy input
    dummy_input = torch.randn(1, 3, 224, 224)

    # Путь к onnx-файлу
    onnx_path = os.path.join(cfg.output_dir, "model.onnx")
    os.makedirs(cfg.output_dir, exist_ok=True)

    # Экспорт
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )

    print(f"ONNX модель сохранена в: {onnx_path}")


if __name__ == "__main__":
    main()
