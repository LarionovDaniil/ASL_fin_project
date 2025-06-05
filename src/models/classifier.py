import os
from datetime import datetime

import matplotlib.pyplot as plt
import mlflow.pytorch
import torch
from pytorch_lightning import LightningModule
from timm import create_model
from torch import nn
from torchmetrics import Accuracy


class SignClassifier(LightningModule):
    def init(self, model_name, num_classes, lr, pretrained):
        super().init()

        self.save_hyperparameters()

        self.model = create_model(
            model_name, pretrained=pretrained, num_classes=num_classes
        )
        self.criterion = nn.CrossEntropyLoss()
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)

        self.train_losses = []
        self.val_losses = []
        self.val_accs = []

        self.train_start_time = None

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.train_losses.append(loss.item())
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = self.val_acc(logits.softmax(dim=-1), y)
        self.val_losses.append(loss.item())
        self.val_accs.append(acc.item())
        self.log("val_loss", loss)
        self.log("val_acc", acc)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def on_train_start(self):
        self.train_start_time = datetime.now()
        os.makedirs("plots", exist_ok=True)
        os.makedirs("model_weights", exist_ok=True)

    def on_train_end(self):
        end_time = datetime.now()

        mlflow.pytorch.log_model(pytorch_model=self.model, artifact_path="model")

        # --- Сохраняем графики ---
        def save_plot(data, name):
            path = os.path.join("plots", f"{name}.png")
            plt.figure()
            plt.plot(data)
            plt.title(name)
            plt.xlabel("Epoch")
            plt.ylabel(name)
            plt.savefig(path)
            plt.close()

        save_plot(self.train_losses, "train_loss")
        save_plot(self.val_losses, "val_loss")
        save_plot(self.val_accs, "val_acc")

        # --- Лог-файл ---
        log_path = os.path.join("plots", "train_log.txt")
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(f"Время начала обучения: {self.train_start_time}\n")
            f.write(f"Время окончания обучения: {end_time}\n")
            f.write(f"Длительность: {end_time - self.train_start_time}\n\n")
            f.write(
                "Train Losses:\n"
                + ", ".join(f"{x:.4f}" for x in self.train_losses)
                + "\n\n"
            )
            f.write(
                "Validation Losses:\n"
                + ", ".join(f"{x:.4f}" for x in self.val_losses)
                + "\n\n"
            )
            f.write(
                "Validation Accuracies:\n"
                + ", ".join(f"{x:.4f}" for x in self.val_accs)
                + "\n"
            )

        print(f"Логи и графики сохранены в {log_path}")

        # --- Сохраняем модель в ONNX ---
        dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
        onnx_path = os.path.join("model_weights", "model.onnx")
        torch.onnx.export(
            self.model,
            dummy_input,
            onnx_path,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            opset_version=11,
        )
        print(f"ONNX-модель сохранена: {onnx_path}")
