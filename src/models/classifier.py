import pytorch_lightning as pl
import timm
import torch
import torch.nn as nn


class SignClassifier(pl.LightningModule):
    def __init__(
        self, model_name: str, num_classes: int, lr: float, pretrained: bool = True
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = timm.create_model(
            model_name, pretrained=pretrained, num_classes=num_classes
        )
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, _):
        images, labels = batch
        logits = self(images)
        loss = self.criterion(logits, labels)
        acc = (logits.argmax(dim=1) == labels).float().mean()
        self.log("train_loss", loss)
        self.log("train_acc", acc)
        return loss

    def validation_step(self, batch, _):
        images, labels = batch
        logits = self(images)
        loss = self.criterion(logits, labels)
        acc = (logits.argmax(dim=1) == labels).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
