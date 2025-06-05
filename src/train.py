import hydra
import mlflow
from omegaconf import DictConfig
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger

from src.data.datamodule import SignDataModule
from src.models.classifier import SignClassifier


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    seed_everything(cfg.train.seed)

    # Устанавливаем MLflow URI и эксперимент
    mlflow.set_tracking_uri(cfg.train.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.train.mlflow.experiment_name)

    # Логгер
    mlflow_logger = MLFlowLogger(
        experiment_name=cfg.mlflow.experiment_name,
        tracking_uri=cfg.mlflow.tracking_uri,
    )

    # DataModule
    datamodule = SignDataModule(
        data_csv_path=cfg.train.data_csv_path,
        images_dir=cfg.train.images_dir,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        val_split=cfg.train.val_split,
        seed=cfg.train.seed,
    )

    # Model
    model = SignClassifier(
        model_name=cfg.model.model_name,
        num_classes=cfg.model.num_classes,
        lr=cfg.train.learning_rate,
        pretrained=cfg.model.pretrained,
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_acc",
        mode="max",
        save_top_k=1,
        dirpath="checkpoints",
        filename="best-checkpoint",
    )

    # Trainer
    trainer = Trainer(
        max_epochs=cfg.train.max_epochs,
        logger=mlflow_logger,
        log_every_n_steps=10,
        accelerator="auto",
        devices=1,
        callbacks=[checkpoint_callback],
    )

    print("Модель начала обучаться.")
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
