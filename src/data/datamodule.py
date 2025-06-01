from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class SignLanguageDataset(Dataset):
    def __init__(self, dataframe, images_dir, transform=None):
        self.df = dataframe
        self.images_dir = Path(images_dir)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = self.images_dir / row["id"]
        label = row["label"]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


class SignDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_csv_path="train.csv",
        images_dir="images/images/train",
        batch_size=32,
        num_workers=4,
        val_split=0.2,
        seed=42,
    ):
        super().__init__()
        self.data_csv_path = data_csv_path
        self.images_dir = images_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.seed = seed

        self.train_transforms = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )

        self.val_transforms = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]
        )

    def setup(self, stage=None):
        df = pd.read_csv(self.data_csv_path)
        train_df, val_df = train_test_split(
            df, test_size=self.val_split, stratify=df["label"], random_state=self.seed
        )

        self.train_dataset = SignLanguageDataset(
            train_df, self.images_dir, transform=self.train_transforms
        )
        self.val_dataset = SignLanguageDataset(
            val_df, self.images_dir, transform=self.val_transforms
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
