import os

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from config import CFG
from utils.augmentation import SmallNoiseImageAugmentation


class ReCaptchaDataset(pl.LightningDataModule):
    def __init__(self, CFG):
        super().__init__()
        self.config = CFG
        self.augmentation = SmallNoiseImageAugmentation()

    def prepare_data(self):
        self.train_dir = os.path.join(self.config["data_dir"], "train/")
        self.test_dir = os.path.join(self.config["data_dir"], "test/")
        self.val_dir = os.path.join(self.config["data_dir"], "val/")

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_set = ImageFolder(
                root=self.train_dir, transform=self.augmentation
            )
            self.val_set = ImageFolder(root=self.val_dir, transform=self.augmentation)
            self.val_set.class_to_idx = self.train_set.class_to_idx.copy()
            self.classes = self.train_set.classes

        if stage == "test" or stage is None:
            self.test_set = ImageFolder(root=self.test_dir, transform=self.augmentation)
            self.test_set.class_to_idx = self.train_set.class_to_idx.copy()

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_set,
            batch_size=self.config["batch_size"],
            num_workers=self.config["num_workers"],
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_set,
            batch_size=self.config["batch_size"],
            num_workers=self.config["num_workers"],
            shuffle=True,
        )


if __name__ == "__main__":
    ReCaptchaData = ReCaptchaDataset(CFG)
    ReCaptchaData.prepare_data()
    ReCaptchaData.setup()
    test = ReCaptchaData.train_dataloader()
    print(ReCaptchaData.classes)
    print(next(iter(test)))
