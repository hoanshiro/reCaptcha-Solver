import pytorch_lightning as pl
import timm
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
from torchmetrics.classification import Accuracy, F1Score

from config import CFG


class ReCaptchaModel(pl.LightningModule):
    def __init__(self, CFG):
        super().__init__()
        self.config = CFG
        self.model = timm.create_model(CFG["model_name"], pretrained=True)
        self.model.classifier = nn.Sequential(
            nn.Linear(self.model.classifier.in_features, 512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512, CFG["num_classes"]),
        )
        self.accuracy = Accuracy(num_classes=CFG["num_classes"], task="multiclass")
        self.f1 = F1Score(
            num_classes=CFG["num_classes"], average="macro", task="multiclass"
        )
        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.parameters(),
            lr=self.config["learning_rate"],
            weight_decay=self.config["weight_decay"],
        )
        return optimizer

    def exec(self, batch, type="val"):
        x, y = batch
        logits = self.model(x)
        loss = F.cross_entropy(logits, y)
        self.log(f"{type}_loss", loss, prog_bar=True)
        self.log(f"{type}_acc", self.accuracy(logits, y), prog_bar=True)
        self.log(f"{type}_f1", self.f1(logits, y), prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self.exec(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.exec(batch, "val")

    def test_step(self, batch, batch_idx):
        return self.exec(batch, "test")


if __name__ == "__main__":
    model = ReCaptchaModel(CFG)
    sample_input = torch.randn(1, 3, 224, 224)
    out = model(sample_input)
    print(out.shape)
