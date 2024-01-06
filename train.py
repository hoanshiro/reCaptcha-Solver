import pytorch_lightning as pl
import torch
from pytorch_lightning import loggers as pl_loggers

from config import CFG
from dataset import ReCaptchaDataset
from model import ReCaptchaModel

if __name__ == "__main__":
    ReCaptchaData = ReCaptchaDataset(CFG)
    model = ReCaptchaModel(CFG)
    tb_logger = pl_loggers.TensorBoardLogger(save_dir="lightning_logs/")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kw_trainer = {"gpus": 1} if device.type == "cuda" else {"accelerator": "cpu"}
    print("Using", kw_trainer)
    trainer = pl.Trainer(
        max_epochs=10, log_every_n_steps=5, logger=tb_logger, **kw_trainer
    )
    trainer.fit(model, ReCaptchaData)
