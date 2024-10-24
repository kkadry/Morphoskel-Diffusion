# %%
import hydra
import sys
import torch
import lightning as L
import wandb
import matplotlib
import gc
from src.utils.logging_utils import flatten_omegaconf, log_wandb_metrics
from hydra.utils import instantiate
from lightning.pytorch.loggers import WandbLogger

# Detecting if python debug mode is on, if so then skip setting backend
if "pydevd" not in sys.modules:
    matplotlib.use("Agg")
    print("setting backend to Agg")

import warnings

warnings.filterwarnings("ignore")

config_path = "configs/"


@hydra.main(version_base=None, config_path=config_path, config_name="sample")
def main(cfg):
    gc.collect()
    # Setup profiler
    logger = WandbLogger(**cfg.wandb, config=flatten_omegaconf(cfg), reinit=False)
    model = instantiate(cfg, _recursive_=False)
    trainer = L.Trainer(**cfg.lightning, logger=logger)
    print("Sampling Start")
    metrics = trainer.predict(
        model=model, dataloaders=model.training_loader, return_predictions=True
    )
    # Saving the metrics
    torch.save(metrics[0], cfg.metrics_path)
    log_wandb_metrics(metrics[0], logger)
    print("Sampling Ended, Shutting down Wandb")
    wandb.finish()
    print("Wandb Shut Down")


if __name__ == "__main__":
    main()
