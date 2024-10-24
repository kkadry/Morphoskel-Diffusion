# %%
import warnings

import lightning as L
import hydra
from hydra.utils import instantiate
from lightning.pytorch.loggers import WandbLogger
from monai.utils import set_determinism

from src.utils.logging_utils import flatten_omegaconf

warnings.filterwarnings("ignore")
seed = 42
set_determinism(seed=seed)


@hydra.main(version_base=None, config_path="configs/", config_name="train_model")
def main(cfg):
    # Setting up the AnatomyEngine model
    model = instantiate(cfg, _recursive_=False)
    wandb_logger = WandbLogger(**cfg.wandb, config=flatten_omegaconf(cfg), reinit=False)
    # Trainer
    trainer = L.Trainer(**cfg.lightning, logger=wandb_logger)
    # Training
    trainer.fit(
        model=model,
        train_dataloaders=model.training_loader,
        val_dataloaders=model.val_loader,
    )


if __name__ == "__main__":
    main()
