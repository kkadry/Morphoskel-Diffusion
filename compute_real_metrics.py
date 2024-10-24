import hydra
import pandas as pd
import numpy as np
import torch
from hydra.utils import instantiate

config_path = "configs/"


@hydra.main(
    version_base=None, config_path=config_path, config_name="compute_real_metrics"
)
def main(cfg):
    model = instantiate(cfg, _recursive_=False).cuda()
    print("Computing anatomic metrics over train set")
    metrics_train = model.compute_real_metrics(
        nsamples=len(model.training_loader), loader=model.training_loader
    )
    print("Computing anatomic metrics over validation set")
    metrics_val = model.compute_real_metrics(
        nsamples=len(model.val_loader), loader=model.val_loader
    )

    # #Saving the metrics
    print(f"Saving metrics to {cfg.real_metrics_folder}{cfg.real_metrics_fname}")
    torch.save(
        metrics_train, f"{cfg.real_metrics_folder}{cfg.real_metrics_fname}_train.pth"
    )
    torch.save(
        metrics_val, f"{cfg.real_metrics_folder}{cfg.real_metrics_fname}_val.pth"
    )

    # Saving quantiles for each metrics
    df = pd.DataFrame(
        [
            {
                "Metric": var_name,
                "Min": np.quantile(da.values, 0.01),
                "Max": np.quantile(da.values, 0.99),
            }
            for var_name, da in metrics_train.data_vars.items()
        ]
    )

    df.to_csv(f"{cfg.real_metrics_folder}{cfg.real_metrics_fname}.csv", index=False)


if __name__ == "__main__":
    main()
