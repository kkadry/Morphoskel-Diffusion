# %%
import torch
from prdc import compute_prdc
import xarray as xr
import numpy as np
from scipy import linalg
import wandb
import pandas as pd
from omegaconf import DictConfig, ListConfig
from omegaconf.errors import InterpolationKeyError


def log_wandb_metrics(metrics, logger, train=False):
    scalar_metrics = dict(metrics["anatomic_summary_metrics"].to_pandas())
    if train:
        logger.experiment.log(scalar_metrics)
    else:
        # Convert scalar_metrics to a pandas DataFrame
        df = pd.DataFrame([scalar_metrics])
        # Log the DataFrame as a table
        logger.experiment.log({"metrics_table": wandb.Table(dataframe=df)})
    # Plotting images
    img_dict = metrics["anatomic_plots"]["imgs"]
    for img_name, img_tensor in img_dict.items():
        logger.experiment.log({img_name: wandb.Image(img_tensor.numpy())})


def flatten_omegaconf(config, parent_key="", sep="."):
    items = {}
    for k, v in config.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        try:
            if isinstance(v, DictConfig):
                items.update(flatten_omegaconf(v, new_key, sep=sep))
            elif isinstance(v, ListConfig):
                for idx, item in enumerate(v):
                    list_key = f"{new_key}{sep}{idx}"
                    if isinstance(item, DictConfig):
                        items.update(flatten_omegaconf(item, list_key, sep=sep))
                    else:
                        items[list_key] = item
            else:
                items[new_key] = v
        except InterpolationKeyError as e:
            # Handle interpolation errors gracefully by skipping unresolved keys
            items[new_key] = f"Interpolation error: {e}"
    return items


def combine_metrics(metrics_1: xr.Dataset, metrics_2: xr.Dataset) -> xr.Dataset:
    """
    Combine two xarray Datasets containing metrics.

    This function combines two metric datasets, handling cases where one or both might be empty.
    If one dataset is empty, it returns the non-empty dataset. If both have data, it concatenates
    them along the 'n' dimension.

    Args:
        metrics_1 (xr.Dataset): The first metrics dataset.
        metrics_2 (xr.Dataset): The second metrics dataset.

    Returns:
        xr.Dataset: A combined dataset of metrics. If one input is empty, returns the non-empty input.
                    If both have data, returns their concatenation along the 'n' dimension.

    Note:
        The function assumes that both inputs are xarray Datasets with compatible structures
        for concatenation along the 'n' dimension when both are non-empty.
    """
    # If either metrics are empty, return the other
    if metrics_1.nbytes == 0:
        return metrics_2
    if metrics_2.nbytes == 0:
        return metrics_1
    # Concatenate the metrics
    return xr.concat([metrics_1, metrics_2], dim="n")


def log_anatomic_summary_metrics(
    metrics_sample: xr.Dataset, metrics_real: xr.Dataset
) -> xr.Dataset:
    """
    Compute and log summary metrics for anatomical data.

    This function calculates summary metrics for morphological, skeletal, and topological features
    by comparing sample metrics with real metrics.

    Args:
        metrics_sample (xr.Dataset): Dataset containing metrics computed from sample data.
        metrics_real (xr.Dataset): Dataset containing metrics computed from real data.

    Returns:
        xr.Dataset: A dataset containing summary metrics for morphological, skeletal, and topological features.

    Note:
        This function relies on separate helper functions to compute specific types of summary metrics.
    """
    summary = xr.Dataset()
    # morphological metrics
    summary.update(compute_morph_summary_metrics(metrics_sample, metrics_real))
    # skeleton metrics
    summary.update(compute_skel_summary_metrics(metrics_sample, metrics_real))
    # topological metrics
    summary.update(compute_topo_summary_metrics(metrics_sample, metrics_real))
    return summary


def log_anatomic_metrics(
    plotter, anatomic_batch_metrics, anatomic_metrics_real, x_hat, seed
):
    # Computing summary metrics at the end of the sampling loop
    if anatomic_metrics_real:
        anatomic_summary_metrics = log_anatomic_summary_metrics(
            anatomic_batch_metrics, anatomic_metrics_real
        )
    else:
        anatomic_summary_metrics = xr.Dataset()
    anatomic_metrics = {
        "anatomic_summary_metrics": anatomic_summary_metrics,
        "anatomic_batch_metrics": anatomic_batch_metrics,
    }
    # Plotting
    anatomic_plots = plotter.log_anatomic_plots(
        metrics=anatomic_metrics, x_hat=x_hat, seed=seed
    )
    anatomic_metrics["anatomic_plots"] = anatomic_plots
    return anatomic_metrics


def compute_morph_summary_metrics(
    metrics_sample: xr.Dataset, metrics_real: xr.Dataset
) -> xr.Dataset:
    """
    Compute summary metrics for morphological features by comparing sample metrics with real metrics.

    This function calculates various morphological summary metrics, including conditional morphological loss,
    3D morphological metrics, and 2D morphological metrics. It processes both sample and real metrics,
    harmonizes them, and computes distribution-based metrics.

    Args:
        metrics_sample (xr.Dataset): Dataset containing morphological metrics computed from sample data.
        metrics_real (xr.Dataset): Dataset containing morphological metrics computed from real data.

    Returns:
        xr.Dataset: A dataset containing summary metrics for morphological features, including:
            - Conditional morphological loss (if applicable)
            - 3D morphological distribution metrics
            - 2D morphological distribution metrics

    Note:
        This function relies on helper functions like `harmonize_metrics` and `compute_morph_dist_metrics`
        to process and compare the metrics.
    """
    summary_metrics = xr.Dataset()
    # Cond metrics
    metrics_loss = metrics_sample.filter_by_attrs(metric_type="morph", loss=True)
    if len(metrics_loss.data_vars) > 0:
        metric_loss = metrics_loss.to_dataarray().values.mean()
        summary_metrics["cond_morph_loss"] = xr.DataArray(
            metric_loss, attrs={"metric_type": "morph", "loss": True, "summary": True}
        )

    # 3D Morph metrics
    metrics_real_3D = metrics_real.filter_by_attrs(
        metric_type="morph", metric_dim=1, log=True
    )
    metrics_sample_3D = metrics_sample.filter_by_attrs(
        metric_type="morph", metric_dim=1, log=True
    )
    metrics_real_3D = harmonize_metrics(metrics_sample_3D, metrics_real_3D)
    # Filter out metrics in real that are not in sample
    if len(metrics_real_3D.data_vars) > 0:
        array_real_3D = metrics_real_3D.to_dataarray().transpose().values
        array_sample_3D = metrics_sample_3D.to_dataarray().transpose().values
        summary_metrics.update(
            compute_morph_dist_metrics(array_real_3D, array_sample_3D, postfix="3D")
        )

    # 2D Morph metrics
    metrics_real_2D = metrics_real.filter_by_attrs(
        metric_type="morph", metric_dim=2, log=True
    )
    metrics_sample_2D = metrics_sample.filter_by_attrs(
        metric_type="morph", metric_dim=2, log=True
    )
    metrics_real_2D = harmonize_metrics(metrics_sample_2D, metrics_real_2D)
    if len(metrics_real_2D.data_vars) > 0:
        array_real_2D = (
            metrics_real_2D.stack({"samples": ["n", "emb_axis"]})
            .to_dataarray()
            .transpose()
            .values
        )
        array_sample_2D = (
            metrics_sample_2D.stack({"samples": ["n", "emb_axis"]})
            .to_dataarray()
            .transpose()
            .values
        )
        summary_metrics.update(
            compute_morph_dist_metrics(array_real_2D, array_sample_2D, postfix="2D")
        )
    return summary_metrics


def harmonize_metrics(metrics_sample, metrics_real):
    common_vars = [
        var for var in metrics_sample.data_vars if var in metrics_real.data_vars
    ]
    metrics_real_filt = metrics_real[common_vars]
    return metrics_real_filt


# compute morph distribution metrics for the batch
def compute_morph_dist_metrics(morph_real, morph_sample, postfix="", nearest_k=5):
    """
    Compute morphological distribution metrics between real and sampled data.

    This function calculates various distribution metrics to compare the morphological features
    of real and sampled data. It includes precision, recall, density, coverage, and Frechet
    Morphological Distance (FMD).

    Args:
        morph_real (numpy.ndarray): Array of morphological features from real data.
        morph_sample (numpy.ndarray): Array of morphological features from sampled data.
        postfix (str, optional): String to append to metric names. Defaults to ''.
        nearest_k (int, optional): Number of nearest neighbors to consider for some metrics. Defaults to 5.

    Returns:
        xarray.Dataset: A dataset containing computed morphological distribution metrics.

    Note:
        If the number of samples in morph_sample is less than or equal to nearest_k,
        an empty dictionary is returned to avoid computational issues.
    """
    # if morph_sample is smaller than nearest_k, return empty metrics
    if morph_sample.shape[0] <= nearest_k:
        return {}
    morph_dist_metrics = xr.Dataset()
    # Apply shape checking and subsampling
    morph_real = check_and_subsample_tensor(morph_real)
    morph_sample = check_and_subsample_tensor(morph_sample)

    # Preprocess the morph features
    features_real_norm, features_sample_norm = preprocess_features(
        morph_real, morph_sample
    )

    # Computing improved precision and recall as well as density and coverage
    prdc_metrics = compute_prdc(
        real_features=features_real_norm,
        fake_features=features_sample_norm,
        nearest_k=nearest_k,
    )
    # loop over and add the metrics to the dataset
    for key, value in prdc_metrics.items():
        morph_dist_metrics[f"{key}_{postfix}"] = xr.DataArray(
            value,
            attrs={
                "metric_type": "morph",
                "dist_dim": 1,
                "dist_metric": True,
                "summary": True,
            },
        )
    # Computing Frechet Morphological Distance
    morph_dist_metrics[f"FMD_{postfix}"] = xr.DataArray(
        compute_FD(features_real_norm, features_sample_norm),
        attrs={
            "metric_type": "morph",
            "dist_dim": 1,
            "dist_metric": True,
            "summary": True,
        },
    )

    return morph_dist_metrics


def preprocess_features(morph_base_df, morph_synth_df):
    # Convert to tensor
    features_base = torch.tensor(morph_base_df)
    features_synth = torch.tensor(morph_synth_df)
    # Compute mean and std
    real_mean = features_base.mean(dim=0)
    real_std = features_base.std(dim=0) + 1e-6

    # Normalizing by mean and std
    features_base_norm = (features_base - real_mean) / real_std
    features_synth_norm = (features_synth - real_mean) / real_std

    return features_base_norm, features_synth_norm


# check and sample the dataframe
def check_and_subsample_tensor(x, max_size=10000):
    if x.shape[0] > max_size:
        indices = torch.randperm(x.shape[0])[:max_size]
        return x[indices]
    return x


# compute summary metrics for the batch
def compute_skel_summary_metrics(metrics_sample, metrics_real):
    summary_metrics = xr.Dataset()
    # Cond metrics
    metric_loss = metrics_sample.filter_by_attrs(metric_type="skel", loss=True)

    if len(metric_loss.data_vars) > 0:
        loss_metric = xr.DataArray(
            metric_loss.to_dataarray().values.mean(),
            attrs={"metric_type": "skel", "loss": True, "summary": True},
        )
        summary_metrics["cond_skel_loss"] = loss_metric
    return summary_metrics


def compute_cond_skel_loss(metrics):
    cond_skel_losses = [
        value for key, value in metrics.items() if "cond_struct_loss" in key
    ]
    if cond_skel_losses:
        return torch.mean(torch.stack(cond_skel_losses))
    else:
        return None  # Return 0 if no matching keys found


def compute_topo_summary_metrics(metrics_sample, metrics_real):
    summary_metrics = xr.Dataset()
    # Cond metrics
    metric_loss = metrics_sample.filter_by_attrs(metric_type="topo", loss=True)
    if len(metric_loss.data_vars) > 0:
        loss_metric = xr.DataArray(
            metric_loss.to_dataarray().values.mean(),
            attrs={"metric_type": "topo", "loss": True, "summary": True},
        )
        summary_metrics["cond_topo_loss"] = loss_metric
    return summary_metrics


def compute_FD(features_1, features_2):
    mu_1, sigma_1 = get_activation_stats(np.array(features_1))
    mu_2, sigma_2 = get_activation_stats(np.array(features_2))
    FID = calculate_frechet_distance(mu_1, sigma_1, mu_2, sigma_2, eps=1e-6)
    return FID


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert (
        mu1.shape == mu2.shape
    ), "Training and test mean vectors have different lengths"
    assert (
        sigma1.shape == sigma2.shape
    ), "Training and test covariances have different dimensions"

    diff = mu1 - mu2
    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = (
            "fid calculation produces singular product; "
            "adding %s to diagonal of cov estimates"
        ) % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


def get_activation_stats(act):
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma
