# %%
from __future__ import annotations
from contextlib import nullcontext
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import xarray as xr
from src.utils.encoder_utils import count_params, disabled_train, expand_dims_like

from typing import Tuple
from omegaconf import DictConfig


class AnatomicEncoder(nn.Module):
    """
    Encoder that combines morphological, skeletal, and topological encodings.

    Attributes:
        OUTPUT_DIM2KEYS (Dict[int, str]): Mapping of output dimensions to keys.
        KEY2CATDIM (Dict[str, int]): Mapping of keys to concatenation dimensions.
        morph_encoder (Optional[GeneralEncoder]): Encoder for morphological features.
        skel_encoder (Optional[GeneralEncoder]): Encoder for skeletal features.
        topo_encoder (Optional[GeneralEncoder]): Encoder for topological features.
        encoders (nn.ModuleDict): Dictionary of all encoders.
    """

    OUTPUT_DIM2KEYS: Dict[int, str] = {
        2: "vector",
        3: "crossattn",
        4: "concat",
        5: "concat",
    }
    KEY2CATDIM: Dict[str, int] = {"vector": 1, "crossattn": 1, "concat": 1}

    def __init__(
        self,
        morph_encoder: Optional[GeneralEncoder] = None,
        skel_encoder: Optional[GeneralEncoder] = None,
        topo_encoder: Optional[GeneralEncoder] = None,
        **kwargs,
    ):
        """
        Initialize the AnatomicEncoder.

        Args:
            morph_encoder (Optional[GeneralEncoder]): Morphological encoder.
            skel_encoder (Optional[GeneralEncoder]): Skeletal encoder.
            topo_encoder (Optional[GeneralEncoder]): Topological encoder.
            **kwargs: Additional keyword arguments.
        """
        super().__init__()
        self.morph_encoder = morph_encoder
        self.skel_encoder = skel_encoder
        self.topo_encoder = topo_encoder
        self.encoders = nn.ModuleDict(
            {
                "morph": self.morph_encoder,
                "skel": self.skel_encoder,
                "topo": self.topo_encoder,
            }
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the AnatomicEncoder.

        Args:
            batch (Dict[str, torch.Tensor]): Input batch containing data for encoding.

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing various encodings.
        """
        encodings = {}
        for key, encoder in self.encoders.items():
            encodings[f"{key}_encoding"] = encoder(batch) if encoder is not None else {}

        encodings["anatomic_encoding"] = self.combine_encodings([*encodings.values()])

        return encodings

    def combine_encodings(
        self, encodings: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """
        Combine different encodings into a single dictionary.

        Args:
            encodings (List[Dict[str, torch.Tensor]]): List of encoding dictionaries.

        Returns:
            Dict[str, torch.Tensor]: Combined encodings.

        Raises:
            ValueError: If an unexpected dimension is encountered in the embeddings.
        """
        output = {"concat": [], "crossattn": [], "vector": []}

        for encodings in encodings:
            for key, value in encodings.items():
                dim = value.dim()
                if dim in self.OUTPUT_DIM2KEYS:
                    output_key = self.OUTPUT_DIM2KEYS[dim]
                    output[output_key].append(value)
                else:
                    raise ValueError(f"Unexpected dimension {dim} for embedding")

        # Concatenate along appropriate dimensions
        for key, tensors in output.items():
            if tensors:
                cat_dim = self.KEY2CATDIM[key]
                output[key] = torch.cat(tensors, dim=cat_dim)

        # Remove empty lists
        output = {k: v for k, v in output.items() if isinstance(v, torch.Tensor)}

        return output

    def log_anatomic_eval_metrics(
        self, batch_sample: Dict[str, torch.Tensor]
    ) -> xr.Dataset:
        """
        Evaluation metrics for anatomic encoders.

        Args:
            batch_sample (Dict[str, torch.Tensor]): Sample batch for metric computation.

        Returns:
            xr.Dataset: Dataset containing evaluation metrics.
        """
        anatomic_metrics = xr.Dataset()
        for encoder in self.encoders.values():
            if encoder is not None:
                anatomic_metrics = anatomic_metrics.merge(
                    encoder.log_eval_metrics(batch_sample)
                )
        return anatomic_metrics

    def log_anatomic_cond_metrics(
        self, batch_sample: Dict[str, torch.Tensor], batch_seed: Dict[str, torch.Tensor]
    ) -> xr.Dataset:
        """
        Log conditional metrics for anatomic encoders.

        Args:
            batch_sample (Dict[str, torch.Tensor]): Sample batch for metric computation.
            batch_seed (Dict[str, torch.Tensor]): Seed batch for metric computation.

        Returns:
            xr.Dataset: Dataset containing logged conditional metrics.
        """
        anatomic_metrics = xr.Dataset()
        for encoder in self.encoders.values():
            if encoder is not None:
                anatomic_metrics = anatomic_metrics.merge(
                    encoder.log_cond_metrics(batch_sample, batch_seed)
                )
        return anatomic_metrics


class GeneralEncoder(nn.Module):
    """
    A general encoder class that manages multiple embedders.
    Adapted from https://github.com/Stability-AI/generative-models/blob/main/sgm/modules/encoders/modules.py

    Attributes:
        OUTPUT_DIM2KEYS (dict): Maps output dimensions to keys.
        KEY2CATDIM (dict): Maps keys to concatenation dimensions.
        embedder_tag (str): Tag for the encoder type.
        embedders (nn.ModuleList): List of embedder models.
    """

    OUTPUT_DIM2KEYS = {2: "vector", 3: "crossattn", 4: "concat", 5: "concat"}
    KEY2CATDIM = {"vector": 1, "crossattn": 1, "concat": 1}
    embedder_tag = "neutral"

    def __init__(self, emb_models: DictConfig, **kwargs):
        """
        Initialize the GeneralEncoder.

        Args:
            emb_models (DictConfig): Configuration for embedder models.
            **kwargs: Additional keyword arguments.
        """
        super().__init__()
        self.embedders = self.init_embedders(emb_models)

    def init_embedders(self, emb_models: DictConfig) -> nn.ModuleList:
        """
        Initialize embedder models.

        Args:
            emb_models (DictConfig): Configuration for embedder models.

        Returns:
            nn.ModuleList: List of initialized embedder models.
        """
        embedders = []
        for n, embedder in enumerate(emb_models.values()):
            assert isinstance(
                embedder, AnatomicEmbModel
            ), f"embedder model {embedder.__class__.__name__} has to inherit from AnatomicEmbModel"
            if not embedder.is_trainable:
                embedder.train = disabled_train
                for param in embedder.parameters():
                    param.requires_grad = False
                embedder.eval()
            print(
                f"Initialized {self.embedder_tag} embedder #{n}: {embedder.__class__.__name__} "
                f"with {count_params(embedder, False)} params. Trainable: {embedder.is_trainable}. "
                f"Tags: {embedder.tags if hasattr(embedder, 'tags') else None}"
            )

            if embedder.input_key is None and embedder.input_keys is None:
                raise KeyError(
                    f"need either 'input_key' or 'input_keys' for embedder {embedder.__class__.__name__}"
                )

            embedders.append(embedder)
        return nn.ModuleList(embedders)

    def forward(
        self,
        batch: Dict,
        force_zero_embeddings: Optional[List] = None,
        track_gradients: bool = False,
    ) -> Dict:
        """
        Forward pass of the GeneralEncoder.

        Args:
            batch (Dict): Input batch.
            force_zero_embeddings (Optional[List]): List of embeddings to force to zero.
            track_gradients (bool): Whether to track gradients.

        Returns:
            Dict: Output embeddings.
        """
        output = dict()
        if force_zero_embeddings is None:
            force_zero_embeddings = []
        for embedder in self.embedders:
            if embedder.__class__.__name__ == "NullEmbedder":
                continue
            embedding_context = (
                nullcontext
                if (embedder.is_trainable or track_gradients)
                else torch.no_grad
            )

            with embedding_context():
                if hasattr(embedder, "input_key") and (embedder.input_key is not None):
                    emb_out = embedder(batch[embedder.input_key])
                elif hasattr(embedder, "input_keys"):
                    emb_out = embedder(*[batch[k] for k in embedder.input_keys])
            assert isinstance(
                emb_out, (torch.Tensor, list, tuple)
            ), f"encoder outputs must be tensors or a sequence, but got {type(emb_out)}"

            if not isinstance(emb_out, (list, tuple)):
                emb_out = [emb_out]
            for emb in emb_out:
                out_key = self.OUTPUT_DIM2KEYS[emb.dim()]

                if embedder.ucg_rate > 0.0:
                    emb = (
                        expand_dims_like(
                            torch.bernoulli(
                                (1.0 - embedder.ucg_rate)
                                * torch.ones(emb.shape[0], device=emb.device)
                            ),
                            emb,
                        )
                        * emb
                    )
                if (
                    hasattr(embedder, "input_key")
                    and embedder.input_key in force_zero_embeddings
                ):
                    emb = torch.zeros_like(emb)
                if out_key in output:
                    output[out_key] = torch.cat(
                        (output[out_key], emb), self.KEY2CATDIM[out_key]
                    )
                else:
                    output[out_key] = emb
        return output

    def get_unconditional_conditioning(
        self,
        batch_c: Dict,
        batch_uc: Optional[Dict] = None,
        force_uc_zero_embeddings: Optional[List[str]] = None,
        force_cond_zero_embeddings: Optional[List[str]] = None,
    ) -> Tuple[Dict, Dict]:
        """
        Get unconditional conditioning for the encoder.

        Args:
            batch_c (Dict): Conditional batch.
            batch_uc (Optional[Dict]): Unconditional batch.
            force_uc_zero_embeddings (Optional[List[str]]): List of unconditional embeddings to force to zero.
            force_cond_zero_embeddings (Optional[List[str]]): List of conditional embeddings to force to zero.

        Returns:
            Tuple[Dict, Dict]: Conditional and unconditional outputs.
        """
        if force_uc_zero_embeddings is None:
            force_uc_zero_embeddings = []
        ucg_rates = list()
        for embedder in self.embedders:
            ucg_rates.append(embedder.ucg_rate)
            embedder.ucg_rate = 0.0
        c = self(batch_c, force_cond_zero_embeddings)
        uc = self(batch_c if batch_uc is None else batch_uc, force_uc_zero_embeddings)

        for embedder, rate in zip(self.embedders, ucg_rates):
            embedder.ucg_rate = rate
        return c, uc

    def log_eval_metrics(self, batch: Dict) -> xr.Dataset:
        """
        Log metrics for evaluation.

        Args:
            batch (Dict): Input batch.

        Returns:
            xr.Dataset: Dataset containing evaluation metrics.
        """
        metrics_ds = xr.Dataset()
        for embedder in self.embedders:
            metric_ds_embedder = embedder.log_metric(
                batch[embedder.input_key], extra_attrs={"log": True}, normalize=False
            )
            metrics_ds = xr.merge([metrics_ds, metric_ds_embedder])
        return metrics_ds

    def log_cond_metrics(self, batch_sample: Dict, batch_seed: Dict) -> xr.Dataset:
        """
        Log conditional metrics for all embedders.

        Args:
            batch_sample (Dict): Sample batch.
            batch_seed (Dict): Seed batch.

        Returns:
            xr.Dataset: Dataset containing logged conditional metrics.
        """
        cond_ds = xr.Dataset()
        for embedder in self.embedders:
            if embedder.null is True:
                continue
            metric_type = embedder.attributes["metric_type"]

            sample_metrics = embedder.log_metric(
                batch_sample["x"], extra_attrs={"cond": True}
            )
            seed_metrics = embedder.log_metric(
                batch_seed["x"], extra_attrs={"cond": True}
            )

            for key in sample_metrics.keys():
                sample_metrics[key].attrs["source"] = "sample"
                seed_metrics[key].attrs["source"] = "seed"

                cond_ds[f"cond_{metric_type}_sample_{key}"] = sample_metrics[key]
                cond_ds[f"cond_{metric_type}_seed_{key}"] = seed_metrics[key]

                cond_loss = embedder.compute_cond_loss(
                    torch.tensor(sample_metrics[key].values),
                    torch.tensor(seed_metrics[key].values),
                )
                if cond_loss is not None:
                    cond_ds[f"cond_{metric_type}_loss_{key}"] = xr.DataArray(
                        cond_loss,
                        dims="n",
                        attrs={
                            **embedder.attributes,
                            "tissue": sample_metrics[key].attrs["tissue"],
                            "loss": True,
                            "cond": True,
                        },
                    )

        return cond_ds


class MorphologicalEncoder(GeneralEncoder):
    embedder_tag = "morph"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class SkeletalEncoder(GeneralEncoder):
    embedder_tag = "skel"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class TopologicalEncoder(GeneralEncoder):
    embedder_tag = "topo"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


# %%
def normalize_metric(x, metric_min, metric_max):
    # If no min or max is set, return the input
    if metric_min is None or metric_max is None:
        return x

    # convert to tensor
    metric_min = torch.tensor(metric_min).to(x)
    metric_max = torch.tensor(metric_max).to(x)

    # Reshape emb_min and emb_max to match x's channel dimension
    shape = [1] * x.dim()
    shape[1] = -1  # Set channel dimension
    metric_min = metric_min.view(*shape)
    metric_max = metric_max.view(*shape)

    # Perform channel-wise normalization
    return (x - metric_min) / (metric_max - metric_min)


# Wrapper class for anatomic embedders
class AnatomicEmbModel(torch.nn.Module):
    """
    A wrapper class for anatomic embedders that processes and embeds anatomical metrics.
    This class takes anatomical data as input, computes a metric, normalizes it,
    and then embeds it using a specified tensor embedder.
    """

    def __init__(self, **kwargs):
        """
        Initialize the AnatomicEmbModel.

        Args:
            tensor_embedder (nn.Module): The tensor embedder to use.
            channels (List[int]): Indices of channels to use.
            metric_min (Union[float, List[float]], optional): Minimum value(s) for normalization.
            metric_max (Union[float, List[float]], optional): Maximum value(s) for normalization.
            attributes (Dict[str, Any], optional): Additional attributes. Defaults to {}.
            dim_tags (List[str], optional): Dimension tags for xarray. Defaults to [].
            tissue_keys (Dict[int, str], optional): Mapping of channel indices to tissue names.
            is_trainable (bool, optional): If the model is trainable. Defaults to False.
            ucg_rate (float, optional): Unconditional guidance rate. Defaults to 0.0.
            input_key (Optional[str], optional): Key for single input in batch dictionary.
            input_keys (Optional[List[str]], optional): Keys for multiple inputs in batch dictionary.
        """
        super().__init__()
        self.tensor_embedder = kwargs.get("tensor_embedder")
        self.channels = list(kwargs.get("channels", []))
        self.metric_min = kwargs.get("metric_min", None)
        self.metric_max = kwargs.get("metric_max", None)

        # Logging parameters
        self.attributes = kwargs["logging"].get("attributes", {})
        self.dim_tags = kwargs["logging"].get("dim_tags", [])
        self.tissue_keys = kwargs["logging"].get("tissue_keys", {})
        self.tags = [self.tissue_keys[i] for i in self.channels]

        # Training parameters
        self.null = False
        self.is_trainable = kwargs["training"].get("is_trainable", False)
        self.ucg_rate = kwargs["training"].get("ucg_rate", 0.0)
        self.input_key = kwargs["training"].get("input_key", None)
        self.input_keys = kwargs["training"].get("input_keys", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the AnatomicEmbModel.

        Args:
            x (torch.Tensor): Input tensor of size BCHWD.

        Returns:
            torch.Tensor: Embedded representation of the anatomic metric.

        Steps:
        1. Calculate and normalize the anatomic metric.
        2. Embed the normalized metric using the configured tensor embedder.
        """
        metric = self.compute_metric_norm(x)
        emb = self.tensor_embedder(metric)
        return emb

    def compute_metric_norm(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute and normalize the anatomic metric.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Normalized anatomic metric.
        """
        metric = self.compute_metric(x)
        metric_norm = normalize_metric(metric, self.metric_min, self.metric_max)
        return metric_norm

    def compute_metric(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the anatomic metric.

        Args:
            x (torch.Tensor): Input tensor.

        Raises:
            NotImplementedError: This method should be implemented by subclasses.
        """
        raise NotImplementedError

    def log_metric(
        self, x: torch.Tensor, extra_attrs: Dict = None, normalize: bool = True
    ) -> xr.Dataset:
        """
        Log the computed metric as an xarray Dataset.

        Args:
            x (torch.Tensor): Input tensor.
            extra_attrs (Dict, optional): Additional attributes to include in the dataarray.
            normalize (bool): Whether to normalize the metric before logging.

        Returns:
            xr.Dataset: Dataset containing the computed metric for each tissue.
        """
        metric = self.compute_metric_norm(x) if normalize else self.compute_metric(x)
        metric_ds = xr.Dataset()
        for i, tag in enumerate(self.tags):
            metric_da = xr.DataArray(
                metric[:, i].cpu().detach(),
                dims=self.dim_tags,
                attrs={**self.attributes, "tissue": tag, **(extra_attrs or {})},
            )
            metric_ds[f'{tag}_{self.attributes["metric_name"]}'] = metric_da
        return metric_ds

    def compute_cond_loss(
        self, sample_cond: torch.Tensor, seed_cond: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the conditional loss between a sample and a seed.

        Args:
            sample_cond (torch.Tensor): Sample tensor representing the conditional measurement.
            seed_cond (torch.Tensor): Seed tensor representing the conditional input.

        Returns:
            torch.Tensor: Mean absolute difference between sample and seed.
        """
        difference = torch.abs(sample_cond.float() - seed_cond.float())
        difference = difference.view(difference.size(0), -1)
        return difference.mean(dim=1)


class NullEmbedder(AnatomicEmbModel):
    """
    Null Embedder used to skip the embedding step.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.null = True

    def forward(self, x: torch.Tensor) -> Dict:
        """
        Forward pass that returns an empty dictionary.

        Args:
            x (torch.Tensor): Input tensor (unused).

        Returns:
            Dict: Empty dictionary.
        """
        return {}

    def log_metric(
        self, x: torch.Tensor, extra_attrs: Dict = None, normalize: bool = True
    ) -> xr.Dataset:
        """
        Log metric method that returns an empty xarray Dataset.

        Args:
            x (torch.Tensor): Input tensor (unused).
            extra_attrs (Dict, optional): Additional attributes (unused).
            normalize (bool): Whether to normalize the metric (unused).

        Returns:
            xr.Dataset: Empty dataset.
        """
        return xr.Dataset()

    def compute_cond_loss(
        self, sample: torch.Tensor, seed: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute conditional loss that returns zeros.

        Args:
            sample (torch.Tensor): Sample tensor.
            seed (torch.Tensor): Seed tensor (unused).

        Returns:
            torch.Tensor: Zero tensor with shape [batch_size].
        """
        return torch.zeros(sample.shape[0], device=sample.device)
