# %%
import torch
import kimimaro
from monai.networks.nets import SegResNet
import sys

sys.path.append("/home/bizon/Karim/data/Karim/anatomygen/")
from src.utils.skel_utils import soft_skel
from src.conditioning.encoders import AnatomicEmbModel
from src.utils.encoder_utils import batchify
from src.utils.skel_utils import (
    blur_voxel_grid_norm,
    consolidate_branch_points,
    croporpad,
)
from typing import Dict, List, Optional, Any, Tuple
from geomloss import SamplesLoss


class SkelEmbModel(AnatomicEmbModel):
    """
    Base class for skeletal embedding models.

    This class extends AnatomicEmbModel and provides a framework for
    computing skeletal metrics on input data.
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize the SkelEmbModel.

        Args:
            **kwargs: Additional keyword arguments to be passed to the parent class.
        """
        super().__init__(**kwargs)

    def compute_metric_norm(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the normalized skeletal metric.

        Args:
            x (torch.Tensor): Input tensor to compute the metric on.

        Returns:
            torch.Tensor: Computed normalized skeletal metric.
        """
        metric = self.compute_metric(x)
        return metric

    def preprocess_labels(self, x: torch.Tensor) -> torch.Tensor:
        """
        Preprocess input labels for skeletal computation.

        Args:
            x (torch.Tensor): Input tensor containing labels.

        Returns:
            torch.Tensor: Preprocessed labels.
        """
        labels = x[:, self.channels]
        labels = torch.nn.functional.interpolate(
            labels.float(), size=self.label_shape, mode="trilinear"
        )
        labels = (
            blur_voxel_grid_norm(labels, self.blur_kernel_size, self.label_sigma) > 0.5
        )
        return labels

    def compute_metric(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the skeletal metric.

        This method should be implemented by subclasses to define
        specific skeletal computations.

        Args:
            x (torch.Tensor): Input tensor to compute the metric on.

        Returns:
            torch.Tensor: Computed skeletal metric.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """
        raise NotImplementedError("Subclasses must implement compute_metric method.")

    def compute_cond_loss(
        self, sample: torch.Tensor, seed: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """
        Compute the conditional loss for skeletal embeddings.

        Args:
            sample (torch.Tensor): Sample tensor.
            seed (torch.Tensor): Seed tensor.

        Returns:
            Optional[torch.Tensor]: Computed conditional loss, or None if not applicable.
        """
        return None


class HardSkelEmbedder(SkelEmbModel):
    """
    A hard skeletal embedder that computes skeletal metrics using kimimaro skeletonization.

    This class extends SkelEmbModel to provide a method for computing skeletal embeddings
    using hard skeletonization techniques.

    Attributes:
        blur_kernel_size (int): Size of the blur kernel for preprocessing.
        label_shape (Tuple[int, int, int]): Shape of the labels after preprocessing.
        label_sigma (float): Sigma value for label blurring.
        skeletonize_params (Dict): Parameters for the skeletonization process.
        vertices_length (int): Target length for vertex padding.
        channels_dict (Dict[int, Optional[List[int]]]): Dictionary mapping channel indices to lists of tissue indices.
        sinkhorn_loss (SamplesLoss): Loss function for optimal transport computation.
    """

    def __init__(
        self,
        label_sigma: float,
        label_shape: Tuple[int, int, int],
        blur_kernel_size: int,
        skeletonize_params: Dict,
        vertices_length: int,
        **kwargs: Any,
    ):
        """
        Initialize the HardSkelEmbedder.

        Args:
            label_sigma (float): Sigma value for label blurring.
            label_shape (Tuple[int, int, int]): Shape of the labels after preprocessing.
            blur_kernel_size (int): Size of the blur kernel for preprocessing.
            skeletonize_params (Dict): Parameters for the skeletonization process.
            vertices_length (int): Target length for vertex padding.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.blur_kernel_size = blur_kernel_size
        self.label_shape = tuple(label_shape)
        self.label_sigma = label_sigma
        self.skeletonize_params = skeletonize_params
        self.vertices_length = vertices_length

        self.channels_dict = {0: None, 1: None}
        self.sinkhorn_loss = SamplesLoss()

    def compute_metric(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the skeletal metric using hard skeletonization.

        Args:
            x (torch.Tensor): Input tensor to compute the metric on.

        Returns:
            torch.Tensor: Computed skeletal metric.
        """
        labels = self.preprocess_labels(x)
        skeleton = self.compute_skeleton(
            labels.float(), channels_dict=self.channels_dict
        )
        return skeleton

    @batchify
    def compute_skeleton(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the skeleton of a binary image.

        Args:
            x (torch.Tensor): A binary image tensor of shape (H, W, D).

        Returns:
            torch.Tensor: The skeleton vertices as a tensor.
        """
        skel = kimimaro.skeletonize(
            x.bool().cpu().numpy(), self.skeletonize_params, progress=False
        )[1]
        vertices = skel.vertices
        vertices_padded = croporpad(vertices, target_length=self.vertices_length)
        return torch.tensor(vertices_padded)

    @batchify
    def compute_optimal_transport(
        self, sample: torch.Tensor, seed: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the optimal transport between two point sets.

        Args:
            sample (torch.Tensor): Sample points of shape (N, 3).
            seed (torch.Tensor): Seed points of shape (M, 3).

        Returns:
            torch.Tensor: The computed optimal transport loss.
        """
        mask_sample = ~(sample == -1).any(dim=-1)
        mask_seed = ~(seed == -1).any(dim=-1)

        active_sample = sample[mask_sample]
        active_seed = seed[mask_seed]

        if active_sample.size(0) == 0 or active_seed.size(0) == 0:
            return torch.tensor(0, device=sample.device) - 1
        return self.sinkhorn_loss(active_sample, active_seed)

    def compute_cond_loss(
        self, sample: torch.Tensor, seed: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the conditional loss for skeletal embeddings.

        Args:
            sample (torch.Tensor): Sample tensor.
            seed (torch.Tensor): Seed tensor.

        Returns:
            torch.Tensor: Computed conditional loss.
        """
        channels_dict = {0: None}
        return self.compute_optimal_transport(sample, seed, channels_dict=channels_dict)


class BranchEmbedder(HardSkelEmbedder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.branch_threshold = kwargs["branch_threshold"]
        self.channels_dict = {0: None, 1: self.channels}

    @batchify
    def compute_branches(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the branch points of a binary image skeleton.

        Args:
            x (torch.Tensor): A binary image tensor of shape (H, W, D).

        Returns:
            torch.Tensor: The branch points as a tensor of shape (vertices_length, 3).
        """
        skel = kimimaro.skeletonize(
            x.bool().cpu().numpy(), self.skeletonize_params, progress=False
        )[1]
        branch_idx = skel.branches()
        branch_idx = consolidate_branch_points(
            branch_idx, threshold=self.branch_threshold
        )
        branch_points = skel.vertices[branch_idx]
        branch_points_padded = croporpad(
            branch_points, target_length=self.vertices_length
        )
        return torch.tensor(branch_points_padded)

    def compute_metric(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the branch points metric for the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W, D).

        Returns:
            torch.Tensor: Computed branch points metric of shape (B, vertices_length, 3).
        """
        labels = self.preprocess_labels(x)
        branches = self.compute_branches(labels, channels_dict=self.channels_dict)
        return branches


class SoftSkelEmbedder(SkelEmbModel):
    """
    A soft skeletal embedder that computes a soft skeleton of input labels.

    This class extends SkelEmbModel to provide soft skeletonization of input labels.

    Attributes:
        soft_skel_iterations (int): Number of iterations for soft skeletonization.
        label_shape (Tuple[int, ...]): Shape of the labels after preprocessing.
    """

    def __init__(
        self, soft_skel_iterations: int, label_shape: List[int], **kwargs: Any
    ) -> None:
        """
        Initialize the SoftSkelEmbedder.

        Args:
            soft_skel_iterations (int): Number of iterations for soft skeletonization.
            label_shape (List[int]): Shape of the labels after preprocessing.
            **kwargs: Additional keyword arguments including embedder, tissues, tags, and metric_dim.
        """
        super().__init__(**kwargs)
        self.soft_skel_iterations = soft_skel_iterations
        self.label_shape = tuple(label_shape)

    def preprocess_labels(self, x: torch.Tensor) -> torch.Tensor:
        """
        Preprocess input labels for soft skeletonization.

        Args:
            x (torch.Tensor): Input tensor containing labels.

        Returns:
            torch.Tensor: Preprocessed labels.
        """
        labels = x[:, self.channels].float()
        labels = torch.nn.functional.interpolate(
            labels, size=self.label_shape, mode="trilinear"
        )
        return labels

    def compute_metric(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the soft skeleton metric for the input tensor.

        Args:
            x (torch.Tensor): Input tensor to compute the metric on.

        Returns:
            torch.Tensor: Computed soft skeleton metric.
        """
        labels = self.preprocess_labels(x)
        skeleton = soft_skel(labels.float(), iter_=self.soft_skel_iterations)
        return skeleton


class NeuralSkelEmbedder(SkelEmbModel):
    """
    A neural skeletal embedder that uses a SegResNet for skeletal computations.

    This class extends SkelEmbModel to provide a neural network-based approach for
    computing skeletal embeddings.

    Attributes:
        unet (SegResNet): The SegResNet model used for skeletal computations.
        label_shape (Tuple[int, ...]): Shape of the labels after preprocessing.
        label_sigma (float): Sigma value for label blurring.
        blur_kernel_size (int): Size of the blur kernel.
    """

    def __init__(
        self,
        unet_params: Dict[str, Any],
        label_sigma: float,
        label_shape: List[int],
        blur_kernel_size: int,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the NeuralSkelEmbedder.

        Args:
            unet_params (Dict[str, Any]): Parameters for the SegResNet model.
            label_sigma (float): Sigma value for label blurring.
            label_shape (List[int]): Shape of the labels after preprocessing.
            blur_kernel_size (int): Size of the blur kernel.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.unet = SegResNet(**unet_params)
        self.label_shape = tuple(label_shape)
        self.label_sigma = label_sigma
        self.blur_kernel_size = blur_kernel_size

    def compute_metric(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the neural skeletal metric for the input tensor.

        Args:
            x (torch.Tensor): Input tensor to compute the metric on.

        Returns:
            torch.Tensor: Computed neural skeletal metric.
        """
        return self.unet(x.float())
