# %%
import torch
import numpy as np
import cc3d
import sys

sys.path.append("/home/bizon/Karim/data/Karim/anatomygen")
from src.conditioning.encoders import AnatomicEmbModel
from src.utils.encoder_utils import batchify
from typing import Any


class TopoEmbModel(AnatomicEmbModel):
    """
    Base class for topological embedding models.

    This class extends AnatomicEmbModel and provides a framework for
    computing topological metrics on input data.
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize the TopoEmbModel.

        Args:
            **kwargs: Additional keyword arguments to be passed to the parent class.
        """
        super().__init__(**kwargs)

    def compute_metric(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculate the topological metric.

        This method should be implemented by subclasses to define
        specific topological computations.

        Args:
            x (torch.Tensor): Input tensor to compute the metric on.

        Returns:
            torch.Tensor: Computed topological metric.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """
        raise NotImplementedError("Subclasses must implement compute_metric method.")


class ComponentEmbedder(TopoEmbModel):
    """
    A component embedder that computes the number of connected components in 3D binary images.
    """

    def compute_metric(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the number of connected components for the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W, D) representing segmentations.

        Returns:
            torch.Tensor: Computed number of connected components for the specified channels.
        """
        channels_dict = {0: None, 1: self.channels}
        metric = self.compute_connected_components(x, channels_dict=channels_dict)
        return metric

    @batchify
    def compute_connected_components(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the number of connected components in a 3D binary image.

        Args:
            x (torch.Tensor): A binary image tensor of shape (H, W, D).

        Returns:
            torch.Tensor: The number of connected components as a scalar tensor.
        """
        x_np = x.long().cpu().numpy()
        x_processed = cc3d.dust(x_np, threshold=100)
        labels_out, N = cc3d.connected_components(x_processed, return_N=True)
        return torch.tensor(N, dtype=torch.float32)


class ComponentEmbedder2D(ComponentEmbedder):
    """
    A component embedder that computes the number of connected components in 2D binary images.
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize the ComponentEmbedder2D.

        Args:
            **kwargs: Additional keyword arguments.
                emb_axis (int): The axis along which to compute the embeddings.
        """
        super().__init__(**kwargs)
        self.emb_axis: int = kwargs["emb_axis"]

    def compute_metric(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the number of connected components for the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W, D) representing segmentations.

        Returns:
            torch.Tensor: Computed number of connected components for the specified channels
                          along the embedding axis.
        """
        channels_dict = {0: None, 1: self.channels, self.emb_axis + 2: None}
        metric = self.compute_connected_components(x, channels_dict=channels_dict)
        return metric


class TopologicalInteractionEmbedder(TopoEmbModel):
    """
    A topological interaction embedder that computes topological relationships between segmentation classes.

    This class is adapted from the TopoInteraction implementation:
    https://github.com/TopoXLab/TopoInteraction

    It extends TopoEmbModel to provide methods for computing topological interactions
    such as inclusion and exclusion between different segmentation classes.
    """

    def __init__(self, dim, connectivity, inclusion, exclusion, min_thick, **kwargs):
        """
        :param dim: 2 or 3
        :param connectivity: 4 or 8 for 2D; 6 or 26 for 3D
        :param inclusion: list of [A,B] classes where A is completely surrounded by B.
        :param exclusion: list of [A,C] classes where A and C exclude each other.
        :param min_thick: Minimum thickness/separation between the two classes. Only used if connectivity is 8 for 2D or 26 for 3D
        """
        super().__init__(**kwargs)

        self.dim = dim
        self.connectivity = connectivity
        self.min_thick = min_thick
        self.interaction_list = []
        self.sum_dim_list = None
        self.conv_op = None
        self.apply_nonlin = lambda x: torch.nn.functional.softmax(x, 1)
        self.ce_loss_func = torch.nn.CrossEntropyLoss(reduction="none")

        if self.dim == 2:
            self.sum_dim_list = [1, 2, 3]
            self.conv_op = torch.nn.functional.conv2d
        elif self.dim == 3:
            self.sum_dim_list = [1, 2, 3, 4]
            self.conv_op = torch.nn.functional.conv3d

        self.set_kernel()

        for inc in inclusion:
            temp_pair = []
            temp_pair.append(True)  # type inclusion
            temp_pair.append(inc[0])
            temp_pair.append(inc[1])
            self.interaction_list.append(temp_pair)

        for exc in exclusion:
            temp_pair = []
            temp_pair.append(False)  # type exclusion
            temp_pair.append(exc[0])
            temp_pair.append(exc[1])
            self.interaction_list.append(temp_pair)

    def set_kernel(self):
        """
        Sets the connectivity kernel based on user's sepcification of dim, connectivity, min_thick
        """
        k = 2 * self.min_thick + 1
        if self.dim == 2:
            if self.connectivity == 4:
                np_kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
            elif self.connectivity == 8:
                np_kernel = np.ones((k, k))

        elif self.dim == 3:
            if self.connectivity == 6:
                np_kernel = np.array(
                    [
                        [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                        [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
                        [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                    ]
                )
            elif self.connectivity == 26:
                np_kernel = np.ones((k, k, k))

        self.kernel = torch_kernel = torch.from_numpy(
            np.expand_dims(np.expand_dims(np_kernel, axis=0), axis=0)
        )

    def topological_interaction_module(self, P):
        """
        Given a discrete segmentation map and the intended topological interactions, this module computes the critical voxels map.
        :param P: Discrete segmentation map
        :return: Critical voxels map
        """

        for ind, interaction in enumerate(self.interaction_list):
            interaction_type = interaction[0]
            label_A = interaction[1]
            label_C = interaction[2]

            # Get Masks
            mask_A = torch.where(P == label_A, 1.0, 0.0).double()
            if interaction_type:
                mask_C = torch.where(P == label_C, 1.0, 0.0).double()
                mask_C = torch.logical_or(mask_C, mask_A).double()
                mask_C = torch.logical_not(mask_C).double()
            else:
                mask_C = torch.where(P == label_C, 1.0, 0.0).double()

            # Get Neighbourhood Information
            neighbourhood_C = self.conv_op(mask_C, self.kernel.double(), padding="same")
            neighbourhood_C = torch.where(neighbourhood_C >= 1.0, 1.0, 0.0)
            neighbourhood_A = self.conv_op(mask_A, self.kernel.double(), padding="same")
            neighbourhood_A = torch.where(neighbourhood_A >= 1.0, 1.0, 0.0)

            # Get the pixels which induce errors
            violating_A = neighbourhood_C * mask_A
            violating_C = neighbourhood_A * mask_C
            violating = violating_A + violating_C
            violating = torch.where(violating >= 1.0, 1.0, 0.0)

            if ind == 0:
                critical_voxels_map = violating
            else:
                critical_voxels_map = torch.logical_or(
                    critical_voxels_map, violating
                ).double()
        return critical_voxels_map

    def crit_voxels(self, x):
        """
        The forward function computes the TI loss value.
        :param x: Likelihood map of shape: b, c, x, y(, z) with c = total number of classes
        :return:  critical voxel map
        """

        if x.device.type == "cuda":
            self.kernel = self.kernel.cuda(x.device.index)

        # Obtain discrete segmentation map
        x_softmax = self.apply_nonlin(x)
        P = torch.argmax(x_softmax, dim=1)
        P = torch.unsqueeze(P.double(), dim=1)
        del x_softmax

        # Call the Topological Interaction Module
        critical_voxels_map = self.topological_interaction_module(P)

        return critical_voxels_map

    def compute_metric(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the topological metric for the input tensor.

        Args:
            x (torch.Tensor): Input tensor to compute the metric on.
                Shape: (B, C, H, W, D), where B is batch size, C is number of channels,
                and H, W, D are spatial dimensions.

        Returns:
            torch.Tensor: Computed topological metric (critical voxels).
                Shape: (B, 1, H, W, D)
        """
        return self.crit_voxels(x.float())

    def compute_cond_loss(
        self, sample: torch.Tensor, seed: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the conditional loss for the topological embeddings.

        Args:
            sample (torch.Tensor): Sample tensor.
                Shape: (B, C, H, W, D), where B is batch size, C is number of channels,
                and H, W, D are spatial dimensions.
            seed (torch.Tensor): Seed tensor (unused in this implementation).
                Shape: Same as sample.

        Returns:
            torch.Tensor: Computed conditional loss (mean of the sample tensor).
                Shape: (B,)
        """
        return torch.mean(sample.float(), dim=tuple(range(1, sample.dim())))


class TopologicalInteractionEmbedder2D(TopologicalInteractionEmbedder):
    """
    A 2D version of the TopologicalInteractionEmbedder that computes topological relationships
    between segmentation classes along a specified embedding axis.
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize the TopologicalInteractionEmbedder2D.

        Args:
            **kwargs: Additional keyword arguments.
                emb_axis (int): The axis along which to compute the embeddings.
        """
        super().__init__(**kwargs)
        self.emb_axis: int = kwargs["emb_axis"]

    @batchify
    def crit_voxels_batch(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute critical voxels for a batch of inputs.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W, D).

        Returns:
            torch.Tensor: Computed critical voxels of shape (B, C, H, W, D).
        """
        return self.crit_voxels(x)

    def compute_metric(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the topological metric for the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W, D).

        Returns:
            torch.Tensor: Computed topological metric (critical voxels) of shape (B, C, H, W, D),
                          with the embedding axis moved to the specified position.
        """
        result = self.crit_voxels_batch(x, channels_dict={self.emb_axis + 2: None})
        return result.movedim(0, self.emb_axis + 2)


class TopologicalInteractionLoss2D(TopologicalInteractionEmbedder2D):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.ce_loss_func = torch.nn.CrossEntropyLoss(reduction="none")

    @torch.enable_grad
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        crit_voxels = super().compute_metric(x)
        gt = torch.ones_like(x, device=x.device).long()[:, 0] * 3
        ce_tensor = self.ce_loss_func(x, gt)
        ce_loss = ce_tensor * crit_voxels[:, 0]
        return ce_loss.sum(dim=[1, 2, 3]).unsqueeze(
            1
        )  # Sum over all dimensions except batch and unsqueeze
