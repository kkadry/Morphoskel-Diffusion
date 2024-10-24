import torch
import scipy
import torch.nn.functional as F
from src.utils.encoder_utils import batchify
from typing import Any, List


class MaskEmbModel(torch.nn.Module):
    """
    Base class for mask embedding models.

    This class provides a framework for computing and applying masks to input data.
    """

    def __init__(self, **kwargs: Any):
        """
        Initialize the MaskEmbModel.

        Args:
            **kwargs: Additional keyword arguments.
                mask_shape (Optional[List[int]]): The shape of the mask to be computed.
        """
        super().__init__()
        self.mask_shape: List[int] = list(kwargs.get("mask_shape", None))

    def compute_mask(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the mask for the input tensor.

        Args:
            x (torch.Tensor): Input tensor to compute the mask for.

        Returns:
            torch.Tensor: Computed mask.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MaskEmbModel.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Computed mask.
        """
        return self.compute_mask(x)

    def compute_channel_mask(self, x: torch.Tensor, matvec: List[int]) -> torch.Tensor:
        """
        Compute a channel mask based on the input tensor and a list of channel indices.

        Args:
            x (torch.Tensor): Input tensor of shape (C, H, W, D).
            matvec (List[int]): List of channel indices to include in the mask.

        Returns:
            torch.Tensor: Computed channel mask of shape (1, H, W, D).
        """
        mask = torch.zeros_like(x)[0:1]
        for m in matvec:
            mask += (x[m : m + 1] == 1).float()
        return mask

    def morphological_operations(
        self, mask: torch.Tensor, iterations_dil: int, iterations_ero: int
    ) -> torch.Tensor:
        """
        Apply morphological operations (dilation and erosion) to the input mask.

        Args:
            mask (torch.Tensor): Input mask of shape (H, W, D).
            iterations_dil (int): Number of dilation iterations.
            iterations_ero (int): Number of erosion iterations.

        Returns:
            torch.Tensor: Processed mask of shape (H, W, D).
        """
        mask = mask.detach().cpu()

        if iterations_dil > 0:
            mask = (
                torch.tensor(
                    scipy.ndimage.binary_dilation(
                        mask, iterations=iterations_dil, border_value=0
                    )
                )
                .to(mask)
                .float()
            )

        if iterations_ero > 0:
            mask = (
                torch.tensor(
                    scipy.ndimage.binary_erosion(
                        mask, iterations=iterations_ero, border_value=0
                    )
                )
                .to(mask)
                .float()
            )

        return mask


class NullMask(MaskEmbModel):
    """
    A mask model that always returns None, effectively applying no mask.
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize the NullMask.

        Args:
            **kwargs: Additional keyword arguments passed to the parent class.
        """
        super().__init__(**kwargs)

    def compute_mask(self, x: torch.Tensor) -> None:
        """
        Compute the mask, which is always None for this class.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            None
        """
        return None

    def forward(self, x: torch.Tensor) -> None:
        """
        Forward pass of the NullMask, always returns None.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            None
        """
        return None


class TissueMask(MaskEmbModel):
    """
    A mask model that computes a tissue mask based on specified channels.
    """

    def __init__(
        self,
        channels: List[int],
        num_channels: int,
        iterations_dil: int = 1,
        iterations_ero: int = 1,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the TissueMask.

        Args:
            channels (List[int]): List of channel indices to exclude from the mask.
            num_channels (int): Total number of channels in the input.
            iterations_dil (int, optional): Number of dilation iterations. Defaults to 1.
            iterations_ero (int, optional): Number of erosion iterations. Defaults to 1.
            **kwargs: Additional keyword arguments passed to the parent class.
        """
        super().__init__(**kwargs)
        self.channels = list(channels)
        self.iterations_dil = iterations_dil
        self.iterations_ero = iterations_ero
        self.matvec = list(range(1, num_channels))
        for m in self.channels:
            self.matvec.remove(m)
        self.channels_dict = {0: None}

    def compute_mask(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the tissue mask for the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W, D).

        Returns:
            torch.Tensor: Computed tissue mask of shape (B, 1, H', W', D'),
                          where H', W', D' are the dimensions of self.mask_shape.
        """
        return self.tissue_mask(x, channels_dict=self.channels_dict)

    @batchify
    def tissue_mask(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the tissue mask for a single input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (C, H, W, D).

        Returns:
            torch.Tensor: Computed tissue mask of shape (1, H', W', D'),
                          where H', W', D' are the dimensions of self.mask_shape.
        """
        # Mask of tissues to maintain
        mask = self.compute_channel_mask(x, self.matvec)

        # Subsampling mask to be same shape as latents
        mask_coarse = (
            F.interpolate(mask.unsqueeze(0), size=self.mask_shape, mode="nearest")
            .squeeze(0)
            .squeeze(0)
        )

        # Dilating the binary mask of frozen tissues
        mask_proc = self.morphological_operations(
            mask_coarse, self.iterations_dil, self.iterations_ero
        )
        return mask_proc.unsqueeze(0)


class BoundingBoxMask(MaskEmbModel):
    """
    A mask embedder that creates a bounding box mask within specified coordinates.

    This class extends MaskEmbModel to provide methods for creating a bounding box mask
    based on given coordinates in x, y, and z dimensions.
    """

    def __init__(self, bbox_x, bbox_y, bbox_z, **kwargs):
        """
        Initialize the BoundingBoxMask.

        Args:
            bbox_x (List[float]): List of two floats specifying the start and end of the bounding box in x-dimension (normalized).
            bbox_y (List[float]): List of two floats specifying the start and end of the bounding box in y-dimension (normalized).
            bbox_z (List[float]): List of two floats specifying the start and end of the bounding box in z-dimension (normalized).
            **kwargs: Additional keyword arguments passed to the parent class.
        """
        super().__init__(**kwargs)
        self.bbox_x_vox = [
            int(bbox_x[0] * self.mask_shape[0]),
            int(bbox_x[1] * self.mask_shape[0]),
        ]
        self.bbox_y_vox = [
            int(bbox_y[0] * self.mask_shape[1]),
            int(bbox_y[1] * self.mask_shape[1]),
        ]
        self.bbox_z_vox = [
            int(bbox_z[0] * self.mask_shape[2]),
            int(bbox_z[1] * self.mask_shape[2]),
        ]

    def compute_mask(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the bounding box mask for the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W, D).

        Returns:
            torch.Tensor: Computed bounding box mask of shape (B, 1, H', W', D'),
                          where H', W', D' are the dimensions of self.mask_shape.
        """
        return self.bounding_box_mask(x)

    def bounding_box_mask(self, x: torch.Tensor) -> torch.Tensor:
        """
        Create a bounding box mask for a single input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W, D).

        Returns:
            torch.Tensor: Computed bounding box mask of shape (B, 1, H', W', D'),
                          where H', W', D' are the dimensions of self.mask_shape.
        """
        mask = torch.zeros(x.shape[0], 1, *self.mask_shape)
        mask[
            ...,
            self.bbox_x_vox[0] : self.bbox_x_vox[1],
            self.bbox_y_vox[0] : self.bbox_y_vox[1],
            self.bbox_z_vox[0] : self.bbox_z_vox[1],
        ] = 1
        return mask.to(x.device)
