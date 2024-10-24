import torch
import numpy as np
import skimage
import edt
from scipy.signal import savgol_filter
from typing import List
from src.conditioning.encoders import AnatomicEmbModel
from src.utils.encoder_utils import batchify
from src.models.autoencoder import AutoencoderKL
from torch.nn import functional as F
from typing import Any, Dict, Optional, Tuple, Union


class MorphEmbModel(AnatomicEmbModel):
    """
    Base class for morphological embedding models.

    This class extends AnatomicEmbModel and provides a framework for
    computing morphological metrics on input data.
    """

    def __init__(self, **kwargs):
        """
        Initialize the MorphEmbModel.

        Args:
            **kwargs: Additional keyword arguments to be passed to the parent class.
        """
        super().__init__(**kwargs)

    def compute_metric(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculate the morphological metric in a batchwise manner.

        This method should be implemented by subclasses to define
        specific morphological computations.

        Args:
            x (torch.Tensor): Input tensor to compute the metric on.

        Returns:
            torch.Tensor: Computed morphological metric.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """
        raise NotImplementedError("Subclasses must implement compute_metric method.")


class VolEmbedder3D(MorphEmbModel):
    def compute_vol(
        self, x: torch.Tensor, tissues: Union[int, List[int]]
    ) -> torch.Tensor:
        """
        Calculate the volume fraction for specified tissue segmentations.

        Args:
            x (torch.Tensor): A binary tensor of size (B, C, H, W, D) representing segmentations.
                B: batch size, C: number of channels/classes, H: height, W: width, D: depth.
            tissues (Union[int, List[int]]): Channel index or list of channel indices to calculate volumes for.

        Returns:
            torch.Tensor: A tensor of size (B, 1) or (B, len(tissues)) containing volume fractions (in percentage) for specified segmentations.

        Note:
            - The input tensor is expected to be binary (0 or 1).
            - The function calculates the mean over spatial dimensions (H, W, D) and multiplies by 100 to get percentage.
            - Values greater than 0.5 are considered as part of the segmentation.
        """
        if isinstance(tissues, int):
            vol = torch.mean((x[:, tissues : tissues + 1]).float(), dim=[2, 3, 4]) * 100
        else:
            vol = torch.mean((x[:, tissues]).float(), dim=[2, 3, 4]) * 100
        return vol

    def compute_metric(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the volume metric for the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W, D) representing segmentations.

        Returns:
            torch.Tensor: Computed volume metric for the specified channels.
        """
        metric = self.compute_vol(x, self.channels)
        return metric


class ElongationEmbedder3D(MorphEmbModel):
    def __init__(self, **kwargs):
        """
        Initialize the ElongationEmbedder3D.

        Args:
            **kwargs: Additional keyword arguments passed to the parent class.
        """
        super().__init__(**kwargs)
        self.channels_dict = {0: None, 1: self.channels}

    @batchify
    def compute_elongation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute elongation for a given 3D image.

        Args:
            x (torch.Tensor): A 3D tensor of shape (H, W, D) representing the segmentation.

        Returns:
            torch.Tensor: A tensor containing the elongation value.
        """
        # Check if x is mostly empty, if so add a single positive voxel clump
        if x.sum() < 30:
            x[0:20, 0:5, 0] = 1.0

        props = skimage.measure.regionprops(x.long().cpu().numpy())
        # Obtains only the first connected component
        if len(props) == 0:
            return torch.tensor(0.0)

        try:
            # Extracting major and minor axis lengths
            major_axis_length = props[0].major_axis_length
            minor_axis_length = props[0].minor_axis_length

            # Calculate elongation
            elongation = major_axis_length / minor_axis_length
        except:
            elongation = 1

        return torch.tensor(elongation, dtype=torch.float32)

    def compute_metric(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the elongation metric for the input tensor.

        Args:
            x (torch.Tensor): A 3D tensor of shape (H, W, D) representing the segmentation.

        Returns:
            torch.Tensor: Computed elongation metric of shape (B,), where B is the batch size.
        """
        return self.compute_elongation(x, channels_dict=self.channels_dict)


class CrossAreaEmbedder(MorphEmbModel):
    def __init__(self, **kwargs):
        """
        Initialize the CrossAreaEmbedder.

        Args:
            **kwargs: Additional keyword arguments.
                emb_axis (int): The axis along which to compute the cross-sectional area.
        """
        super().__init__(**kwargs)
        self.emb_axis: int = kwargs["emb_axis"]

    def compute_cross_area(
        self, x: torch.Tensor, axis: int, tissues: List[int]
    ) -> torch.Tensor:
        """
        Calculate the cross-sectional area for specified tissue segmentations along a chosen axis.

        Args:
            x (torch.Tensor): Tensor of size [B, C, H, W, D], must be binarized and one-hot encoded.
            axis (int): The axis to retain (0 for H, 1 for W, 2 for D).
            tissues (List[int]): List of channel indices to calculate areas for.

        Returns:
            torch.Tensor: Tensor of shape [B, len(tissues), N] representing the area fractions along the chosen axis for specified tissues,
                          where N is the size of the chosen axis (H, W, or D).
        """
        B, C, H, W, D = x.shape

        # Create a default permutation list [B, C, H, W, D]
        perm = [0, 1, 2, 3, 4]

        # Move the selected axis to the last position
        perm[-1] = 2 + axis  # Set the chosen axis (H, W, or D) to the last position
        perm[2 + axis] = 4  # Set the last dimension to the position of the chosen axis

        # Select tissues and permute the tensor
        x_selected = x[:, tissues]
        x_permuted = x_selected.permute(
            perm
        )  # The chosen axis is now the last dimension

        # Calculate area fractions along the chosen axis for specified tissues
        area = torch.mean(x_permuted.float(), dim=[2, 3]) * 100

        # Return the area fractions
        return area

    def compute_metric(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the cross-sectional area metric for the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape [B, C, H, W, D] representing the segmentation.

        Returns:
            torch.Tensor: Computed cross-sectional area metric of shape [B, len(self.channels), N],
                          where N is the size of the chosen axis (H, W, or D).
        """
        metric = self.compute_cross_area(x, self.emb_axis, self.channels)
        return metric


class ThicknessEmbedder2D(MorphEmbModel):
    """
    A 2D thickness embedder that computes the maximum thickness of a binary image.
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize the ThicknessEmbedder2D.

        Args:
            **kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.emb_axis: int = kwargs["emb_axis"]
        self.channels_dict: Dict[int, Optional[List[int]]] = {
            0: None,
            1: self.channels,
            self.emb_axis + 2: None,
        }

    @batchify
    def compute_thickness_2D(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the maximum thickness of a 2D binary image.

        Args:
            x (torch.Tensor): A binary image tensor of shape (H, W).

        Returns:
            torch.Tensor: The maximum thickness as a scalar tensor.
        """
        # Compute the Euclidean Distance Transform
        edf: np.ndarray = edt.edt(
            x.cpu().numpy(), black_border=True, order="C", parallel=0
        )
        edf_tensor: torch.Tensor = torch.from_numpy(edf)
        # Calculate the maximum thickness
        thickness: torch.Tensor = torch.max(edf_tensor)
        return thickness

    def compute_metric(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the thickness metric for the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W, D).

        Returns:
            torch.Tensor: Computed thickness metric of shape (B, C, N), where B is the batch size,
                          C is the number of channels, and N is the size of the chosen axis (H, W, or D)
                          depending on the emb_axis value.
        """
        return self.compute_thickness_2D(x, channels_dict=self.channels_dict)


class CircularityEmbedder2D(ThicknessEmbedder2D):
    """
    A 2D circularity embedder that computes the circularity of binary images.

    """

    @batchify
    def compute_circularity_2D(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the circularity of a 2D binary image.

        Args:
            x (torch.Tensor): A binary image tensor of shape (H, W).

        Returns:
            torch.Tensor: The circularity as a scalar tensor.
        """
        # Ensure x is a binary tensor
        x = (x > 0.5).float()

        # Find contours
        contours = skimage.measure.find_contours(x.cpu().numpy(), 0.5)

        if not contours:
            return torch.tensor(0.0)

        # Select the largest contour
        contour = max(contours, key=len)

        # Smooth the contour
        if len(contour) >= 25:
            contour = savgol_filter(contour.T, 21, 3).T
        else:
            return torch.tensor(0.0)

        # Calculate perimeter
        distances = np.sqrt(np.sum(np.diff(contour, axis=0) ** 2, axis=1))
        perimeter = np.sum(distances) + np.sqrt(np.sum((contour[-1] - contour[0]) ** 2))

        # Calculate area using the Shoelace formula
        x, y = contour[:, 0], contour[:, 1]
        area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

        # Compute circularity
        circularity = 4 * np.pi * area / (perimeter**2)

        return torch.tensor(circularity)

    def compute_metric(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the circularity metric for the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W, D).

        Returns:
            torch.Tensor: Computed circularity metric of shape (B, C, N), where B is the batch size,
                          C is the number of channels, and N is the size of the chosen axis (H, W, or D)
                          depending on the emb_axis value.
        """
        return self.compute_circularity_2D(x, channels_dict=self.channels_dict)


class CentroidEmbedder2D(ThicknessEmbedder2D):
    """
    A 2D centroid embedder that computes the distance of the centroid from the center of the image.

    """

    def compute_metric(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the centroid distance metric for the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W, D).

        Returns:
            torch.Tensor: Computed centroid distance metric of shape (B, C, N), where B is the batch size,
                          C is the number of channels, and N is the size of the chosen axis (H, W, or D)
                          depending on the emb_axis value.
        """
        return self.compute_centroid_2D(x, channels_dict=self.channels_dict)

    @batchify
    def compute_centroid_2D(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the distance of the centroid from the center of a 2D binary image.

        Args:
            x (torch.Tensor): A binary image tensor of shape (H, W).

        Returns:
            torch.Tensor: The distance of the centroid from the center as a scalar tensor.
        """
        # Ensure x is a binary tensor
        x = (x > 0.5).float()

        # Find the centroid
        y_indices, x_indices = torch.where(x > 0)
        if len(y_indices) == 0 or len(x_indices) == 0:
            return torch.tensor(0.0).to(x.device)

        centroid_y = torch.mean(y_indices.float())
        centroid_x = torch.mean(x_indices.float())

        # Calculate the center of the image
        center_y, center_x = x.shape[0] / 2, x.shape[1] / 2

        # Calculate the Euclidean distance from the center
        distance = torch.sqrt(
            (centroid_y - center_y) ** 2 + (centroid_x - center_x) ** 2
        )

        return distance


class ArcLengthEmbedder2D(ThicknessEmbedder2D):
    """
    A 2D arc length embedder that computes the maximum angular span of contours in binary images.

    """

    def compute_metric(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the arc length metric for the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W, D).

        Returns:
            torch.Tensor: Computed arc length metric of shape (B, C, N), where B is the batch size,
                          C is the number of channels, and N is the size of the chosen axis (H, W, or D)
                          depending on the emb_axis value.
        """
        return self.compute_arclength_2D(x, channels_dict=self.channels_dict)

    @batchify
    def compute_arclength_2D(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the maximum angular span of contours in a 2D binary image.

        Args:
            x (torch.Tensor): A binary image tensor of shape (H, W).

        Returns:
            torch.Tensor: The maximum angular span as a scalar tensor.
        """
        contours = skimage.measure.find_contours(x.cpu().numpy(), 0.5)

        if not contours:
            return torch.tensor(0.0)

        arc_lengths = [self.angular_span(contour) for contour in contours]
        max_arc_length = max(arc_lengths)

        return torch.tensor(max_arc_length)

    def angular_span(self, contour: np.ndarray) -> float:
        """
        Calculate the angular span of a contour.

        Args:
            contour (np.ndarray): A contour array of shape (N, 2).

        Returns:
            float: The angular span in degrees.
        """
        # Estimate the center (centroid)
        center = np.array([64, 64])

        angles = np.arctan2(contour[:, 0] - center[0], contour[:, 1] - center[1])

        # Convert angles to complex numbers
        complex_numbers = np.exp(1j * angles)

        # Compute the mean angle
        mean_angle = np.angle(np.mean(complex_numbers))

        # Compute the angular differences from the mean angle, and find the max difference
        angular_differences = np.angle(np.exp(1j * (angles - mean_angle)))
        max_difference = max(angular_differences) - min(angular_differences)

        return np.degrees(max_difference)


class NeuralMorphEmbedder2D(MorphEmbModel):
    """
    A neural morphological embedder that processes 3D input volumes and computes a sequence of 2D morphological metrics.

    This class extends MorphEmbModel to provide a neural network-based approach for computing
    morphological embeddings of 3D volumes, resulting in a sequence of 2D metrics.

    Attributes:
        input_shape (List[int]): The shape of the input volumes.
        emb_axis (int): The axis along which to compute embeddings.
        emb_vox_shape (List[int]): The shape of the embedding voxels.
        emb_axis_shape (int): The size of the embedding axis.
        pool_shape (Tuple[int, int, int]): The shape for adaptive pooling.
        num_morph_channels (int): The number of morphological channels.
        autoencoder (AutoencoderKL): The autoencoder used for encoding inputs.
        pool (nn.AdaptiveAvgPool3d): Adaptive pooling layer.
        fc1 (nn.Linear): First fully connected layer.
        fc2 (nn.Linear): Second fully connected layer.
    """

    def __init__(
        self,
        encoder_params: Dict[str, Any],
        input_shape: Tuple[int, int, int],
        **kwargs: Any,
    ):
        """
        Initialize the NeuralMorphEmbedder2D.

        Args:
            encoder_params (Dict[str, Any]): Parameters for the autoencoder.
            input_shape (Tuple[int, int, int]): The shape of the input volumes.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.input_shape = list(input_shape)
        self.emb_axis = kwargs["emb_axis"]
        self.emb_vox_shape = list(kwargs["emb_vox_shape"])
        self.emb_axis_shape = int(self.emb_vox_shape[self.emb_axis])
        self.pool_shape = tuple(
            1 if i != self.emb_axis else self.emb_axis_shape for i in range(3)
        )
        self.num_morph_channels = len(self.channels)

        # Autoencoder
        self.autoencoder = AutoencoderKL(**encoder_params)

        # Number of voxels after pooling
        n_voxels = self.pool_shape[self.emb_axis]
        n_channels = encoder_params["latent_channels"]
        self.pool = torch.nn.AdaptiveAvgPool3d(self.pool_shape)
        self.fc1 = torch.nn.Linear(n_channels * n_voxels, n_channels * n_voxels)
        self.fc2 = torch.nn.Linear(
            n_channels * n_voxels, self.num_morph_channels * self.emb_axis_shape
        )

    def compute_metric(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute a sequence of 2D morphological metrics for the input 3D volume.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, D, H, W), where B is batch size,
                              C is number of channels, and D, H, W are depth, height, and width.

        Returns:
            torch.Tensor: Computed sequence of 2D morphological metrics of shape
                          (B, num_morph_channels, emb_axis_shape), where num_morph_channels
                          is the number of morphological features and emb_axis_shape is the
                          number of 2D slices in the sequence.
        """
        bsz = x.shape[0]
        # Downsampling
        x = F.interpolate(
            x.float(), size=self.input_shape, mode="trilinear", align_corners=False
        )
        # Encoding to a small latent cube
        x = self.autoencoder.encode(x)[0]
        # Converting to vector
        x = self.pool(x).view(bsz, -1)
        x = torch.nn.ReLU()(self.fc1(x))
        x = self.fc2(x)
        x = x.view(bsz, self.num_morph_channels, self.emb_axis_shape)
        return x
