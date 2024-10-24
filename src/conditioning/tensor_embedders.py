import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils.skel_utils import blur_voxel_grid_norm
from typing import Tuple


class TensorEmbedder(torch.nn.Module):
    """
    Base class for tensor embedders.
    """

    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the embedder.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Embedded tensor.

        Raises:
            NotImplementedError: If not implemented in subclass.
        """
        raise NotImplementedError("Subclasses must implement forward method.")


class IdentityEmbedder(TensorEmbedder):
    """
    Embedder that returns the input tensor unchanged.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that returns the input tensor unchanged.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: The input tensor, unchanged.
        """
        return x


class Scalar2VoxelEmbedder(TensorEmbedder):
    def __init__(self, emb_vox_shape, **kwargs):
        super().__init__(**kwargs)
        self.emb_vox_shape = emb_vox_shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that converts a scalar tensor to a voxel tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C).

        Returns:
            torch.Tensor: Output tensor of shape (B, C, H, W, D).
        """
        B, C = x.shape
        H, W, D = self.emb_vox_shape

        emb = x.view(B, C, 1, 1, 1)
        emb = emb.expand(B, C, H, W, D)

        return emb


class Seq2VoxelEmbedder(TensorEmbedder):
    def __init__(self, emb_vox_shape, emb_axis, **kwargs):
        super().__init__(**kwargs)
        self.emb_vox_shape = emb_vox_shape
        self.emb_axis = emb_axis

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that converts a vector tensor to a voxel tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, N).

        Returns:
            torch.Tensor: Output tensor of shape (B, C, H, W, D).
        """
        B, C, N = x.shape
        H, W, D = self.emb_vox_shape

        target_dim_size = self.emb_vox_shape[self.emb_axis]
        x_interpolated = F.interpolate(
            x, size=target_dim_size, mode="linear", align_corners=False
        )

        x_expanded = x_interpolated.unsqueeze(-1).unsqueeze(-1)

        perm = [0, 1, 2, 3, 4]
        perm[2 + self.emb_axis], perm[2] = 2, 2 + self.emb_axis

        x_permuted = x_expanded.permute(perm)

        shape = [B, C, H, W, D]

        return x_permuted.expand(shape)


class Seq2SeqEmbedder(TensorEmbedder):
    def __init__(self, emb_seq_len, **kwargs):
        super().__init__()
        self.emb_seq_len = emb_seq_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that converts a vector tensor to a sequence tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (B, N, C).

        Returns:
            torch.Tensor: Output tensor of shape (B, N, C').
        """
        B, N, C = x.shape

        # Interpolate along the C dimension
        x = F.interpolate(x, size=self.emb_seq_len, mode="linear", align_corners=False)

        return x


class Scalar2VecEmbedder(TensorEmbedder):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert a scalar tensor to a vector tensor by unsqueezing.

        Args:
            x (torch.Tensor): Input tensor of shape (B,).

        Returns:
            torch.Tensor: Output tensor of shape (B, 1).
        """
        return x.unsqueeze(1)


class Voxel2VoxelEmbedder(TensorEmbedder):
    """
    This class resizes the input voxel grid to the desired embedding shape.
    """

    def __init__(self, emb_vox_shape: tuple, **kwargs):
        """
        Initialize the Voxel2VoxelEmbedder.

        Args:
            emb_vox_shape (tuple): The desired shape of the embedding voxel grid.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.emb_vox_shape = tuple(emb_vox_shape)

    def process_voxel_grid(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process the input voxel grid by resizing it to the final embedding shape.

        Args:
            x (torch.Tensor): Input voxel grid tensor of shape (B, C, H, W, D).

        Returns:
            torch.Tensor: Resized voxel grid tensor of shape (B, C', H', W', D').
        """
        # Resize to final embedding shape
        final_grid = torch.nn.functional.interpolate(
            x, size=self.emb_vox_shape, mode="trilinear", align_corners=False
        )
        return final_grid

    def forward(self, voxel_grid: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Voxel2VoxelEmbedder.

        Args:
            voxel_grid (torch.Tensor): Input voxel grid tensor.

        Returns:
            torch.Tensor: Processed voxel grid tensor.
        """
        return self.process_voxel_grid(voxel_grid)


class Voxel2BlurredVoxelEmbedder(Voxel2VoxelEmbedder):
    """
    This class blurs and resizes the input voxel grid to the desired embedding shape.
    """

    def __init__(self, blur_kernel_size: int, sigma_skel: float, **kwargs):
        """
        Initialize the Voxel2BlurredVoxelEmbedder.

        Args:
            blur_kernel_size (int): The size of the kernel used for blurring.
            sigma_skel (float): The standard deviation for the Gaussian kernel used in blurring.
            **kwargs: Additional keyword arguments passed to the parent class.
        """
        super().__init__(**kwargs)
        self.blur_kernel_size = blur_kernel_size
        self.sigma_skel = sigma_skel

    def process_voxel_grid(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process the input voxel grid by blurring, pooling, and resizing it to the final embedding shape.

        Args:
            x (torch.Tensor): Input voxel grid tensor.

        Returns:
            torch.Tensor: Processed voxel grid tensor.
        """
        # Blur the voxel grid
        blurred_grid = blur_voxel_grid_norm(x, self.blur_kernel_size, self.sigma_skel)

        # Reshape
        scale_factor = tuple(
            max(int(blurred_grid.shape[i + 2] / self.emb_vox_shape[i]), 1)
            for i in range(3)
        )
        max_pool = nn.MaxPool3d(kernel_size=4, stride=scale_factor, padding=0)
        pooled_grid = max_pool(blurred_grid)

        # Use the parent class method to resize to final embedding shape
        return super().process_voxel_grid(pooled_grid)


class Vertex2VoxelEmbedder(Voxel2BlurredVoxelEmbedder):
    """
    This class scatters vertices into a voxel grid, then blurs and resizes the voxel grid to the desired embedding shape.
    """

    def __init__(self, label_shape: Tuple[int, int, int], **kwargs):
        """
        Initialize the Vertex2VoxelEmbedder.

        Args:
            label_shape (Tuple[int, int, int]): The shape of the label voxel grid (H, W, D).
            **kwargs: Additional keyword arguments passed to the parent class.
        """
        super().__init__(**kwargs)
        self.label_shape = tuple(label_shape)

    def forward(self, vertices: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Vertex2VoxelEmbedder.

        Args:
            vertices (torch.Tensor): Input tensor of vertices with shape (B, C, N, 3),
                                     where B is batch size, C is number of channels,
                                     N is number of vertices, and 3 is for x, y, z coordinates.

        Returns:
            torch.Tensor: Processed voxel grid tensor.
        """
        # Scatter vertices into voxel grid
        voxel_grid = self.scatter_vertices(vertices)

        # Process the voxel grid
        processed_grid = self.process_voxel_grid(voxel_grid)

        return processed_grid

    def scatter_vertices(self, vertices: torch.Tensor) -> torch.Tensor:
        """
        Scatter vertices into a voxel grid.

        Args:
            vertices (torch.Tensor): Input tensor of vertices with shape (B, C, N, 3).

        Returns:
            torch.Tensor: Voxel grid tensor with shape (B, C, H, W, D).
        """
        B, C, N, _ = vertices.shape
        H, W, D = self.label_shape
        device = vertices.device

        # Initialize the voxel grid
        voxel_grid = torch.zeros(B, C, H, W, D, dtype=torch.float32, device=device)

        # Create a mask for valid vertices (not equal to -1)
        valid_mask = (vertices != -1).all(dim=-1)

        # Convert valid vertex coordinates to integer indices
        vertices_int = vertices.long()

        # Clamp the indices to ensure they are within the voxel grid bounds
        vertices_int[..., 0] = vertices_int[..., 0].clamp(0, H - 1)
        vertices_int[..., 1] = vertices_int[..., 1].clamp(0, W - 1)
        vertices_int[..., 2] = vertices_int[..., 2].clamp(0, D - 1)

        # Create batch and channel indices for valid vertices
        b_idx = torch.arange(B, device=device).view(B, 1, 1).expand(B, C, N)[valid_mask]
        c_idx = torch.arange(C, device=device).view(1, C, 1).expand(B, C, N)[valid_mask]

        # Get valid vertex indices
        valid_vertices = vertices_int[valid_mask]

        # Scatter the valid vertices into the voxel grid
        voxel_grid[
            b_idx,
            c_idx,
            valid_vertices[:, 0],
            valid_vertices[:, 1],
            valid_vertices[:, 2],
        ] = 1

        return voxel_grid
