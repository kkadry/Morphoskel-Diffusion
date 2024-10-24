import numpy as np
import torch
import torch.nn.functional as F
from typing import List


def gaussian_kernel(kernel_size=3, sigma=1.0):
    x = np.linspace(-sigma, sigma, kernel_size)
    [x, y, z] = np.meshgrid(x, x, x)
    d = np.sqrt(x * x + y * y + z * z)
    g = np.exp(-(d**2) / (2.0 * sigma**2))
    g /= g.sum()
    return torch.from_numpy(g).float()


def blur_voxel_grid_norm(voxel_grid, kernel_size, sigma):
    # Assumes voxel_grid input
    # Generate Gaussian kernel
    kernel = gaussian_kernel(kernel_size, sigma).to(voxel_grid.device)
    kernel = kernel.view(1, 1, kernel_size, kernel_size, kernel_size)

    # Get voxel grid dimensions
    N, C, H, W, D = voxel_grid.shape

    # Extend Gaussian kernel dimensions
    kernel = kernel.repeat(C, 1, 1, 1, 1)

    # Apply 3D convolutional operation
    blurred_grid = F.conv3d(voxel_grid, kernel, padding=(kernel_size // 2), groups=C)
    # normalizing to max 1
    if blurred_grid.max() > 0:
        blurred_grid = blurred_grid / blurred_grid.max()
    return blurred_grid


# The following functions are adapted from https://github.com/jocpae/clDice
def soft_erode(img):
    if len(img.shape) == 4:
        p1 = -F.max_pool2d(-img, (3, 1), (1, 1), (1, 0))
        p2 = -F.max_pool2d(-img, (1, 3), (1, 1), (0, 1))
        return torch.min(p1, p2)
    elif len(img.shape) == 5:
        p1 = -F.max_pool3d(-img, (3, 1, 1), (1, 1, 1), (1, 0, 0))
        p2 = -F.max_pool3d(-img, (1, 3, 1), (1, 1, 1), (0, 1, 0))
        p3 = -F.max_pool3d(-img, (1, 1, 3), (1, 1, 1), (0, 0, 1))
        return torch.min(torch.min(p1, p2), p3)


def soft_dilate(img):
    if len(img.shape) == 4:
        return F.max_pool2d(img, (3, 3), (1, 1), (1, 1))
    elif len(img.shape) == 5:
        return F.max_pool3d(img, (3, 3, 3), (1, 1, 1), (1, 1, 1))


def soft_open(img):
    return soft_dilate(soft_erode(img))


def soft_skel(img, iter_):
    img1 = soft_open(img)
    skel = F.relu(img - img1)
    for j in range(iter_):
        img = soft_erode(img)
        img1 = soft_open(img)
        delta = F.relu(img - img1)
        skel = skel + F.relu(delta - skel * delta)
    return skel


def consolidate_branch_points(indices: List[int], threshold: int = 1) -> List[int]:
    """
    Consolidate a list of integer indices to count the number of unique branch points and their representatives.

    Args:
        indices (List[int]): List of integer indices representing potential branch points.
        threshold (int): Maximum difference between indices to be considered part of the same branch point.

    Returns:
        List[int]: Representative indices for each unique branch point.
    """

    # Sort the list of indices
    sorted_indices = sorted(indices)

    # Initialize variables
    current_cluster = []
    representative_points = []

    # Iterate through the sorted list to identify clusters
    for idx in sorted_indices:
        if not current_cluster:
            current_cluster.append(idx)
        else:
            if idx - current_cluster[-1] <= threshold:
                current_cluster.append(idx)
            else:
                representative_points.append(
                    int(sum(current_cluster) / len(current_cluster))
                )
                current_cluster = [idx]

    # Count the last cluster and its representative if it exists
    if current_cluster:
        representative_points.append(int(sum(current_cluster) / len(current_cluster)))

    return representative_points


def croporpad(
    vertices: np.ndarray, target_length: int = 500, default_value: float = -1
) -> np.ndarray:
    """
    Crop or pad the input vertices to ensure a specific length.

    Args:
        vertices (np.ndarray): Input array of vertices with shape (N, 3).
        target_length (int): Desired length of the output array.
        default_value (float): Value to use for padding.

    Returns:
        np.ndarray: Array of vertices with shape (target_length, 3).
    """
    if vertices.shape[0] < target_length:
        # Pad the array
        padded = np.full((target_length, 3), default_value, dtype=vertices.dtype)
        padded[: vertices.shape[0]] = vertices
        return padded
    elif vertices.shape[0] > target_length:
        # Crop the array
        return vertices[:target_length]
    else:
        # Array is already the correct length
        return vertices
