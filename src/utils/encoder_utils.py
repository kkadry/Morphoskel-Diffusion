import torch
import time
from functools import wraps
from itertools import product


def count_params(model: torch.nn.Module, verbose: bool = False) -> int:
    """
    Count the number of parameters in a PyTorch model.

    Args:
        model (torch.nn.Module): The PyTorch model to count parameters for.
        verbose (bool, optional): If True, print the number of parameters. Defaults to False.

    Returns:
        int: The total number of parameters in the model.
    """
    total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"{model.__class__.__name__} has {total_params * 1.e-6:.2f} M params.")
    return total_params


def disabled_train(self, mode: bool = True) -> torch.nn.Module:
    """
    Overwrite model.train with this function to make sure train/eval mode
    does not change anymore.

    Args:
        self (torch.nn.Module): The model instance.
        mode (bool, optional): The mode to set (ignored). Defaults to True.

    Returns:
        torch.nn.Module: The model instance.
    """
    return self


def expand_dims_like(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Expand the dimensions of tensor x to match the number of dimensions of tensor y.

    Args:
        x (torch.Tensor): The tensor to expand.
        y (torch.Tensor): The tensor to match dimensions with.

    Returns:
        torch.Tensor: The expanded tensor x with the same number of dimensions as y.
    """
    while x.dim() != y.dim():
        x = x.unsqueeze(-1)
    return x


def prob2bool(img: torch.Tensor) -> torch.Tensor:
    """
    Convert a probability map tensor to a binary labelmap tensor.

    Args:
        img (torch.Tensor): Input probability map tensor of shape (B, C, H, W, D).

    Returns:
        torch.Tensor: Binary labelmap tensor of shape (B, C, H, W, D).
    """
    img_argmax = torch.argmax(img, 1)
    img_bool = torch.nn.functional.one_hot(img_argmax).permute(0, 4, 1, 2, 3)
    return img_bool


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f"Function {func.__name__} took {total_time:.4f} seconds")
        return result

    return timeit_wrapper


def batchify(func):
    """
    Decorator to batchify a function over specified channels in given dimensions.

    The decorated function should accept 'channels_dict' as a keyword argument,
    which is a dictionary mapping dimension indices to lists of channel indices.

    Args:
        func (callable): The function to be decorated. Should accept 'channels_dict' as a keyword argument.

    Returns:
        callable: The decorated function with batchify capability.
    """

    @wraps(func)
    def wrapper(self, inputs, *args, channels_dict=None, **kwargs):
        if channels_dict is None:
            raise ValueError(
                "`channels_dict` must be provided as a dictionary mapping dimensions to channels."
            )

        if not isinstance(inputs, torch.Tensor):
            raise TypeError("`inputs` must be a PyTorch tensor.")

        if not isinstance(channels_dict, dict):
            raise TypeError(
                "`channels_dict` must be a dictionary mapping dimensions to channel indices."
            )

        # Sort dimensions to maintain consistent ordering
        dims = sorted(channels_dict.keys())
        channels_per_dim = []

        for dim in dims:
            if dim < 0 or dim >= inputs.dim():
                raise ValueError(
                    f"Dimension {dim} is out of bounds for tensor with {inputs.dim()} dimensions."
                )

            channels = channels_dict[dim]

            if channels in (-1, None):
                # Process all channels in this dimension
                channels = list(range(inputs.size(dim)))
            elif isinstance(channels, (list, tuple)):
                # Validate channel indices
                channels = list(channels)
                for c in channels:
                    if c < 0 or c >= inputs.size(dim):
                        raise ValueError(
                            f"Channel index {c} is out of bounds for dimension {dim}."
                        )
            else:
                raise TypeError(
                    f"Channels for dimension {dim} must be a list, tuple, -1, or None."
                )

            channels_per_dim.append(channels)

        # Generate all combinations of channels across specified dimensions
        channel_combinations = list(product(*channels_per_dim))
        if not channel_combinations:
            raise ValueError("No channel combinations to process.")

        outputs = []
        for combo in channel_combinations:
            # Create slicing indices for each combination
            index = [slice(None)] * inputs.dim()
            for dim, c in zip(dims, combo):
                index[dim] = c
            input_slice = inputs[tuple(index)]

            # Apply the decorated function to the sliced input
            output_slice = func(self, input_slice, *args, **kwargs)

            if not isinstance(output_slice, torch.Tensor):
                raise TypeError("Output of the function must be a PyTorch tensor.")

            outputs.append(output_slice)

        # Stack the outputs along new dimensions corresponding to the channel_dict keys
        stacked_output = torch.stack(outputs, dim=0)

        # Reshape stacked_output to have separate dimensions for each key in channels_dict
        if len(dims) > 1:
            # Calculate the size for each new dimension
            new_dims_sizes = [len(channels) for channels in channels_per_dim]
            # Reshape the first dimensions to match the channel combinations
            stacked_output = stacked_output.view(
                *new_dims_sizes, *stacked_output.shape[1:]
            )
        else:
            # If only one dimension is batchified, keep the stacked dimension as is
            pass  # No additional reshaping needed
        # Move the output to the same device as the input
        stacked_output = stacked_output.to(inputs.device)
        return stacked_output

    return wrapper
