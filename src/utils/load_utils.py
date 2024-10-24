import os
import torch


def load_model_checkpoint(model, checkpoints_folder, checkpoint_path):
    """
    Load a model from a checkpoint.

    Args:
        model: The model to load the state into.
        checkpoints_folder (str): The folder containing the checkpoint.
        checkpoint_path (str): The name of the checkpoint file.

    Returns:
        The model with loaded state.
    """
    if checkpoint_path is not None:
        full_checkpoint_path = os.path.join(checkpoints_folder, checkpoint_path)
        checkpoint = torch.load(full_checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint)
        print(f"\nLoaded model {model.__class__.__name__} from {full_checkpoint_path}")
    return model
