import torch
from tqdm import trange
from k_diffusion.sampling import default_noise_sampler, get_ancestral_step, to_d
import k_diffusion as K


@torch.no_grad()
def sample_euler_ancestral_mask(
    model: torch.nn.Module,
    x: torch.Tensor,
    mask: torch.Tensor,
    latents_0: torch.Tensor,
    sigmas: torch.Tensor,
    extra_args: dict = None,
    callback: callable = None,
    disable: bool = None,
    eta: float = 1.0,
    s_noise: float = 1.0,
    noise_sampler: callable = None,
) -> torch.Tensor:
    """
    Ancestral sampling with Euler method steps and masking functionality.

    This is an adapted version from https://github.com/crowsonkb/k-diffusion/blob/master/k_diffusion/sampling.py
    with masking functionality built in.

    Args:
        model (torch.nn.Module): The model to use for denoising.
        x (torch.Tensor): The initial noise tensor.
        mask (torch.Tensor): The mask to apply during sampling.
        latents_0 (torch.Tensor): The initial latent tensor.
        sigmas (torch.Tensor): The noise levels to use for sampling.
        extra_args (dict, optional): Extra arguments to pass to the model.
        callback (callable, optional): A function to call after each step.
        disable (bool, optional): Whether to disable the progress bar.
        eta (float, optional): The eta parameter for ancestral sampling.
        s_noise (float, optional): The amount of noise to add at each step.
        noise_sampler (callable, optional): A function to sample noise.

    Returns:
        torch.Tensor: The final denoised sample.
    """
    """Ancestral sampling with Euler method steps."""
    extra_args = {} if extra_args is None else extra_args
    noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])
    # Storing the original noise seed
    noise = x.clone().detach()

    disable = True
    for i in trange(len(sigmas) - 1, disable=disable):
        noise = torch.randn_like(noise)
        # Fusing the current denoised output and noised seed image
        if mask is not None:
            x_orig = latents_0 + noise * K.utils.append_dims(sigmas[i], x.ndim)
            x = x_orig * mask + (1.0 - mask) * x

        denoised = model(x, sigmas[i] * s_in, **extra_args)
        sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)
        if callback is not None:
            callback(
                {
                    "x": x,
                    "i": i,
                    "sigma": sigmas[i],
                    "sigma_hat": sigmas[i],
                    "denoised": denoised,
                }
            )
        d = to_d(x, sigmas[i], denoised)
        # Euler method
        dt = sigma_down - sigmas[i]
        x = x + d * dt
        if sigmas[i + 1] > 0:
            x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * sigma_up
    return x


@torch.no_grad()
def sample_heun_mask(
    model,
    x,
    mask,
    latents_0,
    sigmas,
    extra_args=None,
    callback=None,
    disable=None,
    s_churn=0.0,
    s_tmin=0.0,
    s_tmax=float("inf"),
    s_noise=1.0,
):
    """
    Implements Algorithm 2 (Heun steps) from Karras et al. (2022) with masking functionality.

    This is an adapted version of the Heun sampling method with added masking capabilities.

    Args:
        model (torch.nn.Module): The model to use for denoising.
        x (torch.Tensor): The initial noise tensor.
        mask (torch.Tensor): The mask to apply during sampling.
        latents_0 (torch.Tensor): The initial latent tensor.
        sigmas (torch.Tensor): The noise levels to use for sampling.
        extra_args (dict, optional): Extra arguments to pass to the model.
        callback (callable, optional): A function to call after each step.
        disable (bool, optional): Whether to disable the progress bar.
        s_churn (float, optional): The amount of noise to add at each step.
        s_tmin (float, optional): The minimum sigma value for adding noise.
        s_tmax (float, optional): The maximum sigma value for adding noise.
        s_noise (float, optional): The scale of the added noise.

    Returns:
        torch.Tensor: The final denoised sample.
    """
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    noise = x.clone().detach()
    for i in trange(len(sigmas) - 1, disable=disable):
        noise = torch.randn_like(noise)
        # Masking before the denoising
        if mask is not None:
            x_orig = latents_0 + noise * K.utils.append_dims(sigmas[i], x.ndim)
            x = x_orig * mask + (1.0 - mask) * x

        gamma = (
            min(s_churn / (len(sigmas) - 1), 2**0.5 - 1)
            if s_tmin <= sigmas[i] <= s_tmax
            else 0.0
        )
        eps = torch.randn_like(x) * s_noise
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            x = x + eps * (sigma_hat**2 - sigmas[i] ** 2) ** 0.5

        # Denoising
        denoised = model(x, sigma_hat * s_in, **extra_args)
        d = to_d(x, sigma_hat, denoised)
        if callback is not None:
            callback(
                {
                    "x": x,
                    "i": i,
                    "sigma": sigmas[i],
                    "sigma_hat": sigma_hat,
                    "denoised": denoised,
                }
            )
        dt = sigmas[i + 1] - sigma_hat
        if sigmas[i + 1] == 0:
            # Euler method
            x = x + d * dt
        else:
            # Heun's method
            x_2 = x + d * dt
            denoised_2 = model(x_2, sigmas[i + 1] * s_in, **extra_args)
            d_2 = to_d(x_2, sigmas[i + 1], denoised_2)
            d_prime = (d + d_2) / 2
            x = x + d_prime * dt
    return x
