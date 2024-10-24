import torch
import math
import warnings
from omegaconf import OmegaConf
import k_diffusion as K
import lightning as L
from tqdm import tqdm
import xarray as xr
from hydra.utils import instantiate
from typing import Tuple, List, Optional, Any, Dict
import os 
# Prepare functions
from src.utils.plotting_utils import AnatomicPlotter
from src.models.autoencoder import prepare_VAE
from src.models.diffusion_model import prepare_diffusion

# Utils
from src.utils.logging_utils import combine_metrics, log_anatomic_metrics
from src.utils.encoder_utils import prob2bool
from src.utils.load_utils import load_model_checkpoint
from src.utils.sample_utils import sample_euler_ancestral_mask

warnings.filterwarnings("ignore")


class AnatomyEngine(L.LightningModule):
    def __init__(
        self,
        data: OmegaConf,
        conditioner: OmegaConf,
        logger: OmegaConf,
        regressor: OmegaConf,
        autoencoder: OmegaConf,
        diffusion: OmegaConf,
        diffusion_checkpoint: OmegaConf,
        autoencoder_checkpoint: OmegaConf,
        regressor_checkpoint: OmegaConf,
        mask: OmegaConf,
        guidance: OmegaConf,
        sampling: OmegaConf,
        **kwargs: Any,
    ):
        """
        Initialize the AnatomyEngine.

        Args:
            data (OmegaConf): Configuration for the data module.
            conditioner (OmegaConf): Configuration for the conditioner.
            logger (OmegaConf): Configuration for the logger.
            regressor (OmegaConf): Configuration for the regressor.
            autoencoder (OmegaConf): Configuration for the autoencoder.
            diffusion (OmegaConf): Configuration for the diffusion model.
            diffusion_checkpoint (OmegaConf): Configuration for the diffusion model checkpoint.
            autoencoder_checkpoint (OmegaConf): Configuration for the autoencoder checkpoint.
            regressor_checkpoint (OmegaConf): Configuration for the regressor checkpoint.
            mask (OmegaConf): Configuration for the mask encoder.
            guidance (OmegaConf): Configuration for the guidance wrapper.
            sampling (OmegaConf): Configuration for sampling parameters.
            **kwargs (Any): Additional keyword arguments.

        This method sets up the AnatomyEngine by initializing various components
        including data loaders, encoders, models, and sampling configurations.
        """
        super().__init__()
        print("Initializing Anatomy Sampler")

        # Setting up anatomy specific dataloaders
        self.datamod = instantiate(data)
        self.training_loader = self.datamod.train_dataloader()
        self.val_loader = self.datamod.val_dataloader()

        # Setting up anatomic encoder components
        print("\nInitializing Conditioner")
        self.AnatomyConditioner = instantiate(conditioner, _recursive_=True)
        print("\nInitializing Logger")
        self.AnatomyLogger = instantiate(logger, _recursive_=True)
        print("\nInitializing Regressor")
        self.AnatomyRegressor = instantiate(regressor, _recursive_=True)
        self.AnatomyRegressor = load_model_checkpoint(
            self.AnatomyRegressor,
            regressor_checkpoint.folder,
            regressor_checkpoint.path,
        )
        # Setting up mask encoder
        self.MaskEncoder = instantiate(mask)

        # Setting up autoencoder model
        self.VAE = prepare_VAE(autoencoder, autoencoder_checkpoint)

        # Setting up diffusion model
        self.diffusion, self.diffusion_uncond, self.sample_density = prepare_diffusion(
            diffusion, diffusion_checkpoint
        )

        # Setting up sampling configuration
        self.sampling = sampling
        self.decode_bsz = (
            self.sampling.decode_bsz if "decode_bsz" in self.sampling else 1
        )

        # Setting up guidance wrapper
        self.guided_model = instantiate(guidance, AnatomyEngine=self)

        # Loading real metrics
        metrics_path = kwargs["real_metrics_path"]
        if os.path.isfile(metrics_path):
            self.anatomic_metrics_real = torch.load(metrics_path)
        else:
            self.anatomic_metrics_real = None
        # Plotter
        self.plotter = AnatomicPlotter()
        print("\nAnatomy Sampler Initialized")

    def forward(self, batch):
        """
        Forward pass of the AnatomyEngine.

        Args:
            batch (dict): Input batch of data.

        Returns:
            dict: Metrics computed from the sampling experiment.
        """
        metrics = self.sampling_experiment(**self.sampling)
        return metrics

    def sampling_experiment(
        self,
        k_steps: int,
        sigma_min: float,
        sigma_max: float,
        rho: float,
        n_samples: int,
        psi: float,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Run a sampling experiment with the specified parameters.

        Args:
            k_steps (int): Number of sampling steps.
            sigma_min (float): Minimum noise level.
            sigma_max (float): Maximum noise level.
            rho (float): Noise schedule parameter.
            n_samples (int): Number of samples to generate.
            psi (float): Diffuse-denoise scale.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            Dict[str, Any]: Computed anatomic metrics from the sampling experiment.
        """
        # Resetting metrics_batch
        anatomic_batch_metrics = xr.Dataset()
        # Finding the sigmas to denoise over
        sigmas = K.sampling.get_sigmas_karras(
            k_steps,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            rho=rho,
            device=self.device,
        )

        # Setting the loader
        if self.sampling.loader == "train":
            loader = self.training_loader
        elif self.sampling.loader == "val":
            loader = self.val_loader

        for i in range(n_samples):
            # Sampling seed, condition, and mask
            batch = next(iter(loader))
            seed = batch["label"]["data"].to(self.device)
            encodings = self.AnatomyConditioner({"x": seed})
            # Mask
            mask = self.MaskEncoder(seed)

            # Prepare latent in case of masking or diffuse denoise
            sampling_shape = [seed.shape[0]] + self.sampling.noise_dim
            latents_start, latents_clean, sigmas = self.prepare_latent(
                sampling_shape, sigmas, seed, mask, psi
            )

            # Sampling the final denoised latents
            print("Sample Number:", i, "/", n_samples)
            latents, x_hat = self.sample_anatomy(
                sigmas=sigmas,
                latents_start=latents_start,
                latents_clean=latents_clean,
                seed=seed,
                mask=mask,
                encodings=encodings,
            )

            # Computing batch metrics
            anatomic_metrics_single = self.compute_metrics_batch(x_hat, seed)
            # Combining batch metrics
            anatomic_batch_metrics = combine_metrics(
                anatomic_batch_metrics, anatomic_metrics_single
            )

        # Logging summary metrics and plots
        anatomic_metrics = log_anatomic_metrics(
            self.plotter,
            anatomic_batch_metrics,
            self.anatomic_metrics_real,
            x_hat,
            seed,
        )
        return anatomic_metrics

    def prepare_latent(
        self,
        sampling_shape: List[int],
        sigmas: torch.Tensor,
        seed: Optional[torch.Tensor],
        mask: Optional[torch.Tensor],
        psi: Optional[float],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """
        Prepare the latent tensors for sampling.

        Args:
            sampling_shape (List[int]): The shape of the sampling tensor.
            sigmas (torch.Tensor): The noise schedule.
            seed (Optional[torch.Tensor]): The seed tensor, if any.
            mask (Optional[torch.Tensor]): The mask tensor, if any.
            psi (Optional[float]): The diffusion strength parameter.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]: A tuple containing:
                - latents_start: The initial latent tensor to start sampling from.
                - latents_clean: The clean latent tensor (None if not applicable).
                - sigmas: The potentially modified noise schedule.
        """
        # sampling random noise for latents_start
        latents_start = torch.randn(list(sampling_shape)).to(self.device) * sigmas[0]
        # Setting default latents as None
        latents_clean = None
        if seed is None:
            return latents_start, latents_clean, sigmas

        # Setting up diffuse denoise if specified
        if psi is not None:
            crop = max(int(psi * len(sigmas)), 1)
            latent_noise = (
                torch.randn(list(sampling_shape)).to(self.device) * sigmas[-crop]
            )
            with torch.inference_mode():
                # Compute noise schedule and where to start denoising
                latents_clean = self.VAE.get_ldm_inputs(seed)
                latents_start = latents_clean + latent_noise
                sigmas = sigmas[-crop:]

        # If mask is provided encode it into latent
        if mask is not None:
            # Creating the latents
            with torch.inference_mode():
                latents_clean = self.VAE.encode(seed)[0]
        return latents_start, latents_clean, sigmas

    def compute_metrics_batch(
        self, x_hat: torch.Tensor, seed: Optional[torch.Tensor]
    ) -> xr.Dataset:
        """
        Compute evaluation and conditional metrics for a batch of generated samples.

        Args:
            x_hat (torch.Tensor): The generated samples (logits) to evaluate.
            seed (Optional[torch.Tensor]): The original seed images used for conditioning the model.

        Returns:
            xr.Dataset: A dataset containing both evaluation and conditional metrics.
        """
        # Binarizing the logits
        x_bin = prob2bool(x_hat)
        # Computing logging metrics for the sample
        batch_sample = {"x": x_bin}
        anatomic_eval_metrics = self.AnatomyLogger.log_anatomic_eval_metrics(
            batch_sample=batch_sample
        )

        # Computing conditioner metrics
        if seed is not None:
            batch_seed = {"x": seed}
            anatomic_cond_metrics = self.AnatomyConditioner.log_anatomic_cond_metrics(
                batch_sample=batch_sample, batch_seed=batch_seed
            )
        else:
            anatomic_cond_metrics = xr.Dataset()
        # Combining logger and conditioner metrics
        anatomic_metrics_single = xr.merge(
            [anatomic_eval_metrics, anatomic_cond_metrics]
        )
        return anatomic_metrics_single

    def sample_anatomy(
        self,
        sigmas: torch.Tensor,
        latents_start: torch.Tensor,
        latents_clean: Optional[torch.Tensor],
        seed: torch.Tensor,
        mask: torch.Tensor,
        encodings: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample anatomical structures using the diffusion model.

        Args:
            sigmas (torch.Tensor): The reverse noise schedule.
            latents_start (torch.Tensor): Initial latent to be denoised.
            latents_clean (Optional[torch.Tensor]): Clean latents, if any.
            seed (torch.Tensor): The seed tensor.
            mask (torch.Tensor): The mask tensor.
            encodings (Dict[str, torch.Tensor]): Encoded anatomical features.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - latents: The sampled latents.
                - x_hat: The decoded anatomical images.
        """
        callback = None
        # Setting guidance mechanism if any are selected
        model = self.guided_model
        # initializing the seed if needed
        model.init_seed(seed, mask)
        cond = encodings["anatomic_encoding"]
        # Sampling the latents
        latents = sample_euler_ancestral_mask(
            model,
            latents_start,
            mask,
            latents_clean,
            sigmas,
            extra_args={"cond": cond, "encodings": encodings},
            callback=callback,
            disable=False,
        )

        # Decoding the latents
        x_hat = self.decode_anatomy(latents)

        return latents, x_hat

    def decode_anatomy(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representations into anatomical images.

        This method decodes latent representations in batches to manage memory usage.

        Args:
            latents (torch.Tensor): Latent representations to decode.

        Returns:
            torch.Tensor: Decoded anatomical images.
        """
        with torch.no_grad():
            n_samples = self.decode_bsz
            n_rounds = math.ceil(latents.shape[0] / n_samples)
            all_out = []
            for n in range(n_rounds):
                latents_round = latents[n * n_samples : (n + 1) * n_samples]
                out = self.VAE.decode(latents_round)
                all_out.append(out)
        out = torch.cat(all_out, dim=0)
        return out

    def compute_real_metrics(
        self, nsamples: int, loader: torch.utils.data.DataLoader
    ) -> xr.Dataset:
        """
        Compute metrics on real data samples from the provided data loader.

        Args:
            nsamples (int): Number of samples to process.
            loader (torch.utils.data.DataLoader): DataLoader containing the real data samples.

        Returns:
            xr.Dataset: Dataset containing computed metrics for real data samples.
        """
        anatomic_metrics_batch = xr.Dataset()
        # Looping over train set and storing batch wise metrics
        for _ in tqdm(range(nsamples)):
            batch = next(iter(loader))
            x_hat = batch["label"]["data"].to(self.device)
            anatomic_metrics_single = self.compute_metrics_batch(x_hat=x_hat, seed=None)
            anatomic_metrics_batch = combine_metrics(
                anatomic_metrics_batch, anatomic_metrics_single
            )
        return anatomic_metrics_batch


# %%
