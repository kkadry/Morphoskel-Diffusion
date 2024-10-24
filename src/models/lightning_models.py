import matplotlib.pyplot as plt
import monai
import torch
import torch.distributed

import wandb
from src.experiment import AnatomyEngine
from src.utils.logging_utils import log_wandb_metrics


class DiffusionEngine(AnatomyEngine):
    """
    A class representing a diffusion engine for training diffusion models on anatomical segmentation with anatomical conditioning.

    This class extends the AnatomyEngine and implements specific functionality
    for training diffusion-based models in the context of anatomical structures.
    It handles the training process, including loss computation, optimization,
    and checkpoint management for diffusion models, with a focus on incorporating
    anatomical conditioning information derived from anatomical segmentations.

    The diffusion model is trained to generate anatomical images conditioned on
    specific anatomical features or structures extracted from segmentation maps.
    This allows for more controlled and anatomically accurate image synthesis,
    where the generated images are consistent with the provided anatomical segmentations.

    Args:
        **kwargs: Keyword arguments containing configuration parameters.
                  Should include:
                  - training (dict): Configuration parameters for training settings.
                  - diffusion_checkpoint (dict): Configuration for model checkpoint handling.
                  - Other parameters related to anatomical conditioning,
                    particularly those concerning the processing and utilization
                    of anatomical segmentation data.

    Attributes:
        training_config (dict): Configuration parameters for diffusion model training,
                                including settings specific to anatomical conditioning
                                based on segmentation maps.
        checkpoint (dict): Checkpoint information for saving and loading the anatomically
                           conditioned diffusion model.

    """

    def __init__(self, **kwargs):
        """
        Initialize the DiffusionEngine for anatomically conditioned diffusion model training.

        Args:
            **kwargs: Keyword arguments containing configuration parameters.
                      Should include:
                      - training (dict): Configuration parameters for training settings.
                      - diffusion_checkpoint (dict): Configuration for model checkpoint handling.
                      - Other parameters related to anatomical conditioning,
                        particularly those concerning the processing and utilization
                        of anatomical segmentation data.

        """
        super(DiffusionEngine, self).__init__(**kwargs)
        self.training_config = kwargs["training"]
        self.checkpoint = kwargs["diffusion_checkpoint"]

    def forward(self, x):
        out = self.diffusion(x)
        return out

    def shared_step(self, batch):
        x = batch["label"]["data"].float()

        # Encoding with VAE
        with torch.no_grad():
            e = self.VAE.get_ldm_inputs(x)
        # Computing the condition embeddings
        encodings = self.AnatomyConditioner({"x": x})
        cond = encodings["anatomic_encoding"]
        # Drawing a random gaussian sample
        noise = torch.randn_like(e)

        # Drawing random noise level
        sigma = self.sample_density([self.sampling.batch_size], device=x.device)

        # Calculating EDM loss
        loss_batch = self.diffusion.loss(e, noise, sigma, **{"cond": cond})
        loss = loss_batch.mean()

        return loss, loss_batch, sigma

    def training_step(self, batch, batch_idx):
        loss, loss_batch, sigma = self.shared_step(batch)
        # Logging
        self.log_dict({"train_loss": loss})
        return loss

    def validation_step(self, batch, batch_idx):
        loss, loss_batch, sigma = self.shared_step(batch)
        # Logging
        self.log_dict({"val_loss": loss})
        return loss

    def on_validation_epoch_end(self):
        # Saving the checkpoint of the diffusion model
        state_dict = self.diffusion.state_dict()
        torch.save(state_dict, f"{self.checkpoint.folder}/{self.checkpoint.path_save}")

        epochs_thresh = self.training_config.log_thresh_epochs
        epochs_interval = self.training_config.log_interval_epochs
        # Logging
        if (
            self.current_epoch > epochs_thresh
            and self.current_epoch % epochs_interval == 0
        ):
            # Running a sampling loop
            metrics = self.sampling_experiment(**self.sampling)
            log_wandb_metrics(metrics, self.logger)

    def configure_optimizers(self):
        params = self.diffusion.parameters()
        optimizer = torch.optim.Adam(params, lr=self.training_config.lr)
        return optimizer


class AutoencoderEngine(AnatomyEngine):
    """
    AutoencoderEngine class for training and evaluating a Variational Autoencoder (VAE) model.

    This class extends the AnatomyEngine class and provides functionality for training
    and evaluating a VAE model on anatomical data. It includes methods for forward pass,
    training step, and validation step.

    Args:
        **kwargs: Keyword arguments passed to the parent AnatomyEngine class.
                  Should include:
                  - training (dict): Configuration object for training parameters.
                  - autoencoder_checkpoint (dict): Configuration object for autoencoder checkpoints.

    Attributes:
        training_config (dict): Configuration object for training parameters.
        checkpoint (dict): Configuration object for autoencoder checkpoints.
        recons_fn (monai.losses.DiceCELoss): Loss function for reconstruction.
        dice_score (monai.metrics.DiceMetric): Metric for evaluating segmentation quality.
        kl_weight (float): Weight for the KL divergence loss term.
        topo_int_weight (float): Weight for the topological interaction loss term.
    """

    def __init__(self, **kwargs):
        super(AutoencoderEngine, self).__init__(**kwargs)
        self.training_config = kwargs["training"]
        self.checkpoint = kwargs["autoencoder_checkpoint"]

        self.recons_fn = monai.losses.DiceCELoss(
            include_background=True, softmax=True, smooth_dr=0.0001, smooth_nr=0.0001
        )
        self.dice_score = monai.metrics.DiceMetric(include_background=False)

        self.kl_weight = self.training_config.kl_weight
        self.topo_int_weight = self.training_config.topo_int_weight

    def forward(self, x):
        out = self.VAE(x)
        return out

    def training_step(self, batch, batch_idx):
        img = batch["label"]["data"].float()
        img_recon, z_mu, z_sigma = self.VAE(img)
        # Reconstruction loss
        recons_loss = self.recons_fn(img_recon, img)
        # KL Loss
        kl_loss = 0.5 * torch.sum(
            z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1,
            dim=[1, 2, 3, 4],
        )
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
        # Topological loss
        if self.topo_int_weight > 0.0:
            img_recon_bin = torch.nn.functional.softmax(img_recon / 0.1, dim=1)
            # topo_loss=self.AnatomyRegressor.topo_encoder.embedders[0].compute_cond_loss(img_recon_bin)
            topo_loss = self.AnatomyRegressor({"x": img_recon_bin})[
                "anatomic_encoding"
            ]["vector"].mean()
        else:
            topo_loss = torch.tensor(0.0).to(img)
        loss = recons_loss + self.kl_weight * kl_loss + topo_loss * self.topo_int_weight
        # Logging
        self.log_dict(
            {
                "train_recons_loss": recons_loss,
                "train_kl_loss": kl_loss * self.kl_weight,
                "train_topo_loss": topo_loss * self.topo_int_weight,
                "train_loss": loss,
                "latent_std": z_sigma.mean(),
                "latent_mu": z_mu.mean(),
            }
        )
        return loss

    def validation_step(self, batch, batch_idx):
        img = batch["label"]["data"].float()
        img_recon, z_mu, z_sigma = self.VAE(img)
        # Reconstruction loss
        recons_loss = self.recons_fn(img_recon, img)
        # KL Loss
        kl_loss = 0.5 * torch.sum(
            z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1,
            dim=[1, 2, 3, 4],
        )
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
        loss = recons_loss + self.kl_weight * kl_loss

        dice_score = self.dice_score(img_recon, img)
        # Logging the loss
        self.log_dict(
            {
                "val_loss": loss.mean().item(),
                "val_kl_loss": kl_loss.mean().item() * self.kl_weight,
                "val_dice_score": dice_score.mean().item(),
            }
        )
        return loss

    def on_validation_epoch_end(self):
        # Loop over n samples and collect the images
        dataloader = self.trainer.val_dataloaders
        img_compare = []
        for i in range(1):
            batch = next(iter(dataloader))
            img = batch["label"]["data"].float().to(self.device)
            # Autoencoding
            img_recon, z_mu, z_sigma = self.VAE(img)
            # Appending
            img_recon_slice = img_recon[0, :, :, 64].argmax(0).cpu().float()
            img_slice = img[0, :, :, 64].argmax(0).cpu().float()
            img_compare.append(torch.cat([img_slice, img_recon_slice], dim=0))

        # Saving to wandb
        # Log the img_compare list of images to wandb
        self.logger.experiment.log({"val_reconstructions": wandb.Image(img_compare[0])})
        # Saving the checkpoint of the VAE
        state_dict = self.VAE.state_dict()
        torch.save(state_dict, f"{self.checkpoint.folder}/{self.checkpoint.path_save}")

    def configure_optimizers(self):
        params = self.VAE.parameters()
        optimizer = torch.optim.Adam(params, lr=self.training_config.lr)
        return optimizer


class RegressorEngine(AnatomyEngine):
    """
    RegressorEngine is a PyTorch Lightning module for regressing morphological and skeletal attributes.

    This engine trains regressors to predict morphological and skeletal features from input images.
    These predictions are intended to be used later for guiding diffusion models in anatomically-aware
    image generation or manipulation tasks.

    Args:
        **kwargs: Keyword arguments. Should include:
                  - training (dict): Training configuration parameters.
                  - regressor_checkpoint (dict): Checkpoint information for the regressor models.

    Attributes:
        training_config (dict): Configuration for training.
        checkpoint (dict): Checkpoint information for saving and loading models.

    The class inherits from AnatomyEngine and extends its functionality with specific
    methods for morphological and skeletal attribute regression.
    """

    def __init__(self, **kwargs):
        """
        Initialize the RegressorEngine.

        Sets up the training configuration and checkpoint information for the regressor models.

        Args:
            **kwargs: Keyword arguments. Should include:
                      - training (dict): Training configuration parameters.
                      - regressor_checkpoint (dict): Checkpoint information for the regressor models.
        """
        super(RegressorEngine, self).__init__(**kwargs)
        self.training_config = kwargs["training"]
        self.checkpoint = kwargs["regressor_checkpoint"]

    def compute_skel_loss(self, cond_pred, cond_target):
        skel_loss = 0
        for key in cond_pred.keys():
            cond_target_item = cond_target[key]
            cond_pred_item = cond_pred[key]
            # loss = torch.nn.BCEWithLogitsLoss()(cond_pred_item, cond_target_item)
            loss = torch.nn.MSELoss()(cond_pred_item, cond_target_item)
            skel_loss += loss
        return torch.mean(skel_loss)

    def compute_morph_loss(self, cond_pred, cond_target):
        morph_loss = 0
        for key in cond_pred.keys():
            cond_target_item = cond_target[key]
            cond_pred_item = cond_pred[key]
            loss = torch.nn.MSELoss()(cond_pred_item, cond_target_item)
            morph_loss += loss
        return torch.mean(morph_loss)

    def forward(self, x):
        img = x["label"]["data"].float()
        cond_morph = self.morph_regressor(img)
        cond_skel = self.skel_regressor.module(img)
        out = {"cond_morph": cond_morph, "cond_skel": cond_skel}
        return out

    def shared_step(self, batch):
        x = batch["label"]["data"].float()
        # Calculating the conditioning information
        encodings = self.AnatomyConditioner({"x": x})
        # Forward pass of the regressors
        encodings_pred = self.AnatomyRegressor({"x": x.cuda()})

        # Morphological loss
        morph_loss = self.compute_morph_loss(
            encodings_pred["morph_encoding"], encodings["morph_encoding"]
        )
        # Skeletal loss
        skel_loss = self.compute_skel_loss(
            encodings_pred["skel_encoding"], encodings["skel_encoding"]
        )
        loss = morph_loss + skel_loss
        results = {
            "loss": loss,
            "morph_loss": morph_loss,
            "skel_loss": skel_loss,
            "encodings_pred": encodings_pred,
            "encodings": encodings,
        }
        return results

    def training_step(self, batch, batch_idx):
        results = self.shared_step(batch)
        # Logging
        self.log_dict(
            {"morph_loss": results["morph_loss"], "skel_loss": results["skel_loss"]}
        )
        return results["loss"]

    def validation_step(self, batch, batch_idx):
        results = self.shared_step(batch)
        # Logging
        self.log_dict(
            {
                "morph_val_loss": results["morph_loss"],
                "skel_val_loss": results["skel_loss"],
            }
        )
        return results["loss"]

    def on_validation_epoch_end(self):
        # Perform shared step on a validation batch
        val_batch = self.transfer_batch_to_device(
            next(iter(self.datamod.val_dataloader())),
            device=self.device,
            dataloader_idx=0,
        )
        results = self.shared_step(val_batch)

        encodings_pred = results["encodings_pred"]
        encodings = results["encodings"]
        # Plotting morphological encodings
        fig = plt.figure()
        plt.plot(encodings["morph_encoding"]["concat"][0, 0, 0, 0].detach().cpu())
        plt.plot(encodings_pred["morph_encoding"]["concat"][0, 0, 0, 0].detach().cpu())
        self.logger.experiment.log({"morph_encodings": fig})
        plt.clf()
        # Plotting skeletal encodings
        fig, axs = plt.subplots(1, 2)
        axs[0].imshow(encodings["skel_encoding"]["concat"][0, 0, 16].detach().cpu())
        axs[1].imshow(
            encodings_pred["skel_encoding"]["concat"][0, 0, 16].detach().cpu()
        )
        self.logger.experiment.log({"skel_encodings": fig})
        plt.clf()
        # Saving the checkpoint of the morph regressor
        state_dict = self.AnatomyRegressor.state_dict()
        torch.save(state_dict, f"{self.checkpoint.folder}/{self.checkpoint.path_save}")

    def configure_optimizers(self):
        params = self.AnatomyRegressor.parameters()
        optimizer = torch.optim.Adam(params, lr=self.training_config.lr)
        return optimizer


class DiffusionSamplerEngine(AnatomyEngine):
    """
    A class for sampling from a diffusion model.

    This class extends AnatomyEngine to provide functionality for running
    sampling experiments using a diffusion model.

    Attributes:
        Inherits all attributes from AnatomyEngine.

    Methods:
        forward(batch): Runs a sampling experiment and returns metrics.
    """

    def __init__(self, **kwargs):
        super(DiffusionSamplerEngine, self).__init__(**kwargs)

    def forward(self, batch):
        """
        Runs a sampling experiment using the diffusion model.

        Args:
            batch: Input batch of data (not used in this method).

        Returns:
            dict: Dictionary containing metrics from the sampling experiment.
        """
        metrics = self.sampling_experiment(**self.sampling)
        return metrics
