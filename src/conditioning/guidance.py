import torch
from typing import Any, Dict, Optional


# Abstract class
class Guidance:
    def __init__(self, **kwargs):
        pass

    def __call__(self, x: torch.Tensor, sigma: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Apply guidance to the input tensor.

        This method either applies diffusion or computes the denoised tensor based on the weight.

        Args:
            x (torch.Tensor): The input tensor to be processed.
            sigma (torch.Tensor): The noise level.
            **kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor: The processed tensor after applying guidance.
        """
        if self.weight == 1:
            return self.diffusion(x, sigma, **kwargs)
        else:
            return self.compute_denoised(x, sigma, **kwargs)

    def compute_denoised(self, x, sigma):
        raise NotImplementedError

    def init_seed(self, seed, mask):
        return None

    def decode_voxel(self, x, denoised_cond):
        raise NotImplementedError

    def process(self, denoised_voxel):
        raise NotImplementedError

    def compute_grad(self, denoised_voxel, target):
        raise NotImplementedError

    def compute_loss(self, processed, target):
        raise NotImplementedError

    def compute_null_cond(self, denoised_voxel, cond):
        raise NotImplementedError


class IdentityGuidance(Guidance):
    """
    A guidance class that simply applies the diffusion model without any additional processing.
    """

    def __init__(self, AnatomyEngine: Any, **kwargs: Any) -> None:
        """
        Initialize the IdentityGuidance.

        Args:
            AnatomyEngine (Any): The AnatomyEngine object containing the diffusion model.
            **kwargs: Additional keyword arguments passed to the parent class.
        """
        super().__init__(**kwargs)
        self.diffusion = AnatomyEngine.diffusion

    def __call__(
        self, x: torch.Tensor, sigma: torch.Tensor, **kwargs: Any
    ) -> torch.Tensor:
        """
        Apply the diffusion model without any additional guidance.

        Args:
            x (torch.Tensor): The input tensor to be processed.
            sigma (torch.Tensor): The noise level.
            **kwargs: Additional keyword arguments passed to the diffusion model.

        Returns:
            torch.Tensor: The output of the diffusion model.
        """
        return self.diffusion(x, sigma, **kwargs)


class NullGuidance(Guidance):
    def __init__(self, weight, AnatomyEngine, **kwargs):
        super().__init__(**kwargs)
        self.weight = weight
        self.AnatomyEngine = AnatomyEngine
        self.AnatomyConditioner = self.AnatomyEngine.AnatomyConditioner
        self.diffusion = self.AnatomyEngine.diffusion
        self.diffusion_null = self.AnatomyEngine.diffusion
        self.VAE = self.AnatomyEngine.VAE

    def decode_voxel(
        self, x: torch.Tensor, denoised_cond: torch.Tensor
    ) -> torch.Tensor:
        """
        Decode the denoised latent into voxel space and apply soft binarization.

        Args:
            x (torch.Tensor): The input tensor (not used in this method).
            denoised_cond (torch.Tensor): The denoised latent tensor to be decoded.

        Returns:
            torch.Tensor: The decoded and soft-binarized voxel tensor.
        """
        with torch.no_grad():
            denoised_voxel_logits = self.VAE.decode(denoised_cond)
        # Soft binarizing
        denoised_voxel = torch.nn.functional.softmax(denoised_voxel_logits / 0.1, dim=1)
        return denoised_voxel

    def compute_adaptive_encoding(
        self,
        encoding_target: Dict[str, torch.Tensor],
        encoding_sample: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute adaptive null encoding based on the target encoding and optionally a sample encoding.

        Args:
            encoding_target (Dict[str, torch.Tensor]): The target encoding dictionary.
            encoding_sample (Optional[Dict[str, torch.Tensor]]): The sample encoding dictionary, if available.

        Returns:
            Dict[str, torch.Tensor]: The computed null encoding dictionary.
        """
        encoding_null = {}
        for cond_key in encoding_target.keys():
            if encoding_sample is not None:
                encoding_null[cond_key] = (
                    2 * encoding_sample[cond_key] - encoding_target[cond_key]
                )
            else:
                encoding_null[cond_key] = torch.zeros_like(encoding_target[cond_key])
        return encoding_null

    def compute_null_cond(self, denoised_voxel, encodings, **kwargs):
        raise NotImplementedError

    def compute_denoised(
        self, x: torch.Tensor, sigma: torch.Tensor, **kwargs: Any
    ) -> torch.Tensor:
        """
        Compute the denoised output using the diffusion model and apply guidance.

        This method performs the following steps:
        1. Computes the denoised latent using the diffusion model conditioned by kwargs['cond'].
        2. Decodes the denoised output into voxel space.
        3. Computes the null condition based on the denoised voxel and other parameters.
        4. Computes the null-conditioned denoised latent using the null condition.
        5. Combines the conditional and null-conditioned latents using the guidance weight.

        Args:
            x (torch.Tensor): The input tensor to be denoised.
            sigma (torch.Tensor): The noise level tensor.
            **kwargs: Additional keyword arguments, including 'cond' (condition) and 'encodings'.

        Returns:
            torch.Tensor: The weighted combination of conditional and unconditioned denoised outputs.
        """
        # Finding the denoised output and the associated conditional vector
        denoised_cond = self.diffusion(x, sigma, **kwargs)
        # Decoding into voxel space
        denoised_voxel = self.decode_voxel(x, denoised_cond)

        null_cond = self.compute_null_cond(denoised_voxel, **kwargs)

        # Denoise with null condition
        self.kwargs_null_cond = {"cond": null_cond}
        denoised_uncond = self.diffusion_null(x, sigma, **self.kwargs_null_cond)

        # Taking the barycentric mean of the predictions
        denoised_weighted = denoised_uncond + self.weight * (
            denoised_cond - denoised_uncond
        )
        return denoised_weighted


class AdaptiveNullMorphGuidance(NullGuidance):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def compute_null_cond(
        self,
        denoised_voxel: torch.Tensor,
        encodings: Dict[str, torch.Tensor],
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the null condition for adaptive null morphological guidance.

        This method calculates the null condition by:
        1. Computing the morphological encoding from the denoised prediction.
        2. Calculating an adaptive null morphological encoding.
        3. Computing null skeletal and topological encodings as zeros.
        4. Combining all null encodings into a single condition dictionary.

        Args:
            denoised_voxel (torch.Tensor): The denoised voxel prediction.
            encodings (Dict[str, torch.Tensor]): A dictionary containing the original encodings.
            **kwargs: Additional keyword arguments.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing the computed null condition.
        """
        batch_sample = {"x": denoised_voxel}

        # Computing the morphological encoding from the denoised prediction
        synth_morph_encoding = self.AnatomyConditioner.morph_encoder(batch_sample)

        # Computing the null morph condition as 2*synth_cond-cond=synth_cond+(synth_cond-cond)
        null_morph_encoding = self.compute_adaptive_encoding(
            encoding_target=encodings["morph_encoding"],
            encoding_sample=synth_morph_encoding,
        )

        # Computing the null skeletal encoding as zeros
        null_skel_encoding = self.compute_adaptive_encoding(
            encoding_target=encodings["skel_encoding"], encoding_sample=None
        )

        # Computing the null topological encoding as zeros
        null_topo_encoding = self.compute_adaptive_encoding(
            encoding_target=encodings["topo_encoding"], encoding_sample=None
        )

        # Combining the encodings into cond dictionary
        null_cond = self.AnatomyConditioner.combine_encodings(
            [null_morph_encoding, null_skel_encoding, null_topo_encoding]
        )
        return null_cond


class ClassifierFreeGuidance(NullGuidance):
    """
    A guidance class that computes the null condition as zeros
    """

    def __init__(self, weight, AnatomyEngine, **kwargs):
        super().__init__(weight, AnatomyEngine, **kwargs)
        self.weight = weight
        self.diffusion_null = self.AnatomyEngine.diffusion_uncond

    def decode_voxel(self, x, denoised_cond, **kwargs):
        return None

    def compute_null_cond(self, denoised_voxel, encodings, cond, **kwargs):
        # Computing the null encoding as zeros
        null_cond = self.compute_adaptive_encoding(
            encoding_target=cond, encoding_sample=None
        )
        return null_cond


class LossGuidance(Guidance):
    def __init__(self, weight, AnatomyEngine, **kwargs):
        super().__init__(**kwargs)
        self.weight = weight
        self.gamma = weight - 1
        self.AnatomyEngine = AnatomyEngine
        self.AnatomyConditioner = self.AnatomyEngine.AnatomyConditioner
        self.diffusion = self.AnatomyEngine.diffusion
        self.VAE = self.AnatomyEngine.VAE

    def compute_denoised(
        self, x: torch.Tensor, sigma: torch.Tensor, **kwargs: Any
    ) -> torch.Tensor:
        """
        Compute the denoised prediction with gradient-based guidance.

        This method performs the following steps:
        1. Detaches and clones the input tensor, enabling gradient computation.
        2. Applies the diffusion model to get the conditional denoised prediction.
        3. Decodes and soft-binarizes the denoised prediction.
        4. Computes a loss based on the denoised voxel.
        5. Calculates the gradient of the loss with respect to the input.
        6. Applies gradient-based guidance to the denoised prediction.

        Args:
            x (torch.Tensor): The input tensor to be denoised.
            sigma (torch.Tensor): The noise level.
            **kwargs: Additional keyword arguments passed to the diffusion model and other methods.

        Returns:
            torch.Tensor: The denoised prediction with applied guidance.
        """
        with torch.inference_mode(False):
            x = x.detach().clone().requires_grad_()
            # Denoising and decoding
            denoised_cond = self.diffusion(x, sigma, **kwargs)

            # Decoding and softbinarizing
            denoised_voxel = self.decode_voxel(x, denoised_cond)

            loss = self.compute_loss(x, denoised_voxel, **kwargs)
        # Calculating gradient
        cond_grad = self.compute_grad(x, denoised_cond, loss)

        denoised_weighted = denoised_cond.detach() + self.gamma * cond_grad
        return denoised_weighted

    def decode_voxel(self, x, denoised_cond):
        raise NotImplementedError

    def compute_loss(self, x, denoised_voxel, encodings):
        raise NotImplementedError

    def compute_grad(self, x, denoised_cond, loss):
        raise NotImplementedError


class MorphSkelLossGuidance(LossGuidance):
    """
    A guidance class that computes a loss based on the morphological and skeletal encodings using the anatomic conditioner.
    """

    def __init__(self, weight, AnatomyEngine, **kwargs):
        super().__init__(weight, AnatomyEngine, **kwargs)
        self.morph_regressor = self.AnatomyConditioner.morph_encoder
        self.skel_regressor = self.AnatomyConditioner.skel_encoder

    def compute_encoding_loss(self, synth_encoding, target_encoding, sigmoid=False):
        # Assert that the keys are the same length
        assert len(synth_encoding.keys()) == len(target_encoding.keys())
        loss = []
        for key in synth_encoding.keys():
            # Need to make a copy of the target encoding as it is an inference tensor
            target_encoding_copy = target_encoding[key].clone()
            if sigmoid:
                diff_encoding = torch.sigmoid(synth_encoding[key]) - torch.sigmoid(
                    target_encoding_copy
                )
            else:
                diff_encoding = synth_encoding[key] - target_encoding_copy
            loss.append(torch.linalg.norm(diff_encoding.flatten()))
        loss = torch.mean(torch.stack(loss))
        return loss

    def compute_loss(self, x, denoised_voxel, encodings, **kwargs):
        # Computing morphological loss
        batch_sample = {"x": denoised_voxel}
        synth_morph_encoding = self.morph_regressor(batch_sample, track_gradients=True)
        cond_loss_morph = self.compute_encoding_loss(
            synth_morph_encoding, encodings["morph_encoding"]
        )

        # Computing skeletal loss
        synth_skel_encoding = self.skel_regressor(batch_sample, track_gradients=True)
        cond_loss_skel = self.compute_encoding_loss(
            synth_skel_encoding, encodings["skel_encoding"]
        )
        # Combining losses
        loss = cond_loss_morph + cond_loss_skel
        return loss


class MorphSkelLossNNGuidance(MorphSkelLossGuidance):
    """
    A guidance class that computes a loss based on the morphological and skeletal encodings using a neural network regressor.
    """

    def __init__(self, weight, AnatomyEngine, **kwargs):
        super().__init__(weight, AnatomyEngine, **kwargs)
        self.morph_regressor = AnatomyEngine.AnatomyRegressor.morph_encoder
        self.skel_regressor = AnatomyEngine.AnatomyRegressor.skel_encoder

    def compute_loss(self, x, denoised_voxel, encodings, **kwargs):
        # Computing morphological loss
        batch_sample = {"x": denoised_voxel}
        synth_morph_encoding = self.morph_regressor(batch_sample, track_gradients=True)
        cond_loss_morph = self.compute_encoding_loss(
            synth_morph_encoding, encodings["morph_encoding"]
        )

        # Computing skeletal loss
        synth_skel_encoding = self.skel_regressor(batch_sample, track_gradients=True)
        cond_loss_skel = self.compute_encoding_loss(
            synth_skel_encoding, encodings["skel_encoding"], sigmoid=True
        )
        # Combining losses
        loss = cond_loss_morph + cond_loss_skel
        return loss


class DPSGuidance(LossGuidance):
    """
    When calculating the loss, use the decoded clean latent prediction
    """

    def __init__(self, weight, AnatomyEngine, **kwargs):
        super().__init__(weight, AnatomyEngine, **kwargs)

    def decode_voxel(self, x, denoised_cond):
        denoised_voxel_logits = self.VAE.decode(denoised_cond)
        # Soft binarizing
        denoised_voxel = torch.nn.functional.softmax(denoised_voxel_logits / 0.1, dim=1)
        return denoised_voxel

    def compute_grad(self, x, denoised, loss):
        # Taking gradient with respect to denoised latent
        cond_grad = -torch.autograd.grad(loss, denoised)[0]
        return cond_grad


class CGGuidance(DPSGuidance):
    """
    When calculating the loss, use the decoded intermediately noised latent
    """

    def decode_voxel(self, x, denoised_cond):
        denoised_voxel_logits = self.VAE.decode(x)
        # Soft binarizing
        denoised_voxel = torch.nn.functional.softmax(denoised_voxel_logits / 0.1, dim=1)
        return denoised_voxel

    def compute_grad(self, x, denoised, loss):
        # Taking gradient with respect to denoised latent
        cond_grad = -torch.autograd.grad(loss, x)[0]
        return cond_grad


class DPSMorphoSkelGuidance(DPSGuidance, MorphSkelLossGuidance):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class DPSMorphoSkelGuidanceNN(DPSGuidance, MorphSkelLossNNGuidance):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class CGMorphoSkelGuidance(CGGuidance, MorphSkelLossGuidance):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
