from collections.abc import Callable

import cv2
import numpy as np
import torch

from lid import LIDEstimator
from models.diffusions import Sde

from ..datapoint_metric import DatapointMetric


def png_size(pil, compression_ratio=9):
    """Get PNG size of PIL"""
    arr = np.array(pil)
    img_encoded = cv2.imencode(".png", arr, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])
    return 8 * len(img_encoded[1])


class MemorizationMetric(DatapointMetric):
    """A metric for whether a diffusion model has memorized a datapoint."""

    def __init__(self, sde, device=None):
        self.sde = sde
        super().__init__(model=sde, device=device)


class LIDBasedMetric(MemorizationMetric):
    def __init__(self, sde, lid_estimator_partial, **estimate_lid_kwargs):
        super().__init__(sde)
        self.lid_estimator = lid_estimator_partial(sde)
        self.estimate_lid_kwargs = estimate_lid_kwargs


class ConditionalLID(LIDBasedMetric):
    def score_batch(self, batch, **kwargs):
        datapoint, label = batch[0].to(self.device), batch[1].to(self.device)
        return self.lid_estimator.estimate_lid(
            datapoint, encoder_hidden_states=label, **kwargs, **self.estimate_lid_kwargs
        )


class UnconditionalLID(LIDBasedMetric):
    def score_batch(self, batch, **kwargs):
        datapoint, label = batch[0].to(self.device), batch[1].to(self.device)
        return self.lid_estimator.estimate_lid(datapoint, **kwargs, **self.estimate_lid_kwargs)


class ConditionalUnconditionalLIDRatio(LIDBasedMetric):
    def score_batch(self, batch, **kwargs):
        datapoint, label = batch[0].to(self.device), batch[1].to(self.device)
        conditional_lid = self.lid_estimator.estimate_lid(
            datapoint, encoder_hidden_states=label, **kwargs, **self.estimate_lid_kwargs
        )
        unconditional_lid = self.lid_estimator.estimate_lid(
            datapoint, **kwargs, **self.estimate_lid_kwargs
        )
        return conditional_lid / unconditional_lid


class LIDToDecodedPNGRatio(LIDBasedMetric):
    """Ratio between LID and the PNG size.

    Assumes the input is an encoded batch, and decodes this batch with a decoder to get the PNG.
    """

    def __init__(self, sde, decoder, lid_estimator_partial, **estimate_lid_kwargs):
        super().__init__(sde, lid_estimator_partial, **estimate_lid_kwargs)
        self.decoder = decoder.to(self.device)

    def score_batch(self, batch, **kwargs):
        datapoint, label = batch[0].to(self.device), batch[1].to(self.device)

        lid_estimates = self.lid_estimator.estimate_lid(
            datapoint, encoder_hidden_states=label, **kwargs, **self.estimate_lid_kwargs
        )

        decoded_imgs = self.decoder(datapoint)
        png_sizes = torch.tensor([png_size(im) for im in decoded_imgs], device=self.device)

        return lid_estimates / png_sizes


class ReconstructionCost(MemorizationMetric):
    """The reconstruction cost of adding noise and then reconstructing it with the reverse SDE."""

    def __init__(self, sde: Sde, recon_time: float, **reverse_kwargs):
        super().__init__(sde)
        self.recon_time = recon_time
        assert "t_start" not in reverse_kwargs, "Set t_start using recon_time argument"
        self.reverse_kwargs = reverse_kwargs

    def score_batch(self, batch, seed=42, **kwargs):
        datapoint, label = batch
        datapoint, label = datapoint.to(self.device), label.to(self.device)

        with torch.random.fork_rng():
            torch.manual_seed(seed)
            noised_datapoint = self.sde.solve_forward_sde(datapoint, t_end=self.recon_time)
            reconstructed_datapoint = self.sde.solve_reverse_sde(
                datapoint,
                t_start=self.recon_time,
                encoder_hidden_states=label,
                **kwargs,
                **self.reverse_kwargs
            )

        norm_dims = datapoint.shape[1:]
        return torch.linalg.vector_norm(
            datapoint - reconstructed_datapoint, dim=tuple(range(1, datapoint.ndim))
        )


class EpsilonNorm(MemorizationMetric):
    """The norm of the score of the point noised to time `norm_time`."""

    def __init__(self, sde: Sde, norm_time: float, **forward_kwargs):
        super().__init__(sde)
        self.norm_time = norm_time
        self.forward_kwargs = forward_kwargs

    def score_batch(self, batch, **kwargs):
        datapoint, label = batch
        datapoint, label = datapoint.to(self.device), label.to(self.device)

        noised_datapoint = self.sde.solve_forward_ode(
            datapoint, t_end=self.norm_time, encoder_hidden_states=label, **self.forward_kwargs
        )
        score_at_norm_time = self.sde.score_net(
            noised_datapoint, self.norm_time, encoder_hidden_states=label, **kwargs
        )

        norm_dims = datapoint.shape[1:]
        return torch.linalg.vector_norm(score_at_norm_time, dim=tuple(range(1, datapoint.ndim)))


class ClassifierFreeGuidanceNorm(MemorizationMetric):
    """The norm of the classifier-free-guidance term of the point noised to time `norm_time`."""

    def __init__(self, sde: Sde, norm_time: float, **forward_kwargs):
        super().__init__(sde)
        self.norm_time = norm_time
        self.forward_kwargs = forward_kwargs

    def score_batch(self, batch, **kwargs):
        datapoint, label = batch
        datapoint, label = datapoint.to(self.device), label.to(self.device)

        noised_datapoint = self.sde.solve_forward_ode(
            datapoint, t_end=self.norm_time, encoder_hidden_states=label, **self.forward_kwargs
        )
        cond_score_at_norm_time = self.sde.score_net(
            noised_datapoint, self.norm_time, encoder_hidden_states=label, **kwargs
        )
        uncond_score_at_norm_time = self.sde.score_net(noised_datapoint, self.norm_time, **kwargs)

        norm_dims = datapoint.shape[1:]
        return torch.linalg.vector_norm(
            cond_score_at_norm_time - uncond_score_at_norm_time, dim=tuple(range(1, datapoint.ndim))
        )


class DecodedPNGSize(MemorizationMetric):
    def __init__(self, sde, decoder):
        super().__init__(sde)
        self.decoder = decoder.to(self.device)

    def score_batch(self, batch, **kwargs):
        datapoint, label = batch[0].to(self.device), batch[1].to(self.device)
        decoded_imgs = self.decoder(datapoint)
        png_sizes = torch.tensor([png_size(im) for im in decoded_imgs], device=self.device)
        return png_sizes
