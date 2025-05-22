import functools
import math
import numbers
from typing import List, Literal

import numpy as np
import torch

from data.transforms.unpack import UnpackBatch
from lid import ModelBasedLIDEstimator
from lid.utils import fast_regression
from models.diffusions.sdes import VpSde
from models.diffusions.sdes.utils import HUTCHINSON_DATA_DIM_THRESHOLD, compute_trace_of_jacobian


class FokkerPlanckEstimator(ModelBasedLIDEstimator):
    """
    The parent class of all Fokker Planck techniques for LID estimation
    that we employ in our work. This includes FLIPD and FPRegress.

    In all these cases, only a VpSde model is implemented.
    """

    # Base time is non-zero to avoid numerical instability!
    DEFAULT_BASE_TIME: float = 1e-4

    def __init__(
        self,
        model: VpSde,
        ambient_dim: int | None = None,
        device: torch.device | None = None,
        unpack: UnpackBatch | None = None,
    ):
        super().__init__(
            ambient_dim=ambient_dim,
            model=model,
            device=device,
            unpack=unpack,
        )
        assert isinstance(self.model, VpSde), "The model should be a VpSde object."
        self.vpsde: VpSde = self.model

    def _convert_B_t_to_time(self, B_t: float, t: float | None = None) -> float:
        if t is None:
            # B_t is quadretic w.r.t t with the following coefficients
            _A = (self.vpsde.beta_max - self.vpsde.beta_min) / (2 * self.vpsde.t_max)
            _B = self.vpsde.beta_min
            _C = -B_t
            # solve At^2 + Bt + C = 0
            t = (-_B + math.sqrt(_B**2 - 4 * _A * _C)) / (2 * _A)
            t = max(t, 0)
            t = min(t, self.vpsde.t_max)
        return t

    def _convert_delta_to_time(self, delta: float) -> float:
        B_t = math.log(delta**2 + 1)
        return self._convert_B_t_to_time(B_t)

    def _get_all_math_terms(
        self,
        t: float | None = None,
        sigma_t: float | None = None,
        coeff: float | None = None,
    ):
        """
        This inner function is a "numerically optimized"
        way of getting all the math terms that we need from the information
        that we are given
        """
        t = 1e-4 if t is None else t
        # if B_t is not None, we can compute it from t
        B_t = self.vpsde.beta_integral(t_start=0, t_end=t).item()

        coeff = math.exp(-0.5 * B_t) if coeff is None else coeff
        sigma_t = self.vpsde.sigma(t_end=t).item() if sigma_t is None else sigma_t
        return t, sigma_t, coeff

    def _get_laplacian_term(
        self,
        x: torch.Tensor,
        t: float,
        coeff: float,
        method: (
            Literal["hutchinson_gaussian", "hutchinson_rademacher", "deterministic"] | None
        ) = None,
        # The number of samples if one opts for estimation methods to save time:
        hutchinson_sample_count: int = HUTCHINSON_DATA_DIM_THRESHOLD,
        chunk_size: int = 128,
        seed: int = 42,
        verbose: int = 0,
        **score_kwargs,
    ):

        def score_fn(x, t: float):
            if isinstance(t, numbers.Number):
                t = torch.tensor(t).float()
            t: torch.Tensor
            t_repeated = t.repeat(x.shape[0]).to(x.device)
            return self.vpsde.score_net(x, t=t_repeated, **score_kwargs)

        laplacian_term = compute_trace_of_jacobian(
            fn=functools.partial(score_fn, t=t),
            x=coeff * x,
            method=method,
            hutchinson_sample_count=hutchinson_sample_count,
            chunk_size=chunk_size,
            seed=seed,
            verbose=verbose,
        )
        return laplacian_term

    def _get_score_norm_term(self, x, t: float, coeff: float, **score_kwargs):
        if isinstance(t, numbers.Number):
            t = torch.tensor(t).float()
        t: torch.Tensor
        t_repeated = t.repeat(x.shape[0]).to(self.device)
        scores_flattened = self.vpsde.score_net(coeff * x, t=t_repeated, **score_kwargs).reshape(
            x.shape[0], -1
        )
        score_norm_term = torch.sum(scores_flattened * scores_flattened, dim=1)
        return score_norm_term


class FlipdEstimator(FokkerPlanckEstimator):
    """
    An LID estimator based on the connection made between marginal probabilities
    and Gaussian convolution + running the singular value decomposition.

    This is the fastest model-based LID estimator available in the library.

    Args:
        sde: An Sde object containing a trained diffusion model
        ambient_dim: Corresponds to d in the paper. Inferred by estimate_id if not
            specified here.
    """

    @torch.no_grad
    def _estimate_lid(
        self,
        x: torch.Tensor,
        t: float | None = None,
        method: (
            Literal["hutchinson_gaussian", "hutchinson_rademacher", "deterministic"] | None
        ) = None,
        # The number of samples if one opts for estimation methods to save time:
        hutchinson_sample_count: int = HUTCHINSON_DATA_DIM_THRESHOLD,
        chunk_size: int = 128,
        seed: int = 42,
        verbose: int = 0,
        **score_kwargs,
    ) -> torch.Tensor:
        x = x.to(self.device)
        t, sigma_t, coeff = self._get_all_math_terms(
            t=t,
            sigma_t=None,
            coeff=None,
        )
        laplacian_term = self._get_laplacian_term(
            x=x,
            t=t,
            coeff=coeff,
            method=method,
            hutchinson_sample_count=hutchinson_sample_count,
            chunk_size=chunk_size,
            seed=seed,
            verbose=verbose,
            **score_kwargs,
        )
        score_norm_term = self._get_score_norm_term(x=x, t=t, coeff=coeff, **score_kwargs)
        return self.ambient_dim + sigma_t * laplacian_term + score_norm_term


class FPRegressEstimator(FokkerPlanckEstimator):
    """
    Performs the exact LIDL estimation using the regression technique that
    they have proposed, however, instead of training multiple models, it learns
    only one.
    """

    def _update_rho(
        self,
        x_batch: torch.Tensor,
        t: float,
        delta_t: float,
        method: (
            Literal["hutchinson_gaussian", "hutchinson_rademacher", "deterministic"] | None
        ) = None,
        # The number of samples if one opts for estimation methods to save time:
        hutchinson_sample_count: int = HUTCHINSON_DATA_DIM_THRESHOLD,
        chunk_size: int = 128,
        seed: int = 42,
        verbose: int = 0,
        **score_kwargs,
    ):
        t, sigma_t, coeff = self._get_all_math_terms(
            t=t,
            sigma_t=None,
            coeff=None,
        )
        beta_t = self.vpsde.beta(t).item()

        laplacian_term = self._get_laplacian_term(
            x=x_batch,
            t=t,
            coeff=coeff,
            method=method,
            hutchinson_sample_count=hutchinson_sample_count,
            chunk_size=chunk_size,
            seed=seed,
            verbose=verbose,
            **score_kwargs,
        )
        score_norm_term = self._get_score_norm_term(x_batch, t=t, coeff=coeff, **score_kwargs)
        return delta_t * beta_t * 0.5 * (laplacian_term + score_norm_term / sigma_t) / sigma_t

    @torch.no_grad
    def _estimate_lid(
        self,
        x: torch.Tensor,
        delta: float | None = None,
        deltas: List[float] | None = None,
        num_deltas: int | None = None,
        method: (
            Literal["hutchinson_gaussian", "hutchinson_rademacher", "deterministic"] | None
        ) = None,
        # The number of samples if one opts for estimation methods to save time:
        hutchinson_sample_count: int = HUTCHINSON_DATA_DIM_THRESHOLD,
        chunk_size: int = 128,
        seed: int = 42,
        verbose: int = 0,
        **score_kwargs,
    ) -> torch.Tensor:

        x = x.to(self.device)
        # Set deltas "exactly" according to the LIDL codebase:
        # https://github.com/opium-sh/lidl/blob/master/dim_estimators.py
        if deltas is None:
            if delta is not None:
                if num_deltas is None:
                    deltas = [
                        delta / 2.0,
                        delta / 1.41,
                        delta,
                        delta * 1.41,
                        delta * 2.0,
                    ]
                else:
                    deltas = [dlt for dlt in np.geomspace(delta / 2.0, delta * 2.0, num_deltas)]
            else:
                deltas = [
                    0.010000,
                    0.013895,
                    0.019307,
                    0.026827,
                    0.037276,
                    0.051795,
                    0.071969,
                    0.100000,
                ]
        else:
            assert len(deltas) > 1, "The number of deltas should be greater than 1 for regression."

        assert (
            min(deltas) >= FokkerPlanckEstimator.DEFAULT_BASE_TIME
        ), f"The delta should be less than the base time, got: delta = {delta} < {FokkerPlanckEstimator.DEFAULT_BASE_TIME}"
        # Compute the timesteps that we are going to perform regression upon
        timesteps = [self._convert_delta_to_time(delta) for delta in deltas]

        regress_radii = torch.tensor([math.log(delta) for delta in deltas], device=x.device).flip(0)
        regress_rhos = [torch.zeros(x.shape[0]).to(x.device)]
        current_time = timesteps[-1]
        for i in range(len(timesteps) - 2, -1, -1):
            delta_t = current_time - timesteps[i]
            update_value = self._update_rho(  # run an Euler step
                x,
                current_time,
                delta_t,
                method=method,
                hutchinson_sample_count=hutchinson_sample_count,
                chunk_size=chunk_size,
                seed=seed,
                verbose=verbose,
                **score_kwargs,
            )
            regress_rhos.append(regress_rhos[-1].clone() - update_value)
            current_time = timesteps[i]

        regress_rhos = torch.stack(regress_rhos).T  # [batch_size, regression_count]

        return fast_regression(regress_rhos, regress_radii) + self.ambient_dim
