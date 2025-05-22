# testing the interface of model_based LID

import math

import pytest
import torch

from lid.diffusions import FlipdEstimator
from models.diffusions.sdes import VpSde
from models.diffusions.sdes.utils import VpSdeGaussianAnalytical


# This is a very strong test but also takes some time to run!
@pytest.mark.parametrize(
    "setting",
    [
        (42, [0.1, 0.1, 1, 1, 1, 1, 1, 10000, 10000, 10000], 1e-3),
        (100, [0.1, 10, 100], 1e-3),
        (111, [100, 100, 1000, 1000, 5000.0], 1e-3),
    ],
)
def test_fokker_planck(setting):
    seed, cov_eigs, tolerance = setting
    torch.manual_seed(seed)
    device = torch.device("cpu")
    d = len(cov_eigs)
    mean = d * torch.randn(d).to(device)
    # create a covariance matrix that has an almost zero eigenvalue, 3 larger eigenvalues and 6 smaller eigenvalues
    eigvals = torch.tensor(cov_eigs).to(device)
    # create a random orthogonal matrix
    orthogonal = torch.randn(d, d).to(device)
    q, _ = torch.linalg.qr(orthogonal)
    cov = q @ torch.diag(eigvals) @ q.T
    score_net = VpSdeGaussianAnalytical(
        posterior_mean=mean,
        posterior_cov=cov,
    )
    # sample data from a Gaussian multivariate with mean and cov
    data = torch.distributions.MultivariateNormal(mean, cov).sample((1000,)).to(device)

    vpsde = VpSde(score_net=score_net).to(device)
    lid_estimator = FlipdEstimator(
        model=vpsde,
        ambient_dim=d,
        device=device,
    )

    def estimate_exact_lid(
        data: torch.Tensor,
        t: float,
        d: int,
        vpsde: VpSde,
    ) -> torch.Tensor:
        """
        Estimates the LID at a given time t for a given data point.
        """
        t_tensor_reapeated = torch.tensor([t] * data.shape[0], dtype=data.dtype).to(device)
        # activate gradient
        t_tensor_reapeated.requires_grad = True
        # take the gradient of score_net.log_convolution_distribution w.r.t 't' and evaluate t at t_tensor_reapeated
        # write it functional
        log_convolution = score_net.log_convolution_distribution(data, t_tensor_reapeated)
        # set the gradients of t_tensor_reapeated to 0
        t_tensor_reapeated.grad = None
        log_convolution.sum().backward()
        gradients = t_tensor_reapeated.grad.detach().clone()
        beta_t = vpsde.beta(t)
        B_t = vpsde.beta_integral(t_end=t, t_start=0)
        lid = gradients * (2 * (math.exp(B_t) - 1)) / (beta_t * math.exp(B_t)) + d
        # clamp values less than 0 and greater than 10
        return lid

    all_t = torch.linspace(0, vpsde.t_max, 100)
    for t in all_t:
        estimated_lid = lid_estimator.estimate_lid(data, t=t)
        expected_lid = estimate_exact_lid(data, t, d, vpsde)
        avg_estimated_lid = torch.mean(estimated_lid)
        avg_expected_lid = torch.mean(expected_lid)
        diff = torch.abs(avg_estimated_lid - avg_estimated_lid)
        assert (
            diff < tolerance
        ), f"The LID should be close to the exact LID but got {avg_estimated_lid} and {avg_expected_lid}"
