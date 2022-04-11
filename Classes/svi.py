import numpy as np
from functools import partial
import torch

class GaussianSVI:

    def __init__(self, true_posterior, num_samples_per_iter) -> None:
        self.logprob = true_posterior
        self.num_samples = num_samples_per_iter

    def diag_gaussian_sample(self, num_samples, mean, log_std):
        # Take a single sample from a diagonal multivariate Gaussian.
        return mean + np.exp(log_std) * torch.random.normal(0, 1, size=(num_samples, mean.shape))
        # you must use the reparameterization trick.  Also remember that
        # we are parameterizing the _log_ of the standard deviation.

    def diag_gaussian_logpdf(self, x, mean, log_std):
        # Evaluate the density of single point on a diagonal multivariate Gaussian.
        return np.sum(torch.norm.logpdf(x, mean, np.exp(log_std)), axis=-1)

    def elbo(self, logprob, rng, mean, log_std):
        # Single-sample Monte Carlo estimate of the variational lower bound.
        sampled_zs = self.diag_gaussian_sample(rng, mean, log_std)
        return logprob(sampled_zs) - self.diag_gaussian_logpdf(sampled_zs, mean, log_std)

    def batch_elbo(self, logprob, params, num_samples):
        # Average over a batch of random samples. 
        vectorized_elbo = torch.vmap(partial(self.elbo, logprob))
        elbo_estimates = vectorized_elbo(num_samples, *params)
        return np.mean(elbo_estimates)

    def objective(self, params):  # The loss function to be minimized.
        return -self.batch_elbo(self.logprob, params, self.num_samples)
