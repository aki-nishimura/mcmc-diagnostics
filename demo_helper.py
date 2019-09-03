import numpy as np
import time


def random_walk_metropolis_sampler(
        f, x0, proposal_sd, n_samples, Sigma=None, seed=None, thin=1):
    """
    Generates samples via random-walk Metropolis.

    Params:
    ------
    f : function
        Computes the log density of the target density
    n_samples: int
        The number of samples to return. The total number of iteration is
        (thin * n_samples).
    """

    if seed is not None:
        np.random.seed(seed)

    if Sigma is not None:
        L = np.linalg.cholesky(Sigma)
    else:
        L = None

    n_param = len(np.atleast_1d(x0))
    mcmc_samples = np.zeros((n_param, n_samples))
    logp_samples = np.zeros(n_samples)
    accepted = np.zeros(thin)
    logp = f(x0)
    theta = x0
    mcmc_samples[:, 0] = theta
    logp_samples[0] = logp
    accept_rate = 0

    start_time = time.time()
    for mcmc_iter in range(1, n_samples):
        for j in range(thin):
            theta, logp, accepted[j], _ = random_walk_metropolis_step(
                f, theta, logp, proposal_sd, L, chol=True
            )
        mcmc_samples[:, mcmc_iter] = theta
        logp_samples[mcmc_iter] = logp
        weight = (mcmc_iter - 1) / mcmc_iter
        accept_rate = weight * accept_rate + (1 - weight) * np.mean(accepted)
    end_time = time.time()
    time_elapsed = end_time - start_time

    return mcmc_samples, accept_rate, logp_samples, time_elapsed


def random_walk_metropolis_step(f, x0, logp0, proposal_sd, Rho=None, chol=False):
    """
    Params
    ------
    f : function
        Computes the log density of the target density
    proposal_sd : scalar or vector
    Rho : matrix
        (Cholesky factor of) correlation matrix. If given, the proposal
        variance will be 'diag(proposal_sd) * Rho * diag(proposal_sd).'
    chol : bool
        If True, L = Rho is assumed to be the lower triangular cholesky
        factorization of the proposal covariance (which is given by L * L').
    """

    dx = np.random.randn(len(x0))
    if Rho is not None:
        L = Rho if chol else np.linalg.cholesky(Rho)
        dx = np.dot(L, dx)
    dx *= proposal_sd
    x = x0 + dx

    logp = f(x)
    accept_prob = min(1, np.exp(logp - logp0))
    accepted = accept_prob > np.random.uniform()
    if not accepted:
        x = x0
        logp = logp0

    return x, logp, accepted, accept_prob


class ProductGaussian:

    def __init__(self, marginal_sd):
        self.dim = len(np.atleast_1d(marginal_sd))
        self.sigma = marginal_sd

    def compute_logp(self, x):
        """ Computes the log-density (up to an additive constant). """
        return - 1 / 2 * np.sum((x / self.sigma) ** 2)

