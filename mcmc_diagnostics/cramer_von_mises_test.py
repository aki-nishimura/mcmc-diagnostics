"""

References:
----------
Philip Heidelberger and Peter D. Welch (1983). Simulation run length control
in the presence of an initial transient. Operations Research.

Sandor Csorgo and Julian J. Faraway (1996). The exact and asymptotic distributions
of Cramér-von Mises statistics. Journal of the Royal Statistical Society: Series B.

"""

import numpy as np
from math import pi, ceil, sqrt, exp, gamma, factorial
from scipy.special import kv as modified_bessel_2
from . import estimate_ess


def is_stationarity(x, signif_level=.05):
    return calculate_p_value(x) < signif_level


def calculate_p_value(x, ess_chain_frac=.5):
    """ Calculate the p-value under stationarity using Cramer-von-Mises statistics.

    Parameters
    ----------
    x: 1d numpy array
    ess_chain_frac : float between 0 and 1
        Fraction of the tail of the chain used in estimating the effective
        sample size, which is needed in computing the test statistics.
    """
    if (type(x) is not np.ndarray) or x.ndim != 1:
        raise TypeError("The input must be a 1d numpy array.")
    cvm_stat = cramer_von_mises_statistic(x, ess_chain_frac)
    p_val = 1 - cramer_von_mises_cdf(cvm_stat)
    return p_val


def cramer_von_mises_statistic(x, ess_chain_frac=.5):
    bb_stat = _brownian_bridge_statistic(x, 1 - ess_chain_frac)
    cvm_stat = np.trapz(bb_stat ** 2, np.linspace(0, 1, len(bb_stat)))
    return cvm_stat


def _brownian_bridge_statistic(x, frac_discard=.5):
    """
    Parameters
    ----------
    frac_discard : float in [0, 1]
        The non-stationarity can inflate the estimate of the spectrum at zero.
        To avoid this, as proposed in Heidelberger and Welch (1983), discard
        the initial fraction of the time series `x` (and hope that the rest of
        the sequence looks more stationary).
    """
    n_discard = ceil(frac_discard * len(x))
    x_subseq = x[n_discard:]
    spectrum_at_zero = np.var(x_subseq) / estimate_ess(x_subseq, normed=True)
    cumsum = np.concatenate(([0], np.cumsum(x)))
    linear_interp = np.arange(len(x) + 1) * np.mean(x)
    stat = (cumsum - linear_interp) / np.sqrt(len(x) * spectrum_at_zero)
    return stat


def cramer_von_mises_cdf(x, n_summand=None):
    """
    Computes the cumulative distribution function of the (asymptotic)
    Cramer-von-Mises statistics (the integral of a squared Brownian bridge
    process) via a series expansion formula.

    Parameters
    ----------
    x : scalar
    n_summand : optional, int
        The R coda (as of ver 0.19-1) uses 4, but we need more terms for a large
        value of `x`.
    """
    if n_summand is None:
        n_summand = 5 if x < 10 else 10
    cum_density = 0
    for k in range(n_summand):
        cum_density += _cvm_cdf_summand(x, k)
    return cum_density


def _cvm_cdf_summand(x, k):
    temp = (4 * k + 1) ** 2 / 16 / x
    kth_summand = (
        1 / pi ** (3 / 2) / sqrt(x)
        * gamma(k + .5) / factorial(k) * sqrt(4 * k + 1)
        * exp(-temp) * modified_bessel_2(.25, temp)
    ) # Equation (1.3) in Csorgo and Faraway (1996).
    return kth_summand
