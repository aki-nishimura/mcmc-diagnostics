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
    return p_value_under_stationarity(x) < signif_level


def p_value_under_stationarity(x):
    """
    Parameters
    ----------
    x: 1d numpy array
    """
    if (type(x) is not np.ndarray) or x.ndim != 1:
        raise TypeError("The input must be a 1d numpy array.")
    cvm_stat = cramer_von_mises_statistic(x)
    p_val = 1 - cramer_von_mises_cdf(cvm_stat)
    return p_val


def cramer_von_mises_statistic(x):
    bb_stat = _brownian_bridge_statistic(x)
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


def cramer_von_mises_cdf(x, n_summand=4):
    """
    Computes the cumulative distribution function of the (asymptotic)
    Cramer-von-Mises statistics (the integral of a squared Brownian bridge
    process) via a series expansion formula.

    Parameters
    ----------
    x : scalar
    n_summand : int
        The default value of 4 coincides to the R coda implementation.
    """
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
