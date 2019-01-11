"""
Provides some of the most commonly used estimators of the effective sample
sizes in analyzing Markov chain Monte Carlo outputs.

For theories and ideas behind each of the algorithms, see the survey paper
below and the references therein:

    M. Thompson (2010). A Comparison of Methods for Computing Autocorrelation Time.
    https://arxiv.org/abs/1011.0175
"""

import numpy as np
import math
from . import ar_model
import warnings

# Monkey patch warnings
warnings.formatwarning = lambda message, category, filename, lineno, line=None: (
    '{:s}:{:d}: {:s}: {:s}\n'.format(filename, lineno, category.__name__, str(message))
)


def ar_process_fit(samples, max_ar_order=None, axis=0, normed=False):
    """
    Estimates effective sample sizes of samples along the specified axis by
    fitting an autoregressive process via the Yule-Walker equation. The order
    of the AR process is determined via AIC.
    """

    if samples.ndim == 1:
        samples = samples[:, np.newaxis]

    if max_ar_order is None:
        series_length = samples.shape[axis]
        max_ar_order = min(
            series_length - 1, math.ceil(10 * np.log10(series_length))
        )

    n_param = samples.shape[1 - axis]
    ess = np.zeros(n_param)

    for i in range(n_param):

        x = np.take(samples, i, 1 - axis)

        # Determine AR order.
        ar_order, ar_coef = ar_model.fit(x, max_ar_order)

        if ar_order == 0:
            auto_corr_time = 1
        else:
            x_std = (x - np.mean(x)) / np.std(x)
            acorr = np.array([
                _compute_auto_corr(x_std, lag) for lag in range(1, ar_order + 1)
            ])
            auto_corr_time = (1 - np.inner(acorr, ar_coef)) / (1 - np.sum(ar_coef)) ** 2
        ess[i] = 1 / auto_corr_time

    if not normed:
        ess *= samples.shape[axis]

    return ess


def batch_means(samples, n_batch=25, axis=0, normed=False):
    """
    Estimates effective sample sizes of samples along the specified axis
    with the method of batch means.
    """

    if samples.ndim == 1:
        samples = samples[:, np.newaxis]

    batch_index = np.linspace(0, samples.shape[axis], n_batch + 1).astype('int')
    batch_list = [
        np.take(samples, np.arange(batch_index[i], batch_index[i + 1]), axis)
        for i in range(n_batch)
    ]
    batch_mean = np.stack((np.mean(batch, axis) for batch in batch_list), axis)
    mcmc_var = samples.shape[axis] / n_batch * np.var(batch_mean, axis)
    ess = np.var(samples, axis) / mcmc_var
    if not normed: ess *= samples.shape[0]

    return ess


def monotone_sequence(
        samples, axis=0, normed=False, require_acorr=False):
    """
    Estimates effective sample sizes of samples along the specified axis
    with the monotone positive sequence estimator of "Practical Markov
    Chain Monte Carlo" by Geyer (1992). The estimator is ONLY VALID for
    reversible Markov chains. The inputs 'mu' and 'sigma_sq' are optional
    and unnecessary for the most cases in practice.

    Parameters
    ----------
    require_acorr : bool
        If true, a list of estimated auto correlation sequences are returned.

    Returns
    -------
    ess : numpy array
    auto_cor : list of numpy array
        auto-correlation estimates of the chain up to the lag beyond which the
        auto-correlation can be considered insignificant by the monotonicity
        criterion.
    """

    if samples.ndim == 1:
        samples = samples[:, np.newaxis]

    d = samples.shape[1 - axis]
    ess = np.zeros(d)
    auto_cor = []
    for j in range(d):
        if axis == 0:
            x = samples[:, j]
        else:
            x = samples[j, :]
        x_std = (x - np.mean(x)) / np.std(x)
        ess_j, auto_cor_j = _monotone_sequence_1d(x_std, require_acorr)
        ess[j] = ess_j
        if require_acorr:
            auto_cor.append(auto_cor_j)
    if normed:
        ess /= samples.shape[axis]

    return ess, auto_cor


def _monotone_sequence_1d(x, require_acorr):
    """ The time series `x` is assumed to be standardized. """

    auto_corr = []

    # lag in [0, 1] case.
    lag_one_auto_corr = _compute_auto_corr(x, lag=1)
    running_min = 1. + lag_one_auto_corr
    auto_corr_sum = 1. + 2 * lag_one_auto_corr
    if require_acorr:
        auto_corr.extend((1., lag_one_auto_corr))
    curr_lag = 2

    while curr_lag + 2 < len(x):

        even_auto_corr, odd_auto_corr = [
            _compute_auto_corr(x, lag) for lag in [curr_lag, curr_lag + 1]
        ]
        curr_lag += 2
        if even_auto_corr + odd_auto_corr < 0:
            break

        running_min = min(running_min, (even_auto_corr + odd_auto_corr))
        auto_corr_sum += 2 * running_min
        if require_acorr:
            auto_corr.extend((even_auto_corr, odd_auto_corr))

    ess = len(x) / auto_corr_sum
    if auto_corr_sum < 0:
        # Rare, but can happen with floating point errors when the time series
        # `x` shows strong negative correlations.
        ess = float('inf')

    return ess, np.array(auto_corr)


def _compute_auto_corr(x, lag):
    """
    Returns an estimate of the lag 'k' auto-correlation of a time series 'x'.
    The estimator is biased towards zero due to the factor (len(x) - lag) / len(x).
    See Geyer (1992) Section 3.1 and the reference therein for justification.
    """
    acorr = np.mean(x[:-lag] * x[lag:]) * (len(x) - lag) / len(x)
    return acorr


def _r_coda(samples, axis=0, normed=False, R_call='rpy', n_digit=18):
    """
    Estimates effective sample sizes of samples along the specified axis.

    Parameters
    ----------
    R_call : {'rpy', 'external'}
        If 'external', the samples are first saved as a csv file and the
        method runs an R script from bash.
    """
    if R_call == 'rpy':
        ess = _coda_rpy(samples, axis, normed)
    elif R_call == 'external':
        ess = _coda_external(samples, axis, normed, n_digit)
    else:
        raise NotImplementedError()
    return ess


def _coda_rpy(samples, axis=0, normed=False):
    """
    Estimates effective sample sizes of samples along the specified axis by
    calling the R package 'coda' via rpy2. Requires the package rpy2 to be installed.
    """

    try:
        import rpy2.robjects.numpy2ri
        rpy2.robjects.numpy2ri.activate()
        import rpy2.robjects.packages as rpackages
        from rpy2.rinterface import RRuntimeError
    except ImportError:
        warnings.warn(
            'rpy2 needs to be installed to call this function.'
        )
        raise

    try:
        coda = rpackages.importr('coda')
    except RRuntimeError:
        warnings.warn('R CODA package not installed. Installing....')
        utils = rpackages.importr('utils')
        utils.install_packages('coda')
        raise

    ess = np.squeeze(np.array([
        coda.effectiveSize(np.take(samples, i, axis))
        for i in range(samples.shape[axis])
    ]))
    if normed:
        ess = ess / samples.shape[axis]

    return ess


def _coda_external(samples, axis=0, normed=False, n_digit=18,
                   saveto_fname=None, loadfrom_fname=None):
    """
    Estimates effective sample sizes of samples along the specified axis by
    calling the R package 'coda' externally. It is a hacky but convenient way
    to call an R function without having to install rpy2 and its dependencies.
    """

    import os
    import sys

    if not (sys.platform.startswith('linux') or sys.platform == 'darwin'):
        raise OSError(
            "The current implementation uses bash commnads, so runs only on "
            "Linux and Mac."
        )

    filenum = np.random.randint(2 ** 31)
        # Append a random number to a file name to avoid conflicts.
    if saveto_fname is None:
        saveto_fname = 'mchain{:d}.csv'.format(filenum)
    if loadfrom_fname is None:
        loadfrom_fname = 'ess{:d}.csv'.format(filenum)

    if axis == 0:
        np.savetxt(saveto_fname, samples, delimiter=',', fmt='%.{:d}e'.format(n_digit))
    else:
        np.savetxt(saveto_fname, samples.T, delimiter=',', fmt='%.{:d}e'.format(n_digit))

    # Write an R script for computing ESS with the 'coda' package if the script
    # is not already present.
    rscript_name = "compute_ess_with_coda.R"
    r_code = "\"args <- commandArgs(trailingOnly=T) # Read in the input and output file names\n" \
             + "x <- read.csv(args[1], header=F)\n" \
             + "library(coda)\n" \
             + "ess <- unlist(lapply(x, effectiveSize))\n" \
             + "write.table(ess, args[2], sep=',', row.names=F, col.names=F)\""
    os.system(" ".join(["[[ ! -f", rscript_name, "]] && echo", r_code, ">>", rscript_name]))

    # Write the data to a text file, read into the R script, and output
    # the result back into a text file.
    exit_status = os.system(
        " ".join(["Rscript", rscript_name, saveto_fname, loadfrom_fname])
    )
    if exit_status != 0:
        raise RuntimeError("Command line call to an Rscript failed.")

    ess = np.loadtxt(loadfrom_fname, delimiter=',').copy()
    if normed:
        ess = ess / samples.shape[axis]
    os.system(" ".join(["rm -f", saveto_fname, loadfrom_fname]))

    return ess