"""
Provides some of the most commonly used estimators of the effective sample
sizes in analyzing Markov chain Monte Carlo outputs.

For theories and ideas behind each of the algorithms, see the survey paper
below and the references therein:

    M. Thompson (2010). A Comparison of Methods for Computing Autocorrelation Time.
    https://arxiv.org/abs/1011.0175
"""

import math
import warnings
from . import ar_model
import numpy as np

# Monkey patch warnings
warnings.formatwarning = lambda message, category, filename, lineno, line=None: (
    '{:s}:{:d}: {:s}: {:s}\n'.format(filename, lineno, category.__name__, str(message))
)


def estimate_ess(samples, axis=0, method='ar', normed=False, options={}):
    """
    Wrapper for calling a function specified by the `method` argument.
    Estimates effective sample sizes (ESS) of samples along the specified axis.

    Parameters
    ----------
    samples : numpy 1d or 2d array
    axis : int, {0, 1}
    method : str, {'ar', 'monotone-sequence', 'batch-means'}
    normed : bool
        If True, the efficiency (ESS per MCMC sample) is returned.
    options : dict with keys in ['max_ar_order', 'n_batch']

    Returns
    -------
    ess : numpy array
    """

    if type(samples) is not np.ndarray:
        raise TypeError("The first argument must be a numpy array.")
    if samples.ndim > 2:
        raise ValueError("The function only supports a 1d or 2d array.")
    if samples.ndim == 2 and axis == -1:
        axis = 1
    if axis not in [0, 1]:
        raise ValueError("Invalid axis value.")

    if samples.shape[axis] <= 25:
        warnings.warn(
            "The number of samples is extremely small. The estimated ESS "
            "will likely be not reliable at all."
        )

    if method == 'ar':
        max_ar_order = options.get('max_ar_order', None)
        ess = ar_process_fit(samples, axis, normed, max_ar_order=max_ar_order)

    elif method == 'monotone-sequence':
        ess, _ = monotone_sequence(samples, axis, normed)

    elif method == 'batch-means':
        n_batch = options.get('n_batch', 25)
        ess = batch_means(samples, axis, normed, n_batch=n_batch)

    else:
        raise NotImplementedError("Method {:s} is not supported.".format(method))

    return ess


def ar_process_fit(samples, axis=0, normed=False, max_ar_order=None):
    """
    Estimates effective sample sizes of samples along the specified axis by
    fitting an autoregressive process via the Yule-Walker equation. The order
    of the AR process is determined via AIC.
    """

    if samples.ndim == 1:
        samples = samples[:, np.newaxis]

    series_length = samples.shape[axis]
    if max_ar_order is None:
        if series_length <= 100:
            max_ar_order = math.ceil(series_length / 5)
        else:
            max_ar_order = math.ceil(10 * np.log10(series_length))

    n_param = samples.shape[1 - axis]

    if series_length == 1: # Edge case
        ess = np.zeros(n_param)
    else:
        if axis == 0:
            samples = samples.T
        ess = np.array([
            _ar_process_fit_1d(x, max_ar_order) for x in samples
            # Loop is over the rows of samples.
        ])

    if normed: ess /= series_length

    return ess


def _ar_process_fit_1d(x, max_ar_order):

    ar_order, ar_coef = ar_model.fit(x, max_ar_order)

    if ar_order == 0:
        auto_corr_time = 1
    else:
        x_std = (x - np.mean(x)) / np.std(x)
        acorr = np.array([
            _compute_auto_corr(x_std, lag) for lag in range(1, ar_order + 1)
        ])
        auto_corr_time = (1 - np.inner(acorr, ar_coef)) / (1 - np.sum(ar_coef)) ** 2

    ess = len(x) / auto_corr_time
    return ess


def batch_means(samples, axis=0, normed=False, n_batch=25):
    """
    Estimates effective sample sizes of samples along the specified axis
    with the method of batch means.
    """

    if samples.ndim == 1:
        samples = samples[:, np.newaxis]

    n_sample = samples.shape[axis]
    if 2 * n_batch > n_sample:
        raise ValueError(
            "The number of batches must be less than twice the number of samples."
        )

    batch_index = np.linspace(0, n_sample, n_batch + 1).astype('int')
    batch_list = [
        np.take(samples, np.arange(batch_index[i], batch_index[i + 1]), axis)
        for i in range(n_batch)
    ]
    batch_mean = np.stack((np.mean(batch, axis) for batch in batch_list), axis)
    mcmc_var = n_sample / n_batch * np.var(batch_mean, axis)
    ess = np.var(samples, axis) / mcmc_var
    if not normed: ess *= n_sample

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

    n_param = samples.shape[1 - axis]
    n_sample = samples.shape[axis]
    ess = np.zeros(n_param)
    if n_sample <= 2: # Edge case.
        return ess

    auto_cor = []
    for j in range(n_param):
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
        ess /= n_sample

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
        import rpy2
    except ImportError:
        warnings.warn(
            'rpy2 needs to be installed to call this function.'
        )
        raise

    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    import rpy2.robjects.packages as rpackages
    from rpy2.rinterface import RRuntimeError

    try:
        coda = rpackages.importr('coda')
    except RRuntimeError:
        warnings.warn('R CODA package not installed. Installing....')
        utils = rpackages.importr('utils')
        utils.install_packages('coda')

    ess = np.squeeze(np.array([
        coda.effectiveSize(np.take(samples, i, axis))
        for i in range(samples.shape[axis])
    ]))
    if normed:
        ess = ess / samples.shape[axis]

    return ess


def _coda_external(samples, axis=0, normed=False, n_digit=18,
                   saveto_fname=None, loadfrom_fname=None, rscript_name=None):
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
    if rscript_name is None:
        rscript_name = "compute_ess_with_coda.R"

    if axis == 0:
        np.savetxt(saveto_fname, samples, delimiter=',', fmt='%.{:d}e'.format(n_digit))
    else:
        np.savetxt(saveto_fname, samples.T, delimiter=',', fmt='%.{:d}e'.format(n_digit))

    # Write an R script for computing ESS with the 'coda' package if the script
    # is not already present.
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
    os.system(" ".join(["rm -f", rscript_name]))

    return ess