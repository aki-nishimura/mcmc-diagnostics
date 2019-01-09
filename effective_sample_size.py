"""
Provides some of the most commonly used estimators of the effective sample
sizes in analyzing Markov chain Monte Carlo outputs.

For theories and ideas behind each of the algorithms, see the survey paper
below and the references therein:

    M. Thompson (2010). A Comparison of Methods for Computing Autocorrelation Time.
    https://arxiv.org/abs/1011.0175
"""

import numpy as np
from statsmodels.tsa.ar_model import AR
import os # Necessary for coda_ess_external


def r_coda(samples, axis=0, normed=False, R_call='rpy', n_digit=18):
    """
    Estimates effective sample sizes of samples along the specified axis.

    Parameters
    ----------
    R_call : {'rpy', 'external'}
        If 'external', the samples are first saved as a csv file and the
        method runs an R script from bash.
    """
    if R_call == 'rpy':
        ess = coda_rpy(samples, axis, normed)
    elif R_call == 'external':
        ess = coda_external(samples, axis, normed, n_digit)
    else:
        raise NotImplementedError()
    return ess


def coda_rpy(samples, axis=0, normed=False):

    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    import rpy2.robjects.packages as rpackages

    install_coda = False
    if install_coda:
        utils = rpackages.importr('utils')
        utils.install_packages('coda')

    coda = rpackages.importr('coda')
    ess = np.squeeze(np.array([
        coda.effectiveSize(np.take(samples, i, axis))
        for i in range(samples.shape[axis])
    ]))
    if normed:
        ess = ess / samples.shape[axis]
    return ess


def coda_external(samples, axis=0, normed=False, n_digit=18):
    """
    Estimates effective sample sizes of samples along the specified axis by
    calling the R package 'coda' externally. It is a hacky but convenient way
    to call an R function without having to install rpy2 and its dependencies.
    """

    filenum = np.random.randint(2 ** 31)
    # Append a random number to a file name to avoid conflicts.
    saveto = 'mchain{:d}.csv'.format(filenum)
    loadfrom = 'ess{:d}.csv'.format(filenum)
    if axis == 0:
        np.savetxt(saveto, samples, delimiter=',', fmt='%.{:d}e'.format(n_digit))
    else:
        np.savetxt(saveto, samples.T, delimiter=',', fmt='%.{:d}e'.format(n_digit))

    # Write an R script for computing ESS with the 'coda' package if the script
    # is not already present.
    r_code = "\"args <- commandArgs(trailingOnly=T) # Read in the input and " \
             "output file names\n" \
             + "x <- read.csv(args[1], header=F)\n" \
             + "library(coda)\n" \
             + "ess <- unlist(lapply(x, effectiveSize))\n" \
             + "write.table(ess, args[2], sep=',', row.names=F, col.names=F)\""
    os.system(" ".join(["[[ ! -f compute_coda_ess.R ]] && echo", r_code, ">>", "compute_coda_ess.R"]))

    # Write the data to a text file, read into the R script, and output
    # the result back into a text file.
    os.system(" ".join(["Rscript compute_coda_ess.R", saveto, loadfrom]))
    ess = np.loadtxt(loadfrom, delimiter=',').copy()
    if normed:
        ess = ess / samples.shape[axis]
    os.system(" ".join(["rm -f", saveto, loadfrom]))
    return ess


def ar_process_fit(samples, maxlag=50, axis=0, normed=False):

    if samples.ndim == 1:
        samples = samples[:, np.newaxis]

    n_param = samples.shape[1 - axis]
    ess = np.zeros(n_param)

    for i in range(n_param):

        x = np.take(samples, i, 1 - axis)

        # Determine AR order.
        ar_order, ar_coef = _fit_ar_process(x, maxlag)

        mean = np.mean(x)
        var = np.var(x)
        acorr = np.array([
            compute_acorr(x, lag, mean, var) for lag in range(1, ar_order + 1)
        ])
        auto_corr_time = (1 - np.inner(acorr, ar_coef)) / (1 - np.sum(ar_coef)) ** 2
        ess[i] = 1 / auto_corr_time

    if not normed:
        ess *= samples.shape[axis]

    return ess


def _fit_ar_process(x, maxlag, ic='aic'):
    model = AR(x)
    ar_result = model.fit(maxlag, ic=ic)
    ar_order = ar_result.k_ar
    ar_coef = ar_result.params[1:] # Exclude the mean
    return ar_order, ar_coef


def batch_means(samples, n_batch=25, axis=0, normed=False):
    """
    Estimates effective sample sizes of samples along the specified axis
    with the method of batch means.
    """
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
        samples, axis=0, normed=False, mu=None, sigma_sq=None, req_acorr=False):
    """
    Estimates effective sample sizes of samples along the specified axis
    with the monotone positive sequence estimator of "Practical Markov
    Chain Monte Carlo" by Geyer (1992). The estimator is ONLY VALID for
    reversible Markov chains. The inputs 'mu' and 'sigma_sq' are optional
    and unnecessary for the most cases in practice.

    Parameters
    ----------
    mu, sigma_sq : vectors for the mean E(x) and variance Var(x) if the
        analytical (or accurately estimated) value is available. If provided,
        it can stabilize the estimate of auto-correlations and hence ESS.
        This is intended for research uses when one wants to
        accurately quantify the asymptotic efficiency of a MCMC algorithm.
    req_acorr : bool
        If true, a list of estimated auto correlation sequences are returned as the
        second output.

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
    if mu is None:
        mu = np.mean(samples, axis)
    if sigma_sq is None:
        sigma_sq = np.var(samples, axis)

    d = samples.shape[1 - axis]
    ess = np.zeros(d)
    auto_cor = []
    for j in range(d):
        if axis == 0:
            x = samples[:, j]
        else:
            x = samples[j, :]
        ess_j, auto_cor_j = _monotone_sequence_1d(x, mu[j], sigma_sq[j])
        ess[j] = ess_j
        auto_cor.append(auto_cor_j)
    if normed:
        ess /= samples.shape[axis]

    if req_acorr:
        return ess, auto_cor
    return ess


def _monotone_sequence_1d(x, mean, var):

    auto_cor = []

    lag = 0
    auto_cor_sum = 0
    even_auto_cor = compute_acorr(x, lag, mean, var)
    auto_cor.append(even_auto_cor)
    auto_cor_sum -= even_auto_cor

    lag += 1
    odd_auto_cor = compute_acorr(x, lag, mean, var)
    auto_cor.append(odd_auto_cor)
    running_min = even_auto_cor + odd_auto_cor

    while (even_auto_cor + odd_auto_cor > 0) and (lag + 2 < len(x)):
        running_min = min(running_min, (even_auto_cor + odd_auto_cor))
        auto_cor_sum = auto_cor_sum + 2 * running_min

        lag += 1
        even_auto_cor = compute_acorr(x, lag, mean, var)
        auto_cor.append(even_auto_cor)

        lag = lag + 1
        odd_auto_cor = compute_acorr(x, lag, mean, var)
        auto_cor.append(odd_auto_cor)

    ess = len(x) / auto_cor_sum
    if auto_cor_sum < 0:
        # Rare, but can happen with floating point errors when 'x' shows
        # strong negative correlations.
        ess = float('inf')

    return ess, np.array(auto_cor)


def compute_acorr(x, lag, mean, var):
    """
    Returns an estimate of the lag 'k' auto-correlation of a time series 'x'.
    The estimator is biased towards zero due to the factor (n - k) / n.
    See Geyer (1992) Section 3.1 and the reference therein for justification.
    """
    n = len(x)
    acorr = (x[:(n - lag)] - mean) * (x[lag:] - mean)
    acorr = np.mean(acorr) / var * (n - lag) / n
    return acorr