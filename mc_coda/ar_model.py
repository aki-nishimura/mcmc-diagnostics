"""
Fits an auto-regressive process to a univariate time series. The order is
determined by minimizing AIC and the coefficients are estimated via the
Yule-Walker equation.

The code is based on the function
    statsmodels.regression.linear_model.yule_walker
from the statsmodels package and has been adapted by Aki Nishimura.
"""

# License for the original Statsmodels code (BSD 3-clause)
# --------------------------------------------------------
#
# Copyright (C) 2006, Jonathan E. Taylor
# All rights reserved.
#
# Copyright (c) 2006-2008 Scipy Developers.
# All rights reserved.
#
# Copyright (c) 2009-2018 Statsmodels Developers.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#   a. Redistributions of source code must retain the above copyright notice,
#      this list of conditions and the following disclaimer.
#   b. Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#   c. Neither the name of Statsmodels nor the names of its contributors
#      may be used to endorse or promote products derived from this software
#      without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL STATSMODELS OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
# OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
# DAMAGE.


import numpy as np
import math
from scipy.linalg import solve_toeplitz


def fit(x, max_order=None):

    if max_order is None:
        max_order = min(
            len(x) - 1, math.ceil(10 * np.log10(len(x)))
        )

    aic_best = float('inf')
    ar_order_best = 0
    ar_coef_best = None
    for order in range(max_order + 1):
        ar_coef, aic = fit_via_yule_walker(x, order)
        if aic < aic_best:
            aic_best = aic
            ar_order_best = order
            ar_coef_best = ar_coef

    return ar_order_best, ar_coef_best


def fit_via_yule_walker(x, order, acf_method="mle", demean=True):
    """
    Estimate AR(p) parameters of a sequence x using the Yule-Walker equation.

    Parameters
    ----------
    x : 1d numpy array
    order : integer
        The order of the autoregressive process.
    acf_method : {'unbiased', 'mle'}, optional
       Method can be 'unbiased' or 'mle' and this determines denominator in
       estimating autocorrelation function (ACF) at lag k. If 'mle', the
       denominator is  `n = x.shape[0]`, if 'unbiased' the denominator is `n - k`.
    demean : bool
        True, the mean is subtracted from `x` before estimation.
    """

    if demean:
        x = x.copy()
        x -= x.mean()

    if acf_method == "unbiased":
        denom = lambda lag: len(x) - lag
    else:
        denom = lambda lag: len(x)
    if x.ndim > 1 and x.shape[1] != 1:
        raise ValueError("expecting a vector to estimate AR parameters")

    auto_cov = np.zeros(order + 1, np.float64)
    auto_cov[0] = (x ** 2).sum() / denom(0)
    for lag in range(1, order + 1):
        auto_cov[lag] = np.sum(x[0:-lag] * x[lag:]) / denom(lag)

    if order == 0:
        ar_coef = None
        innovation_var = auto_cov[0]
    else:
        ar_coef = _solve_yule_walker(auto_cov)
        innovation_var = auto_cov[0] - (auto_cov[1:] * ar_coef).sum()

    aic = compute_aic(innovation_var, order, len(x))

    return ar_coef, aic


def _solve_yule_walker(auto_cov):
    ar_coef = solve_toeplitz(auto_cov[:-1], auto_cov[1:])
    return ar_coef


def compute_aic(innovation_var, ar_order, n_obs):
    # I don't quite understand the formula (the maximized likelihood part), but
    # it agrees with the one used in R's ar.yw.default as well as Python's statsmodels.
    # Also, described in http://pages.stern.nyu.edu/~churvich/TimeSeries/Handouts/AICC.pdf
    return n_obs * np.log(innovation_var) + 2 * (1 + ar_order)

