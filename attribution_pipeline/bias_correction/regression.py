"""
Core bias-correction math -- ported verbatim from
Exploratory_Work/Reduced_Att_Set_Processing/reduced_set_HG_bias_correction.py.
Do not change the formulas; only the surrounding data plumbing has been
generalized.

`bias_correct` is the key trick: given a SINGLE member/year scalar value and
the baseline-year-offset array `t` (baseline_years - target_year), it uses the
fitted obs/sim trend lines to project that one realized value across the
whole baseline period, producing a synthetic len(t) "pseudo-baseline" column.
"""

import numpy as np
import statsmodels.api as sm


def soft_log(x):
    """log(exp(x) - 1); avoids negative FWI/DSR values under log transform."""
    x = np.asarray(x, dtype=float)
    return np.log(np.exp(x) - 1)


def inverse_soft_log(x):
    """Inverse of soft_log: log(exp(x) + 1)."""
    x = np.asarray(x, dtype=float)
    return np.log(np.exp(x) + 1)


def find_regression_parameters(fwi, t):
    """Fit fwi ~ fwi0 + delta*t via OLS. Returns (fwi0, delta, std_residual)."""
    X = sm.add_constant(t)
    model = sm.OLS(fwi, X)
    results = model.fit()
    fwi0, delta = results.params
    return fwi0, delta, np.std(fwi - delta * t)


def bias_correct(scalar_log, t, fwi0_obs, delta_sim, fwi0_sim):
    """Detrend/rescale a member's single log-space scalar against the fitted
    sim trend, then re-centre on the fitted obs trend. Broadcasts across `t`
    (length = number of baseline years) to produce the pseudo-baseline series."""
    return fwi0_obs + (scalar_log - delta_sim * t - fwi0_sim)
