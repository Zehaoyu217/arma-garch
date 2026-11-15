"""Locate the mismatch: the LR null distribution (AR1 + x, no GARCH, T = 120) produced three ways.
  A  true DGP (common.simulate)  + the wrapper's fits (as garch_test computes the observed LR)
  B  true DGP (common.simulate)  + _core.fit_joint direct (as garch_test computes the bootstrap LR*)
  C  _simulate_null(true params) + _core.fit_joint direct (the bootstrap DGP)"""
import warnings
from multiprocessing import Pool

import numpy as np

import common
from common import _core
from sarimax_garch import SARIMAXGARCH
from sarimax_garch.compare import _simulate_null

import e11_valid_inference as e


def fit_lr(y, X):
    f0 = _core.fit_joint(y, X=X, p=1, q=0, d=0, dist="normal", compute_cov=False, fixed={"alpha": 0.0, "beta": 0.0})
    f1 = _core.fit_joint(y, X=X, p=1, q=0, d=0, dist="normal", compute_cov=False)
    return max(0.0, 2 * (f1.loglik - f0.loglik))


def one(seed):
    rng = np.random.default_rng(seed)
    y, X, order = e.series("AR1", "none", 120, rng)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        np.seterr(all="ignore")
        mod = SARIMAXGARCH(y, exog=X, order=order, trend="c", garch=(1, 1))
        a = max(0.0, 2 * (mod.fit().llf - mod.fit(fixed_params={"alpha.L1": 0.0, "beta.L1": 0.0}).llf))
        b = fit_lr(y, X)
        true = dict(const=0.1, phi1=0.5, gamma1=0.5, omega=1.0, alpha=0.0, beta=0.0)
        rt = _core.fit_joint(y, X=X, p=1, q=0, d=0, dist="normal", fixed=true, compute_cov=False)
        yc = _simulate_null(rt, np.random.default_rng(seed + 7).standard_normal((1, 120 + 200)))[0]
        c = fit_lr(yc, X)
    return a, b, c


if __name__ == "__main__":
    with Pool(12) as pool:
        rows = np.array(pool.map(one, range(6_000_000, 6_001_200)))
    for j, lab in enumerate(("A true DGP + wrapper fits", "B true DGP + fit_joint", "C bootstrap DGP + fit_joint")):
        v = rows[:, j]
        print(f"{lab:30} mean {v.mean():.3f}  P(=0) {np.mean(v < 1e-6):.3f}  q90 {np.quantile(v, .9):.2f}  "
              f"q95 {np.quantile(v, .95):.2f}  P(>5.99) {np.mean(v > 5.99):.3f}")
    print(f"max |A - B| = {np.max(np.abs(rows[:, 0] - rows[:, 1])):.2e}")
