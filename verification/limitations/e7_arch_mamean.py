"""E7 -- MA-GARCH cross-check against Python arch >= 9.0 (MAMean, added in arch 9.0.0, 2026-11-11).

Fits the same simulated MA(q)-GARCH(1,1) series (T = 3000) with arch's MAMean and with the engine
(fit_joint(p=0, q=q)), normal and Student-t, and reports the largest parameter difference in units of the
engine's robust SE, both log-likelihoods, and the engine's log-likelihood at arch's estimate.
Needs a Python with arch >= 9.0 (not a dependency of the package):  pip install "arch>=9.0"
"""
import warnings

import numpy as np

import common  # noqa: F401  (sys.path to the package)
from sarimax_garch import _core

from arch.univariate import GARCH, MAMean, Normal, StudentsT  # noqa: E402

CASES = [(1, [0.5], "normal"), (1, [0.5], "t"), (2, [0.4, -0.3], "normal"), (1, [0.9], "normal"), (1, [-0.6], "t")]

if __name__ == "__main__":
    for seed, (q, theta, dist) in enumerate(CASES):
        y, _, _ = _core.simulate(3000, 0.1, [], theta, 0.05, 0.08, 0.90, dist=dist, nu=6.0, seed=100 + seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            am = MAMean(y, lags=q, volatility=GARCH(1, 0, 1),
                        distribution=Normal() if dist == "normal" else StudentsT(), rescale=False).fit(disp="off")
            en = _core.fit_joint(y, p=0, q=q, dist=dist)
        a, mine = am.params.values, en.params      # const, ma.., omega, alpha, beta, (nu) in both
        d = np.abs(a - mine)
        ll_at_arch = -_core._neg_loglik_raw(a, en.y, None, 0, q, dist)
        print(f"MA({q}) theta={theta} {dist:6}: max |param diff| {d.max():.1e} = {(d / en.se).max():.3f} SE | "
              f"loglik arch {am.loglikelihood:.4f}, engine {en.loglik:.4f}, engine at arch's estimate {ll_at_arch:.4f}")
