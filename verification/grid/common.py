"""Shared design for the validation grid: ARMA orders, GARCH parameter sets, data generator.

Data are simulated on the differenced scale (w_t and x_t) as regression with ARMA-GARCH errors,
then integrated d times to levels -- so a fit with d on the levels must recover the same truth."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.environ.get("JAG_DIR", os.path.join(HERE, "..", "..", "fixed")))
import joint_armax_garch as J  # noqa: E402

OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)

ORDERS = {(0, 0): ([], []), (1, 0): ([0.5], []), (0, 1): ([], [0.4]), (1, 1): ([0.5], [0.3]),
          (2, 1): ([0.5, -0.25], [0.3]), (1, 2): ([0.5], [0.3, 0.2]), (2, 2): ([0.5, -0.25], [0.3, 0.2])}
GARCH = {"typical": (0.05, 0.08, 0.90), "low-pers": (0.20, 0.10, 0.60), "high-alpha": (0.10, 0.20, 0.70),
         "high-pers": (0.01, 0.05, 0.94), "arch-like": (0.50, 0.30, 0.20)}
C0, GAMMA, H = 0.1, [1.0, -0.5], 10


def make_X(n, seed):
    return np.column_stack([np.sin(2 * np.pi * np.arange(n) / 12), np.random.default_rng(seed).normal(size=n)])


def garch_params(garch):
    """A GARCH-set name from GARCH, or an explicit (omega, alpha, beta) tuple."""
    return GARCH[garch] if isinstance(garch, str) else tuple(garch)


def truth(order, garch, dist, nu, use_x):
    phi, theta = ORDERS[order]
    return np.r_[C0, phi, theta, (GAMMA if use_x else []), garch_params(garch), ([nu] if dist == "t" else [])]


def make_data(T, order, d, garch, dist="t", nu=6.0, use_x=True, seed=0):
    """Returns y_levels (T,), X_levels (T, 2) or None, X_future_levels (H, 2) or None."""
    phi, theta = ORDERS[order]
    om, al, be = garch_params(garch)
    Xw = make_X(T + H, seed + 7) if use_x else None
    w, _, _ = J.simulate(T, C0, phi, theta, om, al, be, dist=dist, nu=nu,
                         X=None if Xw is None else Xw[:T], gamma=GAMMA if use_x else None, seed=seed)
    y, X = w, Xw
    for _ in range(d):
        y = 100.0 + np.cumsum(y)
        X = None if X is None else np.cumsum(X, axis=0)
    return y, (None if X is None else X[:T]), (None if X is None else X[T:])


def diff_future_X(X_levels, Xf_levels, d):
    """Future regressors on the differenced scale (what rugarch is given)."""
    if Xf_levels is None:
        return None
    return Xf_levels if d == 0 else np.diff(np.vstack([X_levels[-d:], Xf_levels]), n=d, axis=0)
