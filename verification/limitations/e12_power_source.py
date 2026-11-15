"""E12 -- is the low power of GARCH tests due to GARCH itself, or to the SARIMAX mean model in front of it?

One GARCH(1,1) shock series eps_t drives every mean model (paired comparison):
  oracle     the shocks themselves, zero mean known (no mean estimation at all)
  mean       y = 0.1 + eps
  OLS        y = 0.1 + 0.5 x + eps                            (x exogenous AR(1))
  AR(1)      y_t = 0.1 + 0.5 y_{t-1} + eps_t
  ARMA(1,1)  ... + eps_t + 0.3 eps_{t-1}
  MA(1) .9   y_t = 0.1 + eps_t + 0.9 eps_{t-1}               (near-unit-root MA)
  ARIMA(1,1,1) the ARMA(1,1) on the first difference (with drift)
  airline    (1-L)(1-L^12) y = (1 - 0.4L)(1 - 0.6L^12) eps    (two-stage only)
GARCH: none (size) | typical (alpha .10, beta .75) | persistent (alpha .08, beta .90); T = 120, 240, 480.
Tests at 5% on each model: LR one-stage (the engine's joint fit, GARCH vs alpha = beta = 0), LR two-stage
(zero-mean GARCH on the SARIMAX/OLS residuals), ARCH-LM(4) and ARCH-LM(12) on those residuals.  The oracle runs
the same tests on the true shocks.
"""
import json
import sys
import warnings
from multiprocessing import Pool

import numpy as np

import common
from common import _core
from sarimax_garch import SARIMAXGARCH, arch_lm_test

GARCHS = {"none": (1.0, 0.0, 0.0), "typical": (0.15, 0.10, 0.75), "persistent": (0.02, 0.08, 0.90)}
MODELS = {  # name: (engine p, q, d, uses x) for one-stage, SARIMAX (order, seasonal, trend) for two-stage
    "mean": ((0, 0, 0, False), ((0, 0, 0), (0, 0, 0, 0), "c")),
    "OLS": ((0, 0, 0, True), ((0, 0, 0), (0, 0, 0, 0), "c")),
    "AR(1)": ((1, 0, 0, False), ((1, 0, 0), (0, 0, 0, 0), "c")),
    "ARMA(1,1)": ((1, 1, 0, False), ((1, 0, 1), (0, 0, 0, 0), "c")),
    "MA(1) .9": ((0, 1, 0, False), ((0, 0, 1), (0, 0, 0, 0), "c")),
    "ARIMA(1,1,1)": ((1, 1, 1, False), ((1, 1, 1), (0, 0, 0, 0), "c")),
    "airline": (None, ((0, 1, 1), (0, 1, 1, 12), None)),
}
BURN = 600


def build(eps, x):
    n = len(eps)
    ys = {"mean": 0.1 + eps, "OLS": 0.1 + 0.5 * x + eps}
    ar = np.zeros(n); arma = np.zeros(n); ma9 = 0.1 + eps.copy()
    ma9[1:] += 0.9 * eps[:-1]
    for t in range(1, n):
        ar[t] = 0.1 + 0.5 * ar[t - 1] + eps[t]
        arma[t] = 0.1 + 0.5 * arma[t - 1] + eps[t] + 0.3 * eps[t - 1]
    w = eps.copy()
    w[1:] += -0.4 * eps[:-1]; w[12:] += -0.6 * eps[:-12]; w[13:] += 0.24 * eps[:-13]
    zz = np.zeros(n)
    for t in range(n):
        zz[t] = w[t] + (zz[t - 12] if t >= 12 else 0.0)
    ys.update({"AR(1)": ar, "ARMA(1,1)": arma, "MA(1) .9": ma9, "ARIMA(1,1,1)": np.cumsum(0.05 + arma - 0.1),
               "airline": np.cumsum(zz)})
    return ys


def tests_on(r):
    """Two-stage LR and ARCH-LM on a residual series r (zero mean)."""
    g = _core.fit_joint(r, p=0, q=0, dist="normal", fixed={"const": 0.0}, compute_cov=False)
    g0 = _core.fit_joint(r, p=0, q=0, dist="normal", fixed={"const": 0.0, "alpha": 0.0, "beta": 0.0}, compute_cov=False)
    return {"LR two-stage": 2 * (g.loglik - g0.loglik) > 5.99, "ARCH-LM(4)": arch_lm_test(r, 4)[1] < 0.05,
            "ARCH-LM(12)": arch_lm_test(r, 12)[1] < 0.05}


def one(args):
    gk, T, seed = args
    om, al, be = GARCHS[gk]
    rng = np.random.default_rng(seed)
    n = T + BURN
    z = rng.standard_normal(n)
    eps, s2 = np.zeros(n), np.zeros(n)
    s2[0] = om / (1 - al - be)
    eps[0] = np.sqrt(s2[0]) * z[0]
    for t in range(1, n):
        s2[t] = om + al * eps[t - 1] ** 2 + be * s2[t - 1]
        eps[t] = np.sqrt(s2[t]) * z[t]
    x = common.ar1_exog(n, 0.7, rng)
    ys = build(eps, x)
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        np.seterr(all="ignore")
        e = eps[-T:]
        o = tests_on(e)
        o["LR one-stage"] = o["LR two-stage"]                       # oracle: the mean is known, nothing to estimate
        out["oracle"] = o
        for name, (eng, (order, seas, trend)) in MODELS.items():
            y = ys[name][-T:]
            X = x[-T:, None] if name == "OLS" else None
            sm = SARIMAXGARCH(y, exog=X, order=order, seasonal_order=seas, trend=trend).fit()
            burn = int(sm.loglikelihood_burn)
            r = np.asarray(sm.standardized_forecasts_error)[0, burn:] * np.sqrt(sm.params["sigma2"])
            o = tests_on(r[np.isfinite(r)])
            if eng is not None:
                p, q, d, _ = eng
                g = _core.fit_joint(y, X=X, p=p, q=q, d=d, dist="normal", compute_cov=False)
                g0 = _core.fit_joint(y, X=X, p=p, q=q, d=d, dist="normal", fixed={"alpha": 0.0, "beta": 0.0},
                                     compute_cov=False)
                o["LR one-stage"] = 2 * (g.loglik - g0.loglik) > 5.99
            out[name] = o
    return out


def main():
    R = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    res = {}
    with Pool(workers) as pool:
        for i, gk in enumerate(GARCHS):
            for j, T in enumerate((120, 240, 480)):
                rows = pool.map(one, [(gk, T, 5_000_000 + 100_000 * i + 10_000 * j + r) for r in range(R)])
                key = f"GARCH {gk} | T={T}"
                res[key] = {m: {t: round(float(np.mean([r[m][t] for r in rows if t in r[m]])), 3)
                                for t in ("LR one-stage", "LR two-stage", "ARCH-LM(4)", "ARCH-LM(12)") if t in rows[0][m]}
                            for m in rows[0]}
                print(f"{key}: " + " | ".join(
                    f"{m} " + "/".join(f"{res[key][m].get(t, float('nan')):.2f}" for t in ("LR one-stage", "LR two-stage", "ARCH-LM(4)"))
                    for m in res[key]), flush=True)
    with open(f"{common.OUT}/e12_power_source.json", "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
