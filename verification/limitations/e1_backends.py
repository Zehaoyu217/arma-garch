"""E1 -- what switching garch on changes besides the variance model.

Homoskedastic data (no ARCH at all): regression with AR(1) errors, T in {120, 500}, d in {0, 1}.
Three fits per series, all through the public wrapper:
  sm  = SARIMAXGARCH(..., garch=None)                       -> statsmodels SARIMAX (exact likelihood)
  e0  = SARIMAXGARCH(..., garch=(1,1)).fit(fixed alpha=beta=0) -> the engine without GARCH (same null)
  e1  = SARIMAXGARCH(..., garch=(1,1)).fit()                -> the engine with GARCH
Reports the backend gap llf(e0) - llf(sm), how often a GARCH-vs-not comparison picks GARCH when it is
done across backends (e1 vs sm, what you get by toggling garch) or within the engine (e1 vs e0), and
how far the mean estimates move between the two backends on the same data.
"""
import json
import warnings
from multiprocessing import Pool

import numpy as np

import common  # noqa: F401  (sets one BLAS thread, sys.path)
from sarimax_garch import SARIMAXGARCH

R = 200
DESIGNS = [(120, 0), (500, 0), (120, 1), (500, 1)]


def one(args):
    T, d, seed = args
    rng = np.random.default_rng(seed)
    x = common.ar1_exog(T + 1, 0.7, rng)
    w, *_ = common.simulate(T + 1, 0.2, 0.5, 1.0, 0.0, 0.0, rng.standard_normal(T + 1 + 500), X=x[:, None], g=[0.5])
    if d == 1:                               # levels: the model differences y and X itself
        y, X = np.cumsum(w), np.cumsum(x)[:, None]
    else:
        y, X = w, x[:, None]
    y, X = y[1:], X[1:]
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sm = SARIMAXGARCH(y, exog=X, order=(1, d, 0), trend="c").fit()
        mod = SARIMAXGARCH(y, exog=X, order=(1, d, 0), trend="c", garch=(1, 1))
        e0 = mod.fit(fixed_params={"alpha.L1": 0.0, "beta.L1": 0.0})
        e1 = mod.fit()
    for name, r in (("sm", sm), ("e0", e0), ("e1", e1)):
        out[name] = dict(llf=float(r.llf), aic=float(r.aic), nobs=int(r.nobs))
    se = np.asarray(sm.bse)[:3]
    out["dparam_se"] = list(np.abs(np.asarray(e0.params)[:3] - np.asarray(sm.params)[:3]) / se)   # intercept, x1, ar.L1
    return out


def main():
    res = {}
    with Pool(12) as pool:
        for T, d in DESIGNS:
            rows = pool.map(one, [(T, d, 1000 * T + 10 * d + i) for i in range(R)])
            llf = {k: np.array([r[k]["llf"] for r in rows]) for k in ("sm", "e0", "e1")}
            aic = {k: np.array([r[k]["aic"] for r in rows]) for k in ("sm", "e0", "e1")}
            gap = llf["e0"] - llf["sm"]
            lr_w, lr_x = 2 * (llf["e1"] - llf["e0"]), 2 * (llf["e1"] - llf["sm"])
            dp = np.array([r["dparam_se"] for r in rows])
            res[f"T={T},d={d}"] = dict(
                nobs_sm=rows[0]["sm"]["nobs"], nobs_engine=rows[0]["e0"]["nobs"],
                gap_mean=float(gap.mean()), gap_sd=float(gap.std()),
                lr_within_gt_5_99=float(np.mean(lr_w > 5.99)), lr_cross_gt_5_99=float(np.mean(lr_x > 5.99)),
                aic_picks_garch_within=float(np.mean(aic["e1"] < aic["e0"])),
                aic_picks_garch_cross=float(np.mean(aic["e1"] < aic["sm"])),
                median_dparam_in_sm_se=dict(zip(["intercept", "x1", "ar.L1"], np.median(dp, 0).round(3).tolist())),
                max_dparam_in_sm_se=dict(zip(["intercept", "x1", "ar.L1"], np.max(dp, 0).round(3).tolist())))
            print(f"T={T} d={d}: " + json.dumps(res[f"T={T},d={d}"]), flush=True)
    with open(f"{common.OUT}/e1_backends.json", "w") as f:
        json.dump(res, f, indent=1)


def scale_one(args):
    """The backend gap llf(engine, alpha = beta = 0) - llf(SARIMAX) for the same series in different units."""
    scale, seed = args
    rng = np.random.default_rng(seed)
    x = common.ar1_exog(501, 0.7, rng)
    w, *_ = common.simulate(501, 0.2, 0.5, 1.0, 0.0, 0.0, rng.standard_normal(501 + 500), X=x[:, None], g=[0.5])
    y, X = scale * w[1:], scale * x[1:, None]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sm = SARIMAXGARCH(y, exog=X, order=(1, 0, 0), trend="c").fit()
        e0 = SARIMAXGARCH(y, exog=X, order=(1, 0, 0), trend="c", garch=(1, 1)).fit(
            fixed_params={"alpha.L1": 0.0, "beta.L1": 0.0})
    return float(e0.llf - sm.llf)


def scale_check():
    """The gap is the log-density of the first (unscored) observation, so its SIGN depends on the units of y:
    about -0.5*log(2*pi*e*Var(u)), positive for Var(u) > 1/(2*pi*e) = 0.059 (sd 0.24), negative below."""
    out = {}
    with Pool(12) as pool:
        for s in (10.0, 1.0, 0.24, 0.1, 0.01):
            gaps = np.array(pool.map(scale_one, [(s, 7000 + i) for i in range(100)]))
            out[str(s)] = dict(gap_mean=float(gaps.mean()), gap_sd=float(gaps.std()))
            print(f"scale {s:>5}: sd(y) ~ {s * np.sqrt(1 / 0.75):.3f}   gap engine - statsmodels = {gaps.mean():+.2f} "
                  f"(sd {gaps.std():.2f})", flush=True)
    with open(f"{common.OUT}/e1_scale.json", "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    import sys as _sys
    scale_check() if "scale" in _sys.argv[1:] else main()
