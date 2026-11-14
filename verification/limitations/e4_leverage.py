"""E4 -- asymmetric VOLATILITY (leverage): GJR data fitted with the package's symmetric GARCH(1,1).

DGP: y_t = g x_t + u_t, u_t = c + phi u_{t-1} + eps_t,
     sigma_t^2 = omega + (alpha + lev 1[eps_{t-1} < 0]) eps_{t-1}^2 + beta sigma_{t-1}^2,  z ~ N(0, 1)
     persistence alpha + lev/2 + beta = 0.97 in every design; lev = 0 is the symmetric control.
Estimators: SM (statsmodels SARIMAX, no GARCH), GN / GT (engine, symmetric GARCH, normal / t),
GJR (prototype: the engine's mean filter + GJR variance, Gaussian; not in the package).
The shocks are symmetric, so any mean bias here comes from the mis-specified VARIANCE leaking into the
mean through the joint likelihood (the variance equation depends on eps_{t-1} = f(c, phi, g)).
"""
import json
import sys
import warnings
from multiprocessing import Pool

import numpy as np

import common
from common import _core
from sarimax_garch import SARIMAXGARCH

PNAMES = ["c", "phi", "g", "mu"]
BASE = dict(c=0.1, phi=0.5, g=0.5, omega=0.03, beta=0.90)


def one(args):
    T, lev, seed = args
    alpha = 0.07 - lev / 2
    rng = np.random.default_rng(seed)
    x = common.ar1_exog(T, 0.7, rng)
    y, u, eps, sig2 = common.simulate(T, BASE["c"], BASE["phi"], BASE["omega"], alpha, BASE["beta"],
                                      rng.standard_normal(T + 500), lev=lev, X=x[:, None], g=[BASE["g"]])
    X = x[:, None]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sm = SARIMAXGARCH(y, exog=X, order=(1, 0, 0), trend="c").fit()
        gn = _core.fit_joint(y, X=X, p=1, q=0, dist="normal")
        gt = _core.fit_joint(y, X=X, p=1, q=0, dist="t", compute_cov=False)
        gj, gj_ll, *_ = common.fit_gjr(y, X, 1, 0, gn.params)
    est = {"SM": [sm.params["intercept"], sm.params["ar.L1"], sm.params["x1"]],
           "GN": [gn["const"], gn["phi1"], gn["gamma1"]], "GT": [gt["const"], gt["phi1"], gt["gamma1"]],
           "GJR": list(gj[:3])}
    est = {k: [float(a) for a in v] + [float(v[0] / (1 - v[1]))] for k, v in est.items()}
    lo, hi = gn.params[:3] - 1.96 * gn.se[:3], gn.params[:3] + 1.96 * gn.se[:3]
    tv = np.array([BASE["c"], BASE["phi"], BASE["g"]])
    return dict(est=est, cover_gn=((lo <= tv) & (tv <= hi)).tolist(), lev_hat=float(gj[-1]),
                lr_lev=float(2 * (gj_ll - gn.loglik)), alpha_gn=float(gn["alpha"]))


def main():
    R = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    T = 2000
    truth = np.array([BASE["c"], BASE["phi"], BASE["g"], BASE["c"] / (1 - BASE["phi"])])
    results = {}
    with Pool(12) as pool:
        for lev in (0.0, 0.06, 0.14):
            rows = pool.map(one, [(T, lev, 500_000 + int(lev * 1000) * 1000 + r) for r in range(R)])
            key = f"T={T} lev={lev:.2f} alpha={0.07 - lev / 2:.2f}"
            res = dict(R=R, lev_hat_mean=float(np.mean([r["lev_hat"] for r in rows])),
                       lr_lev_reject_5pct=float(np.mean([r["lr_lev"] > 3.84 for r in rows])),
                       gn_coverage95=dict(zip(["c", "phi", "g"], np.mean([r["cover_gn"] for r in rows], 0).round(3).tolist())),
                       params={})
            print(f"\n=== {key} (R={R}): GJR lev_hat mean {res['lev_hat_mean']:.3f}, LR(lev=0) rejects "
                  f"{res['lr_lev_reject_5pct']:.2f}; GN 95% coverage {res['gn_coverage95']}")
            print(f"  {'':4}" + "".join(f"{p:>27}" for p in PNAMES))
            for nm in ("SM", "GN", "GT", "GJR"):
                s = common.summarise(np.array([r["est"][nm] for r in rows]), truth)
                res["params"][nm] = {p: dict(bias=round(float(s["bias"][j]), 4), sd=round(float(s["sd"][j]), 4),
                                             rmse=round(float(s["rmse"][j]), 4), bias_over_sd=round(float(s["bias_sd"][j]), 2))
                                     for j, p in enumerate(PNAMES)}
                print(f"  {nm:4}" + "".join(f"{s['bias'][j]:+.4f} ({s['sd'][j]:.4f}) {s['bias_sd'][j]:+.2f}sd".rjust(27)
                                            for j in range(len(PNAMES))), flush=True)
            results[key] = res
    with open(f"{common.OUT}/e4_leverage.json", "w") as f:
        json.dump(results, f, indent=1)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "time":
        import time
        t0 = time.time()
        o = one((2000, 0.14, 1))
        print(f"one replication: {time.time() - t0:.1f}s", {k: [round(v, 3) for v in vv] for k, vv in o["est"].items()},
              "lev_hat", round(o["lev_hat"], 3))
    else:
        main()
