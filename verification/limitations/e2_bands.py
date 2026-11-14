"""E2 -- point forecasts and bands under GARCH-t: what get_forecast() gives and how well it covers.

DGP: u_t = c + phi u_{t-1} + eps_t, GARCH(1,1), standardised t(5) shocks (D1, D2) or normal (D3).
Per replication: fit on T observations, forecast H = 20 steps, and score every band against the TRUE
predictive distribution at that origin (M = 20,000 paths from the true parameters and the true
end-of-sample state) -- so each replication contributes its exact conditional coverage, and the average
over replications is the unconditional coverage.

Bands (the wrapper's GARCHPredictionResults.conf_int is mean +/- z*se_mean for method="normal", the
default, and quantiles of fc.sims for method="simulated"):
  fit-t default     dist="t" fit, method="normal"      (what conf_int() / summary_frame() return)
  fit-t simulated   dist="t" fit, method="simulated"   (n_sims = 10,000, the get_forecast default)
  fit-n default     dist="normal" fit, method="normal"
  fit-n simulated   dist="normal" fit, method="simulated"
  oracle normal     true parameters, normal shape      (isolates the shape error)
  oracle simulated  true parameters, simulated         (sanity: ~nominal)
Also: the point forecast error of the fitted mean against the true conditional mean, by horizon.
"""
import json
import sys
import warnings
from multiprocessing import Pool

import numpy as np
from scipy import stats

import common
from common import _core

H, M, NSIM = 20, 20_000, 10_000
LEVELS = (0.80, 0.90, 0.95, 0.99)
TRUE0 = dict(c=0.05, phi=0.5, omega=0.05, alpha=0.08, beta=0.90)
DESIGNS = {"D1 t5 T=1000": dict(T=1000, z=dict(dist="t", nu=5.0)),
           "D2 t5 T=250": dict(T=250, z=dict(dist="t", nu=5.0)),
           "D3 normal T=1000": dict(T=1000, z=dict(dist="normal")),
           "D4 t5 T=250 phi=0.9": dict(T=250, z=dict(dist="t", nu=5.0), true=dict(c=0.01, phi=0.9))}
BANDS = ["fit-t default", "fit-t simulated", "fit-n default", "fit-n simulated", "oracle normal", "oracle simulated"]


def bands_from(fc, kind):
    """(levels, 2, H) lower/upper from a core forecast dict."""
    out = np.empty((len(LEVELS), 2, H))
    for i, lv in enumerate(LEVELS):
        a = (1 - lv) / 2
        if kind == "normal":
            z = stats.norm.ppf(1 - a)
            sd = np.sqrt(fc["variance"])
            out[i, 0], out[i, 1] = fc["mean"] - z * sd, fc["mean"] + z * sd
        else:
            out[i, 0], out[i, 1] = np.quantile(fc["sims"], [a, 1 - a], axis=0)
    return out


def one(args):
    name, seed = args
    D = DESIGNS[name]
    T, zd = D["T"], D["z"]
    TRUE = {**TRUE0, **D.get("true", {})}
    rng = np.random.default_rng(seed)
    z = common.draw_z(rng, T + 500, **zd)
    y, u, eps, sig2 = common.simulate(T, TRUE["c"], TRUE["phi"], TRUE["omega"], TRUE["alpha"], TRUE["beta"], z)
    truth = common.true_future((u[-1], eps[-1], sig2[-1]), H, M, rng, **TRUE, zdist=zd)
    cond_mean = truth.mean(0)
    res = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ft = _core.fit_joint(y, p=1, q=0, dist="t", compute_cov=False)
        fn = _core.fit_joint(y, p=1, q=0, dist="normal", compute_cov=False)
        fixed = dict(const=TRUE["c"], phi1=TRUE["phi"], omega=TRUE["omega"], alpha=TRUE["alpha"], beta=TRUE["beta"])
        if zd["dist"] == "t":
            fixed["nu"] = zd["nu"]
        fo = _core.fit_joint(y, p=1, q=0, dist=zd["dist"], fixed=fixed, compute_cov=False)
    fc_t = _core.forecast(ft, H, n_sims=NSIM, seed=seed + 1)
    fc_n = _core.forecast(fn, H, n_sims=NSIM, seed=seed + 2)
    fc_o = _core.forecast(fo, H, n_sims=NSIM, seed=seed + 3)
    B = {"fit-t default": bands_from(fc_t, "normal"), "fit-t simulated": bands_from(fc_t, "sim"),
         "fit-n default": bands_from(fc_n, "normal"), "fit-n simulated": bands_from(fc_n, "sim"),
         "oracle normal": bands_from(fc_o, "normal"), "oracle simulated": bands_from(fc_o, "sim")}
    cov = np.empty((len(BANDS), len(LEVELS), 3, H))          # inside, below, above
    for b, nm in enumerate(BANDS):
        for i in range(len(LEVELS)):
            lo, hi = B[nm][i]
            below = np.mean(truth < lo, 0)
            above = np.mean(truth > hi, 0)
            cov[b, i] = np.vstack([1 - below - above, below, above])
    res["cov"] = cov
    res["mean_err_t"] = fc_t["mean"] - cond_mean               # point forecast vs true conditional mean
    res["mean_err_n"] = fc_n["mean"] - cond_mean
    res["true_sd"] = truth.std(0)
    res["nu_hat"] = float(ft["nu"])
    res["converged"] = (bool(ft.converged), bool(fn.converged))
    # is the predictive distribution normal?  excess kurtosis of the true h-step distribution
    res["true_kurt"] = stats.kurtosis(truth, axis=0)
    return res


def main():
    R = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    summary = {}
    with Pool(12) as pool:
        for d_i, name in enumerate(DESIGNS):
            rows = pool.map(one, [(name, 10_000 * (d_i + 1) + r) for r in range(R)])
            cov = np.mean([r["cov"] for r in rows], 0)           # (band, level, 3, H)
            met = np.array([r["mean_err_t"] for r in rows])
            sd = np.mean([r["true_sd"] for r in rows], 0)
            s = dict(R=R, nu_hat_median=float(np.median([r["nu_hat"] for r in rows])),
                     not_converged_t=float(np.mean([not r["converged"][0] for r in rows])),
                     not_converged_n=float(np.mean([not r["converged"][1] for r in rows])),
                     true_excess_kurtosis_h=dict((f"h{h}", float(np.mean([r["true_kurt"][h - 1] for r in rows])))
                                                 for h in (1, 5, 10, 20)),
                     point_rmse_over_sd=dict((f"h{h}", float(np.sqrt(np.mean(met[:, h - 1] ** 2)) / sd[h - 1]))
                                             for h in (1, 5, 10, 20)),
                     coverage={})
            for b, bn in enumerate(BANDS):
                s["coverage"][bn] = {}
                for i, lv in enumerate(LEVELS):
                    s["coverage"][bn][f"{int(lv * 100)}"] = {
                        f"h{h}": [round(float(cov[b, i, k, h - 1]), 4) for k in range(3)] for h in (1, 5, 10, 20)}
            summary[name] = s
            print(f"\n=== {name} (R={R}) nu_hat median {s['nu_hat_median']:.2f}  not-converged t/n "
                  f"{s['not_converged_t']:.2f}/{s['not_converged_n']:.2f}")
            print("  true predictive excess kurtosis:", {k: round(v, 2) for k, v in s["true_excess_kurtosis_h"].items()})
            print("  point forecast RMSE (vs true cond. mean) / predictive sd:",
                  {k: round(v, 3) for k, v in s["point_rmse_over_sd"].items()})
            print(f"  {'band':<17}" + "".join(f"{f'{int(lv*100)}% h1/h5/h10/h20':>31}" for lv in LEVELS))
            for b, bn in enumerate(BANDS):
                row = f"  {bn:<17}"
                for i in range(len(LEVELS)):
                    row += f"{' / '.join(f'{cov[b, i, 0, h - 1]:.3f}' for h in (1, 5, 10, 20)):>31}"
                print(row, flush=True)
    with open(f"{common.OUT}/e2_bands.json", "w") as f:
        json.dump(summary, f, indent=1)


if __name__ == "__main__":
    main()
