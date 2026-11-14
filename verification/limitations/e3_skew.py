"""E3 -- what asymmetric (skewed) innovations do to the mean estimates and the 1-step bands.

DGP: y_t = g x_t + u_t,  u_t = c + phi u_{t-1} + eps_t,  GARCH(1,1),  z ~ Hansen skew-t(nu=6, lam)
     (lam < 0: long left tail; lam = 0 is the symmetric t6 control).  x: exogenous AR(1).
Estimators on the same series:
  SM  statsmodels SARIMAX through the wrapper (garch=None): homoskedastic Gaussian exact MLE
  GN  engine, GARCH(1,1), dist="normal"  (Gaussian QMLE)
  GT  engine, GARCH(1,1), dist="t"       (symmetric-t MLE: the package's fat-tail option)
  GS  prototype: the engine's filter with a Hansen skew-t likelihood (common.fit_skewt) -- not in the package
Reported: bias / SD / RMSE of c, phi, g, mu = c/(1-phi), omega, alpha, beta; 95% robust-Wald coverage
for GN and GT; the 1-step point forecast error against the true conditional mean (in units of the true
sigma_{T+1}); and the TRUE probability that y_{T+1} falls below / above each model's 1% and 5% quantiles.
Theory check: the pseudo-true location shift of a location-scale t fitted to i.i.d. skew-t draws.
"""
import json
import sys
import warnings
from multiprocessing import Pool

import numpy as np
from scipy import optimize, stats

import common
from common import _core
from sarimax_garch import SARIMAXGARCH

TRUE = dict(c=0.1, phi=0.5, g=0.5, omega=0.05, alpha=0.08, beta=0.90)
NU = 6.0
TAILS = (0.01, 0.05)
PNAMES = ["c", "phi", "g", "mu", "omega", "alpha", "beta"]


def one(args):
    T, lam, seed = args
    rng = np.random.default_rng(seed)
    x = common.ar1_exog(T + 1, 0.7, rng)
    z = common.draw_z(rng, T + 500, "skewt", NU, lam)
    y, u, eps, sig2 = common.simulate(T, TRUE["c"], TRUE["phi"], TRUE["omega"], TRUE["alpha"], TRUE["beta"], z,
                                      X=x[:T, None], g=[TRUE["g"]])
    X = x[:T, None]
    # true 1-step predictive: location m, scale s, shape skew-t(NU, lam)
    m_true = TRUE["g"] * x[T] + TRUE["c"] + TRUE["phi"] * u[-1]
    s_true = np.sqrt(TRUE["omega"] + TRUE["alpha"] * eps[-1] ** 2 + TRUE["beta"] * sig2[-1])
    out = dict(est={}, cover={}, loc_err={}, tails={}, conv={})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sm = SARIMAXGARCH(y, exog=X, order=(1, 0, 0), trend="c").fit()
        gn = _core.fit_joint(y, X=X, p=1, q=0, dist="normal")
        gt = _core.fit_joint(y, X=X, p=1, q=0, dist="t")
        gs_th, gs_ll, gs_eps, gs_sig2 = common.fit_skewt(y, X, 1, 0, gt.params)
    truth = np.array([TRUE["c"], TRUE["phi"], TRUE["g"], TRUE["c"] / (1 - TRUE["phi"]),
                      TRUE["omega"], TRUE["alpha"], TRUE["beta"]])

    def pack(c, phi, g, om=np.nan, al=np.nan, be=np.nan):
        return [c, phi, g, c / (1 - phi), om, al, be]

    smp = sm.params
    out["est"]["SM"] = pack(smp["intercept"], smp["ar.L1"], smp["x1"])
    for nm, r in (("GN", gn), ("GT", gt)):
        out["est"][nm] = pack(r["const"], r["phi1"], r["gamma1"], r["omega"], r["alpha"], r["beta"])
        lo, hi = r.params - 1.96 * r.se, r.params + 1.96 * r.se
        idx = [r.names.index(n) for n in ("const", "phi1", "gamma1")]
        tv = np.array([TRUE["c"], TRUE["phi"], TRUE["g"]])
        out["cover"][nm] = ((lo[idx] <= tv) & (tv <= hi[idx])).tolist()
        out["conv"][nm] = bool(r.converged)
    out["est"]["GS"] = pack(gs_th[0], gs_th[1], gs_th[2], gs_th[3], gs_th[4], gs_th[5])
    out["nu_t"], out["nu_s"], out["lam_s"] = float(gt["nu"]), float(gs_th[6]), float(gs_th[7])
    out["llr_skew"] = float(2 * (gs_ll - gt.loglik))        # LR test of lam = 0 within the t family
    # check available today: does the t intercept move away from the Gaussian-QMLE intercept?
    out["dc_se"] = float((gt["const"] - gn["const"]) / gn.se[0])
    out["skew_z"] = float(stats.skew(gn.std_resid))

    # 1-step forecasts: (location, scale, quantile function of the standardised shock)
    def onestep(c, phi, g, om, al, be, e_last, s2_last):
        m = g * x[T] + c + phi * (y[-1] - g * x[T - 1])
        return m, np.sqrt(om + al * e_last ** 2 + be * s2_last)

    zr_gn = gn.std_resid
    nu_t, nu_s, lam_s = out["nu_t"], out["nu_s"], out["lam_s"]
    fc = {
        "SM normal": (smp["intercept"] + smp["x1"] * x[T] + smp["ar.L1"] * (y[-1] - smp["x1"] * x[T - 1]),
                      np.sqrt(smp["sigma2"]), lambda a: stats.norm.ppf(a)),
        "GN normal": (*onestep(gn["const"], gn["phi1"], gn["gamma1"], gn["omega"], gn["alpha"], gn["beta"],
                               gn.eps[-1], gn.sig2[-1]), lambda a: stats.norm.ppf(a)),
        "GN + empirical z (FHS)": (*onestep(gn["const"], gn["phi1"], gn["gamma1"], gn["omega"], gn["alpha"],
                                            gn["beta"], gn.eps[-1], gn.sig2[-1]), lambda a: np.quantile(zr_gn, a)),
        "GT t": (*onestep(gt["const"], gt["phi1"], gt["gamma1"], gt["omega"], gt["alpha"], gt["beta"],
                          gt.eps[-1], gt.sig2[-1]), lambda a: stats.t.ppf(a, nu_t) * np.sqrt((nu_t - 2) / nu_t)),
        "GS skew-t": (*onestep(*gs_th[:6], gs_eps[-1], gs_sig2[-1]), lambda a: common.skewt_ppf(a, nu_s, lam_s)),
    }
    for nm, (m, s, qf) in fc.items():
        out["loc_err"][nm] = float((m - m_true) / s_true)
        row = []
        for a in TAILS:
            L, U = m + s * qf(a), m + s * qf(1 - a)
            row += [float(common.skewt_cdf((L - m_true) / s_true, NU, lam)),
                    float(1 - common.skewt_cdf((U - m_true) / s_true, NU, lam))]
        out["tails"][nm] = row                                # [P(below q01), P(above q99), P(below q05), P(above q95)]
    return out


def pseudo_true_t_location(lam, n=1_000_000, seed=7):
    """Location-scale t fitted by ML to i.i.d. standardised skew-t draws: where symmetric-t MLE puts the
    'centre' of a skewed distribution (in units of its standard deviation)."""
    z = common.draw_z(np.random.default_rng(seed), n, "skewt", NU, lam)
    nll = lambda th: -np.sum(stats.t.logpdf(z, th[2], loc=th[0], scale=th[1])) if th[1] > 0 and th[2] > 2 else 1e20
    r = optimize.minimize(nll, [0.0, 0.8, 6.0], method="Nelder-Mead", options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 4000})
    return dict(loc=float(r.x[0]), scale=float(r.x[1]), nu=float(r.x[2]), median=float(np.median(z)),
                mode=float(optimize.minimize_scalar(lambda v: -common.skewt_logpdf(v, NU, lam), bounds=(-2, 2),
                                                    method="bounded").x))


def main():
    designs = [(2000, 0.0), (2000, -0.1), (2000, -0.25), (2000, -0.5), (500, -0.25)]
    R = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    truth = np.array([TRUE["c"], TRUE["phi"], TRUE["g"], TRUE["c"] / (1 - TRUE["phi"]),
                      TRUE["omega"], TRUE["alpha"], TRUE["beta"]])
    results = {}
    with Pool(12) as pool:
        for T, lam in designs:
            rows = pool.map(one, [(T, lam, int(1e6 * (T == 500) + 1e4 * abs(lam) * 100 + r)) for r in range(R)])
            mo = common.skewt_moments(NU, lam)
            pt = pseudo_true_t_location(lam)
            key = f"T={T} lam={lam:+.2f}"
            res = dict(R=R, skew=mo["skew"], kurt=mo["kurt"], pseudo_true_t=pt,
                       nu_t_median=float(np.median([r["nu_t"] for r in rows])),
                       lam_s_mean=float(np.mean([r["lam_s"] for r in rows])),
                       nu_s_median=float(np.median([r["nu_s"] for r in rows])),
                       lr_skew_reject_5pct=float(np.mean([r["llr_skew"] > 3.84 for r in rows])),
                       not_converged={nm: float(np.mean([not r["conv"][nm] for r in rows])) for nm in ("GN", "GT")},
                       check_t_vs_normal_intercept=dict(
                           mean_dc_over_se=float(np.mean([r["dc_se"] for r in rows])),
                           share_dc_over_2se=float(np.mean([r["dc_se"] > 2 for r in rows])),
                           share_abs_dc_over_2se=float(np.mean([abs(r["dc_se"]) > 2 for r in rows])),
                           sd_dc_over_se=float(np.std([r["dc_se"] for r in rows])),
                           share_abs_dc_over_0_5se=float(np.mean([abs(r["dc_se"]) > 0.5 for r in rows])),
                           share_abs_dc_over_1se=float(np.mean([abs(r["dc_se"]) > 1 for r in rows])),
                           mean_skew_std_resid=float(np.mean([r["skew_z"] for r in rows])),
                           sd_skew_std_resid=float(np.std([r["skew_z"] for r in rows]))),
                       params={}, coverage95={}, loc_err={}, tails={})
            for nm in ("SM", "GN", "GT", "GS"):
                est = np.array([r["est"][nm] for r in rows])
                s = common.summarise(est, truth)
                res["params"][nm] = {p: dict(bias=round(float(s["bias"][j]), 4), sd=round(float(s["sd"][j]), 4),
                                             rmse=round(float(s["rmse"][j]), 4), bias_over_sd=round(float(s["bias_sd"][j]), 2))
                                     for j, p in enumerate(PNAMES) if np.isfinite(s["bias"][j])}
            for nm in ("GN", "GT"):
                res["coverage95"][nm] = dict(zip(["c", "phi", "g"], np.mean([r["cover"][nm] for r in rows], 0).round(3).tolist()))
            for nm in rows[0]["loc_err"]:
                le = np.array([r["loc_err"][nm] for r in rows])
                res["loc_err"][nm] = dict(mean=round(float(le.mean()), 4), rmse=round(float(np.sqrt(np.mean(le ** 2))), 4))
                res["tails"][nm] = np.mean([r["tails"][nm] for r in rows], 0).round(4).tolist()
            results[key] = res
            print(f"\n=== {key}  (z skew {mo['skew']:+.2f}, kurt {mo['kurt']:.1f}; R={R})", flush=True)
            print(f"  pseudo-true symmetric-t centre of z: {pt['loc']:+.3f} sd (median {pt['median']:+.3f}, mode {pt['mode']:+.3f})")
            print(f"  GT nu median {res['nu_t_median']:.2f} | GS lam mean {res['lam_s_mean']:+.3f}, nu median "
                  f"{res['nu_s_median']:.2f} | LR(lam=0) rejects {res['lr_skew_reject_5pct']:.2f} | "
                  f"not converged GN {res['not_converged']['GN']:.2f} GT {res['not_converged']['GT']:.2f}")
            print(f"  {'':4}" + "".join(f"{p:>26}" for p in PNAMES))
            for nm in ("SM", "GN", "GT", "GS"):
                pr = res["params"][nm]
                print(f"  {nm:4}" + "".join(
                    (f"{pr[p]['bias']:+.4f} ({pr[p]['sd']:.4f}) " + f"{pr[p]['bias_over_sd']:+.1f}sd").rjust(26)
                    if p in pr else " " * 26 for p in PNAMES))
            print("  95% Wald coverage c/phi/g:", res["coverage95"])
            print("  check (c_t - c_normal) / se(c_normal):", {k: round(v, 3) for k, v in res["check_t_vs_normal_intercept"].items()})
            print(f"  1-step: {'model':<24}{'loc err/sigma':>14}{'rmse':>8}   P(y<q01) P(y>q99) P(y<q05) P(y>q95)")
            for nm in res["loc_err"]:
                t = res["tails"][nm]
                print(f"          {nm:<24}{res['loc_err'][nm]['mean']:>+14.4f}{res['loc_err'][nm]['rmse']:>8.4f}   "
                      + "  ".join(f"{v:.4f}" for v in t))
    with open(f"{common.OUT}/e3_skew.json", "w") as f:
        json.dump(results, f, indent=1)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "time":
        import time
        t0 = time.time()
        o = one((2000, -0.25, 1))
        print(f"one replication: {time.time() - t0:.1f}s")
        print({k: [round(v, 3) for v in o["est"][k]] for k in o["est"]}, o["lam_s"], o["nu_s"], o["nu_t"])
    else:
        main()
