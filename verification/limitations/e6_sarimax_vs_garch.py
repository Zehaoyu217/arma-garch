"""E6 -- SARIMAX vs SARIMAX-GARCH, both ways round: where GARCH gains an edge, and what it costs when the
data have no GARCH (including MA-GARCH).

DGP: y_t = 0.5 x_t + u_t,  u_t = 0.1 + phi u_{t-1} + eps_t + theta eps_{t-1},  eps_t = sigma_t z_t,
     z ~ N(0, 1), x exogenous AR(1) (rho 0.7, future path known), unconditional Var(eps) = 1 in every design.
     mean:  AR1 (phi .5) | MA1 (theta .5) | MA1 theta .9 (near non-invertible, stress)
     GARCH: none (omega 1) | typical monthly (alpha .10, beta .75) | persistent (alpha .08, beta .90)
     T:     120 | 250 | 500 | 1000
Fits on every series (all through the public wrapper):
  SM  SARIMAX (garch=None)                                -- statsmodels, exact likelihood
  E0  garch=(1,1) with alpha = beta = 0 fixed             -- the engine's constant-variance null
  GN  garch=(1,1), dist="normal"                          -- the engine's GARCH fit
Scores:
  mean parameters   bias, RMSE and 95% CI coverage (SM with statsmodels' default 'opg' SEs and with
                    cov_params_robust; GN with its default robust sandwich)
  forecasts         h = 1, 6, 12 with the realised x path: point error vs the true conditional mean, expected
                    CRPS against 200 true future draws, exact 90% coverage against 1,000 true future paths
                    (SM: its normal band; GN: simulated quantiles), and h = 1 coverage in calm vs turbulent
                    origins (bottom / top third of the true sigma^2_{T+1})
  detection         P(choose GARCH) by ARCH-LM(4) and ARCH-LM(12) on SM's residuals (5%), LR within the
                    engine (GN vs E0 > 5.99), AIC and BIC within the engine, and AIC across backends (GN vs SM)
  over-fit cost     GN on a boundary (alpha or beta < 1e-4), at the persistence cap, not invertible
"""
import json
import sys
import warnings
from multiprocessing import Pool

import numpy as np
from scipy import stats

import common
from sarimax_garch import SARIMAXGARCH, arch_lm_test

H, HS, M, MC, NSIM = 12, (1, 6, 12), 1000, 200, 2000
C, G = 0.1, 0.5
MEANS = {"AR1": dict(phi=0.5, theta=0.0), "MA1": dict(phi=0.0, theta=0.5), "MA1 theta=.9": dict(phi=0.0, theta=0.9)}
GARCHS = {"none": dict(omega=1.0, alpha=0.0, beta=0.0), "typical": dict(omega=0.15, alpha=0.10, beta=0.75),
          "persistent": dict(omega=0.02, alpha=0.08, beta=0.90)}
TS = (120, 250, 500, 1000)
DESIGNS = ([(m, g, T) for m in ("AR1", "MA1") for g in GARCHS for T in TS]
           + [("MA1 theta=.9", g, T) for g in ("none", "persistent") for T in (250, 1000)])
Z90 = stats.norm.ppf(0.95)


def crps_normal(mean, sd, x):
    z = (x - mean) / sd
    return sd * (z * (2 * stats.norm.cdf(z) - 1) + 2 * stats.norm.pdf(z) - 1 / np.sqrt(np.pi))


def crps_sample(sample, xs):
    """Mean CRPS of one predictive sample against several realisations xs."""
    s = np.sort(sample)
    n = len(s)
    spread = np.sum((2 * np.arange(1, n + 1) - n - 1) * s) / n ** 2
    return float(np.mean(np.abs(s[:, None] - xs[None, :]).mean(0)) - spread)


def one(args):
    mean_key, garch_key, T, seed = args
    mp, gp = MEANS[mean_key], GARCHS[garch_key]
    phi, theta = mp["phi"], mp["theta"]
    p, q = (1, 0) if phi else (0, 1)
    rng = np.random.default_rng(seed)
    x = common.ar1_exog(T + H, 0.7, rng)
    y, u, eps, sig2 = common.simulate(T, C, phi, gp["omega"], gp["alpha"], gp["beta"], rng.standard_normal(T + 500),
                                      X=x[:T, None], g=[G], theta=theta)
    X, Xf = x[:T, None], x[T:T + H, None]
    truth = common.true_future((u[-1], eps[-1], sig2[-1]), H, M, rng, C, phi, gp["omega"], gp["alpha"], gp["beta"],
                               dict(dist="normal"), xg_future=G * x[T:T + H], theta=theta)
    m = np.empty(H)
    m[0] = C + phi * u[-1] + theta * eps[-1]
    for h in range(1, H):
        m[h] = C + phi * m[h - 1]
    cond_mean = m + G * x[T:T + H]
    s2_next = gp["omega"] + gp["alpha"] * eps[-1] ** 2 + gp["beta"] * sig2[-1]
    arma = "ar.L1" if p else "ma.L1"
    tv = {"intercept": C, "x1": G, arma: phi if p else theta}
    out = dict(s2_next=float(s2_next), est={}, cover={}, fc={}, det={}, flags={})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sm = SARIMAXGARCH(y, exog=X, order=(p, 0, q), trend="c").fit()
        mod = SARIMAXGARCH(y, exog=X, order=(p, 0, q), trend="c", garch=(1, 1))
        e0 = mod.fit(fixed_params={"alpha.L1": 0.0, "beta.L1": 0.0})
        gn = mod.fit()
    names = list(tv)
    sm_p = np.asarray(sm.params)[[list(sm.model.param_names).index(n) for n in names]]
    idx = [list(sm.model.param_names).index(n) for n in names]
    se_opg = np.sqrt(np.diag(np.asarray(sm.cov_params_opg)))[idx]
    se_rob = np.sqrt(np.diag(np.asarray(sm.cov_params_robust)))[idx]
    gn_p, gn_se = gn.params[names].values, gn.bse[names].values
    t_arr = np.array([tv[n] for n in names])
    out["est"] = {"SM": sm_p.tolist(), "GN": gn_p.tolist()}
    out["cover"] = {"SM opg": (np.abs(sm_p - t_arr) <= 1.96 * se_opg).tolist(),
                    "SM robust": (np.abs(sm_p - t_arr) <= 1.96 * se_rob).tolist(),
                    "GN robust": (np.abs(gn_p - t_arr) <= 1.96 * gn_se).tolist()}
    # forecasts
    f_sm = sm.get_forecast(H, exog=Xf)
    sm_mean, sm_se = np.asarray(f_sm.predicted_mean), np.asarray(f_sm.se_mean)
    f_gn = gn.get_forecast(H, exog=Xf, n_sims=NSIM, seed=seed + 7)
    gn_mean, sims = f_gn.predicted_mean.values, f_gn.sims
    draws = truth[:MC]
    for h in HS:
        k = h - 1
        lo_s, hi_s = sm_mean[k] - Z90 * sm_se[k], sm_mean[k] + Z90 * sm_se[k]
        lo_g, hi_g = np.quantile(sims[:, k], [0.05, 0.95])
        out["fc"][f"h{h}"] = dict(
            err_sm=float(sm_mean[k] - cond_mean[k]), err_gn=float(gn_mean[k] - cond_mean[k]),
            crps_sm=float(crps_normal(sm_mean[k], sm_se[k], draws[:, k]).mean()),
            crps_gn=crps_sample(sims[:, k], draws[:, k]),
            cov_sm=float(np.mean((truth[:, k] >= lo_s) & (truth[:, k] <= hi_s))),
            cov_gn=float(np.mean((truth[:, k] >= lo_g) & (truth[:, k] <= hi_g))),
            sd_true=float(truth[:, k].std()))
    # detection
    r = np.asarray(sm.resid)
    p4, p12 = arch_lm_test(r, 4)[1], arch_lm_test(r, 12)[1]
    out["lm_nan"] = bool(not (np.isfinite(p4) and np.isfinite(p12)))
    out["det"] = {"ARCH-LM(4)": p4 < 0.05, "ARCH-LM(12)": p12 < 0.05,
                  "LR within engine": 2 * (gn.llf - e0.llf) > 5.99, "AIC within engine": gn.aic < e0.aic,
                  "BIC within engine": gn.bic < e0.bic, "AIC across backends": gn.aic < sm.aic}
    out["flags"] = {"boundary": bool(gn._res.at_boundary), "cap": bool(gn._res.at_persistence_cap),
                    "not_invertible": bool(not gn.invertible), "gn_not_converged": bool(not gn.converged),
                    "persistence": float(gn.persistence)}
    return out


def summarise(rows, mean_key):
    arma = "phi" if MEANS[mean_key]["phi"] else "theta"
    tv = np.array([C, G, MEANS[mean_key]["phi"] or MEANS[mean_key]["theta"]])
    pn = ["intercept", "x1", arma]
    res = dict(R=len(rows), params={}, coverage={}, forecast={}, detection={}, flags={})
    for nm in ("SM", "GN"):
        est = np.array([r["est"][nm] for r in rows])
        bias, sd = est.mean(0) - tv, est.std(0, ddof=1)
        res["params"][nm] = {p: dict(bias=round(float(bias[j]), 4), sd=round(float(sd[j]), 4),
                                     rmse=round(float(np.sqrt(bias[j] ** 2 + sd[j] ** 2)), 4)) for j, p in enumerate(pn)}
    for nm in ("SM opg", "SM robust", "GN robust"):
        res["coverage"][nm] = dict(zip(pn, np.mean([r["cover"][nm] for r in rows], 0).round(3).tolist()))
    s2 = np.array([r["s2_next"] for r in rows])
    lo_t, hi_t = np.quantile(s2, [1 / 3, 2 / 3])
    calm, turb = s2 <= lo_t, s2 >= hi_t
    for h in HS:
        f = [r["fc"][f"h{h}"] for r in rows]
        g = lambda key: np.array([v[key] for v in f])
        sdt = g("sd_true")
        d = dict(rmse_sm=float(np.sqrt(np.mean((g("err_sm") / sdt) ** 2))),
                 rmse_gn=float(np.sqrt(np.mean((g("err_gn") / sdt) ** 2))),
                 crps_ratio=float(g("crps_gn").mean() / g("crps_sm").mean()),
                 cov_sm=float(g("cov_sm").mean()), cov_gn=float(g("cov_gn").mean()))
        if h == 1:
            d.update(cov_sm_calm=float(g("cov_sm")[calm].mean()), cov_sm_turb=float(g("cov_sm")[turb].mean()),
                     cov_gn_calm=float(g("cov_gn")[calm].mean()), cov_gn_turb=float(g("cov_gn")[turb].mean()),
                     crps_ratio_calm=float(g("crps_gn")[calm].mean() / g("crps_sm")[calm].mean()),
                     crps_ratio_turb=float(g("crps_gn")[turb].mean() / g("crps_sm")[turb].mean()))
        res["forecast"][f"h{h}"] = {k: round(v, 4) for k, v in d.items()}
    for rule in rows[0]["det"]:
        res["detection"][rule] = round(float(np.mean([r["det"][rule] for r in rows])), 3)
    for fl in ("boundary", "cap", "not_invertible", "gn_not_converged"):
        res["flags"][fl] = round(float(np.mean([r["flags"][fl] for r in rows])), 3)
    res["flags"]["persistence_median"] = round(float(np.median([r["flags"]["persistence"] for r in rows])), 3)
    res["flags"]["arch_lm_nan"] = int(sum(r["lm_nan"] for r in rows))
    return res


def main():
    R = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    designs = DESIGNS if len(sys.argv) <= 2 else [d for d in DESIGNS if f"{d[0]}|{d[1]}|{d[2]}" in sys.argv[2:]]
    out = {}
    with Pool(12) as pool:
        for i, (mk, gk, T) in enumerate(designs):
            rows = pool.map(one, [(mk, gk, T, 600_000 + 10_000 * DESIGNS.index((mk, gk, T)) + r) for r in range(R)])
            key = f"{mk} | GARCH {gk} | T={T}"
            out[key] = summarise(rows, mk)
            s = out[key]
            f1, f12 = s["forecast"]["h1"], s["forecast"]["h12"]
            arma = "phi" if MEANS[mk]["phi"] else "theta"
            print(f"{key:38} | {arma} RMSE SM {s['params']['SM'][arma]['rmse']:.3f} GN {s['params']['GN'][arma]['rmse']:.3f}"
                  f" | x1 RMSE SM {s['params']['SM']['x1']['rmse']:.3f} GN {s['params']['GN']['x1']['rmse']:.3f}"
                  f" | cover x1 opg/rob/GN {s['coverage']['SM opg']['x1']:.2f}/{s['coverage']['SM robust']['x1']:.2f}/"
                  f"{s['coverage']['GN robust']['x1']:.2f}"
                  f" | CRPS GN/SM h1 {f1['crps_ratio']:.3f} h12 {f12['crps_ratio']:.3f}"
                  f" | cov90 h1 calm SM {f1['cov_sm_calm']:.2f} GN {f1['cov_gn_calm']:.2f} turb SM {f1['cov_sm_turb']:.2f}"
                  f" GN {f1['cov_gn_turb']:.2f} | pick GARCH: LM4 {s['detection']['ARCH-LM(4)']:.2f} LR {s['detection']['LR within engine']:.2f}"
                  f" AICw {s['detection']['AIC within engine']:.2f} BICw {s['detection']['BIC within engine']:.2f}"
                  f" AICx {s['detection']['AIC across backends']:.2f} | boundary {s['flags']['boundary']:.2f}", flush=True)
    path = f"{common.OUT}/e6_sarimax_vs_garch.json"
    if len(sys.argv) <= 2:
        with open(path, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
