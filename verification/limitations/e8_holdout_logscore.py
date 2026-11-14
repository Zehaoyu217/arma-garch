"""E8 -- holdout log-likelihood vs CRPS: which out-of-sample score tells the right model apart?

The prequential one-step holdout log score is the ranking metric of microprediction's `skaters` (Cotton).
DGP: y = 0.5 x + u, u_t = 0.1 + 0.5 u_{t-1} + eps_t; GARCH none / typical (alpha .10, beta .75) /
persistent (alpha .08, beta .90), normal shocks, plus persistent GARCH with t(5) shocks; unconditional
Var(eps) = 1.  Train / holdout sizes 250 / 60, 500 / 120, 1000 / 250.  Models are fitted on the training
part only (SM = SARIMAX, GN = GARCH normal, GT = GARCH t), their parameters frozen, and every held-out
observation is scored one step ahead given everything before it (SM: SARIMAXResults.extend; GN / GT: the
engine's filter with fixed parameters):
  log score  log of the one-step predictive density (holdout_logscore)
  CRPS       exact for the same predictive distributions (normal closed form; location-scale t closed form)
Per series a Diebold-Mariano test (Newey-West) on the paired per-observation differences decides
"significantly better" at 5%; the in-sample LR test of GARCH (within the engine, training data) is the
reference.  Reported: how often each rule picks GN over SM (and SM over GN), and GT over GN.
"""
import json
import sys
import warnings
from multiprocessing import Pool

import numpy as np
from scipy import stats
from scipy.special import beta as B

import common
from common import _core
from sarimax_garch import SARIMAXGARCH, dm_test, holdout_logscore

DGPS = {"none": dict(omega=1.0, alpha=0.0, beta=0.0, z=dict(dist="normal")),
        "typical": dict(omega=0.15, alpha=0.10, beta=0.75, z=dict(dist="normal")),
        "persistent": dict(omega=0.02, alpha=0.08, beta=0.90, z=dict(dist="normal")),
        "persistent, t5 shocks": dict(omega=0.02, alpha=0.08, beta=0.90, z=dict(dist="t", nu=5.0))}
SIZES = ((250, 60), (500, 120), (1000, 250))


def crps_normal(y, m, s):
    z = (y - m) / s
    return s * (z * (2 * stats.norm.cdf(z) - 1) + 2 * stats.norm.pdf(z) - 1 / np.sqrt(np.pi))


def crps_t(y, m, s, nu):
    """CRPS of a location-scale Student t (nu > 1) at y (closed form, as in R scoringRules::crps_t)."""
    z = (y - m) / s
    return s * (z * (2 * stats.t.cdf(z, nu) - 1) + 2 * stats.t.pdf(z, nu) * (nu + z ** 2) / (nu - 1)
                - 2 * np.sqrt(nu) * B(0.5, nu - 0.5) / ((nu - 1) * B(0.5, nu / 2) ** 2))


def engine_holdout(y, X, res, dist, n_ho):
    """One-step log scores and CRPS of the held-out observations, parameters frozen at res."""
    full = SARIMAXGARCH(y, exog=X, order=(1, 0, 0), trend="c", garch=(1, 1), dist=dist).fit(
        fixed_params=dict(res.params))
    r = full._res
    ls = _core._loglik_obs(r.params, r.y, r.X, r.p, r.q, r.dist, r.init, r.xreg)[-n_ho:]
    yte = y[-n_ho:]
    m, s = yte - r.eps[-n_ho:], np.sqrt(r.sig2[-n_ho:])          # one-step mean and sd of each held-out y
    if dist == "t":
        nu = r["nu"]
        cr = crps_t(yte, m, s * np.sqrt((nu - 2) / nu), nu)
    else:
        cr = crps_normal(yte, m, s)
    return ls, cr


def one(args):
    dgp, n_tr, n_ho, seed = args
    g = DGPS[dgp]
    T = n_tr + n_ho
    rng = np.random.default_rng(seed)
    x = common.ar1_exog(T, 0.7, rng)
    z = common.draw_z(rng, T + 500, **g["z"])
    y, *_ = common.simulate(T, 0.1, 0.5, g["omega"], g["alpha"], g["beta"], z, X=x[:, None], g=[0.5])
    X = x[:, None]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        np.seterr(all="ignore")
        sm = SARIMAXGARCH(y[:n_tr], exog=X[:n_tr], order=(1, 0, 0), trend="c").fit()
        ext = sm.extend(y[n_tr:], exog=X[n_tr:])
        ls_sm = np.asarray(ext.llf_obs)
        cr_sm = crps_normal(y[n_tr:], np.asarray(ext.forecasts)[0], np.sqrt(np.asarray(ext.forecasts_error_cov)[0, 0]))
        mod = SARIMAXGARCH(y[:n_tr], exog=X[:n_tr], order=(1, 0, 0), trend="c", garch=(1, 1))
        e0 = mod.fit(fixed_params={"alpha.L1": 0.0, "beta.L1": 0.0})
        gn = mod.fit()
        gt = SARIMAXGARCH(y[:n_tr], exog=X[:n_tr], order=(1, 0, 0), trend="c", garch=(1, 1), dist="t").fit()
        ls_gn, cr_gn = engine_holdout(y, X, gn, "normal", n_ho)
        ls_gt, cr_gt = engine_holdout(y, X, gt, "t", n_ho)
    out = dict(lr=bool(2 * (gn.llf - e0.llf) > 5.99), d_ls=float(np.mean(ls_gn - ls_sm)),
               r_crps=float(np.mean(cr_gn) / np.mean(cr_sm)), d_ls_t=float(np.mean(ls_gt - ls_gn)),
               r_crps_t=float(np.mean(cr_gt) / np.mean(cr_gn)))
    for key, dd in (("ls", ls_gn - ls_sm), ("crps", cr_sm - cr_gn), ("ls_t", ls_gt - ls_gn), ("crps_t", cr_gn - cr_gt)):
        st, pv = dm_test(dd)                     # + = the GARCH (or t) model better; HLN-corrected, t(n-1)
        out[f"{key}_win"], out[f"{key}_loss"] = bool(pv < 0.05 and st > 0), bool(pv < 0.05 and st < 0)
        out[f"{key}_stat"] = float(st)
    if seed % 1000 == 0:                         # consistency: the package's holdout_logscore gives the same scores
        ref = holdout_logscore(y, exog=X, order=(1, 0, 0), garch=(1, 1), holdout=n_ho)
        out["check"] = float(np.max(np.abs(ref - ls_gn)) + np.max(np.abs(
            holdout_logscore(y, exog=X, order=(1, 0, 0), holdout=n_ho) - ls_sm)))
    return out


def check_crps_t():
    rng = np.random.default_rng(0)
    nu, m, s, y = 5.0, 0.3, 1.7, 1.1
    draws = m + s * rng.standard_t(nu, 400_000)
    sample = float(np.mean(np.abs(draws - y)) - 0.5 * np.mean(np.abs(draws[:200_000] - draws[200_000:])))
    print(f"crps_t closed form {crps_t(y, m, s, nu):.4f} vs sample {sample:.4f}")


def main():
    R = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    check_crps_t()
    res, dm_stats = {}, {}
    with Pool(12) as pool:
        for i, dgp in enumerate(DGPS):
            for j, (n_tr, n_ho) in enumerate(SIZES):
                rows = pool.map(one, [(dgp, n_tr, n_ho, 900_000 + 100_000 * i + 10_000 * j + r) for r in range(R)])
                f = lambda k: float(np.mean([r[k] for r in rows]))
                key = f"{dgp} | train {n_tr} / holdout {n_ho}"
                dm_stats[key] = {s: [r[f"{s}_stat"] for r in rows] for s in ("ls", "crps", "ls_t", "crps_t")}
                res[key] = dict(R=R, lr_in_sample=f("lr"), d_logscore=f("d_ls"), crps_ratio=f("r_crps"),
                                ls_win=f("ls_win"), ls_loss=f("ls_loss"), crps_win=f("crps_win"), crps_loss=f("crps_loss"),
                                d_logscore_t=f("d_ls_t"), crps_ratio_t=f("r_crps_t"), ls_t_win=f("ls_t_win"),
                                ls_t_loss=f("ls_t_loss"), crps_t_win=f("crps_t_win"), crps_t_loss=f("crps_t_loss"),
                                max_check=max([r.get("check", 0.0) for r in rows]))
                s = res[key]
                print(f"{key:42} | LR in-sample {s['lr_in_sample']:.2f} | GN vs SM: dLS {s['d_logscore']:+.4f} "
                      f"CRPS ratio {s['crps_ratio']:.4f} | DM picks GN: logscore {s['ls_win']:.2f} (SM {s['ls_loss']:.2f}) "
                      f"CRPS {s['crps_win']:.2f} (SM {s['crps_loss']:.2f}) | GT vs GN: logscore {s['ls_t_win']:.2f} "
                      f"(GN {s['ls_t_loss']:.2f}) CRPS {s['crps_t_win']:.2f} (GN {s['crps_t_loss']:.2f})", flush=True)
    with open(f"{common.OUT}/e8_holdout_logscore.json", "w") as fh:
        json.dump(res, fh, indent=1)
    with open(f"{common.OUT}/e8_dm_stats.json", "w") as fh:      # per-series DM statistics, read by E9
        json.dump(dm_stats, fh)


if __name__ == "__main__":
    main()
