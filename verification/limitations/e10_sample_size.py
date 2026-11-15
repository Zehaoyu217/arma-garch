"""E10 -- how many observations does GARCH need on top of a fitted SARIMAX, one-stage vs two-stage, and are the
p-values sized right under their nulls?

Families (monthly-style, normal shocks, unconditional Var(eps) = 1):
  ARIMA(1,1,1)        dy_t = 0.05 + 0.5 dy_{t-1} + eps_t + 0.3 eps_{t-1}                         (trend "c" = drift)
  reg + ARIMA(1,1,1)  y_t = 0.5 x_t + u_t, du_t = 0.05 + 0.5 du_{t-1} + eps_t + 0.3 eps_{t-1}, x a random walk
  airline             (1-L)(1-L^12) y_t = (1 - 0.4 L)(1 - 0.6 L^12) eps_t      (SARIMA(0,1,1)(0,1,1)12, no constant)
GARCH: none | typical monthly (alpha .10, beta .75) | persistent (alpha .08, beta .90).  T = 60, 120, 240, 360, 600.
Fits on every series:
  SM  statsmodels SARIMAX (for airline with seasonal_order (0,1,1,12)); its residuals after the burn-in
      (loglikelihood_burn), standardised by the Kalman forecast-error variance and rescaled by sigma2
  OS  one stage: the engine's joint ARIMA-GARCH MLE (not available for airline: no seasonal ARMA with GARCH)
  TS  two stage: zero-mean GARCH(1,1) fitted by the engine to SM's residuals
Scores:
  detection  ARCH-LM(12) on SM's residuals; LR GARCH vs constant variance within OS and within TS (chi2(2) 5.99)
  identification  share of GARCH fits with alpha or beta on the boundary; RMSE of the persistence alpha + beta
  value      one-step variance forecast, QLIKE loss against the true sigma^2_{T+1}: GARCH (OS, TS) / constant (SM)
  p-values   size of the 5% Wald test at the TRUE mean parameters: SM opg, SM robust, OS robust
"""
import json
import sys
import warnings
from multiprocessing import Pool

import numpy as np

import common
from common import _core
from sarimax_garch import SARIMAXGARCH, arch_lm_test

FAMS = {"ARIMA(1,1,1)": dict(order=(1, 1, 1), seasonal=(0, 0, 0, 0), trend="c", x=False,
                             truth={"intercept": 0.05, "ar.L1": 0.5, "ma.L1": 0.3}),
        "reg + ARIMA(1,1,1)": dict(order=(1, 1, 1), seasonal=(0, 0, 0, 0), trend="c", x=True,
                                   truth={"intercept": 0.05, "x1": 0.5, "ar.L1": 0.5, "ma.L1": 0.3}),
        "airline": dict(order=(0, 1, 1), seasonal=(0, 1, 1, 12), trend=None, x=False,
                        truth={"ma.L1": -0.4, "ma.S.L12": -0.6})}
GARCHS = {"none": (1.0, 0.0, 0.0), "typical": (0.15, 0.10, 0.75), "persistent": (0.02, 0.08, 0.90)}
TS_ = (60, 120, 240, 360, 600)
BURN = 600


def simulate(fam, T, g, rng):
    om, al, be = g
    n = T + BURN
    z = rng.standard_normal(n)
    eps, s2 = np.zeros(n), np.zeros(n)
    s2[0] = om / (1 - al - be)
    eps[0] = np.sqrt(s2[0]) * z[0]
    for t in range(1, n):
        s2[t] = om + al * eps[t - 1] ** 2 + be * s2[t - 1]
        eps[t] = np.sqrt(s2[t]) * z[t]
    s2_next = om + al * eps[-1] ** 2 + be * s2[-1]               # the true sigma^2_{T+1}
    x = None
    if fam == "airline":
        w = eps.copy()
        w[1:] += -0.4 * eps[:-1]
        w[12:] += -0.6 * eps[:-12]
        w[13:] += 0.24 * eps[:-13]
        zz = np.zeros(n)
        for t in range(n):                                       # seasonal integration (1 - L^12)^-1
            zz[t] = w[t] + (zz[t - 12] if t >= 12 else 0.0)
        y = np.cumsum(zz)
    else:
        du = np.zeros(n)
        for t in range(1, n):
            du[t] = 0.05 + 0.5 * du[t - 1] + eps[t] + 0.3 * eps[t - 1]
        y = np.cumsum(du)
        if fam == "reg + ARIMA(1,1,1)":
            x = np.cumsum(rng.standard_normal(n))
            y = y + 0.5 * x
    return y[BURN:], (None if x is None else x[BURN:]), s2_next


def qlike(true, fc):
    r = true / fc
    return float(r - np.log(r) - 1)


def one(args):
    fam, gk, T, seed = args
    f = FAMS[fam]
    rng = np.random.default_rng(seed)
    y, x, s2n = simulate(fam, T, GARCHS[gk], rng)
    out = dict(det={}, bnd={}, pers={}, ql={}, cover={})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        np.seterr(all="ignore")
        sm = SARIMAXGARCH(y, exog=x, order=f["order"], seasonal_order=f["seasonal"], trend=f["trend"]).fit()
        burn = int(sm.loglikelihood_burn)
        r = np.asarray(sm.standardized_forecasts_error)[0, burn:] * np.sqrt(sm.params["sigma2"])
        r = r[np.isfinite(r)]
        out["det"]["ARCH-LM(12)"] = arch_lm_test(r, 12)[1] < 0.05
        out["ql"]["const"] = qlike(s2n, float(sm.params["sigma2"]))
        # two stage: zero-mean GARCH(1,1) on SARIMAX's residuals
        ts = _core.fit_joint(r, p=0, q=0, dist="normal", fixed={"const": 0.0})
        ts0 = _core.fit_joint(r, p=0, q=0, dist="normal", fixed={"const": 0.0, "alpha": 0.0, "beta": 0.0},
                              compute_cov=False)
        out["det"]["LR two-stage"] = 2 * (ts.loglik - ts0.loglik) > 5.99
        out["bnd"]["TS"] = bool(ts.at_boundary)
        out["pers"]["TS"] = float(ts.persistence)
        out["ql"]["TS"] = qlike(s2n, float(_core.forecast(ts, 1, n_sims=0)["variance_innovation"][0]))
        names = list(f["truth"])
        idx = [list(sm.model.param_names).index(nm) for nm in names]
        p_sm = np.asarray(sm.params)[idx]
        tv = np.array([f["truth"][nm] for nm in names])
        for lab, cov in (("SM opg", sm.cov_params_opg), ("SM robust", sm.cov_params_robust)):
            se = np.sqrt(np.diag(np.asarray(cov)))[idx]
            out["cover"][lab] = (np.abs(p_sm - tv) <= 1.96 * se).tolist()
        if fam != "airline":                                     # one stage: joint ARIMA-GARCH
            mod = SARIMAXGARCH(y, exog=x, order=f["order"], trend=f["trend"], garch=(1, 1))
            os0 = mod.fit(fixed_params={"alpha.L1": 0.0, "beta.L1": 0.0})
            os_ = mod.fit()
            out["det"]["LR one-stage"] = 2 * (os_.llf - os0.llf) > 5.99
            out["bnd"]["OS"] = bool(os_._res.at_boundary)
            out["pers"]["OS"] = float(os_.persistence)
            out["ql"]["OS"] = qlike(s2n, float(os_.get_forecast(1, exog=None if x is None else x[-1:] + 0.0,
                                                                 n_sims=0).conditional_variance.iloc[0]))
            p_os, se_os = os_.params[names].values, os_.bse[names].values
            out["cover"]["OS robust"] = (np.abs(p_os - tv) <= 1.96 * se_os).tolist()
    return out


def summarise(rows, fam, gk):
    names = list(FAMS[fam]["truth"])
    s = dict(R=len(rows), detection={}, boundary={}, persistence_rmse={}, qlike_ratio={}, garch_wins={}, wald_size={})
    for k in rows[0]["det"]:
        s["detection"][k] = round(float(np.mean([r["det"][k] for r in rows])), 3)
    for k in rows[0]["bnd"]:
        s["boundary"][k] = round(float(np.mean([r["bnd"][k] for r in rows])), 3)
        if gk != "none":
            om, al, be = GARCHS[gk]
            s["persistence_rmse"][k] = round(float(np.sqrt(np.mean([(r["pers"][k] - al - be) ** 2 for r in rows]))), 3)
    qc = np.mean([r["ql"]["const"] for r in rows])
    for k in ("OS", "TS"):
        if k in rows[0]["ql"]:
            s["qlike_ratio"][k] = round(float(np.mean([r["ql"][k] for r in rows]) / qc), 3)
            s["garch_wins"][k] = round(float(np.mean([r["ql"][k] < r["ql"]["const"] for r in rows])), 3)
    for k in rows[0]["cover"]:
        s["wald_size"][k] = dict(zip(names, (1 - np.mean([r["cover"][k] for r in rows], 0)).round(3).tolist()))
    return s


def main():
    R = int(sys.argv[1]) if len(sys.argv) > 1 else 250
    res = {}
    with Pool(12) as pool:
        for i, fam in enumerate(FAMS):
            for j, gk in enumerate(GARCHS):
                for k, T in enumerate(TS_):
                    seeds = [2_000_000 + 100_000 * i + 10_000 * j + 1000 * k + r for r in range(R)]
                    rows = pool.map(one, [(fam, gk, T, sd) for sd in seeds])
                    key = f"{fam} | GARCH {gk} | T={T}"
                    res[key] = summarise(rows, fam, gk)
                    s = res[key]
                    print(f"{key:40} det {s['detection']} | bnd {s['boundary']} | QLIKE GARCH/const {s['qlike_ratio']} "
                          f"(wins {s['garch_wins']}) | pers RMSE {s['persistence_rmse']} | Wald size {s['wald_size']}",
                          flush=True)
    with open(f"{common.OUT}/e10_sample_size.json", "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "time":
        import time
        for fam in FAMS:
            t0 = time.time()
            o = one((fam, "persistent", 240, 1))
            print(f"{fam}: {time.time() - t0:.2f}s", o["det"], o["bnd"], {k: round(v, 3) for k, v in o["ql"].items()})
    else:
        main()
