"""ARIMA part vs statsmodels SARIMAX (regression with ARIMA errors), 7 orders x d in {0,1,2} x {no X, X}.

GARCH is switched off (alpha = beta = 0 fixed, normal errors) so SARIMAX is literally the same model:
  (a) residuals and per-obs loglik at identical parameters      (Kalman filter vs conditional recursion)
  (b) level forecasts, mean and variance, h = 1..10            (SARIMAX integrates inside the state space)
  (c) free fits                                                 (exact MLE vs conditional MLE)
Then, with GARCH on: (d) forecast mean/variance vs simulation for d = 0, 1, 2, and (e) a fit with d on
levels is identical to a fit with d = 0 on the differenced data."""
import json
import os
import warnings

import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

from common import J, ORDERS, OUT, H, make_data

warnings.filterwarnings("ignore")
T, BURN, NOG = 1500, 100, {"alpha": 0.0, "beta": 0.0}
rows, j = [], 0
print(f"{'case':24s} {'(a) resid':>10s} {'ll_t':>9s} | {'(b) mean':>9s} {'var rel':>9s} | {'(c) max|d|/SE':>13s}")
for order in ORDERS:
    p, q = order
    for d in (0, 1, 2):
        for use_x in (False, True):
            name = f"arima{p}{d}{q}_{'X' if use_x else 'noX'}"
            y, X, Xf = make_data(T, order, d, (1.0, 0.0, 0.0), "normal", 6.0, use_x, seed=5000 + j)
            j += 1
            r = J.fit_joint(y, X=X, p=p, q=q, d=d, dist="normal", fixed=NOG)
            k = 0 if X is None else 2
            val = {"intercept": r["const"], "sigma2": r["omega"]}
            val.update({f"ar.L{i + 1}": r[f"phi{i + 1}"] for i in range(p)})
            val.update({f"ma.L{i + 1}": r[f"theta{i + 1}"] for i in range(q)})
            val.update({f"x{i + 1}": r[f"gamma{i + 1}"] for i in range(k)})
            # (a) identical parameters, SARIMAX on the differenced data (same indexing as the script)
            m_sd = SARIMAX(y, exog=X, order=(p, d, q), trend="c", simple_differencing=True)
            par = np.array([val[n] for n in m_sd.param_names])
            err = m_sd.filter(par).forecasts_error[0][p:]
            a_res = np.abs(err - r.eps)[BURN:].max()
            lls = J._loglik_obs(r.params, r.y, r.X, p, q, "normal", r.init, r.xreg)
            a_ll = np.abs(m_sd.loglikeobs(par)[p:] - lls)[BURN:].max()
            # (b) level forecasts at identical parameters
            m_lv = SARIMAX(y, exog=X, order=(p, d, q), trend="c")
            f_sm = m_lv.filter(par).get_forecast(H, exog=Xf)
            f_j = J.forecast(r, H, X_future=Xf, n_sims=0)
            b_mean = np.abs(f_sm.predicted_mean - f_j["mean"]).max()
            b_var = np.abs(f_sm.var_pred_mean / f_j["variance"] - 1).max()
            # (c) free fits
            fit = m_lv.fit(disp=False)
            est = dict(zip(fit.model.param_names, fit.params))
            bse = dict(zip(fit.model.param_names, fit.bse))
            c_dse = max(abs(val[n] - est[n]) / bse[n] for n in fit.model.param_names)
            rows.append(dict(name=name, a_res=a_res, a_ll=a_ll, b_mean=b_mean, b_var=b_var, c_dse=c_dse))
            print(f"{name:24s} {a_res:10.1e} {a_ll:9.1e} | {b_mean:9.1e} {b_var:9.1e} | {c_dse:13.3f}")

print(f"\nSUMMARY over {len(rows)} ARIMA cases (GARCH off)")
for key, lab in (("a_res", "(a) residuals at identical params, max |diff|"),
                 ("a_ll", "(a) per-obs loglik at identical params, max |diff|"),
                 ("b_mean", "(b) level forecast mean, max |diff|"),
                 ("b_var", "(b) level forecast variance, max relative diff"),
                 ("c_dse", "(c) free fit, max |script - SARIMAX| / SE")):
    print(f"  {lab:52s} {max(r[key] for r in rows):.2e}")

print("\n(d) GARCH on (typical, t6, X): analytic forecast vs 200,000 simulated paths, h = 1..10")
zs = []
for i, order in enumerate(((1, 1), (2, 2), (0, 1))):
    for d in (0, 1, 2):
        y, X, Xf = make_data(3000, order, d, "typical", "t", 6.0, True, seed=7000 + 10 * d + order[0])
        r = J.fit_joint(y, X=X, p=order[0], q=order[1], d=d)
        fc = J.forecast(r, H, X_future=Xf, n_sims=200_000, seed=10 * i + d + 1)
        mc_se = np.sqrt(fc["variance"] / 200_000)
        zmean = np.abs(fc["mean"] - fc["sims"].mean(0)) / mc_se
        vrel = np.abs(fc["variance"] / fc["sims"].var(0) - 1)
        zs.append((fc["mean"][0] - fc["sims"].mean(0)[0]) / mc_se[0])
        print(f"  ARIMA({order[0]},{d},{order[1]}): max |mean - sim| = {zmean.max():.2f} MC SE   "
              f"max |var/sim var - 1| = {vrel.max():.3f}   var grows with h: {bool(np.all(np.diff(fc['variance']) > 0)) if d else 'n/a'}")

print(f"  h=1 z-scores across the 9 cases: mean {np.mean(zs):+.2f}, sd {np.std(zs):.2f}   (unbiased => ~N(0,1))")

print("\n(e) fit with d on levels == fit with d=0 on the differenced data (all 7 orders, d = 1, 2)")
worst = 0.0
for order in ORDERS:
    for d in (1, 2):
        y, X, Xf = make_data(1500, order, d, "typical", "t", 6.0, True, seed=9000 + d)
        a = J.fit_joint(y, X=X, p=order[0], q=order[1], d=d, compute_cov=False)
        b = J.fit_joint(np.diff(y, n=d), X=np.diff(X, n=d, axis=0), p=order[0], q=order[1], d=0, compute_cov=False)
        worst = max(worst, np.abs(a.params - b.params).max())
print(f"  max |params(d on levels) - params(d=0 on differences)| = {worst:.1e}")
json.dump(rows, open(os.path.join(OUT, "statsmodels_summary.json"), "w"), indent=1)
