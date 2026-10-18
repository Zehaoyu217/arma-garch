"""Simulate/load cases, fit with joint_armax_garch.py, export data + script results for R."""
import sys, json, warnings, numpy as np
import os; sys.path.insert(0, os.environ.get("JAG_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))
import joint_armax_garch as J
import inspect
ARMAX = {"xreg": "armax"} if "xreg" in inspect.signature(J.simulate).parameters else {}   # v2.1 simulate is ARMAX
warnings.filterwarnings("ignore", category=RuntimeWarning)
OUT = os.environ.get("CMP_DIR", "cmp"); H = 10
T = 3000
tt = np.arange(T + H)
Xall = np.column_stack([np.sin(2*np.pi*tt/12), np.random.default_rng(99).normal(size=T + H)])

cases = {}
y, _, _ = J.simulate(T, 0.2, [0.5], [0.3], 0.05, 0.08, 0.88, dist="t", nu=6.0, seed=11)
cases["A_arma11_t"] = dict(y=y, X=None, p=1, q=1, d=0, dist="t")
y, _, _ = J.simulate(T, 0.1, [0.6], [-0.3], 0.1, 0.10, 0.85, dist="normal", seed=12)
cases["B_arma11_norm"] = dict(y=y, X=None, p=1, q=1, d=0, dist="normal")
y, _, _ = J.simulate(T, 0.1, [0.5, -0.2], [0.3], 0.05, 0.07, 0.90, dist="t", nu=8.0, seed=13)
cases["C_arma21_t"] = dict(y=y, X=None, p=2, q=1, d=0, dist="t")
y, _, _ = J.simulate(T, 0.2, [], [0.4], 0.05, 0.08, 0.88, dist="t", nu=6.0, X=Xall[:T], gamma=[1.0, -0.5], seed=14)
cases["D_ma1_X_t"] = dict(y=y, X=Xall[:T], Xf=Xall[T:], p=0, q=1, d=0, dist="t")
y, _, _ = J.simulate(T, 0.2, [0.5], [0.3], 0.05, 0.08, 0.88, dist="t", nu=6.0, X=Xall[:T], gamma=[1.0, -0.5], seed=15, **ARMAX)
cases["E_arma11_X_t"] = dict(y=y, X=Xall[:T], Xf=Xall[T:], p=1, q=1, d=0, dist="t")   # ARMAX data, fitted with the default (regression) model
w, _, _ = J.simulate(T, 0.0, [0.4], [0.35], 0.05, 0.07, 0.90, dist="t", nu=7.0, seed=5)
cases["F_arima111_t_levels"] = dict(y=100.0 + np.cumsum(w), X=None, p=1, q=1, d=1, dist="t")
dem = np.loadtxt("dem2gbp_benchmark.csv")
cases["G_dem2gbp_arma11_t"] = dict(y=dem, X=None, p=1, q=1, d=0, dist="t")

meta = {}
for name, c in cases.items():
    r = J.fit_joint(c["y"], X=c["X"], p=c["p"], q=c["q"], d=c["d"], dist=c["dist"])
    ll_obs = J._loglik_obs(r.params, r.y, r.X, r.p, r.q, r.dist, r.init)
    fc = J.forecast(r, H, X_future=c.get("Xf"), n_sims=0)
    np.savetxt(f"{OUT}/{name}_y.csv", r.y)                       # the (differenced) series actually fitted
    if r.X is not None:
        np.savetxt(f"{OUT}/{name}_X.csv", r.X, delimiter=",")
        np.savetxt(f"{OUT}/{name}_Xf.csv", c["Xf"], delimiter=",")
    np.savetxt(f"{OUT}/{name}_script_filter.csv", np.column_stack([r.eps, np.sqrt(r.sig2), ll_obs]), delimiter=",")
    meta[name] = dict(p=c["p"], q=c["q"], d=c["d"], dist=c["dist"], k=0 if r.X is None else r.X.shape[1],
                      names=r.names, params=r.params.tolist(), se=r.se.tolist(), se_hess=r.se_hessian.tolist(),
                      loglik=r.loglik, nobs=r.nobs, converged=r.converged,
                      fc_mean_diff=(fc["mean_diff"] if r.d else fc["mean"]).tolist(),
                      fc_sigma=np.sqrt(fc["variance_innovation"]).tolist())
    print(f"{name:22s} loglik={r.loglik:10.3f} converged={r.converged} params={np.round(r.params, 4)}")
json.dump(meta, open(f"{OUT}/script_results.json", "w"), indent=1)
