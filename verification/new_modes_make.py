import os, sys, json, warnings, numpy as np
sys.path.insert(0, os.environ.get("JAG_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fixed")))
import joint_armax_garch as J
warnings.filterwarnings("ignore", category=RuntimeWarning)
C = os.environ.get("CMP_DIR", "cmp"); O = f"{C}/new_modes"; os.makedirs(O, exist_ok=True)
T, H = 3000, 10
Xall = np.column_stack([np.sin(2*np.pi*np.arange(T+H)/12), np.random.default_rng(99).normal(size=T+H)])
yR, _, _ = J.simulate(T, 0.2, [0.6], [0.3], 0.05, 0.08, 0.88, dist="t", nu=6.0, X=Xall[:T], gamma=[1.0, -0.5],
                      seed=21, xreg="regression")
yA = np.loadtxt(f"{C}/A_arma11_t_y.csv")
cases = {"R_reg_arma11_X_t": dict(y=yR, X=Xall[:T], Xf=Xall[T:], kw=dict(p=1, q=1, xreg="regression")),
         "N_nomean_arma11_t": dict(y=yA, X=None, Xf=None, kw=dict(p=1, q=1, fixed={"const": 0.0}))}
meta = {}
for nm, c in cases.items():
    r = J.fit_joint(c["y"], X=c["X"], dist="t", **c["kw"])
    fc = J.forecast(r, H, X_future=c["Xf"], n_sims=0)
    np.savetxt(f"{O}/{nm}_y.csv", r.y)
    if r.X is not None:
        np.savetxt(f"{O}/{nm}_X.csv", r.X, delimiter=","); np.savetxt(f"{O}/{nm}_Xf.csv", c["Xf"], delimiter=",")
    ll = J._loglik_obs(r.params, r.y, r.X, r.p, r.q, "t", r.init, r.xreg)
    np.savetxt(f"{O}/{nm}_script_filter.csv", np.column_stack([r.eps, np.sqrt(r.sig2), ll]), delimiter=",")
    meta[nm] = dict(p=r.p, q=r.q, k=0 if r.X is None else r.X.shape[1], names=r.names, params=r.params.tolist(),
                    se=np.nan_to_num(r.se, nan=-1).tolist(), loglik=r.loglik, fixed=r.fixed,
                    fc_mean=fc["mean"].tolist(), fc_sigma=np.sqrt(fc["variance_innovation"]).tolist())
    print(nm, "LL=%.4f" % r.loglik, np.round(r.params, 4))
json.dump(meta, open(f"{O}/script.json", "w"), indent=1)
