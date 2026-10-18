"""GARCH part vs Python `arch` (Sheppard): AR(p)-GARCH(1,1), p in {0,1,2} x {normal, t} x 4 GARCH sets,
plus 2 ARMAX cases (arch's ARX with exog is ARMAX, so these use xreg="armax").
  (a) sigma_t and residuals at identical parameters   (b) free fits: estimates, robust and Hessian SEs"""
import json
import os
import warnings

import numpy as np
from arch import arch_model

from common import GARCH, J, OUT, make_X

warnings.filterwarnings("ignore")
T, BURN = 2000, 300
PHI = {0: [], 1: [0.5], 2: [0.5, -0.25]}
cases = [(p, dist, g, False) for p in (0, 1, 2) for dist in ("normal", "t") for g in list(GARCH)[:4]]
cases += [(1, "t", "typical", True), (1, "t", "high-alpha", True)]
rows = []
print(f"{'case':34s} {'(a) sigma':>9s} {'resid':>8s} | {'max|d|/SE':>9s} {'robSE ratio':>12s} {'hesSE ratio':>12s} {'LL diff':>8s}")
for j, (p, dist, g, use_x) in enumerate(cases):
    om, al, be = GARCH[g]
    X = make_X(T, 300 + j) if use_x else None
    y, _, _ = J.simulate(T, 0.1, PHI[p], [], om, al, be, dist=dist, nu=6.0, X=X,
                         gamma=[1.0, -0.5] if use_x else None, seed=3000 + j, xreg="armax")
    r = J.fit_joint(y, X=X, p=p, q=0, dist=dist, xreg="armax")
    mean = "ARX" if p else ("LS" if use_x else "Constant")
    am = arch_model(y, x=X, mean=mean, lags=p if p else None, vol="GARCH", p=1, q=1,
                    dist="normal" if dist == "normal" else "t", rescale=False)
    fit_r = am.fit(disp="off")                         # robust (Bollerslev-Wooldridge) SEs by default
    fit_h = am.fit(disp="off", cov_type="classic")     # inverse-Hessian SEs
    names = list(fit_r.params.index)                   # Const|mu, y[1..p], x0.., omega, alpha[1], beta[1], nu
    vec = np.r_[r.params[:1 + p], r.params[1 + p:1 + p + r.X.shape[1]] if use_x else [], r.params[1 + p + (2 if use_x else 0):]]
    fx = am.fix(vec)
    sig = np.asarray(fx.conditional_volatility)[p:]
    res = np.asarray(fx.resid)[p:]
    a_sig = np.abs(sig - r.sigma)[BURN:].max()
    a_res = np.abs(res - r.eps)[BURN:].max()
    est = np.asarray(fit_r.params)
    dse = np.abs(vec - est) / r.se[[*range(len(vec))]]
    rob = r.se / np.asarray(fit_r.std_err)
    hes = r.se_hessian / np.asarray(fit_h.std_err)
    lld = r.loglik - fit_r.loglikelihood
    name = f"AR({p}) {dist:6s} {g:10s}" + (" ARMAX-X" if use_x else "")
    rows.append(dict(name=name, a_sig=a_sig, a_res=a_res, dse=dse.max(), rob_lo=rob.min(), rob_hi=rob.max(),
                     hes_lo=hes.min(), hes_hi=hes.max(), lld=lld))
    print(f"{name:34s} {a_sig:9.1e} {a_res:8.1e} | {dse.max():9.3f} {rob.min():5.3f}-{rob.max():5.3f} "
          f"{hes.min():5.3f}-{hes.max():5.3f} {lld:8.3f}")
print(f"\nSUMMARY over {len(rows)} cases")
print(f"  (a) identical params, t>={BURN}: max |sigma diff| = {max(r['a_sig'] for r in rows):.1e}, "
      f"max |resid diff| = {max(r['a_res'] for r in rows):.1e}")
print(f"  (b) free fits: max |script - arch| / SE = {max(r['dse'] for r in rows):.3f}")
print(f"      robust SE ratio script/arch  : {min(r['rob_lo'] for r in rows):.3f} - {max(r['rob_hi'] for r in rows):.3f}")
print(f"      Hessian SE ratio script/arch : {min(r['hes_lo'] for r in rows):.3f} - {max(r['hes_hi'] for r in rows):.3f}")
print(f"      loglik script - arch         : {min(r['lld'] for r in rows):.3f} - {max(r['lld'] for r in rows):.3f}")
json.dump(rows, open(os.path.join(OUT, "arch_summary.json"), "w"), indent=1)
