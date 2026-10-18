"""28 cases (7 ARMA orders x {normal, t} x {no X, X}; d cycles 0/1/2, GARCH set cycles over 5):
fit with the script, export the (differenced) data and the script's results for rugarch."""
import json
import os
import warnings

import numpy as np

from common import GARCH, J, ORDERS, OUT, H, diff_future_X, make_data, truth

warnings.filterwarnings("ignore", category=RuntimeWarning)
D = os.path.join(OUT, "rugarch")
os.makedirs(D, exist_ok=True)
T = 2000
cases, j = {}, 0
for order in ORDERS:
    for dist in ("normal", "t"):
        for use_x in (False, True):
            d, garch = j % 3, list(GARCH)[j % 5]
            name = f"c{j:02d}_arma{order[0]}{order[1]}_d{d}_{dist}_{'X' if use_x else 'noX'}_{garch}"
            y, X, Xf = make_data(T, order, d, garch, dist, 6.0, use_x, seed=1000 + j)
            r = J.fit_joint(y, X=X, p=order[0], q=order[1], d=d, dist=dist)
            fc = J.forecast(r, H, X_future=Xf, n_sims=0)
            ll = J._loglik_obs(r.params, r.y, r.X, r.p, r.q, r.dist, r.init, r.xreg)
            np.savetxt(f"{D}/{name}_y.csv", r.y)
            if use_x:
                np.savetxt(f"{D}/{name}_X.csv", r.X, delimiter=",")
                np.savetxt(f"{D}/{name}_Xf.csv", diff_future_X(X, Xf, d), delimiter=",")
            np.savetxt(f"{D}/{name}_script_filter.csv", np.column_stack([r.eps, r.sigma, ll]), delimiter=",")
            cases[name] = dict(p=r.p, q=r.q, d=d, dist=dist, k=0 if r.X is None else r.X.shape[1], garch=garch,
                               names=r.names, params=r.params.tolist(), truth=truth(order, garch, dist, 6.0, use_x).tolist(),
                               se=r.se.tolist(), se_hess=r.se_hessian.tolist(), loglik=r.loglik,
                               fc_mean=(fc["mean_diff"] if d else fc["mean"]).tolist(),
                               fc_sigma=np.sqrt(fc["variance_innovation"]).tolist())
            print(f"{name:42s} LL={r.loglik:10.3f} conv={r.converged}")
            j += 1
json.dump(cases, open(f"{D}/script.json", "w"), indent=1)
