"""Independent, deliberately naive loop implementation of the documented likelihood,
written from the docstring equations only (not from the vectorised code)."""
import sys, numpy as np
from math import lgamma, log, pi
import os; sys.path.insert(0, os.environ.get("JAG_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))
import joint_armax_garch as J

def ref_loglik(theta, y, X, p, q, dist, init, xreg="armax"):
    k = 0 if X is None else X.shape[1]
    i = 0
    c = theta[i]; i += 1
    phi = theta[i:i+p]; i += p
    th = theta[i:i+q]; i += q
    g = theta[i:i+k]; i += k
    om, al, be = theta[i:i+3]; i += 3
    nu = theta[i] if dist == "t" else None
    T = len(y)
    # regression with ARMA errors: the ARMA runs on u_t = y_t - x_t'g;  ARMAX: on y_t, with x_t'g added
    u = [y[t] - (float(X[t] @ g) if (k and xreg == "regression") else 0.0) for t in range(T)]
    eps = []
    for t in range(p, T):
        m = c + sum(phi[a]*u[t-1-a] for a in range(p))
        if k and xreg == "armax": m += float(X[t] @ g)
        n = len(eps)
        m += sum(th[b]*(eps[n-1-b] if n-1-b >= 0 else 0.0) for b in range(q))
        eps.append(u[t] - m)
    eps = np.array(eps)
    if init == "sample":
        v0 = np.mean(eps**2)
    else:
        tau = min(75, len(eps)); w = 0.94**np.arange(tau); w /= w.sum()
        v0 = np.sum(w*eps[:tau]**2)
    ll = 0.0; e2_prev = v0; s2_prev = v0
    for t in range(len(eps)):
        s2 = om + al*e2_prev + be*s2_prev
        z2 = eps[t]**2/s2
        if dist == "normal":
            ll += -0.5*(log(2*pi) + log(s2) + z2)
        else:
            ll += (lgamma((nu+1)/2) - lgamma(nu/2) - 0.5*log(pi*(nu-2))
                   - 0.5*log(s2) - (nu+1)/2*np.log1p(z2/(nu-2)))
        e2_prev, s2_prev = eps[t]**2, s2
    return ll

import inspect
modes = ["armax", "regression"] if "xreg" in inspect.signature(J._loglik_obs).parameters else [None]
for xreg in modes:
  rng = np.random.default_rng(0)
  worst = 0.0
  for trial in range(60):
    p, q, k = rng.integers(0, 3), rng.integers(0, 3), rng.integers(0, 3)
    dist = ["normal", "t"][trial % 2]; init = ["backcast", "sample"][(trial//2) % 2]
    T = 400
    X = rng.normal(size=(T, k)) if k else None
    y = rng.normal(size=T)
    theta = np.r_[rng.normal(0, .3), rng.uniform(-.4, .4, p), rng.uniform(-.4, .4, q),
                  rng.normal(0, .5, k), rng.uniform(.05, .3), rng.uniform(.02, .2), rng.uniform(.5, .75),
                  ([rng.uniform(3, 20)] if dist == "t" else [])]
    a = (J._loglik_obs(theta, y, X, p, q, dist, init) if xreg is None
         else J._loglik_obs(theta, y, X, p, q, dist, init, xreg)).sum()
    b = ref_loglik(theta, y, X, p, q, dist, init, xreg or "armax")
    worst = max(worst, abs(a-b)/abs(b))
  print(f"xreg={xreg or 'armax (v2.x)'}: 60 random (p,q,k,dist,init) configs: "
        f"max relative |loglik_vectorised - loglik_loop| = {worst:.2e}")

