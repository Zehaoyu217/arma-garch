"""Shared helpers for the limitation experiments (verification/limitations/).

- Hansen (1994) skewed Student-t, standardised to mean 0 and variance 1 (the `arch` package's
  "skewt"; lam < 0 = left-skewed / long left tail), with ppf, cdf, logpdf and random draws.
- A simulator for regression with AR(1)-GARCH(1,1) errors that takes ANY standardised shock
  distribution and an optional GJR leverage term (the engine's own simulate() only does normal / t).
- A prototype skew-t likelihood on top of the engine's own filter (_core._filter): the model the
  package does not have, used to show what a dist="skewt" option would change.
"""
from __future__ import annotations

import os
import sys

for _v in ("VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")             # one BLAS thread per worker process

import numpy as np
from scipy import optimize, stats
from scipy.special import gammaln

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "sarimax_garch", "src"))
from sarimax_garch import _core  # noqa: E402

OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)


# -----------------------------------------------------------------------------
# Hansen skew-t (standardised: mean 0, variance 1)
# -----------------------------------------------------------------------------
def _skewt_abc(nu, lam):
    c = np.exp(gammaln((nu + 1) / 2) - gammaln(nu / 2)) / np.sqrt(np.pi * (nu - 2))
    a = 4 * lam * c * (nu - 2) / (nu - 1)
    b = np.sqrt(1 + 3 * lam ** 2 - a ** 2)
    return a, b, c


def skewt_ppf(u, nu, lam):
    a, b, _ = _skewt_abc(nu, lam)
    u = np.asarray(u, float)
    lo = u < (1 - lam) / 2
    q = np.where(lo, stats.t.ppf(np.where(lo, u / (1 - lam), 0.5), nu),
                 stats.t.ppf(np.where(lo, 0.5, 0.5 + (u - (1 - lam) / 2) / (1 + lam)), nu))
    q = q * np.where(lo, 1 - lam, 1 + lam) * np.sqrt((nu - 2) / nu)
    return (q - a) / b


def skewt_cdf(z, nu, lam):
    a, b, _ = _skewt_abc(nu, lam)
    z = np.asarray(z, float)
    w = b * z + a
    lo = w < 0
    r = np.sqrt(nu / (nu - 2))
    return np.where(lo, (1 - lam) * stats.t.cdf(w / (1 - lam) * r, nu),
                    (1 - lam) / 2 + (1 + lam) * (stats.t.cdf(w / (1 + lam) * r, nu) - 0.5))


def skewt_logpdf(z, nu, lam):
    a, b, c = _skewt_abc(nu, lam)
    w = b * np.asarray(z, float) + a
    s = np.where(w < 0, 1 - lam, 1 + lam)
    return np.log(b) + np.log(c) - (nu + 1) / 2 * np.log1p((w / s) ** 2 / (nu - 2))


def draw_z(rng, size, dist="normal", nu=6.0, lam=0.0):
    """Standardised shocks (mean 0, variance 1)."""
    if dist == "normal":
        return rng.standard_normal(size)
    if dist == "t":
        return rng.standard_t(nu, size) * np.sqrt((nu - 2) / nu)
    if dist == "skewt":
        return skewt_ppf(rng.random(size), nu, lam)
    raise ValueError(dist)


def skewt_moments(nu, lam):
    """Skewness and kurtosis by numerical integration (for reporting the designs)."""
    from scipy import integrate
    f = lambda z, k: z ** k * np.exp(skewt_logpdf(z, nu, lam))
    m = [integrate.quad(f, -np.inf, np.inf, args=(k,), limit=200)[0] for k in range(5)]
    return dict(mass=m[0], mean=m[1], var=m[2] - m[1] ** 2, skew=m[3], kurt=m[4])


# -----------------------------------------------------------------------------
# DGP: y_t = x_t' g + u_t,  u_t = c + phi u_{t-1} + eps_t,  eps_t = sigma_t z_t,
#      sigma_t^2 = omega + (alpha + lev * 1[eps_{t-1} < 0]) eps_{t-1}^2 + beta sigma_{t-1}^2
# -----------------------------------------------------------------------------
def simulate(T, c, phi, omega, alpha, beta, z, lev=0.0, X=None, g=None, burn=500):
    """z: standardised shocks of length T + burn.  Returns y, u, eps, sig2 (post burn-in) and the
    end-of-sample state needed to simulate the true predictive distribution."""
    n = T + burn
    assert len(z) == n
    persist = alpha + lev / 2 + beta
    eps = np.zeros(n); sig2 = np.zeros(n); u = np.zeros(n)
    sig2[0] = omega / (1 - persist)
    eps[0] = np.sqrt(sig2[0]) * z[0]
    u[0] = c / (1 - phi) + eps[0]
    for t in range(1, n):
        e1 = eps[t - 1]
        sig2[t] = omega + (alpha + lev * (e1 < 0)) * e1 * e1 + beta * sig2[t - 1]
        eps[t] = np.sqrt(sig2[t]) * z[t]
        u[t] = c + phi * u[t - 1] + eps[t]
    u, eps, sig2 = u[burn:], eps[burn:], sig2[burn:]
    y = u + (0.0 if X is None else np.asarray(X) @ np.asarray(g))
    return y, u, eps, sig2


def true_future(state, H, M, rng, c, phi, omega, alpha, beta, zdist, lev=0.0, xg_future=None):
    """M paths of y_{T+1..T+H} from the TRUE model, given the true end-of-sample state
    (u_T, eps_T, sig2_T).  zdist: dict(dist=..., nu=..., lam=...)."""
    uT, eT, s2T = state
    u = np.full(M, uT); e = np.full(M, eT); s2 = np.full(M, s2T)
    out = np.empty((M, H))
    for h in range(H):
        s2 = omega + (alpha + lev * (e < 0)) * e * e + beta * s2
        e = np.sqrt(s2) * draw_z(rng, M, **zdist)
        u = c + phi * u + e
        out[:, h] = u + (0.0 if xg_future is None else xg_future[h])
    return out


def ar1_exog(T, rho, rng, burn=200):
    x = np.zeros(T + burn)
    sd = np.sqrt(1 - rho ** 2)
    for t in range(1, T + burn):
        x[t] = rho * x[t - 1] + sd * rng.standard_normal()
    return x[burn:]


# -----------------------------------------------------------------------------
# prototype: regression with ARMA-GARCH(1,1)-skew-t errors, on the engine's own filter
# -----------------------------------------------------------------------------
def fit_skewt(y, X, p, q, start_t, init="backcast"):
    """Joint MLE with Hansen skew-t innovations.  start_t: a parameter vector in the engine's
    dist="t" packing (const, phi.., theta.., gamma.., omega, alpha, beta, nu); lam is appended.
    Returns (params incl. lam, loglik, eps, sig2)."""
    k = 0 if X is None else X.shape[1]
    ia = 1 + p + q + k + 1                      # index of alpha

    def nll(th):
        core, lam = th[:-1], th[-1]
        omega, alpha, beta, nu = core[ia - 1], core[ia], core[ia + 1], core[-1]
        if (omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= _core._PERSIST_CAP
                or not (2.05 <= nu <= _core._NU_MAX) or abs(lam) >= 0.99):
            return 1e10
        eps, sig2, _ = _core._filter(core, y, X, p, q, "t", init, "regression")
        if not np.all(np.isfinite(sig2)) or np.any(sig2 <= 0):
            return 1e10
        ll = skewt_logpdf(eps / np.sqrt(sig2), nu, lam) - 0.5 * np.log(sig2)
        return -float(ll.sum()) if np.all(np.isfinite(ll)) else 1e10

    nfree = len(start_t) + 1
    bounds = ([(None, None)] * (1 + p + q + k) + [(1e-10, None), (0.0, 1.0), (0.0, 1.0), (2.05, _core._NU_MAX)]
              + [(-0.98, 0.98)])
    best = None
    for lam0 in (0.0, -0.2, 0.2):               # three starts: the skewness direction is not known
        x0 = np.r_[start_t, lam0]
        r = optimize.minimize(nll, x0, method="L-BFGS-B", bounds=bounds,
                              options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-8})
        r2 = optimize.minimize(nll, r.x, method="Nelder-Mead",
                               options={"maxiter": 4000 * nfree, "fatol": 1e-10, "xatol": 1e-8})
        cand = r2 if r2.fun < r.fun else r
        if best is None or cand.fun < best.fun:
            best = cand
    th = best.x
    eps, sig2, _ = _core._filter(th[:-1], y, X, p, q, "t", init, "regression")
    return th, -best.fun, eps, sig2


# -----------------------------------------------------------------------------
# prototype: GJR-GARCH(1,1) variance (leverage) on the engine's own mean filter
# -----------------------------------------------------------------------------
def gjr_sig2(eps, omega, alpha, beta, lev):
    """sigma_t^2 = omega + (alpha + lev 1[eps_{t-1}<0]) eps_{t-1}^2 + beta sigma_{t-1}^2, with the
    engine's backcast start.  Linear in sigma^2 given eps, so lfilter does it."""
    from scipy import signal
    n = len(eps)
    tau = min(75, n)
    w = 0.94 ** np.arange(tau)
    v0 = float(np.sum(w / w.sum() * eps[:tau] ** 2))
    u = np.empty(n)
    u[0] = omega + (alpha + lev / 2 + beta) * v0
    u[1:] = omega + (alpha + lev * (eps[:-1] < 0)) * eps[:-1] ** 2
    return signal.lfilter([1.0], [1.0, -beta], u)


def fit_gjr(y, X, p, q, start_n, init="backcast"):
    """Gaussian joint MLE of regression with ARMA-GJR-GARCH(1,1) errors.  start_n: engine dist="normal"
    packing (const, phi.., theta.., gamma.., omega, alpha, beta); lev is appended."""
    k = 0 if X is None else X.shape[1]
    ia = 1 + p + q + k + 1

    def nll(th):
        core, lev = th[:-1], th[-1]
        omega, alpha, beta = core[ia - 1], core[ia], core[ia + 1]
        if omega <= 0 or alpha < 0 or beta < 0 or alpha + lev < 0 or alpha + lev / 2 + beta >= _core._PERSIST_CAP:
            return 1e10
        eps, _, _ = _core._filter(core, y, X, p, q, "normal", init, "regression")
        sig2 = gjr_sig2(eps, omega, alpha, beta, lev)
        if not np.all(np.isfinite(sig2)) or np.any(sig2 <= 0):
            return 1e10
        return float(0.5 * np.sum(np.log(2 * np.pi) + np.log(sig2) + eps ** 2 / sig2))

    best = None
    for lev0 in (0.0, 0.1):
        x0 = np.r_[start_n, lev0]
        r = optimize.minimize(nll, x0, method="Nelder-Mead",
                              options={"maxiter": 6000 * len(x0), "fatol": 1e-10, "xatol": 1e-8})
        r = optimize.minimize(nll, r.x, method="Powell", options={"maxiter": 20000, "ftol": 1e-12, "xtol": 1e-8})
        if best is None or r.fun < best.fun:
            best = r
    th = best.x
    eps, _, _ = _core._filter(th[:-1], y, X, p, q, "normal", init, "regression")
    return th, -best.fun, eps, gjr_sig2(eps, th[ia - 1], th[ia], th[ia + 1], th[-1])


def summarise(est, truth):
    """est: (R, k) estimates, truth: (k,) -> dict of bias, sd, rmse, bias/sd, MC SE of the bias."""
    est = np.asarray(est, float)
    bias = np.nanmean(est, 0) - truth
    sd = np.nanstd(est, 0, ddof=1)
    R = np.sum(np.isfinite(est), 0)
    return dict(bias=bias, sd=sd, rmse=np.sqrt(bias ** 2 + sd ** 2), z=bias / (sd / np.sqrt(R)), bias_sd=bias / sd)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    for nu, lam in ((6, 0.0), (6, -0.25), (6, -0.5), (30, -0.5)):
        mo = skewt_moments(nu, lam)
        zz = draw_z(rng, 2_000_000, "skewt", nu, lam)
        u = rng.random(5)
        rt = np.max(np.abs(skewt_cdf(skewt_ppf(u, nu, lam), nu, lam) - u))
        print(f"nu={nu:>2} lam={lam:+.2f}: integral mass {mo['mass']:.6f} mean {mo['mean']:+.2e} var {mo['var']:.6f} "
              f"skew {mo['skew']:+.3f} kurt {mo['kurt']:.2f} | draws mean {zz.mean():+.4f} var {zz.var():.4f} "
              f"P(z<0)={np.mean(zz < 0):.3f} median {np.median(zz):+.3f} | cdf(ppf(u))-u {rt:.1e}")
