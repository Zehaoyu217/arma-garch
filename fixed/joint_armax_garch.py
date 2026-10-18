"""
joint_armax_garch.py  --  joint MLE of regression with ARIMA(p,d,q)-GARCH(1,1) errors
=====================================================================================
Pure numpy/scipy. No `arch`, no statsmodels needed (they are only used,
optionally, in the self-test for cross-checking).

Model (on w_t = (1-L)^d y_t and x_t = (1-L)^d X_t, the d-th differences):

    mean:      u_t = w_t - x_t' gamma                          (statsmodels SARIMAX form)
               u_t = c + sum_i phi_i u_{t-i} + sum_j theta_j eps_{t-j} + eps_t
    variance:  eps_t = sigma_t z_t,   sigma_t^2 = omega + alpha eps_{t-1}^2 + beta sigma_{t-1}^2
    errors:    z_t ~ N(0,1)  or  standardized Student-t(nu)  (unit variance)

With xreg="armax" the mean is ARMAX instead: w_t = c + sum phi_i w_{t-i} + x_t'gamma + MA + eps_t.

Everything -- mean coefficients, GARCH coefficients, nu -- is estimated in ONE
likelihood, so the mean estimates are implicitly GLS-weighted by 1/sigma_t^2
(this moves point estimates, not just standard errors, relative to two-step).
Standard errors are QMLE sandwich (Bollerslev-Wooldridge): H^{-1} (S'S) H^{-1}.

ARIMA + GARCH:  pass the LEVEL series with d>0.  The script differences y (and
X, if X is given in levels), fits the ARMA-GARCH on the differences, and
`forecast()` integrates the results back to levels -- mean, variance (via the
integrated psi-weights), and simulated paths/quantiles.

Usage
-----
    from joint_armax_garch import fit_joint, forecast

    res = fit_joint(y, X=X, p=1, q=1, d=0, dist="t")     # y stationary
    res = fit_joint(y_levels, p=1, q=1, d=1, dist="t")   # ARIMA(1,1,1)-GARCH-t
    print(res.summary())
    fc  = forecast(res, horizon=10, X_future=Xf, n_sims=10_000, seed=0)
    fc["mean"], fc["variance"], fc["quantiles"](0.05), fc["sims"]

    res = fit_joint(y, X=X, xreg="armax")                 # ARMAX mean instead of SARIMAX form
    res = fit_joint(y, fixed={"const": 0.0})              # no mean (rugarch include.mean=FALSE)
    new = fit_joint(y_new, X=X_new, xreg=res.xreg,        # apply fitted params to new data
                    fixed=dict(zip(res.names, res.params)))   # (rugarch ugarchfilter)

    # small samples (< ~250 obs): test for ARCH before fitting GARCH, bootstrap the CIs
    base = fit_joint(y, X=X, p=1, q=1, dist="normal", fixed={"alpha": 0.0, "beta": 0.0})
    arch_lm_test(base.eps, lags=12)                       # (statistic, p-value)
    ljung_box(res.std_resid, 12, df_adjust=2), ljung_box(res.std_resid ** 2, 12)
    bs = bootstrap(res, n_boot=199, seed=0); bs["ci"]     # 95% percentile CIs

    # levels (d=0) or differences (d=1)?  GARCH or not?  Compare out of sample at your horizons:
    lv = backtest(y, X=X, p=1, q=0, d=0, horizon=36, start=120, step=3)
    df = backtest(y, X=X, p=1, q=1, d=1, horizon=36, start=120, step=3)
    lv["crps"], lv["coverage90"], df["crps"], df["coverage90"]   # by horizon; lower CRPS wins

Run `python joint_armax_garch.py` for the self-test (parameter recovery,
arch cross-check if arch is installed, ARIMA-GARCH level-forecast check, and --
if dem2gbp_benchmark.csv sits next to this file -- the published Fiorentini-
Calzolari-Panattoni (1996) GARCH(1,1) benchmark, which this code reproduces to
6 digits with init="sample").

Same ARMA/GARCH recursions and t density as R rugarch (sGARCH), fGarch,
MATLAB arima+garch, Stata `arch ... ar() ma()`: at identical parameters the
residuals, sigma_t and per-observation loglik equal rugarch's to ~1e-14 once the
start-up has decayed, and forecasts match to ~1e-15.  The start-up differs:
rugarch keeps the first max(p,q,1) observations with h = mean(res^2), so its
loglik differs by O(1) and its estimates by O(1/T).  rugarch's `mu` is the
unconditional mean: const = mu * (1 - sum(phi)).
Python's `arch` is the same estimator but has no MA terms in the mean.

Exogenous X: by default (xreg="regression", since v3.0) this is regression with
ARMA errors -- the model of statsmodels SARIMAX/ARIMA(exog=), rugarch
`external.regressors` and R arima(xreg=); SARIMAX trend="c" is `const` here.
xreg="armax" (the v2.x default) puts x_t next to the AR terms instead, a
different model when p > 0.  With p = 0 the two coincide.

Conditioning: the likelihood is conditional on the first p observations of w;
pre-sample eps are 0; sigma^2 is initialised with an exponentially weighted
backcast of eps^2 (as arch does).  Rescale w to std ~ 0.1-10 before fitting.
"""
from __future__ import annotations

import warnings
import numpy as np
from scipy import optimize, signal, stats
from scipy.special import gammaln

__version__ = "3.2"

_PERSIST_CAP = 0.9999      # alpha + beta < cap (covariance stationarity, with margin)
_NU_MAX = 200.0            # upper bound for nu: t(200) is normal to 3 decimals; beyond it the
                           # likelihood is flat and an unbounded simplex wanders off to 1e6


# -----------------------------------------------------------------------------
# parameter packing
# -----------------------------------------------------------------------------
def _unpack(theta, p, q, k, dist):
    i = 0
    c = theta[i]; i += 1
    phi = theta[i:i + p]; i += p
    th = theta[i:i + q]; i += q
    gamma = theta[i:i + k]; i += k
    omega, alpha, beta = theta[i], theta[i + 1], theta[i + 2]; i += 3
    nu = theta[i] if dist == "t" else None
    return c, phi, th, gamma, omega, alpha, beta, nu


def _names(p, q, k, dist):
    return (["const"] + [f"phi{i + 1}" for i in range(p)]
            + [f"theta{j + 1}" for j in range(q)]
            + [f"gamma{i + 1}" for i in range(k)]
            + ["omega", "alpha", "beta"] + (["nu"] if dist == "t" else []))


# -----------------------------------------------------------------------------
# filter: ARMA residuals + GARCH variances (vectorised with lfilter)
# -----------------------------------------------------------------------------
def _filter(theta, y, X, p, q, dist, init="backcast", xreg="regression"):
    """Return eps (T-p,), sig2 (T-p,), nu for the conditioning sample t = p..T-1.

    init : pre-sample value v0 used for both eps_{t-1}^2 and sigma_{t-1}^2 at the first
           in-sample date, so sigma^2_first = omega + (alpha + beta) v0.
           "backcast" -- exponentially weighted (0.94) mean of the first 75 eps^2 (arch's rule)
           "sample"   -- plain mean of eps^2 over the sample (FCP 1996 / fGarch "mci" rule)
    xreg : "regression" -- regression with ARMA errors (statsmodels SARIMAX exog, rugarch
                           external.regressors, R arima xreg): z_t = w_t - x_t'gamma
                           follows the ARMA with constant c
           "armax"      -- w_t = c + sum phi_i w_{t-i} + x_t'gamma + MA + eps_t
    """
    T = len(y)
    k = 0 if X is None else X.shape[1]
    c, phi, th, gamma, omega, alpha, beta, nu = _unpack(theta, p, q, k, dist)

    base = y - X @ gamma if (k and xreg == "regression") else y
    rhs = base[p:] - c
    for i in range(p):
        rhs = rhs - phi[i] * base[p - 1 - i: T - 1 - i]
    if k and xreg == "armax":
        rhs = rhs - X[p:] @ gamma
    # eps_t = rhs_t - sum_j theta_j eps_{t-j}   (zero pre-sample eps)
    eps = signal.lfilter([1.0], np.r_[1.0, th], rhs) if q else rhs

    n = len(eps)
    if init == "sample":
        v0 = float(np.mean(eps ** 2))
    else:
        tau = min(75, n)
        w = 0.94 ** np.arange(tau)
        w /= w.sum()
        v0 = float(np.sum(w * eps[:tau] ** 2))
    u = np.empty(n)
    u[0] = omega + (alpha + beta) * v0             # pre-sample eps^2 = sigma^2 = v0
    u[1:] = omega + alpha * eps[:-1] ** 2
    sig2 = signal.lfilter([1.0], [1.0, -beta], u)  # sig2_t = u_t + beta sig2_{t-1}
    return eps, sig2, nu


def _loglik_obs(theta, y, X, p, q, dist, init="backcast", xreg="regression"):
    eps, sig2, nu = _filter(theta, y, X, p, q, dist, init, xreg)
    if not np.all(np.isfinite(sig2)) or np.any(sig2 <= 0):
        return None
    z2 = eps ** 2 / sig2
    if dist == "normal":
        return -0.5 * (np.log(2 * np.pi) + np.log(sig2) + z2)
    return (gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(np.pi * (nu - 2))
            - 0.5 * np.log(sig2) - (nu + 1) / 2 * np.log1p(z2 / (nu - 2)))


def _neg_loglik(theta, y, X, p, q, dist, init="backcast", xreg="regression"):
    k = 0 if X is None else X.shape[1]
    _, _, _, _, omega, alpha, beta, nu = _unpack(theta, p, q, k, dist)
    if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= _PERSIST_CAP:
        return 1e10
    if dist == "t" and not (2.0 < nu <= _NU_MAX):
        return 1e10
    return _neg_loglik_raw(theta, y, X, p, q, dist, init, xreg)


def _neg_loglik_raw(theta, y, X, p, q, dist, init="backcast", xreg="regression"):
    """-loglik without the parameter-region penalty (only invalid sigma^2 / nu -> 1e10).
    Used where the region is enforced elsewhere: SLSQP constraints, numerical derivatives."""
    if dist == "t" and _unpack(theta, p, q, 0 if X is None else X.shape[1], dist)[-1] <= 2.0:
        return 1e10
    ll = _loglik_obs(theta, y, X, p, q, dist, init, xreg)
    if ll is None or not np.all(np.isfinite(ll)):
        return 1e10
    return -float(ll.sum())


# -----------------------------------------------------------------------------
# starting values: Hannan-Rissanen for the mean, generic GARCH(1,1) start
# -----------------------------------------------------------------------------
def _lagmat(v, lags, start):
    """Columns v_{t-1}..v_{t-lags} for t = start..T-1."""
    return np.column_stack([v[start - j: len(v) - j] for j in range(1, lags + 1)]) \
        if lags else np.empty((len(v) - start, 0))


def _ols(Z, y):
    beta, *_ = np.linalg.lstsq(Z, y, rcond=None)
    return beta, y - Z @ beta


def _hr_start(y, X, p, q, dist, xreg="regression"):
    if X is not None and xreg == "regression":
        # OLS for gamma, then Hannan-Rissanen on the regression errors
        g0, _ = _ols(np.column_stack([np.ones(len(y)), X]), y)
        s = _hr_start(y - X @ g0[1:], None, p, q, dist)
        return np.r_[s[:1 + p + q], g0[1:], s[1 + p + q:]]
    T = len(y)
    k = 0 if X is None else X.shape[1]
    if q == 0:
        s = p
        Z = np.column_stack([np.ones(T - s), _lagmat(y, p, s)] + ([X[s:]] if k else []))
        b, e = _ols(Z, y[s:])
        c, phi, th, gamma = b[0], b[1:1 + p], np.empty(0), b[1 + p:]
    else:
        m = min(max(p + q + 5, 10), max(T // 10, p + q + 1))        # long-AR order
        Z = np.column_stack([np.ones(T - m), _lagmat(y, m, m)] + ([X[m:]] if k else []))
        _, e_long = _ols(Z, y[m:])
        e_full = np.zeros(T)
        e_full[m:] = e_long
        s = m + q
        Z = np.column_stack([np.ones(T - s), _lagmat(y, p, s), _lagmat(e_full, q, s)]
                            + ([X[s:]] if k else []))
        b, e = _ols(Z, y[s:])
        c, phi, th, gamma = b[0], b[1:1 + p], b[1 + p:1 + p + q], b[1 + p + q:]
        th = np.clip(th, -0.95, 0.95)
    s2 = float(np.var(e))
    alpha0, beta0 = 0.05, 0.90
    garch = [s2 * (1 - alpha0 - beta0), alpha0, beta0]
    start = np.r_[c, phi, th, gamma, garch, ([8.0] if dist == "t" else [])]
    return start


# -----------------------------------------------------------------------------
# QMLE sandwich covariance
# -----------------------------------------------------------------------------
def _fd_typ(y, X, p, q, dist):
    """Typical magnitude of each parameter; finite-difference steps are rel * max(|theta_j|, typ_j).
    A floor of 1.0 for every parameter made the omega step absolute (1e-4), which crossed
    omega = 0 -- and hit the 1e10 penalty -- whenever omega < 2e-4, i.e. for any series with
    std below ~0.2 or with high persistence.  Mean-equation floors scale with the data;
    omega's step is purely relative."""
    k = 0 if X is None else X.shape[1]
    sd = float(np.std(y)) or 1.0
    sx = np.std(X, axis=0) if k else np.empty(0)
    sx = np.where(sx > 0, sx, 1.0)
    return np.r_[sd, np.ones(p + q), sd / sx, 1e-8 * sd ** 2, 1.0, 1.0, ([1.0] if dist == "t" else [])]


def _num_scores(theta, y, X, p, q, dist, init="backcast", rel=1e-5, typ=None, xreg="regression", free=None):
    """Per-observation score matrix S (n, len(free)) by central differences (one-sided if a step
    is invalid).  free: indices of the parameters to differentiate (default: all)."""
    k = len(theta)
    typ = np.ones(k) if typ is None else typ
    free = np.arange(k) if free is None else free
    base = _loglik_obs(theta, y, X, p, q, dist, init, xreg)
    S = np.empty((len(base), len(free)))
    for col, j in enumerate(free):
        h = rel * max(typ[j], abs(theta[j]))
        tp, tm = theta.copy(), theta.copy()
        tp[j] += h; tm[j] -= h
        lp = _loglik_obs(tp, y, X, p, q, dist, init, xreg)
        lm = _loglik_obs(tm, y, X, p, q, dist, init, xreg)
        if lp is None and lm is None:
            raise FloatingPointError(f"score step for parameter {j} gives invalid sigma^2 on both sides")
        if lp is None:
            S[:, col] = (base - lm) / h
        elif lm is None:
            S[:, col] = (lp - base) / h
        else:
            S[:, col] = (lp - lm) / (2 * h)
    return S


def _num_hessian(f, theta, rel=1e-4, typ=None, free=None):
    """Hessian of scalar f at theta, over the `free` indices, by central differences of central gradients."""
    k = len(theta)
    typ = np.ones(k) if typ is None else typ
    free = np.arange(k) if free is None else free

    def grad(t):
        g = np.empty(len(free))
        for col, j in enumerate(free):
            h = rel * max(typ[j], abs(t[j]))
            tp, tm = t.copy(), t.copy()
            tp[j] += h; tm[j] -= h
            g[col] = (f(tp) - f(tm)) / (2 * h)
        return g

    H = np.empty((len(free), len(free)))
    for col, j in enumerate(free):
        h = rel * max(typ[j], abs(theta[j]))
        tp, tm = theta.copy(), theta.copy()
        tp[j] += h; tm[j] -= h
        H[:, col] = (grad(tp) - grad(tm)) / (2 * h)
    return 0.5 * (H + H.T)


def _sandwich_cov(theta, y, X, p, q, dist, init="backcast", xreg="regression", free=None):
    """Return (robust sandwich cov, Hessian-based cov, OPG-based cov), each (k, k);
    rows/columns of fixed (non-free) parameters are NaN."""
    k = len(theta)
    free = np.arange(k) if free is None else free
    typ = _fd_typ(y, X, p, q, dist)
    S = _num_scores(theta, y, X, p, q, dist, init, typ=typ, xreg=xreg, free=free)
    OPG = S.T @ S
    # raw loglik: a step across the alpha+beta cap must see the likelihood, not the 1e10 wall
    H = _num_hessian(lambda t: _neg_loglik_raw(t, y, X, p, q, dist, init, xreg), theta, typ=typ, free=free)
    try:
        Hinv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        Hinv = np.linalg.pinv(H)

    def embed(V):
        out = np.full((k, k), np.nan)
        out[np.ix_(free, free)] = V
        return out
    return embed(Hinv @ OPG @ Hinv), embed(Hinv), embed(np.linalg.pinv(OPG))


# -----------------------------------------------------------------------------
# result object
# -----------------------------------------------------------------------------
class JointResult:
    def __init__(self, params, names, cov, loglik, p, q, d, dist, y, X, eps, sig2,
                 y_levels_tail=None, X_levels_tail=None, converged=True,
                 cov_hessian=None, cov_opg=None, init="backcast", xreg="regression", fixed=None):
        self.params = np.asarray(params, float)
        self.names = list(names)
        self.xreg = xreg
        self.fixed = dict(fixed or {})              # name -> value held fixed in the fit
        self.n_free = len(self.params) - len(self.fixed)
        self.cov = cov                              # QMLE sandwich (robust) -- used by .se
        self.cov_hessian = cov_hessian              # inverse Hessian (classical ML SEs)
        self.cov_opg = cov_opg                      # inverse outer-product-of-gradients
        self.init = init
        self.loglik = float(loglik)
        self.p, self.q, self.d, self.dist = p, q, d, dist
        self.y, self.X = y, X                       # the (differenced) series used in the fit
        self.eps, self.sig2 = eps, sig2             # aligned with y[p:]
        self.y_levels_tail = y_levels_tail          # last d levels (d > 0)
        self.X_levels_tail = X_levels_tail          # last d rows of X in levels, or None
        self.converged = converged
        self.k = len(self.params)
        self.nobs = len(eps)

    # --- accessors ---
    def __getitem__(self, name):
        return self.params[self.names.index(name)]

    @property
    def se(self):
        return np.sqrt(np.diag(self.cov)) if self.cov is not None else np.full(self.k, np.nan)

    @property
    def se_hessian(self):
        return (np.sqrt(np.diag(self.cov_hessian)) if self.cov_hessian is not None
                else np.full(self.k, np.nan))

    @property
    def aic(self):
        return -2 * self.loglik + 2 * self.n_free

    @property
    def bic(self):
        return -2 * self.loglik + self.n_free * np.log(self.nobs)

    @property
    def uncond_mean(self):
        """Unconditional mean of the ARMA part -- rugarch's `mu`: const / (1 - sum(phi))."""
        return self["const"] / (1 - np.sum(self.params[1:1 + self.p]))

    @property
    def sigma(self):
        return np.sqrt(self.sig2)

    @property
    def std_resid(self):
        return self.eps / np.sqrt(self.sig2)

    @property
    def persistence(self):
        return self["alpha"] + self["beta"]

    @property
    def half_life(self):
        per = self.persistence
        return np.log(0.5) / np.log(per) if 0 < per < 1 else np.inf

    @property
    def uncond_var(self):
        per = self.persistence
        return self["omega"] / (1 - per) if per < 1 else np.inf

    @property
    def kurtosis_eta(self):
        """E[eta^4] implied by the fitted error distribution (inf if nu <= 4)."""
        if self.dist == "normal":
            return 3.0
        nu = self["nu"]
        return 3.0 * (nu - 2) / (nu - 4) if nu > 4 else np.inf

    @property
    def fourth_moment_exists(self):
        """E[eps^4] < inf  <=>  kappa_eta*alpha^2 + 2*alpha*beta + beta^2 < 1.
        This is the condition for asymptotic normality of the ARMA-GARCH QMLE
        (Francq & Zakoian 2004). If False, the ARMA-parameter SEs below rest on theory
        that does not apply; see Ling (2007) self-weighted QMLE."""
        a, b = self["alpha"], self["beta"]
        return bool(self.kurtosis_eta * a ** 2 + 2 * a * b + b ** 2 < 1)

    @property
    def at_boundary(self):
        """alpha or beta numerically at 0: standard Wald/LR inference is invalid there
        (Francq & Zakoian 2007; Andrews 2001), and with alpha = 0 beta is unidentified."""
        return bool(self["alpha"] < 1e-4 or self["beta"] < 1e-4)

    @property
    def at_persistence_cap(self):
        """alpha + beta pinned at the stationarity cap: the unconstrained optimum is (near-)
        integrated, so SEs, half-life and uncond_var are not meaningful."""
        return bool(self.persistence > _PERSIST_CAP - 1e-6)

    def _poly_roots_ok(self, coefs, sign):
        if len(coefs) == 0:
            return True
        roots = np.roots(np.r_[(sign * coefs)[::-1], 1.0])  # 1 + sign*c1 z + ... = 0
        return bool(np.all(np.abs(roots) > 1.0 + 1e-8))

    @property
    def stationary(self):
        return self._poly_roots_ok(self.params[1:1 + self.p], -1.0)

    @property
    def invertible(self):
        return self._poly_roots_ok(self.params[1 + self.p:1 + self.p + self.q], +1.0)

    def _roots(self, coefs, sign):
        coefs = np.asarray(coefs, float)
        if len(coefs) == 0 or not np.any(coefs):
            return np.empty(0)
        return np.roots(np.r_[(sign * coefs)[::-1], 1.0])

    @property
    def ar_roots(self):
        return self._roots(self.params[1:1 + self.p], -1.0)

    @property
    def ma_roots(self):
        return self._roots(self.params[1 + self.p:1 + self.p + self.q], +1.0)

    @property
    def near_unit_root(self):
        """An AR root within 2% of the unit circle: the series may need another difference."""
        return bool(self.ar_roots.size and np.min(np.abs(self.ar_roots)) < 1.02)

    @property
    def ma_near_unit_root(self):
        """An MA root within 5% of the unit circle: typically over-differencing.  The band is
        wider than for AR because the conditional likelihood pushes MA roots away from 1
        (an over-differenced series at T=1000 gives |root| ~ 1.02-1.03)."""
        return bool(self.ma_roots.size and np.min(np.abs(self.ma_roots)) < 1.05)

    @property
    def arma_cancellation(self):
        """An AR and an MA inverse root closer than 0.1: the ARMA orders are too high and the
        AR/MA coefficients are weakly identified (large SEs, optimiser-dependent values)."""
        ar, ma = self.ar_roots, self.ma_roots
        if not (ar.size and ma.size):
            return False
        return bool(np.min(np.abs(1 / ar[:, None] - 1 / ma[None, :])) < 0.1)

    @property
    def nu_at_bound(self):
        return bool(self.dist == "t" and "nu" not in self.fixed and self["nu"] > _NU_MAX - 1.0)

    def summary(self):
        se = self.se
        z = self.params / se
        pv = 2 * stats.norm.sf(np.abs(z))
        kind = ("ARIMA" if self.X is None else "ARIMAX" if self.xreg == "armax"
                else "Regression with ARIMA errors")
        lines = [f"Joint {kind}({self.p},{self.d},{self.q})-GARCH(1,1)-{self.dist} MLE   "
                 f"nobs={self.nobs}  loglik={self.loglik:.3f}  AIC={self.aic:.2f}  BIC={self.bic:.2f}",
                 f"{'param':>8} {'coef':>12} {'rob.SE':>10} {'z':>8} {'p':>8}"]
        for n, b, s, zz, pp in zip(self.names, self.params, se, z, pv):
            if n in self.fixed:
                lines.append(f"{n:>8} {b:>12.5f} {'fixed':>10}")
            else:
                lines.append(f"{n:>8} {b:>12.5f} {s:>10.5f} {zz:>8.2f} {pp:>8.4f}")
        lines.append(f"persistence={self.persistence:.4f}  half-life={self.half_life:.1f}  "
                     f"uncond.var={self.uncond_var:.4f}")
        lines.append(f"AR stationary={self.stationary}  MA invertible={self.invertible}  "
                     f"converged={self.converged}")
        lines.append(f"E[eps^4]<inf (needed for asymptotic normality of these SEs): "
                     f"{self.fourth_moment_exists}   kappa_eta={self.kurtosis_eta:.2f}")
        if not self.fourth_moment_exists:
            lines.append("  WARNING: fourth moment infinite -> mean-equation SEs are not backed by "
                         "standard QMLE theory (Francq-Zakoian 2004); consider Ling (2007).")
        if self.at_boundary:
            lines.append("  WARNING: GARCH coefficient at the boundary -> Wald/LR p-values invalid; "
                         "alpha=0 leaves beta unidentified.")
        if self.at_persistence_cap:
            lines.append(f"  WARNING: alpha+beta at the {_PERSIST_CAP} stationarity cap -> the data want an "
                         "integrated GARCH; SEs, half-life and uncond.var are not meaningful.")
        if not self.invertible:
            lines.append("  WARNING: MA not invertible -> over-differencing or misspecification; "
                         "the conditional likelihood used here assumes invertibility.")
        elif self.ma_near_unit_root:
            lines.append("  WARNING: MA root near the unit circle -> likely over-differenced (try d-1); "
                         "the conditional likelihood is unreliable there.")
        if self.near_unit_root:
            lines.append("  WARNING: AR root near the unit circle -> the series may need another "
                         "difference (try d+1).")
        if self.arma_cancellation:
            lines.append("  WARNING: AR and MA roots nearly cancel -> ARMA orders too high; AR/MA "
                         "coefficients are weakly identified (try lower p, q).")
        if self.nu_at_bound:
            lines.append("  WARNING: nu at its upper bound -> no fat tails in the data; dist='normal' "
                         "is the same model with one parameter less.")
        return "\n".join(lines)


# -----------------------------------------------------------------------------
# estimation
# -----------------------------------------------------------------------------
def _check_finite(y, X):
    bad = ~np.isfinite(y)
    if bad.any():
        raise ValueError(f"y has {int(bad.sum())} NaN/inf value(s), first at index {int(np.argmax(bad))}: "
                         "the conditional likelihood needs a gap-free series -- drop or fill them first")
    if X is not None:
        badx = ~np.all(np.isfinite(X), axis=1)
        if badx.any():
            raise ValueError(f"X has NaN/inf in {int(badx.sum())} row(s), first at row {int(np.argmax(badx))}")


def _check_design(y, X, p, fixed):
    const_fixed = "const" in fixed
    if np.ptp(y) == 0:
        raise ValueError("the series is constant (after differencing): there is nothing to fit")
    if X is not None:
        Z = X if const_fixed else np.column_stack([np.ones(len(X)), X])
        if np.linalg.matrix_rank(Z) < Z.shape[1]:
            flat = [f"gamma{j + 1}" for j in range(X.shape[1]) if np.ptp(X[:, j]) == 0]
            hint = f" ({', '.join(flat)} constant after differencing -- duplicates const)" \
                if flat and not const_fixed else ""
            raise ValueError(f"X is rank-deficient{hint}: drop the collinear column(s)")
    n = len(y) - p
    garch_free = not ("alpha" in fixed and "beta" in fixed)
    if n < 100:
        warnings.warn(f"only {n} observations: GARCH estimates are unreliable this short "
                      "(rugarch refuses fewer than 100).")
    elif garch_free and n < 250:
        warnings.warn(f"{n} observations: GARCH parameters are weakly identified below ~250 (Monte Carlo at "
                      "120 obs: half the fits end on a boundary and the alpha/beta Wald CIs cover ~60%). "
                      "Test for ARCH first: arch_lm_test() on a fit with fixed={'alpha': 0, 'beta': 0}; "
                      "use bootstrap() for the GARCH CIs; prefer dist='normal'.")


def fit_joint(y, X=None, p=1, q=1, d=0, dist="t", start=None, polish=True, compute_cov=True,
              init="backcast", xreg="regression", fixed=None):
    """Joint MLE of ARIMAX(p,d,q)-GARCH(1,1)-{normal,t}.

    y     : 1-d array-like. If d > 0 this is the LEVEL series; it is differenced d times.
            Must be gap-free (no NaN/inf).
    X     : (T, k) exogenous regressors or None. With d > 0, X with T rows is taken as
            levels and differenced too (as statsmodels / R arima do with xreg); X with
            T-d rows is taken as already aligned with the differenced series.
    p, q  : ARMA orders.  d : differencing order.  dist : "normal" | "t".
    start : optional initial parameter vector (packing = result.names); two-step
            estimates make a good start when the MA+GARCH+nu surface is multimodal.
    polish      : Nelder-Mead refinement after L-BFGS-B (helps MA terms; off for MC loops).
    compute_cov : QMLE sandwich covariance (off for MC loops that need only point estimates).
    init  : pre-sample variance rule, "backcast" (arch) or "sample" (FCP 1996 / fGarch).
            Only affects the first few sigma_t^2; estimates differ in the 4th decimal.
    xreg  : "regression" (default) -- regression with ARMA errors: y_t - x_t'gamma follows the
                                      ARMA (statsmodels SARIMAX exog, rugarch external.regressors,
                                      R arima(xreg=)).
            "armax"                -- x_t enters the mean equation next to the AR terms (the
                                      v2.x default).  Same model as "regression" when p = 0.
    fixed : {name: value} held fixed instead of estimated (rugarch fixed.pars, statsmodels
            fix_params); names as in result.names.  fixed={"const": 0.0} is rugarch
            include.mean=FALSE / statsmodels trend="n".  Fixing every parameter evaluates
            the model without estimating it (rugarch ugarchfilter), e.g. on new data.
    """
    if dist not in ("normal", "t"):
        raise ValueError(f"dist must be 'normal' or 't', got {dist!r}")
    if xreg not in ("armax", "regression"):
        raise ValueError(f"xreg must be 'armax' or 'regression', got {xreg!r}")
    y = np.asarray(y, float).ravel()
    if X is not None:
        X = np.asarray(X, float)
        if X.ndim == 1:
            X = X[:, None]
    _check_finite(y, X)
    y_levels_tail = X_levels_tail = None
    if d > 0:
        y_levels_tail = y[-d:].copy()
        if X is not None and X.shape[0] == len(y):
            X_levels_tail = X[-d:].copy()
            X = np.diff(X, n=d, axis=0)
        y = np.diff(y, n=d)
    if X is not None and X.shape[0] != len(y):
        raise ValueError(f"X has {X.shape[0]} rows, series (after differencing) has {len(y)}")
    k = 0 if X is None else X.shape[1]
    names = _names(p, q, k, dist)
    fixed = {} if fixed is None else {n: float(v) for n, v in fixed.items()}
    unknown = sorted(set(fixed) - set(names))
    if unknown:
        raise ValueError(f"unknown fixed parameter(s) {unknown}; this model's parameters are {names}")
    _check_design(y, X, p, fixed)
    sd = y.std()
    if not (0.05 <= sd <= 50):
        warnings.warn(f"std(series)={sd:.3g}; rescale toward 0.1-10 for a well-behaved optimiser.")

    if start is None:
        start = _hr_start(y, X, p, q, dist, xreg)
    start = np.asarray(start, float).ravel()
    if len(start) != len(names):
        raise ValueError(f"start has {len(start)} values; this model has {len(names)}: {names}")
    fidx = np.array([names.index(n) for n in fixed], int)
    free = np.setdiff1d(np.arange(len(names)), fidx)
    tmpl = start.copy()
    tmpl[fidx] = [fixed[n] for n in fixed]

    def full(v):
        t = tmpl.copy()
        t[free] = v
        return t

    bounds_all = ([(None, None)] * (1 + p + q + k)
                  + [(1e-10, None), (0.0, 1.0), (0.0, 1.0)]
                  + ([(2.05, _NU_MAX)] if dist == "t" else []))
    bounds = [bounds_all[i] for i in free]
    obj = lambda v: _neg_loglik(full(v), y, X, p, q, dist, init, xreg)
    ia = 1 + p + q + k + 1
    if free.size == 0:                                  # everything fixed: filter only
        fun = obj(np.empty(0))
        if fun >= 1e10:
            raise ValueError("the fixed parameters give an invalid likelihood "
                             "(need omega > 0, alpha, beta >= 0, alpha+beta < cap, nu > 2)")
        theta, converged = tmpl, True
    else:
        r = optimize.minimize(obj, tmpl[free], method="L-BFGS-B", bounds=bounds,
                              options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-8})
        best = r
        if polish:
            r2 = optimize.minimize(obj, r.x, method="Nelder-Mead",
                                   options={"maxiter": 5000, "fatol": 1e-10, "xatol": 1e-8})
            if r2.fun < r.fun:
                best = r2
        # alpha+beta < cap is a 1e10 wall for L-BFGS-B / Nelder-Mead, and both can stall against it
        # (DEM/GBP ARMA(1,1)-t: 0.24 loglik short of the constrained max).  SLSQP treats it as a
        # linear constraint; its answer is kept only if it is feasible and strictly better.
        garch_free = ia in free or ia + 1 in free
        if garch_free and (polish or full(best.x)[ia] + full(best.x)[ia + 1] > _PERSIST_CAP - 1e-3):
            con = {"type": "ineq", "fun": lambda v: _PERSIST_CAP - 1e-9 - full(v)[ia] - full(v)[ia + 1]}
            r3 = optimize.minimize(lambda v: _neg_loglik_raw(full(v), y, X, p, q, dist, init, xreg), best.x,
                                   method="SLSQP", bounds=bounds, constraints=[con],
                                   options={"maxiter": 1000, "ftol": 1e-12})
            f3 = obj(r3.x)
            if f3 < best.fun - 1e-9:
                r3.fun = f3
                best = r3
        theta, fun, converged = full(best.x), best.fun, bool(r.success)
    eps, sig2, _ = _filter(theta, y, X, p, q, dist, init, xreg)
    cov = cov_h = cov_o = None
    if compute_cov and free.size:
        cov, cov_h, cov_o = _sandwich_cov(theta, y, X, p, q, dist, init, xreg, free)
    return JointResult(theta, names, cov, -fun, p, q, d, dist, y, X,
                       eps, sig2, y_levels_tail, X_levels_tail, converged=converged,
                       cov_hessian=cov_h, cov_opg=cov_o, init=init, xreg=xreg, fixed=fixed)


# -----------------------------------------------------------------------------
# forecasting
# -----------------------------------------------------------------------------
def _psi_weights(phi, theta, h):
    """MA(inf) weights psi_0..psi_{h-1} of ARMA(p,q): psi_j = sum_i phi_i psi_{j-i} + theta_j."""
    p, q = len(phi), len(theta)
    psi = np.zeros(h)
    psi[0] = 1.0
    for j in range(1, h):
        s = theta[j - 1] if j <= q else 0.0
        for i in range(1, min(p, j) + 1):
            s += phi[i - 1] * psi[j - i]
        psi[j] = s
    return psi


def _integrate(fc_diff, y_tail, d):
    """Undo d-fold differencing. fc_diff: (..., h) forecasts of the d-th difference;
    y_tail: the last >= d LEVEL observations."""
    out = np.asarray(fc_diff, float)
    diffs = [np.asarray(y_tail, float)]
    for _ in range(1, d):
        diffs.append(np.diff(diffs[-1]))
    for kk in range(d - 1, -1, -1):          # diffs[kk][-1] = last value of the kk-th difference
        out = diffs[kk][-1] + np.cumsum(out, axis=-1)
    return out


def forecast(res, horizon, X_future=None, n_sims=10_000, seed=None):
    """h-step forecasts from a JointResult.

    Returns dict with
      mean                : E[y_{T+h}]          (levels if d > 0)
      variance            : Var(y_{T+h} | F_T)  (psi-accumulated; integrated psi if d > 0)
      variance_innovation : E[sigma^2_{T+h}]    (GARCH path, differenced scale)
      sims                : (n_sims, h) simulated paths (levels if d > 0), or None
      quantiles(a)        : callable -> h-vector of simulated a-quantiles
      mean_diff / variance_diff : the differenced-scale versions when d > 0
    Multi-step quantiles under t errors are NOT analytic -- use the simulated ones.
    """
    p, q, d, dist = res.p, res.q, res.d, res.dist
    y, X = res.y, res.X
    k = 0 if X is None else X.shape[1]
    c, phi, th, gamma, omega, alpha, beta, nu = _unpack(res.params, p, q, k, dist)
    T, h = len(y), int(horizon)

    if k:
        if X_future is None:
            raise ValueError("model has exogenous regressors: pass X_future (h rows)")
        Xf = np.asarray(X_future, float)
        if Xf.ndim == 1:
            Xf = Xf[:, None]
        if res.X_levels_tail is not None:          # X was given in levels -> difference it
            Xf = np.diff(np.vstack([res.X_levels_tail, Xf]), n=d, axis=0)
        if Xf.shape != (h, k):
            raise ValueError(f"X_future must be ({h}, {k}) after alignment, got {Xf.shape}")
        xg = Xf @ gamma
    else:
        xg = np.zeros(h)
    # ARMAX: x'gamma inside the ARMA recursion.  Regression errors: the recursion runs on
    # z = y - x'gamma and x'gamma is added back afterwards.
    regression = k and res.xreg == "regression"
    xg_in, xg_out = (np.zeros(h), xg) if regression else (xg, np.zeros(h))
    if regression:
        y = y - X @ gamma

    eps_full = np.zeros(T)
    eps_full[p:] = res.eps

    # --- analytic conditional mean path (future eps = 0) ---
    y_ext, e_ext = list(y), list(eps_full)
    mean = np.empty(h)
    for i in range(h):
        t = T + i
        m = c + xg_in[i]
        for j in range(p):
            m += phi[j] * y_ext[t - 1 - j]
        for j in range(q):
            m += th[j] * e_ext[t - 1 - j]
        mean[i] = m + xg_out[i]
        y_ext.append(m)
        e_ext.append(0.0)

    # --- GARCH innovation-variance path ---
    vinn = np.empty(h)
    vinn[0] = omega + alpha * res.eps[-1] ** 2 + beta * res.sig2[-1]
    for i in range(1, h):
        vinn[i] = omega + (alpha + beta) * vinn[i - 1]

    # --- Var(y_{T+h}) = sum_j psi_j^2 E[sigma^2_{T+h-j}], psi integrated d times for levels ---
    psi = _psi_weights(phi, th, h)
    var_diff = np.array([np.sum(psi[:i + 1] ** 2 * vinn[i::-1]) for i in range(h)])
    psi_lvl = psi.copy()
    for _ in range(d):
        psi_lvl = np.cumsum(psi_lvl)
    var_lvl = np.array([np.sum(psi_lvl[:i + 1] ** 2 * vinn[i::-1]) for i in range(h)])

    # --- simulation ---
    sims = None
    if n_sims and n_sims > 0:
        rng = np.random.default_rng(seed)
        if dist == "t":
            z = rng.standard_t(nu, size=(n_sims, h)) * np.sqrt((nu - 2) / nu)
        else:
            z = rng.standard_normal(size=(n_sims, h))
        y_lag = [np.full(n_sims, y[T - 1 - j]) for j in range(p)]
        e_lag = [np.full(n_sims, eps_full[T - 1 - j]) for j in range(q)]
        e_prev = np.full(n_sims, res.eps[-1])
        s2_prev = np.full(n_sims, res.sig2[-1])
        sims = np.empty((n_sims, h))
        for i in range(h):
            s2 = omega + alpha * e_prev ** 2 + beta * s2_prev
            e = np.sqrt(s2) * z[:, i]
            m = c + xg_in[i]
            for j in range(p):
                m = m + phi[j] * y_lag[j]
            for j in range(q):
                m = m + th[j] * e_lag[j]
            yi = m + e
            sims[:, i] = yi + xg_out[i]
            if p:
                y_lag = [yi] + y_lag[:-1]
            if q:
                e_lag = [e] + e_lag[:-1]
            e_prev, s2_prev = e, s2

    out = {"variance_innovation": vinn, "psi": psi}
    if d > 0:
        out["mean_diff"], out["variance_diff"] = mean, var_diff
        out["mean"] = _integrate(mean, res.y_levels_tail, d)
        out["variance"] = var_lvl
        out["sims"] = _integrate(sims, res.y_levels_tail, d) if sims is not None else None
    else:
        out["mean"], out["variance"], out["sims"] = mean, var_diff, sims
    s_ = out["sims"]
    out["quantiles"] = (lambda a: np.quantile(s_, a, axis=0)) if s_ is not None else None
    return out


# -----------------------------------------------------------------------------
# simulation helpers (used by the self-test; handy for Monte Carlo)
# -----------------------------------------------------------------------------
def simulate(T, c, phi, theta, omega, alpha, beta, dist="t", nu=6.0, X=None, gamma=None,
             burn=500, seed=None, xreg="regression"):
    """Simulate regression with ARMA(p,q)-GARCH(1,1)-{normal,t} errors (xreg="armax": an
    ARMAX mean). Returns y (T,), eps, sig2 (post burn-in)."""
    rng = np.random.default_rng(seed)
    phi, theta = np.atleast_1d(phi), np.atleast_1d(theta)
    p, q = len(phi), len(theta)
    n = T + burn
    if dist == "t":
        z = rng.standard_t(nu, n) * np.sqrt((nu - 2) / nu)
    else:
        z = rng.standard_normal(n)
    xg = np.zeros(n)
    if X is not None:
        xg[burn:] = np.asarray(X) @ np.asarray(gamma)
    xg_in = xg if xreg == "armax" else np.zeros(n)
    eps = np.zeros(n); sig2 = np.zeros(n); y = np.zeros(n)   # y holds the ARMA part
    sig2[0] = omega / (1 - alpha - beta)
    eps[0] = np.sqrt(sig2[0]) * z[0]
    for t in range(1, n):
        sig2[t] = omega + alpha * eps[t - 1] ** 2 + beta * sig2[t - 1]
        eps[t] = np.sqrt(sig2[t]) * z[t]
        m = c + xg_in[t]
        for i in range(p):
            m += phi[i] * y[t - 1 - i]
        for j in range(q):
            m += theta[j] * eps[t - 1 - j]
        y[t] = m + eps[t]
    if xreg == "regression":
        y = y + xg
    return y[burn:], eps[burn:], sig2[burn:]


# -----------------------------------------------------------------------------
# diagnostics and bootstrap (small samples: see verification/README.md, "Small samples")
# -----------------------------------------------------------------------------
def arch_lm_test(resid, lags=12):
    """Engle's (1982) ARCH-LM test: regress e_t^2 on a constant and e_{t-1}^2 .. e_{t-lags}^2;
    LM = n R^2 ~ chi2(lags) under no ARCH.  Returns (statistic, p_value).

    Use it on the residuals of a fit with fixed={"alpha": 0, "beta": 0} to decide whether GARCH
    is warranted at all (essential below ~250 obs), or on res.std_resid after a GARCH fit to
    check for remaining ARCH."""
    e2 = np.asarray(resid, float) ** 2
    Z = np.column_stack([np.ones(len(e2) - lags), _lagmat(e2, lags, lags)])
    yv = e2[lags:]
    _, u = _ols(Z, yv)
    r2 = 1.0 - (u @ u) / np.sum((yv - yv.mean()) ** 2)
    stat = len(yv) * r2
    return float(stat), float(stats.chi2.sf(stat, lags))


def ljung_box(x, lags=12, df_adjust=0):
    """Ljung-Box Q on a series: Q = n(n+2) sum_k rho_k^2 / (n-k) ~ chi2(lags - df_adjust).
    x = res.std_resid tests for remaining autocorrelation (set df_adjust = p + q),
    x = res.std_resid**2 for remaining ARCH.  Returns (statistic, p_value)."""
    x = np.asarray(x, float)
    x = x - x.mean()
    n = len(x)
    acov = np.array([x[k:] @ x[:n - k] for k in range(lags + 1)]) / n
    rho = acov[1:] / acov[0]
    q = n * (n + 2) * np.sum(rho ** 2 / (n - np.arange(1, lags + 1)))
    return float(q), float(stats.chi2.sf(q, max(lags - df_adjust, 1)))


def bootstrap(res, n_boot=199, method="residual", seed=None, burn=200):
    """Bootstrap SEs and 95% percentile CIs by re-estimating on series simulated from the fit.

    method : "residual"   -- shocks resampled from the standardised residuals (no distributional
                             assumption; Pascual, Romo & Ruiz 2006)
             "parametric" -- shocks drawn from the fitted normal / t
    Fixed parameters stay fixed; with d > 0 everything happens on the differenced scale.
    Returns dict(params (n_ok, k), se (k,), ci (2, k), names, n_ok).

    Small samples (T = 120-180): the Wald CIs for omega/alpha/beta cover only 50-75%; these
    percentile CIs cover 85-95% for omega and beta and 75-85% for alpha.  Keep the Wald CIs for
    the mean parameters -- the percentile bootstrap under-covers there (0.81-0.91) because it
    inherits the AR small-sample bias.  See verification/README.md, "Small samples"."""
    rng = np.random.default_rng(seed)
    p, q, dist, X = res.p, res.q, res.dist, res.X
    k0 = 0 if X is None else X.shape[1]
    c, phi, th, gamma, omega, alpha, beta, nu = _unpack(res.params, p, q, k0, dist)
    T = len(res.y)
    n = T + burn
    if method == "residual":
        z = res.std_resid
        z = (z - z.mean()) / z.std()
        Z = z[rng.integers(0, len(z), size=(n_boot, n))]
    elif dist == "t":
        Z = rng.standard_t(nu, size=(n_boot, n)) * np.sqrt((nu - 2) / nu)
    else:
        Z = rng.standard_normal(size=(n_boot, n))
    xg = np.zeros(n)
    if k0:
        xg[burn:] = X @ gamma
    xg_in = xg if res.xreg == "armax" else np.zeros(n)
    eps = np.zeros((n_boot, n)); s2 = np.zeros((n_boot, n)); y = np.zeros((n_boot, n))
    s2[:, 0] = res.uncond_var if res.persistence < 0.999 else float(np.mean(res.sig2))
    eps[:, 0] = np.sqrt(s2[:, 0]) * Z[:, 0]
    for t in range(1, n):                       # vectorised over replicates
        s2[:, t] = omega + alpha * eps[:, t - 1] ** 2 + beta * s2[:, t - 1]
        eps[:, t] = np.sqrt(s2[:, t]) * Z[:, t]
        m = c + xg_in[t]
        for i in range(p):
            m = m + phi[i] * y[:, t - 1 - i]
        for j in range(q):
            m = m + th[j] * eps[:, t - 1 - j]
        y[:, t] = m + eps[:, t]
    y = y[:, burn:]
    if k0 and res.xreg == "regression":
        y = y + X @ gamma
    out = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for b in range(n_boot):
            try:
                rb = fit_joint(y[b], X=X, p=p, q=q, d=0, dist=dist, start=res.params, compute_cov=False,
                               init=res.init, xreg=res.xreg, fixed=res.fixed)
                out.append(rb.params)
            except (ValueError, FloatingPointError, np.linalg.LinAlgError):
                pass
    P = np.array(out)
    return dict(params=P, se=P.std(0, ddof=1), ci=np.percentile(P, [2.5, 97.5], axis=0),
                names=res.names, n_ok=len(P))


def _crps(sample, x):
    """CRPS of a predictive sample against the realisation x: E|S - x| - E|S - S'|/2.
    A proper score for the whole predictive distribution (lower is better; equals the absolute
    error for a point forecast)."""
    s = np.sort(np.asarray(sample, float))
    n = len(s)
    return float(np.mean(np.abs(s - x)) - np.sum((2 * np.arange(1, n + 1) - n - 1) * s) / n ** 2)


def backtest(y, X=None, p=1, q=0, d=0, dist="normal", horizon=12, start=None, step=1, levels=(0.9,),
             n_sims=2000, seed=0, **fit_kw):
    """Rolling-origin out-of-sample evaluation of one specification.

    For each origin t0 = start, start + step, ... (t0 + horizon <= T): fit on y[:t0] (and X[:t0]),
    forecast `horizon` steps with the realised X[t0:t0 + horizon] -- the "macro path given"
    setting -- and score against the realised y.  fit_kw go to fit_joint (xreg, fixed, init, ...).
    Returns dict(h, rmse, mae, crps, coverage90 (one per level), n_origins, origins): arrays by horizon.

    Use it to decide between a levels model (d=0) and a differences model (d=1), and between GARCH
    and fixed={"alpha": 0, "beta": 0}, by CRPS and coverage at the horizons you care about -- AIC
    and BIC cannot compare d=0 with d=1 (different data).  See verification/README.md,
    "Multi-step, macro-conditioned forecasts"."""
    y = np.asarray(y, float).ravel()
    T = len(y)
    if X is not None:
        X = np.asarray(X, float)
        if X.ndim == 1:
            X = X[:, None]
    if start is None:
        start = max(T // 2, 100 + p + d)
    origins = list(range(start, T - horizon + 1, step))
    if not origins:
        raise ValueError(f"no forecast origins: need start + horizon <= T (T={T}, start={start}, horizon={horizon})")
    err = np.empty((len(origins), horizon))
    crp = np.empty((len(origins), horizon))
    cov = {lv: np.empty((len(origins), horizon)) for lv in levels}
    prev = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i, t0 in enumerate(origins):
            r = fit_joint(y[:t0], X=None if X is None else X[:t0], p=p, q=q, d=d, dist=dist,
                          start=prev, compute_cov=False, **fit_kw)
            prev = r.params
            fc = forecast(r, horizon, X_future=None if X is None else X[t0:t0 + horizon],
                          n_sims=n_sims, seed=seed + i)
            actual = y[t0:t0 + horizon]
            err[i] = fc["mean"] - actual
            for h in range(horizon):
                crp[i, h] = _crps(fc["sims"][:, h], actual[h])
                for lv in levels:
                    lo, hi = np.quantile(fc["sims"][:, h], [(1 - lv) / 2, 1 - (1 - lv) / 2])
                    cov[lv][i, h] = lo <= actual[h] <= hi
    out = dict(h=np.arange(1, horizon + 1), rmse=np.sqrt(np.mean(err ** 2, 0)), mae=np.mean(np.abs(err), 0),
               crps=np.mean(crp, 0), n_origins=len(origins), origins=np.array(origins))
    for lv in levels:
        out[f"coverage{int(round(lv * 100))}"] = np.mean(cov[lv], 0)
    return out


# -----------------------------------------------------------------------------
# published benchmark: Fiorentini, Calzolari & Panattoni (1996), J. Applied Econometrics
# 11:399-417, on the Bollerslev-Ghysels (1996, JBES 14:139-151) DEM/GBP returns;
# tabulated in McCullough & Renfro (1999) and Brooks, Burke & Persand (2001, IJF 17:45-56).
# Constant mean + GARCH(1,1), Gaussian, pre-sample eps^2 = sigma^2 = sample variance.
# -----------------------------------------------------------------------------
FCP_BENCHMARK = dict(
    coef=np.array([-0.00619041, 0.0107613, 0.153134, 0.805974]),      # mu, omega, alpha, beta
    se_hessian=np.array([0.00846212, 0.00285271, 0.0265228, 0.0335527]),
    se_robust=np.array([0.00918935, 0.00649319, 0.0535317, 0.0724614]),
    loglik=-1106.60788,
)


def fcp_benchmark(path="dem2gbp_benchmark.csv", verbose=True):
    """Re-run the FCP (1996) benchmark. `path` is a one-column text file of the 1974 DEM/GBP
    returns (shipped as dem2gbp_benchmark.csv; also fGarch::dem2gbp / rugarch::dmbp in R).
    Returns True when coefficients match to 1e-6, loglik to 1e-5 and SEs to 3 sig. digits."""
    y = np.loadtxt(path, comments="#")
    r = fit_joint(y, p=0, q=0, dist="normal", init="sample")
    b = FCP_BENCHMARK
    ok = (np.all(np.abs(r.params - b["coef"]) < 1e-6)
          and abs(r.loglik - b["loglik"]) < 1e-5
          and np.all(np.abs(r.se_hessian / b["se_hessian"] - 1) < 2e-3)
          and np.all(np.abs(r.se / b["se_robust"] - 1) < 2e-3))
    if verbose:
        print(f"  coef      mine={r.params}\n            FCP ={b['coef']}")
        print(f"  loglik    mine={r.loglik:.5f}  FCP={b['loglik']:.5f}")
        print(f"  se(hess)  mine={r.se_hessian}\n            FCP ={b['se_hessian']}")
        print(f"  se(robust)mine={r.se}\n            FCP ={b['se_robust']}")
    return ok


# -----------------------------------------------------------------------------
# self-test
# -----------------------------------------------------------------------------
def _selftest():
    import time
    np.set_printoptions(precision=4, suppress=True)
    ok_all = True

    def check(name, cond):
        nonlocal ok_all
        ok_all &= bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    # ---- 1. ARMAX(1,1)-GARCH(1,1)-t parameter recovery -------------------------
    print("\n1) regression with ARMA(1,1)-GARCH(1,1)-t errors: recovery, T=3000, sinusoid exog")
    T = 3000
    t_idx = np.arange(T)
    X = np.column_stack([np.sin(2 * np.pi * t_idx / 12), np.cos(2 * np.pi * t_idx / 12)])
    truth = dict(c=0.2, phi=[0.5], theta=[0.3], gamma=[1.0, -0.6],
                 omega=0.05, alpha=0.08, beta=0.88, nu=6.0)
    y, _, _ = simulate(T, truth["c"], truth["phi"], truth["theta"], truth["omega"],
                       truth["alpha"], truth["beta"], dist="t", nu=truth["nu"],
                       X=X, gamma=truth["gamma"], seed=11)
    t0 = time.time()
    res = fit_joint(y, X=X, p=1, q=1, dist="t")
    print(f"  fit time {time.time() - t0:.1f}s")
    print(res.summary())
    true_vec = np.r_[truth["c"], truth["phi"], truth["theta"], truth["gamma"],
                     truth["omega"], truth["alpha"], truth["beta"], truth["nu"]]
    zdev = (res.params - true_vec) / res.se
    print("  (est-true)/SE:", dict(zip(res.names, zdev.round(2))))
    check("all |est-true| < 3 SE", np.all(np.abs(zdev) < 3))
    check("MA invertible, AR stationary", res.invertible and res.stationary)

    # ---- 2. cross-check against arch on the nested AR(1) case --------------------
    print("\n2) cross-check vs arch: AR(1)-GARCH(1,1), normal and t")
    try:
        from arch import arch_model
        y2, _, _ = simulate(3000, 0.1, [0.6], [], 0.1, 0.10, 0.85, dist="t", nu=5.0, seed=2)
        for dist, nu_name in (("normal", None), ("t", "nu")):
            am = arch_model(y2, mean="ARX", lags=1, vol="GARCH", p=1, q=1, dist=dist,
                            rescale=False).fit(disp="off")
            mine = fit_joint(y2, p=1, q=0, dist=dist)
            a = am.params
            arch_vec = np.r_[a["Const"], a["y[1]"], a["omega"], a["alpha[1]"], a["beta[1]"],
                             ([a["nu"]] if dist == "t" else [])]
            diff = np.abs(mine.params - arch_vec)
            print(f"  {dist:>6}: mine={mine.params}  arch={arch_vec}  max|diff|={diff.max():.5f}"
                  f"  loglik mine={mine.loglik:.3f} arch={am.loglikelihood:.3f}")
            check(f"{dist}: params match arch to 1e-2 (init-condition noise only)", diff.max() < 1e-2)
            # psi-accumulated variance path vs arch fc.variance (arch's is Var(y_{T+h}))
            h = 5
            fa = am.forecast(horizon=h, reindex=False)
            fm = forecast(mine, horizon=h, n_sims=0)
            rel = np.abs(fm["variance"] - fa.variance.values[0]) / fa.variance.values[0]
            print(f"          Var(y_T+h) mine={fm['variance']} arch={fa.variance.values[0]}")
            check(f"{dist}: multi-step Var(y) within 2% of arch", rel.max() < 0.02)
            relm = np.abs(fm["mean"] - fa.mean.values[0]).max()
            check(f"{dist}: mean path matches arch (<1e-2)", relm < 1e-2)
    except ImportError:
        print("  arch not installed -- skipped")

    # ---- 3. ARIMA(1,1,1)-GARCH(1,1)-t: fit on levels with d=1, level forecasts ------
    print("\n3) ARIMA(1,1,1)-GARCH(1,1)-t on levels (d=1): recovery + level-forecast variance")
    w, _, _ = simulate(3000, 0.0, [0.4], [0.35], 0.05, 0.07, 0.90, dist="t", nu=7.0, seed=5)
    y_lvl = 100.0 + np.cumsum(w)
    res3 = fit_joint(y_lvl, p=1, q=1, d=1, dist="t")
    print(res3.summary())
    true3 = np.r_[0.0, 0.4, 0.35, 0.05, 0.07, 0.90, 7.0]
    z3 = (res3.params - true3) / res3.se
    print("  (est-true)/SE:", dict(zip(res3.names, z3.round(2))))
    check("ARIMA params within 3 SE", np.all(np.abs(z3) < 3))
    h = 10
    fc = forecast(res3, horizon=h, n_sims=60_000, seed=1)
    sim_mean, sim_var = fc["sims"].mean(0), fc["sims"].var(0)
    print(f"  level mean  analytic={fc['mean']}\n              simulated={sim_mean}")
    print(f"  level var   analytic={fc['variance']}\n              simulated={sim_var}")
    mc_se = np.sqrt(fc["variance"]) / np.sqrt(60_000)
    check("level mean: analytic vs simulated within 4 MC standard errors",
          np.all(np.abs(fc["mean"] - sim_mean) < 4 * mc_se))
    check("level variance: analytic vs simulated within 4%",
          np.all(np.abs(fc["variance"] / sim_var - 1) < 0.04))
    check("level variance grows with h (integrated process)", np.all(np.diff(fc["variance"]) > 0))
    check("first level forecast = last level + first diff forecast",
          abs(fc["mean"][0] - (y_lvl[-1] + fc["mean_diff"][0])) < 1e-12)
    q05, q95 = fc["quantiles"](0.05), fc["quantiles"](0.95)
    check("quantiles ordered and bracket the mean", np.all(q05 < fc["mean"]) and np.all(fc["mean"] < q95))

    # ---- 4. d=2 integration sanity (pure algebra) --------------------------------
    print("\n4) d=2 integration algebra")
    lvl = np.array([1.0, 3.0, 7.0])             # y_T-2, y_T-1, y_T ; w1_T = 4, w2_T = 2
    fc2 = np.array([1.0, 1.0, 1.0])             # forecasts of the 2nd difference
    want = np.array([12.0, 18.0, 25.0])         # w1: 5,6,7 ; y: 12,18,25
    got = _integrate(fc2, lvl, 2)
    check("d=2 undifferencing exact", np.allclose(got, want))

    # ---- 5. published FCP (1996) benchmark, if the data file is next to the script ----
    print("\n5) Fiorentini-Calzolari-Panattoni (1996) GARCH(1,1) benchmark on DEM/GBP")
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    cand = [os.path.join(here, "dem2gbp_benchmark.csv"), "dem2gbp_benchmark.csv"]
    found = next((c for c in cand if os.path.exists(c)), None)
    if found is None:
        print("  dem2gbp_benchmark.csv not found -- skipped (ship it next to this script)")
    else:
        check("reproduces FCP coefficients (1e-6), loglik (1e-5), SEs (3 sig. digits)",
              fcp_benchmark(found))

    # ---- 6. SEs are scale-invariant (v2.1: omega SE = 0 at std 0.2, TypeError at std 0.02) ----
    print("\n6) scale invariance of estimates and SEs (y vs 0.1*y vs 0.01*y)")
    y6, _, _ = simulate(2500, 0.05, [0.3], [0.2], 0.02, 0.06, 0.93, dist="t", nu=6.0, seed=3)
    base = fit_joint(y6, p=1, q=1, dist="t")
    for s in (0.1, 0.01):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            rs = fit_joint(s * y6, p=1, q=1, dist="t")
        un = np.r_[1 / s, 1, 1, 1 / s ** 2, 1, 1, 1]
        check(f"scale {s}: params rescale exactly (1e-4 rel)",
              np.allclose(rs.params * un, base.params, rtol=1e-4, atol=1e-6))
        check(f"scale {s}: robust and Hessian SEs rescale (1% rel)",
              np.allclose(rs.se * un, base.se, rtol=1e-2) and np.allclose(rs.se_hessian * un, base.se_hessian, rtol=1e-2))

    # ---- 7. persistence cap: reach the constrained max and say so ----------------------
    if found is not None:
        print("\n7) DEM/GBP ARMA(1,1)-GARCH-t: alpha+beta pinned at the cap")
        r7 = fit_joint(np.loadtxt(found, comments="#"), p=1, q=1, dist="t")
        print(f"  loglik={r7.loglik:.4f}  persistence={r7.persistence:.6f}")
        check("reaches the constrained max (v2.1 stalled at -987.03)", r7.loglik > -986.80)
        check("flags the cap in summary()", r7.at_persistence_cap and "stationarity cap" in r7.summary())

    # ---- 8. input validation: clear errors instead of LinAlgError / silent nonsense ----
    print("\n8) input validation")
    y8, _, _ = simulate(1000, 0.1, [0.5], [0.3], 0.05, 0.08, 0.88, dist="t", seed=8)
    x8 = np.random.default_rng(8).normal(size=1000)

    def raises(fn, exc=ValueError):
        try:
            fn()
        except exc:
            return True
        except Exception:
            return False
        return False
    y_nan = y8.copy(); y_nan[10] = np.nan
    check("NaN in y -> ValueError", raises(lambda: fit_joint(y_nan)))
    check("constant series -> ValueError", raises(lambda: fit_joint(np.full(500, 3.0))))
    check("X column of ones -> ValueError (duplicates const)",
          raises(lambda: fit_joint(y8, X=np.column_stack([np.ones(1000), x8]))))
    check("X collinear columns -> ValueError", raises(lambda: fit_joint(y8, X=np.column_stack([x8, 2 * x8]))))
    check("time trend in X with d=1 (constant after differencing) -> ValueError",
          raises(lambda: fit_joint(np.cumsum(y8), X=np.arange(1000.0), d=1)))
    check("start of the wrong length -> ValueError", raises(lambda: fit_joint(y8, start=np.zeros(3))))
    check("unknown fixed name -> ValueError", raises(lambda: fit_joint(y8, fixed={"mu": 0.0})))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        fit_joint(y8[:80], p=1, q=0, compute_cov=False)
    check("fewer than 100 obs -> warning", any("unreliable this short" in str(x.message) for x in w))

    # ---- 9. xreg="regression": regression with ARMA-GARCH errors (rugarch/statsmodels xreg) ----
    print("\n9) xreg='regression' (regression with ARMA errors)")
    T9 = 3000
    X9 = np.column_stack([np.sin(2 * np.pi * np.arange(T9 + 10) / 12),
                          np.random.default_rng(9).normal(size=T9 + 10)])
    y9, _, _ = simulate(T9, 0.2, [0.6], [0.3], 0.05, 0.08, 0.88, dist="t", nu=6.0,
                        X=X9[:T9], gamma=[1.0, -0.5], seed=9, xreg="regression")
    r9 = fit_joint(y9, X=X9[:T9], p=1, q=1, xreg="regression")
    print(r9.summary().splitlines()[0])
    z9 = (r9.params - np.r_[0.2, 0.6, 0.3, 1.0, -0.5, 0.05, 0.08, 0.88, 6.0]) / r9.se
    check("regression-mode params within 3 SE of truth", np.all(np.abs(z9) < 3))
    c9, ph9, th9, g9 = r9["const"], r9["phi1"], r9["theta1"], r9.params[3:5]
    Xaug = np.column_stack([X9[:T9], np.vstack([np.zeros((1, 2)), X9[:T9 - 1]])])
    ll_armax = _loglik_obs(np.r_[c9, ph9, th9, g9, -ph9 * g9, r9.params[5:]], y9, Xaug, 1, 1, "t", "backcast", "armax").sum()
    check("== ARMAX with [x_t, x_t-1] and gamma_lag = -phi*gamma (loglik to 1e-8)",
          abs(ll_armax - r9.loglik) < 1e-8)
    fc9 = forecast(r9, 10, X_future=X9[T9:], n_sims=40_000, seed=2)
    se9 = np.sqrt(fc9["variance"] / 40_000)
    check("regression-mode forecast mean == simulated mean (4 MC SE)",
          np.all(np.abs(fc9["mean"] - fc9["sims"].mean(0)) < 4 * se9))
    ra = fit_joint(y9, X=X9[:T9], p=0, q=1, compute_cov=False, xreg="armax")
    rr = fit_joint(y9, X=X9[:T9], p=0, q=1, compute_cov=False, xreg="regression")
    check("p = 0: 'armax' and 'regression' are the same model", abs(ra.loglik - rr.loglik) < 1e-6)

    # ---- 10. fixed parameters: include.mean=FALSE and filter-only ------------------------
    print("\n10) fixed parameters")
    r10 = fit_joint(y9, X=X9[:T9], xreg="regression", fixed={"const": 0.0})
    check("fixed const stays 0, its SE is NaN, others estimated",
          r10["const"] == 0.0 and np.isnan(r10.se[0]) and np.all(np.isfinite(r10.se[1:])))
    check("AIC counts only estimated parameters", abs(r10.aic - (-2 * r10.loglik + 2 * 8)) < 1e-9)
    rf = fit_joint(y9, X=X9[:T9], xreg="regression", fixed=dict(zip(r9.names, r9.params)))
    check("all fixed == filter: same loglik, eps, sigma as the fit",
          abs(rf.loglik - r9.loglik) < 1e-9 and np.allclose(rf.eps, r9.eps) and np.allclose(rf.sig2, r9.sig2))

    # ---- 11. specification diagnostics ----------------------------------------------------
    print("\n11) specification diagnostics")
    iid = np.random.default_rng(11).normal(size=2000)
    r11 = fit_joint(iid, p=1, q=1, dist="t")
    check("ARMA(1,1) on white noise -> AR/MA cancellation flagged", r11.arma_cancellation)
    check("normal data with dist='t' -> nu at bound flagged", r11.nu_at_bound)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        nus = [fit_joint(simulate(120, 0.1, [0.5], [], 0.15, 0.10, 0.75, dist="normal", seed=300 + i)[0],
                         p=1, q=0, dist="t", compute_cov=False)["nu"] for i in range(30)]
    check(f"nu never escapes its upper bound in 30 short-sample t fits (max {max(nus):.0f}; v3.0 reached 1e6)",
          max(nus) <= _NU_MAX + 1e-9)
    r11b = fit_joint(np.cumsum(y8), p=1, q=1, d=2, compute_cov=False)
    check("over-differenced (d=2 on I(1)) -> MA near unit root flagged",
          r11b.ma_near_unit_root or not r11b.invertible)

    # ---- 12. small-sample helpers: ARCH-LM, Ljung-Box, bootstrap --------------------------
    print("\n12) arch_lm_test / ljung_box / bootstrap")
    rng12 = np.random.default_rng(12)
    rej = np.mean([arch_lm_test(rng12.normal(size=300), 12)[1] < 0.05 for _ in range(200)])
    check(f"ARCH-LM size under no ARCH: 5%-level rejection rate {rej:.3f} in [0.02, 0.09]", 0.02 <= rej <= 0.09)
    sims = [simulate(1000, 0.0, [], [], 0.15, 0.10, 0.75, dist="normal", seed=200 + i)[1] for i in range(20)]
    power = np.mean([arch_lm_test(e, 12)[1] < 0.05 for e in sims])
    check(f"ARCH-LM power on GARCH(.10, .75) at T=1000: {power:.2f} >= 0.8", power >= 0.8)
    try:
        from statsmodels.stats.diagnostic import het_arch
        sm_lm = het_arch(sims[0], nlags=12)
        check("ARCH-LM statistic and p-value equal statsmodels het_arch (1e-8)",
              np.allclose(arch_lm_test(sims[0], 12), sm_lm[:2], atol=1e-8))
    except ImportError:
        print("  statsmodels not installed -- het_arch cross-check skipped")
    wn = rng12.normal(size=1000)
    check("Ljung-Box on white noise: p > 0.01", ljung_box(wn, 12)[1] > 0.01)
    check("Ljung-Box on an AR(1): p < 0.01", ljung_box(simulate(500, 0.0, [0.5], [], 1.0, 0.0, 0.0, "normal", seed=3)[0], 12)[1] < 0.01)
    y12, _, _ = simulate(400, 0.1, [0.5], [], 0.15, 0.10, 0.75, dist="normal", seed=120)
    r12 = fit_joint(y12, p=1, q=0, dist="normal")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bs = bootstrap(r12, n_boot=49, seed=0)
    check("bootstrap: 49/49 refits succeed, shapes right",
          bs["n_ok"] == 49 and bs["params"].shape == (49, r12.k) and bs["ci"].shape == (2, r12.k))
    ratio = bs["se"][:2] / r12.se[:2]
    check("bootstrap SEs of const, phi1 within a factor 2 of the robust SEs", np.all((0.5 < ratio) & (ratio < 2.0)))

    # ---- 13. backtest: rolling-origin out-of-sample scores ---------------------------------
    print("\n13) backtest (rolling origin, macro path given)")
    X13 = np.column_stack([np.sin(2 * np.pi * np.arange(400) / 12), np.cos(2 * np.pi * np.arange(400) / 12)])
    y13, _, _ = simulate(400, 0.1, [0.5], [], 0.15, 0.10, 0.75, dist="normal", X=X13, gamma=[1.0, -0.5], seed=13)
    bt = backtest(y13, X=X13, p=1, q=0, d=0, horizon=6, start=200, step=2, n_sims=500)
    check("levels backtest: 98 origins (200..394 step 2), per-horizon arrays, all finite",
          bt["n_origins"] == 98 and bt["crps"].shape == (6,) and np.all(np.isfinite(bt["rmse"])) and np.all(np.isfinite(bt["crps"])))
    check(f"90% interval coverage averages {bt['coverage90'].mean():.2f} (nominal 0.90, within [0.75, 1.0])",
          0.75 <= bt["coverage90"].mean() <= 1.0)
    check("CRPS <= RMSE at every horizon (CRPS of a calibrated sample is below its absolute error scale)",
          np.all(bt["crps"] <= bt["rmse"]))
    bt1 = backtest(100 + np.cumsum(y13), X=np.cumsum(X13, axis=0), p=1, q=0, d=1, horizon=3, start=350, step=10, n_sims=200)
    check("differences backtest (d=1, X in levels) runs: 5 origins, shape (3,)", bt1["n_origins"] == 5 and bt1["crps"].shape == (3,))
    check("backtest with start + horizon > T -> ValueError", raises(lambda: backtest(y13, p=1, q=0, horizon=12, start=395)))

    print(f"\nOVERALL: {'ALL PASS' if ok_all else 'SOME FAILURES'}")
    return ok_all


if __name__ == "__main__":
    _selftest()
