"""
joint_armax_garch.py  --  joint MLE of ARIMAX(p,d,q) mean + GARCH(1,1) variance
================================================================================
Pure numpy/scipy. No `arch`, no statsmodels needed (they are only used,
optionally, in the self-test for cross-checking).

Model (on w_t = (1-L)^d y_t, the d-th difference of the level series):

    mean:      w_t = c + sum_i phi_i w_{t-i} + sum_j theta_j eps_{t-j} + x_t' gamma + eps_t
    variance:  eps_t = sigma_t z_t,   sigma_t^2 = omega + alpha eps_{t-1}^2 + beta sigma_{t-1}^2
    errors:    z_t ~ N(0,1)  or  standardized Student-t(nu)  (unit variance)

Everything -- mean coefficients, GARCH coefficients, nu -- is estimated in ONE
likelihood, so the mean estimates are implicitly GLS-weighted by 1/sigma_t^2
(this moves point estimates, not just standard errors, relative to two-step).
Standard errors are QMLE sandwich (Bollerslev-Wooldridge): H^{-1} (S'S) H^{-1}.

ARIMA + GARCH:  pass the LEVEL series with d>0.  The script differences y (and
X, if X is given in levels), fits the ARMAX-GARCH on the differences, and
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

    # diagnostics: Ljung-Box on res.std_resid and res.std_resid**2 (statsmodels)

Run `python joint_armax_garch.py` for the self-test (parameter recovery,
arch cross-check if arch is installed, ARIMA-GARCH level-forecast check, and --
if dem2gbp_benchmark.csv sits next to this file -- the published Fiorentini-
Calzolari-Panattoni (1996) GARCH(1,1) benchmark, which this code reproduces to
6 digits with init="sample").

Same conditional likelihood as R rugarch (rec.init='all' == init="sample"),
fGarch (init.rec="mci"), MATLAB arima+garch, Stata `arch ... ar() ma()`.
Python's `arch` is the same estimator but has no MA terms in the mean.

Conditioning: the likelihood is conditional on the first p observations of w;
pre-sample eps are 0; sigma^2 is initialised with an exponentially weighted
backcast of eps^2 (as arch does).  Rescale w to std ~ 0.1-10 before fitting.
"""
from __future__ import annotations

import warnings
import numpy as np
from scipy import optimize, signal, stats
from scipy.special import gammaln

__version__ = "2.1"


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
def _filter(theta, y, X, p, q, dist, init="backcast"):
    """Return eps (T-p,), sig2 (T-p,), nu for the conditioning sample t = p..T-1.

    init : pre-sample value v0 used for both eps_{t-1}^2 and sigma_{t-1}^2 at the first
           in-sample date, so sigma^2_first = omega + (alpha + beta) v0.
           "backcast" -- exponentially weighted (0.94) mean of the first 75 eps^2 (arch's rule)
           "sample"   -- plain mean of eps^2 over the sample (FCP 1996 / fGarch "mci" rule)
    """
    T = len(y)
    k = 0 if X is None else X.shape[1]
    c, phi, th, gamma, omega, alpha, beta, nu = _unpack(theta, p, q, k, dist)

    rhs = y[p:] - c
    for i in range(p):
        rhs = rhs - phi[i] * y[p - 1 - i: T - 1 - i]
    if k:
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


def _loglik_obs(theta, y, X, p, q, dist, init="backcast"):
    eps, sig2, nu = _filter(theta, y, X, p, q, dist, init)
    if not np.all(np.isfinite(sig2)) or np.any(sig2 <= 0):
        return None
    z2 = eps ** 2 / sig2
    if dist == "normal":
        return -0.5 * (np.log(2 * np.pi) + np.log(sig2) + z2)
    return (gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(np.pi * (nu - 2))
            - 0.5 * np.log(sig2) - (nu + 1) / 2 * np.log1p(z2 / (nu - 2)))


def _neg_loglik(theta, y, X, p, q, dist, init="backcast"):
    k = 0 if X is None else X.shape[1]
    _, _, _, _, omega, alpha, beta, nu = _unpack(theta, p, q, k, dist)
    if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 0.9999:
        return 1e10
    if dist == "t" and nu <= 2.0:
        return 1e10
    ll = _loglik_obs(theta, y, X, p, q, dist, init)
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


def _hr_start(y, X, p, q, dist):
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
def _num_scores(theta, y, X, p, q, dist, init="backcast", rel=1e-5):
    """Per-observation score matrix S (n, k) by central differences."""
    k = len(theta)
    base = _loglik_obs(theta, y, X, p, q, dist, init)
    S = np.empty((len(base), k))
    for j in range(k):
        h = rel * max(1.0, abs(theta[j]))
        tp, tm = theta.copy(), theta.copy()
        tp[j] += h; tm[j] -= h
        lp = _loglik_obs(tp, y, X, p, q, dist, init)
        lm = _loglik_obs(tm, y, X, p, q, dist, init)
        S[:, j] = (lp - lm) / (2 * h)
    return S


def _num_hessian(f, theta, rel=1e-4):
    """Hessian of scalar f at theta by central differences of central gradients."""
    k = len(theta)

    def grad(t):
        g = np.empty(k)
        for j in range(k):
            h = rel * max(1.0, abs(t[j]))
            tp, tm = t.copy(), t.copy()
            tp[j] += h; tm[j] -= h
            g[j] = (f(tp) - f(tm)) / (2 * h)
        return g

    H = np.empty((k, k))
    for j in range(k):
        h = rel * max(1.0, abs(theta[j]))
        tp, tm = theta.copy(), theta.copy()
        tp[j] += h; tm[j] -= h
        H[:, j] = (grad(tp) - grad(tm)) / (2 * h)
    return 0.5 * (H + H.T)


def _sandwich_cov(theta, y, X, p, q, dist, init="backcast"):
    """Return (robust sandwich cov, Hessian-based cov, OPG-based cov)."""
    S = _num_scores(theta, y, X, p, q, dist, init)
    OPG = S.T @ S
    H = _num_hessian(lambda t: _neg_loglik(t, y, X, p, q, dist, init), theta)
    try:
        Hinv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        Hinv = np.linalg.pinv(H)
    return Hinv @ OPG @ Hinv, Hinv, np.linalg.pinv(OPG)


# -----------------------------------------------------------------------------
# result object
# -----------------------------------------------------------------------------
class JointResult:
    def __init__(self, params, names, cov, loglik, p, q, d, dist, y, X, eps, sig2,
                 y_levels_tail=None, X_levels_tail=None, converged=True,
                 cov_hessian=None, cov_opg=None, init="backcast"):
        self.params = np.asarray(params, float)
        self.names = list(names)
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
        return -2 * self.loglik + 2 * self.k

    @property
    def bic(self):
        return -2 * self.loglik + self.k * np.log(self.nobs)

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

    def summary(self):
        se = self.se
        z = self.params / se
        pv = 2 * stats.norm.sf(np.abs(z))
        lines = [f"Joint ARIMAX({self.p},{self.d},{self.q})-GARCH(1,1)-{self.dist} MLE   "
                 f"nobs={self.nobs}  loglik={self.loglik:.3f}  AIC={self.aic:.2f}  BIC={self.bic:.2f}",
                 f"{'param':>8} {'coef':>12} {'rob.SE':>10} {'z':>8} {'p':>8}"]
        for n, b, s, zz, pp in zip(self.names, self.params, se, z, pv):
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
        if not self.invertible:
            lines.append("  WARNING: MA not invertible -> over-differencing or misspecification; "
                         "the conditional likelihood used here assumes invertibility.")
        return "\n".join(lines)


# -----------------------------------------------------------------------------
# estimation
# -----------------------------------------------------------------------------
def fit_joint(y, X=None, p=1, q=1, d=0, dist="t", start=None, polish=True, compute_cov=True,
              init="backcast"):
    """Joint MLE of ARIMAX(p,d,q)-GARCH(1,1)-{normal,t}.

    y     : 1-d array-like. If d > 0 this is the LEVEL series; it is differenced d times.
    X     : (T, k) exogenous regressors or None. With d > 0, X with T rows is taken as
            levels and differenced too (regression-with-ARIMA-errors convention); X with
            T-d rows is taken as already aligned with the differenced series.
    p, q  : ARMA orders.  d : differencing order.  dist : "normal" | "t".
    start : optional initial parameter vector (packing = result.names); two-step
            estimates make a good start when the MA+GARCH+nu surface is multimodal.
    polish      : Nelder-Mead refinement after L-BFGS-B (helps MA terms; off for MC loops).
    compute_cov : QMLE sandwich covariance (off for MC loops that need only point estimates).
    init  : pre-sample variance rule, "backcast" (arch) or "sample" (FCP 1996 / fGarch).
            Only affects the first few sigma_t^2; estimates differ in the 4th decimal.
    """
    y = np.asarray(y, float).ravel()
    if X is not None:
        X = np.asarray(X, float)
        if X.ndim == 1:
            X = X[:, None]
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
    sd = y.std()
    if not (0.05 <= sd <= 50):
        warnings.warn(f"std(series)={sd:.3g}; rescale toward 0.1-10 for a well-behaved optimiser.")

    if start is None:
        start = _hr_start(y, X, p, q, dist)
    start = np.asarray(start, float)

    bounds = ([(None, None)] * (1 + p + q + k)
              + [(1e-10, None), (0.0, 1.0), (0.0, 1.0)]
              + ([(2.05, 200.0)] if dist == "t" else []))
    obj = lambda th: _neg_loglik(th, y, X, p, q, dist, init)
    r = optimize.minimize(obj, start, method="L-BFGS-B", bounds=bounds,
                          options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-8})
    best = r
    if polish:
        r2 = optimize.minimize(obj, r.x, method="Nelder-Mead",
                               options={"maxiter": 5000, "fatol": 1e-10, "xatol": 1e-8})
        if r2.fun < r.fun:
            best = r2
    eps, sig2, _ = _filter(best.x, y, X, p, q, dist, init)
    cov = cov_h = cov_o = None
    if compute_cov:
        cov, cov_h, cov_o = _sandwich_cov(best.x, y, X, p, q, dist, init)
    return JointResult(best.x, _names(p, q, k, dist), cov, -best.fun, p, q, d, dist, y, X,
                       eps, sig2, y_levels_tail, X_levels_tail, converged=bool(r.success),
                       cov_hessian=cov_h, cov_opg=cov_o, init=init)


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

    eps_full = np.zeros(T)
    eps_full[p:] = res.eps

    # --- analytic conditional mean path (future eps = 0) ---
    y_ext, e_ext = list(y), list(eps_full)
    mean = np.empty(h)
    for i in range(h):
        t = T + i
        m = c + xg[i]
        for j in range(p):
            m += phi[j] * y_ext[t - 1 - j]
        for j in range(q):
            m += th[j] * e_ext[t - 1 - j]
        mean[i] = m
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
            m = c + xg[i]
            for j in range(p):
                m = m + phi[j] * y_lag[j]
            for j in range(q):
                m = m + th[j] * e_lag[j]
            yi = m + e
            sims[:, i] = yi
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
             burn=500, seed=None):
    """Simulate ARMAX(p,q)-GARCH(1,1)-{normal,t}. Returns y (T,), eps, sig2 (post burn-in)."""
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
    eps = np.zeros(n); sig2 = np.zeros(n); y = np.zeros(n)
    sig2[0] = omega / (1 - alpha - beta)
    eps[0] = np.sqrt(sig2[0]) * z[0]
    for t in range(1, n):
        sig2[t] = omega + alpha * eps[t - 1] ** 2 + beta * sig2[t - 1]
        eps[t] = np.sqrt(sig2[t]) * z[t]
        m = c + xg[t]
        for i in range(p):
            m += phi[i] * y[t - 1 - i]
        for j in range(q):
            m += theta[j] * eps[t - 1 - j]
        y[t] = m + eps[t]
    return y[burn:], eps[burn:], sig2[burn:]


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
    print("\n1) ARMAX(1,1)-GARCH(1,1)-t recovery, T=3000, sinusoid exog")
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

    print(f"\nOVERALL: {'ALL PASS' if ok_all else 'SOME FAILURES'}")
    return ok_all


if __name__ == "__main__":
    _selftest()
