"""SARIMAX-style front end for regression with ARIMA(p,d,q) errors, GARCH(1,1) opt-in.

Backend rule
------------
garch=None and dist="normal"  ->  statsmodels.tsa.statespace.sarimax.SARIMAX; fit() returns its own
                                  results object (nothing re-implemented).
garch=(1, 1) or dist="t"      ->  joint conditional MLE of regression with ARIMA-GARCH(1,1) errors
                                  (sarimax_garch._core); fit() returns SARIMAXGARCHResults, which carries
                                  the statsmodels attribute names (params, bse, pvalues, llf, aic, resid,
                                  summary(), conf_int(), cov_params(), get_forecast() ...).

Model (regression with ARIMA errors, the SARIMAX formulation):
    y_t = x_t' beta + u_t,   (1-L)^d u_t = c + phi(L)-AR + theta(L)-MA + eps_t,
    eps_t = sigma_t z_t,  sigma_t^2 = omega + alpha eps_{t-1}^2 + beta sigma_{t-1}^2,  z ~ N(0,1) or t(nu)
"""
from __future__ import annotations

import contextlib
import warnings

import numpy as np
import pandas as pd
from scipy import stats

from . import _core

__all__ = ["SARIMAXGARCH", "SARIMAXGARCHResults", "GARCHPredictionResults"]


# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------
def _forecast_index(index, steps):
    """Continue a pandas index `steps` periods past its end (dates when the frequency is known)."""
    if isinstance(index, pd.DatetimeIndex):
        freq = index.freq or (pd.infer_freq(index) if len(index) >= 3 else None)
        if freq is not None:
            return pd.date_range(index[-1], periods=steps + 1, freq=freq)[1:]
    if isinstance(index, pd.PeriodIndex):
        return pd.period_range(index[-1], periods=steps + 1, freq=index.freq)[1:]
    if isinstance(index, pd.RangeIndex) or (len(index) and np.issubdtype(np.asarray(index).dtype, np.integer)):
        last = int(index[-1])
        return pd.RangeIndex(last + 1, last + 1 + steps)
    return pd.RangeIndex(len(index), len(index) + steps)


def _crps_normal(mean, sd, x):
    z = (x - mean) / sd
    return sd * (z * (2 * stats.norm.cdf(z) - 1) + 2 * stats.norm.pdf(z) - 1 / np.sqrt(np.pi))


# -----------------------------------------------------------------------------
# model
# -----------------------------------------------------------------------------
class SARIMAXGARCH:
    """Regression with ARIMA(p,d,q) errors, SARIMAX-style, with GARCH(1,1) opt-in.

    Parameters (statsmodels SARIMAX names where they exist)
    -------------------------------------------------------
    endog, exog          : array-like / pandas; a pandas index is kept for resid and forecasts.
    order                : (p, d, q).
    seasonal_order       : statsmodels only (raises with GARCH; put seasonal dummies / Fourier terms in exog).
    trend                : as in SARIMAX, the trend polynomial of the (differenced) error equation.
                           None / "n": no constant -- the SARIMAX default!  "c": a constant, named
                           "intercept" for every d; with d >= 1 it is the drift of the level.  "t" and
                           "ct" (a time trend inside the differenced equation, SARIMAX's "drift"
                           parameter) are statsmodels-only; with GARCH / t errors they raise -- put a
                           time column in exog instead.
    garch                : None (off) or (1, 1) / True.
    dist                 : "normal" or "t" (Student-t errors, nu estimated; needs the joint backend).
    enforce_stationarity, enforce_invertibility : passed to statsmodels; with GARCH they only warn.
    **sm_kwargs          : any other statsmodels SARIMAX option (statsmodels backend only).
    """

    def __init__(self, endog, exog=None, order=(1, 0, 0), seasonal_order=(0, 0, 0, 0), trend=None,
                 garch=None, dist="normal", enforce_stationarity=True, enforce_invertibility=True, **sm_kwargs):
        if isinstance(endog, pd.DataFrame):
            if endog.shape[1] != 1:
                raise ValueError("endog must be one series")
            endog = endog.iloc[:, 0]
        self.endog_name = str(getattr(endog, "name", None) or "y")
        self._index = endog.index if isinstance(endog, pd.Series) else pd.RangeIndex(len(endog))
        self.endog = np.asarray(endog, float).ravel()
        if exog is not None:
            if isinstance(exog, pd.DataFrame):
                self.exog_names = [str(c) for c in exog.columns]
            elif isinstance(exog, pd.Series):
                self.exog_names = [str(exog.name or "x1")]
            else:
                exog = np.asarray(exog, float)
                self.exog_names = [f"x{i + 1}" for i in range(1 if exog.ndim == 1 else exog.shape[1])]
            self.exog = np.asarray(exog, float).reshape(-1, len(self.exog_names))
        else:
            self.exog, self.exog_names = None, []
        self.order = tuple(int(v) for v in order)
        self.seasonal_order = tuple(seasonal_order)
        self.trend = trend
        self.garch = (1, 1) if garch is True else (tuple(garch) if garch else None)
        self.dist = dist
        self.enforce_stationarity = enforce_stationarity
        self.enforce_invertibility = enforce_invertibility
        self._sm_kwargs = sm_kwargs
        self._fixed = {}
        if self.garch is not None and self.garch != (1, 1):
            raise NotImplementedError("only GARCH(1,1) is implemented: garch=(1, 1)")
        if dist not in ("normal", "t"):
            raise ValueError("dist must be 'normal' or 't'")
        self.backend = "statsmodels" if (self.garch is None and dist == "normal") else "garch"
        p, d, q = self.order
        if self.backend == "garch":
            if self.seasonal_order != (0, 0, 0, 0):
                raise NotImplementedError("GARCH or seasonal: pick one. seasonal_order is a statsmodels-SARIMAX "
                                          "feature and the joint ARMA-GARCH estimator has no seasonal ARMA "
                                          "terms; with garch=(1, 1) model seasonality through exog "
                                          "(month dummies or sin/cos Fourier terms)")
            if sm_kwargs:
                raise NotImplementedError(f"{sorted(sm_kwargs)}: statsmodels-only options, not available "
                                          "with GARCH / t errors")
        self._trend_kind = self._parse_trend(trend, d, self.backend == "garch")
        self._build_names()

    # --- names -------------------------------------------------------------------
    @staticmethod
    def _parse_trend(trend, d, strict):
        """SARIMAX semantics: the trend sits in the equation of the d-th difference of the errors, so
        "c" is a constant there (statsmodels names it "intercept" for every d) and "t" is b*t there
        (statsmodels' "drift" -- with d = 1 a quadratic trend in the level)."""
        if trend in (None, "n"):
            return None
        if trend == "c":
            return "intercept"
        if strict:
            raise NotImplementedError(
                f"trend={trend!r}: with GARCH / t errors only None or 'c' is supported (as in SARIMAX, 'c' is "
                "the constant of the differenced equation -- with d >= 1 that is the drift of the level). "
                "SARIMAX's 't'/'ct' time-trend terms are statsmodels-only; add a time column to exog instead")
        return trend

    def _build_names(self):
        p, d, q = self.order
        k = len(self.exog_names)
        garch_on = self.garch is not None
        self._core_names = _core._names(p, q, k, self.dist)
        disp = {"const": self._trend_kind or "const",
                "omega": "omega" if garch_on else "sigma2", "alpha": "alpha.L1", "beta": "beta.L1", "nu": "nu"}
        disp.update({f"phi{i + 1}": f"ar.L{i + 1}" for i in range(p)})
        disp.update({f"theta{j + 1}": f"ma.L{j + 1}" for j in range(q)})
        disp.update({f"gamma{i + 1}": self.exog_names[i] for i in range(k)})
        self._to_display = disp
        self._to_core = {v: k_ for k_, v in disp.items()}
        order = (["const"] + [f"gamma{i + 1}" for i in range(k)] + [f"phi{i + 1}" for i in range(p)]
                 + [f"theta{j + 1}" for j in range(q)] + ["omega", "alpha", "beta"] + (["nu"] if self.dist == "t" else []))
        hidden = set()
        if self._trend_kind is None:
            hidden.add("const")
        if not garch_on:
            hidden.update(("alpha", "beta"))
        self._display_core = [n for n in order if n not in hidden]
        self.param_names = [disp[n] for n in self._display_core]

    def _core_fixed(self, fixed):
        out = {}
        for name, value in fixed.items():
            core = self._to_core.get(name, name if name in self._core_names else None)
            if core is None:
                raise ValueError(f"unknown parameter {name!r}; this model's parameters are {self.param_names}")
            out[core] = float(value)
        if self._trend_kind is None:
            out["const"] = 0.0
        if self.garch is None:
            out.update(alpha=0.0, beta=0.0)
        return out

    def _core_start(self, start):
        if start is None:
            return None
        if isinstance(start, (pd.Series, dict)):
            vals = dict(start)
        else:
            start = np.asarray(start, float).ravel()
            if len(start) != len(self.param_names):
                raise ValueError(f"start_params has {len(start)} values; expected {len(self.param_names)}: {self.param_names}")
            vals = dict(zip(self.param_names, start))
        core = {self._to_core.get(n, n): float(v) for n, v in vals.items()}
        defaults = dict(const=0.0, alpha=0.0, beta=0.0, nu=8.0)
        return np.array([core.get(n, defaults.get(n, np.nan)) for n in self._core_names])

    def _future_exog(self, exog, steps):
        if self.exog is None:
            return None
        if exog is None:
            raise ValueError(f"the model has exog ({self.exog_names}); pass exog with {steps} rows")
        if isinstance(exog, pd.DataFrame) and set(self.exog_names) <= set(map(str, exog.columns)):
            exog = exog[self.exog_names]
        Xf = np.asarray(exog, float).reshape(-1, len(self.exog_names))
        if Xf.shape[0] != steps:
            raise ValueError(f"exog for the forecast must have {steps} rows, got {Xf.shape[0]}")
        return Xf

    # --- statsmodels-style API ---------------------------------------------------
    @contextlib.contextmanager
    def fix_params(self, params):
        """Hold parameters fixed during fit():  with mod.fix_params({"ar.L1": 0.5}): res = mod.fit()"""
        old = self._fixed
        self._fixed = {**old, **params}
        try:
            yield self
        finally:
            self._fixed = old

    def fit(self, cov_type=None, fixed_params=None, start_params=None, disp=False, polish=True,
            init="backcast", **kwargs):
        """Estimate.  statsmodels backend: returns the SARIMAX results object (kwargs go to its fit());
        unless you pass maxiter or method, the fit gets 1000 L-BFGS iterations and, if it still has not
        converged, a Nelder-Mead polish (see _sm_fit) -- check res.mle_retvals["converged"].
        GARCH backend: cov_type "robust" (default; QMLE sandwich), "oim"/"approx" (inverse Hessian) or
        "opg"; fixed_params {name: value}; polish / init as in the core estimator."""
        fixed = {**self._fixed, **(fixed_params or {})}
        if self.backend == "statsmodels":
            from statsmodels.tsa.statespace.sarimax import SARIMAX
            endog = pd.Series(self.endog, index=self._index, name=self.endog_name)
            exog = None if self.exog is None else pd.DataFrame(self.exog, index=self._index, columns=self.exog_names)
            mod = SARIMAX(endog, exog=exog, order=self.order, seasonal_order=self.seasonal_order, trend=self.trend,
                          enforce_stationarity=self.enforce_stationarity,
                          enforce_invertibility=self.enforce_invertibility, **self._sm_kwargs)
            self._sm_model = mod
            kw = dict(disp=disp, **kwargs)
            if cov_type is not None:
                kw["cov_type"] = cov_type
            if start_params is not None:
                kw["start_params"] = np.asarray(start_params, float)
                if fixed:
                    kw["includes_fixed"] = True                  # start vectors from res.params are full
            robust = "maxiter" not in kwargs and "method" not in kwargs
            if fixed:
                with mod.fix_params(fixed):
                    return self._sm_fit(mod, kw, robust, has_fixed=True)
            return self._sm_fit(mod, kw, robust)
        if kwargs:
            raise TypeError(f"unexpected fit() options for the GARCH backend: {sorted(kwargs)}")
        p, d, q = self.order
        res = _core.fit_joint(self.endog, X=self.exog, p=p, q=q, d=d, dist=self.dist,
                              start=self._core_start(start_params), polish=polish, compute_cov=True,
                              init=init, xreg="regression", fixed=self._core_fixed(fixed))
        if self.enforce_stationarity and not res.stationary:
            warnings.warn("AR part is not stationary (enforce_stationarity is advisory with GARCH)")
        if self.enforce_invertibility and not res.invertible:
            warnings.warn("MA part is not invertible (enforce_invertibility is advisory with GARCH)")
        return SARIMAXGARCHResults(self, res, cov_type or "robust", fixed)

    @staticmethod
    def _sm_fit(mod, kw, robust, has_fixed=False):
        """statsmodels' SARIMAX.fit stops after 50 L-BFGS iterations by default.  On regressions with
        trending or badly scaled exog that is often too few, and the 'fit' it returns can sit far from
        the MLE (in testing: phi 0.895 vs 0.733, loglik -237.9 vs -233.2) with only a ConvergenceWarning.
        Unless the caller chose maxiter / method: allow 1000 iterations; if L-BFGS still has not
        converged, polish with Nelder-Mead and restart L-BFGS from there, keeping the best fit.  The
        result is always statsmodels' own results object; a fit that still has not converged warns."""
        from statsmodels.tools.sm_exceptions import ConvergenceWarning
        if not robust:
            return mod.fit(**kw)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            res = mod.fit(maxiter=1000, **kw)
            if not res.mle_retvals.get("converged", True):
                rest = {k: v for k, v in kw.items() if k not in ("start_params", "includes_fixed")}
                extra = {"includes_fixed": True} if has_fixed else {}     # res.params includes fixed values
                nm = mod.fit(start_params=res.params.values, method="nm", maxiter=10_000, **extra, **rest)
                res2 = mod.fit(start_params=nm.params.values, maxiter=1000, **extra, **rest)
                cands = [r for r in (res2, nm, res) if np.isfinite(r.llf)]
                best_llf = max(r.llf for r in cands)
                near = [r for r in cands if r.llf >= best_llf - 1e-6]
                lbfgs_ok = [r for r in near if r is not nm and r.mle_retvals.get("converged", False)]
                res = lbfgs_ok[0] if lbfgs_ok else max(near, key=lambda r: r.llf)
        if not res.mle_retvals.get("converged", True):
            warnings.warn("statsmodels SARIMAX did not converge (1000 L-BFGS iterations plus a Nelder-Mead "
                          "polish); check res.mle_retvals, and try centring / rescaling exog", ConvergenceWarning)
        return res

    def backtest(self, horizon=12, start=None, step=1, alpha=0.1, n_sims=2000, seed=0, fixed_params=None):
        """Rolling-origin out-of-sample scores by horizon, with the realised exog path given.

        For origins t0 = start, start+step, ... (t0 + horizon <= T): refit on the first t0 observations
        with this model's backend, forecast `horizon` steps, score against the realised values.
        Returns a DataFrame indexed by horizon: rmse, mae, crps (proper score of the predictive
        distribution -- Gaussian closed form for the statsmodels backend, simulated for GARCH) and
        coverage of the (1 - alpha) interval.  Compare levels (d=0) vs differences (d=1) and GARCH vs
        not with it: AIC/BIC cannot compare models fitted to differently differenced data.
        result.attrs["n_origins"] / ["n_not_converged"] record how many refits did not converge
        (a warning is raised when any did)."""
        T = len(self.endog)
        p, d, q = self.order
        if start is None:
            start = max(T // 2, 100 + p + d)
        origins = list(range(start, T - horizon + 1, step))
        if not origins:
            raise ValueError(f"no forecast origins: need start + horizon <= T (T={T}, start={start}, horizon={horizon})")
        y_s = pd.Series(self.endog, index=self._index, name=self.endog_name)
        X_f = None if self.exog is None else pd.DataFrame(self.exog, index=self._index, columns=self.exog_names)
        z = stats.norm.ppf(1 - alpha / 2)
        err = np.empty((len(origins), horizon)); crps = np.empty_like(err); cov = np.empty_like(err)
        prev, not_converged = None, 0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for i, t0 in enumerate(origins):
                sub = SARIMAXGARCH(y_s.iloc[:t0], exog=None if X_f is None else X_f.iloc[:t0], order=self.order,
                                   seasonal_order=self.seasonal_order, trend=self.trend, garch=self.garch,
                                   dist=self.dist, enforce_stationarity=self.enforce_stationarity,
                                   enforce_invertibility=self.enforce_invertibility, **self._sm_kwargs)
                res = sub.fit(disp=False, fixed_params=fixed_params, start_params=prev)
                ok = res.converged if sub.backend == "garch" else res.mle_retvals.get("converged", True)
                not_converged += not ok
                prev = res.params
                Xf = None if X_f is None else X_f.iloc[t0:t0 + horizon]
                actual = self.endog[t0:t0 + horizon]
                if sub.backend == "garch":
                    fc = res.get_forecast(horizon, exog=Xf, n_sims=n_sims, seed=seed + i)
                    mean = fc.predicted_mean.values
                    sims = fc.sims
                    for h in range(horizon):
                        crps[i, h] = _core._crps(sims[:, h], actual[h])
                        lo, hi = np.quantile(sims[:, h], [alpha / 2, 1 - alpha / 2])
                        cov[i, h] = lo <= actual[h] <= hi
                else:
                    fc = res.get_forecast(horizon, exog=Xf)
                    mean, se = np.asarray(fc.predicted_mean), np.asarray(fc.se_mean)
                    crps[i] = _crps_normal(mean, se, actual)
                    cov[i] = np.abs(actual - mean) <= z * se
                err[i] = mean - actual
        out = pd.DataFrame({"rmse": np.sqrt(np.mean(err ** 2, 0)), "mae": np.mean(np.abs(err), 0),
                            "crps": crps.mean(0), f"coverage{int(round((1 - alpha) * 100))}": cov.mean(0)},
                           index=pd.Index(np.arange(1, horizon + 1), name="h"))
        out.attrs.update(n_origins=len(origins), n_not_converged=int(not_converged))
        if not_converged:
            warnings.warn(f"backtest: {not_converged} of {len(origins)} refits did not converge; their forecasts "
                          "may be off -- see the fit() notes (centre / rescale exog, or a larger step)")
        return out


# -----------------------------------------------------------------------------
# results
# -----------------------------------------------------------------------------
class SARIMAXGARCHResults:
    """Results of the GARCH / t backend, with the statsmodels results attribute names."""

    _cov_types = {"robust": "cov", "oim": "cov_hessian", "approx": "cov_hessian", "opg": "cov_opg"}

    def __init__(self, model, res, cov_type, user_fixed):
        if cov_type not in self._cov_types:
            raise ValueError(f"cov_type must be one of {sorted(self._cov_types)}")
        self.model, self._res, self.cov_type = model, res, cov_type
        idx = [res.names.index(n) for n in model._display_core]
        names = model.param_names
        self.param_names = names
        self.params = pd.Series(res.params[idx], index=names, name="params")
        full = getattr(res, self._cov_types[cov_type])
        if full is None:                                  # every parameter fixed: a filter, no inference
            full = np.full((len(res.params), len(res.params)), np.nan)
        C = full[np.ix_(idx, idx)]
        self._cov = pd.DataFrame(C, index=names, columns=names)
        self.bse = pd.Series(np.sqrt(np.diag(C)), index=names, name="bse")
        self.zvalues = self.params / self.bse
        self.tvalues = self.zvalues                      # statsmodels name; these are z statistics
        self.pvalues = pd.Series(2 * stats.norm.sf(np.abs(self.zvalues.values)), index=names, name="pvalues")
        self.llf = float(res.loglik)
        self.nobs = int(res.nobs)
        self.df_model = int(res.n_free)
        self.aic = -2 * self.llf + 2 * self.df_model
        self.bic = -2 * self.llf + self.df_model * np.log(self.nobs)
        self.hqic = -2 * self.llf + 2 * self.df_model * np.log(np.log(self.nobs))
        self.fixed_params = dict(user_fixed)
        self.converged = bool(res.converged)
        self._offset = model.order[1] + model.order[0]          # d + p leading observations are not scored

    # --- series on the original index ---------------------------------------------
    def _aligned(self, values, name):
        full = np.full(len(self.model.endog), np.nan)
        full[self._offset:] = values
        return pd.Series(full, index=self.model._index, name=name)

    @property
    def resid(self):
        return self._aligned(self._res.eps, "resid")

    @property
    def fittedvalues(self):
        return pd.Series(self.model.endog, index=self.model._index) - self.resid.values

    @property
    def conditional_variance(self):
        return self._aligned(self._res.sig2, "conditional_variance")

    @property
    def conditional_volatility(self):
        return np.sqrt(self.conditional_variance).rename("conditional_volatility")

    @property
    def std_resid(self):
        return self._aligned(self._res.std_resid, "std_resid")

    # --- GARCH summaries ---------------------------------------------------------------
    persistence = property(lambda self: self._res.persistence)
    half_life = property(lambda self: self._res.half_life)
    unconditional_variance = property(lambda self: self._res.uncond_var)
    stationary = property(lambda self: self._res.stationary)
    invertible = property(lambda self: self._res.invertible)
    # statsmodels names: roots of 1 - phi_1 z - ... and 1 + theta_1 z + ...; |root| > 1 = stationary / invertible
    arroots = property(lambda self: self._res.ar_roots)
    maroots = property(lambda self: self._res.ma_roots)

    # --- inference ------------------------------------------------------------------
    def cov_params(self):
        return self._cov

    def conf_int(self, alpha=0.05):
        z = stats.norm.ppf(1 - alpha / 2)
        return pd.DataFrame({0: self.params - z * self.bse, 1: self.params + z * self.bse})

    def summary(self, alpha=0.05):
        from statsmodels.iolib.summary2 import Summary
        from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
        from statsmodels.stats.stattools import jarque_bera
        m = self.model
        p, d, q = m.order
        spec = (("Regression with " if m.exog is not None else "") + f"ARIMA({p}, {d}, {q})"
                + ("-GARCH(1, 1)" if m.garch else "") + (" errors" if m.exog is not None else "")
                + f", {'Student-t' if m.dist == 't' else 'normal'}")
        z = self._res.std_resid
        lags = min(12, max(1, self.nobs // 5))
        lb = acorr_ljungbox(z, lags=[lags], return_df=True).iloc[0]
        lm = het_arch(z, nlags=lags)
        jb = jarque_bera(z)
        smry = Summary()
        smry.add_title("SARIMAX-GARCH Results" if m.garch else "SARIMAX (Student-t) Results")
        info = {"Dep. Variable:": m.endog_name, "Model:": spec, "No. Observations:": str(self.nobs),
                "Estimator:": "joint conditional MLE", "Cov. Type:": self.cov_type,
                "Log Likelihood:": f"{self.llf:.3f}", "AIC:": f"{self.aic:.3f}", "BIC:": f"{self.bic:.3f}",
                "HQIC:": f"{self.hqic:.3f}", "Converged:": str(self.converged)}
        if self.fixed_params:
            info["Fixed:"] = ", ".join(f"{k}={v:g}" for k, v in self.fixed_params.items())
        smry.add_dict(info, ncols=2)
        ci = self.conf_int(alpha)
        table = pd.DataFrame({"coef": self.params, "std err": self.bse, "z": self.zvalues, "P>|z|": self.pvalues,
                              f"[{alpha / 2:.3f}": ci[0], f"{1 - alpha / 2:.3f}]": ci[1]})
        smry.add_df(table, float_format="%.4f")
        diag = {f"Ljung-Box (Q{lags}):": f"{lb['lb_stat']:.2f}", "Prob(Q):": f"{lb['lb_pvalue']:.2f}",
                f"ARCH-LM ({lags}):": f"{lm[0]:.2f}", "Prob(ARCH-LM):": f"{lm[1]:.2f}",
                "Jarque-Bera (JB):": f"{jb[0]:.2f}", "Prob(JB):": f"{jb[1]:.2f}",
                "Skew:": f"{jb[2]:.2f}", "Kurtosis:": f"{jb[3]:.2f}"}
        if m.garch:
            diag.update({"Persistence:": f"{self.persistence:.4f}", "Half-life:": f"{self.half_life:.1f}",
                         "Uncond. variance:": f"{self.unconditional_variance:.4g}",
                         "AR stationary / MA invertible:": f"{self.stationary} / {self.invertible}"})
        smry.add_dict(diag, ncols=2)
        notes = ["Diagnostics are computed on the standardized residuals eps_t / sigma_t."]
        if m.garch is None and m.dist == "t":
            notes.append("GARCH off: sigma2 is the constant error variance (as statsmodels' sigma2).")
        notes += [ln.strip() for ln in self._res.summary().splitlines() if "WARNING" in ln]
        smry.add_text("\n".join(notes))
        return smry

    # --- diagnostics ------------------------------------------------------------------
    def test_serial_correlation(self, lags=None, method="ljungbox"):
        """Ljung-Box on the standardized residuals (DataFrame: lb_stat, lb_pvalue by lag)."""
        from statsmodels.stats.diagnostic import acorr_ljungbox
        if method != "ljungbox":
            raise NotImplementedError("only method='ljungbox'")
        return acorr_ljungbox(self._res.std_resid, lags=lags or min(12, self.nobs // 5), return_df=True)

    def test_arch(self, lags=12):
        """Engle's ARCH-LM on the standardized residuals (remaining ARCH): Series lm_stat, lm_pvalue, f_stat, f_pvalue."""
        from statsmodels.stats.diagnostic import het_arch
        return pd.Series(het_arch(self._res.std_resid, nlags=lags), index=["lm_stat", "lm_pvalue", "f_stat", "f_pvalue"])

    def test_normality(self):
        """Jarque-Bera on the standardized residuals: Series jb_stat, jb_pvalue, skew, kurtosis."""
        from statsmodels.stats.stattools import jarque_bera
        return pd.Series(jarque_bera(self._res.std_resid), index=["jb_stat", "jb_pvalue", "skew", "kurtosis"])

    def plot_diagnostics(self, lags=24, figsize=(10, 8)):
        """statsmodels-style 2x2 panel: standardized residuals, histogram + N(0,1), normal Q-Q, correlogram."""
        import matplotlib.pyplot as plt
        from statsmodels.graphics.tsaplots import plot_acf
        z = self.std_resid.dropna()
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes[0, 0].plot(z.index, z.values, lw=0.8)
        axes[0, 0].axhline(0, color="k", lw=0.5)
        axes[0, 0].set_title("Standardized residual")
        axes[0, 1].hist(z.values, bins=40, density=True, alpha=0.6, label="hist")
        grid = np.linspace(-4, 4, 200)
        axes[0, 1].plot(grid, stats.norm.pdf(grid), label="N(0,1)")
        if self.model.dist == "t":
            nu = self._res["nu"]
            s = np.sqrt((nu - 2) / nu)
            axes[0, 1].plot(grid, stats.t.pdf(grid / s, nu) / s, label=f"t({nu:.1f})")
        axes[0, 1].legend()
        axes[0, 1].set_title("Histogram plus estimated density")
        stats.probplot(z.values, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title("Normal Q-Q")
        plot_acf(z.values, lags=lags, ax=axes[1, 1])
        axes[1, 1].set_title("Correlogram")
        fig.tight_layout()
        return fig

    def bootstrap(self, n_boot=199, method="residual", seed=None):
        """Bootstrap SEs and 95% percentile CIs (DataFrame: se, ci_lower, ci_upper by parameter).
        Use for the GARCH parameters in small samples; keep the Wald SEs for the mean parameters."""
        bs = _core.bootstrap(self._res, n_boot=n_boot, method=method, seed=seed)
        idx = [bs["names"].index(n) for n in self.model._display_core]
        return pd.DataFrame({"se": bs["se"][idx], "ci_lower": bs["ci"][0, idx], "ci_upper": bs["ci"][1, idx]},
                            index=self.param_names)

    # --- forecasting ------------------------------------------------------------------
    def get_forecast(self, steps=1, exog=None, n_sims=10_000, seed=None):
        """Out-of-sample forecast.  Returns GARCHPredictionResults (predicted_mean, se_mean,
        var_pred_mean, conditional_variance, conditional_volatility, conf_int(), summary_frame(), sims)."""
        steps = int(steps)
        fc = _core.forecast(self._res, steps, X_future=self.model._future_exog(exog, steps), n_sims=n_sims, seed=seed)
        index = _forecast_index(self.model._index, steps)
        return GARCHPredictionResults(fc["mean"], fc["variance"], index, self.model.endog_name, fc["sims"],
                                      cond_var=fc["variance_innovation"])

    def forecast(self, steps=1, exog=None):
        return self.get_forecast(steps, exog=exog, n_sims=0).predicted_mean

    def get_prediction(self):
        """In-sample one-step-ahead predictions (fitted values) with se = conditional volatility."""
        mean, var = self.fittedvalues.values, self.conditional_variance.values
        return GARCHPredictionResults(mean, var, self.model._index, self.model.endog_name, None, cond_var=var)

    def simulate(self, nsimulations, repetitions=1, exog=None, seed=None):
        """Simulated future paths (DataFrame nsimulations x repetitions, forecast index)."""
        fc = self.get_forecast(nsimulations, exog=exog, n_sims=repetitions, seed=seed)
        return pd.DataFrame(fc.sims.T, index=fc.predicted_mean.index)

    def __repr__(self):
        return f"<SARIMAXGARCHResults: {self.model.endog_name}, {len(self.params)} params, llf={self.llf:.3f}>"


class GARCHPredictionResults:
    """Mirror of statsmodels' PredictionResults: predicted_mean, var_pred_mean, se_mean, conf_int(),
    summary_frame(); plus the GARCH volatility path, the simulated paths (sims) and quantile().

    var_pred_mean          Var(y_{T+h} | F_T): forecast-error variance of y -- accumulates the ARMA
                           dynamics (and, with d > 0, the integration) over the horizon.
    conditional_variance   E[sigma^2_{T+h} | F_T]: the GARCH variance forecast of the innovation -- what a
                           "volatility forecast" means.  Equals var_pred_mean at h = 1.
    conditional_volatility its square root."""

    def __init__(self, mean, var, index, name, sims=None, cond_var=None):
        self.predicted_mean = pd.Series(np.asarray(mean, float), index=index, name="predicted_mean")
        self.var_pred_mean = pd.Series(np.asarray(var, float), index=index, name="var_pred_mean")
        self.se_mean = pd.Series(np.sqrt(np.asarray(var, float)), index=index, name="se_mean")
        cv = None if cond_var is None else np.asarray(cond_var, float)
        self.conditional_variance = None if cv is None else pd.Series(cv, index=index, name="conditional_variance")
        self.conditional_volatility = None if cv is None else pd.Series(np.sqrt(cv), index=index,
                                                                        name="conditional_volatility")
        self.sims = None if sims is None else np.asarray(sims)
        self._name = name

    def conf_int(self, alpha=0.05, method="normal"):
        """method 'normal': mean +/- z se (as statsmodels); 'simulated': quantiles of the simulated
        paths (exact for GARCH / t at multi-step horizons)."""
        if method == "simulated":
            if self.sims is None:
                raise ValueError("no simulated paths: call get_forecast with n_sims > 0")
            lo, hi = np.quantile(self.sims, [alpha / 2, 1 - alpha / 2], axis=0)
        else:
            z = stats.norm.ppf(1 - alpha / 2)
            lo, hi = self.predicted_mean - z * self.se_mean, self.predicted_mean + z * self.se_mean
        return pd.DataFrame({f"lower {self._name}": np.asarray(lo), f"upper {self._name}": np.asarray(hi)},
                            index=self.predicted_mean.index)

    def summary_frame(self, alpha=0.05, method="normal"):
        ci = self.conf_int(alpha, method)
        return pd.DataFrame({"mean": self.predicted_mean, "mean_se": self.se_mean,
                             "mean_ci_lower": ci.iloc[:, 0].values, "mean_ci_upper": ci.iloc[:, 1].values})

    def quantile(self, q):
        if self.sims is None:
            raise ValueError("no simulated paths: call get_forecast with n_sims > 0")
        return pd.Series(np.quantile(self.sims, q, axis=0), index=self.predicted_mean.index, name=f"q{q}")
