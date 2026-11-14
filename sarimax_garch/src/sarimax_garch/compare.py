"""compare(): should this SARIMAX model have GARCH (and Student-t) errors?  The workflow of GUIDE.md as code.

Every comparison is made on one likelihood: the statsmodels SARIMAX fit is the reference model and the
source of the ARCH-LM pre-test, while GARCH-vs-not is decided only between fits of the package's own
engine (the constant-variance null is the same engine with alpha = beta = 0 fixed).  Comparing the
SARIMAX log-likelihood or AIC with a GARCH fit's picks GARCH for about half of all series that have no
ARCH (verification/limitations, E1), so compare() never does it.

Flags carry the limitation codes of GUIDE.md (L1-L15) and the fallback to use.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats

from . import _core
from .model import SARIMAXGARCH

__all__ = ["compare", "CompareResults", "sign_bias_test", "holdout_logscore", "dm_test"]


def _split(obj, n_train):
    if obj is None:
        return None, None
    if isinstance(obj, (pd.Series, pd.DataFrame)):
        return obj.iloc[:n_train], obj.iloc[n_train:]
    a = np.asarray(obj, float)
    return a[:n_train], a[n_train:]


def holdout_logscore(endog, exog=None, order=(1, 0, 0), trend="c", garch=None, dist="normal", holdout=0.2):
    """Prequential one-step holdout log-likelihood, with the parameters frozen at the training fit: fit on all
    but the last `holdout` observations (an int, or a fraction of T), then score each held-out observation by
    the log of its one-step-ahead predictive density given everything before it -- the log score that
    microprediction's `skaters` ranks forecasters by (Gneiting & Raftery 2007: strictly proper).

    Unlike in-sample llf / AIC it is comparable across the two backends, across d, GARCH or not, and normal
    or t: every model scores the same observations in the same units.  statsmodels: SARIMAXResults.extend
    (Kalman state carried over); engine: the full series filtered with the training parameters.
    Returns the per-observation log scores (nats; their mean is the holdout log-likelihood per observation)."""
    T = len(endog)
    n = int(round(holdout * T)) if isinstance(holdout, float) and holdout < 1 else int(holdout)
    if not 10 <= n <= T - 50:
        raise ValueError(f"holdout must leave >= 50 training and >= 10 holdout observations (T = {T}, holdout = {n})")
    y_tr, y_te = _split(endog, T - n)
    x_tr, x_te = _split(exog, T - n)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mod = SARIMAXGARCH(y_tr, exog=x_tr, order=order, trend=trend, garch=garch, dist=dist)
        res = mod.fit()
        if mod.backend == "statsmodels":
            ext = res.extend(np.asarray(y_te, float), exog=None if x_te is None else np.asarray(x_te, float))
            return np.asarray(ext.llf_obs, float)
        full = SARIMAXGARCH(endog, exog=exog, order=order, trend=trend, garch=garch, dist=dist).fit(
            fixed_params=dict(res.params))                                  # every parameter fixed = filter only
    r = full._res
    return _core._loglik_obs(r.params, r.y, r.X, r.p, r.q, r.dist, r.init, r.xreg)[-n:]


def dm_test(diff, h=1, lag=None, small_sample=True):
    """Diebold & Mariano (1995) test of equal expected score for two forecasts scored on the same observations.

    diff : d_t = S(F_A,t, y_t) - S(F_B,t, y_t), e.g. log score A - log score B (+ = A better), t = 1..n
    H0   : E[d_t] = 0  (for the log score: equal Kullback-Leibler distance of A and B from the truth)
    DM   = mean(d) / sqrt(LRV / n),  LRV = Newey-West (Bartlett) long-run variance of d_t with
           lag = max(h - 1, floor(4 (n/100)^(2/9)))  (h-step forecasts overlap: d_t is MA(h - 1))
    Under H0, DM -> N(0, 1) as n -> infinity (d_t stationary and mixing; with parameters estimated on a fixed or
    rolling window the null is about the estimated models -- Giacomini & White 2006 -- and nested models are
    allowed).  small_sample (default): Harvey, Leybourne & Newbold (1997) -- DM * sqrt((n + 1 - 2h + h(h-1)/n) / n)
    against t(n - 1); E9 (verification/limitations) shows it brings the size closer to nominal.
    Returns (statistic, two-sided p-value).  Reject H0 at level a when p < a; then DM > 0 means A is better."""
    d = np.asarray(diff, float)
    n = len(d)
    lag = max(int(h) - 1, int(np.floor(4 * (n / 100) ** (2 / 9)))) if lag is None else int(lag)
    dc = d - d.mean()
    lrv = dc @ dc / n + 2 * sum((1 - k / (lag + 1)) * (dc[k:] @ dc[:-k]) / n for k in range(1, lag + 1))
    if lrv <= 0:
        return 0.0, 1.0
    stat = d.mean() / np.sqrt(lrv / n)
    if small_sample:
        stat *= np.sqrt(max(n + 1 - 2 * h + h * (h - 1) / n, 1.0) / n)
        return float(stat), float(2 * stats.t.sf(abs(stat), n - 1))
    return float(stat), float(2 * stats.norm.sf(abs(stat)))


def sign_bias_test(std_resid):
    """Engle & Ng (1993) joint sign-bias test on standardized residuals z_t: regress z_t^2 on a constant,
    S-_{t-1}, S-_{t-1} z_{t-1} and S+_{t-1} z_{t-1} (S- = 1[z < 0], S+ = 1 - S-); LM = n R^2 ~ chi2(3)
    under no remaining asymmetry.  Returns (statistic, p_value)."""
    z = np.asarray(std_resid, float)
    z = z[np.isfinite(z)]
    zl, z2 = z[:-1], z[1:] ** 2
    neg = (zl < 0).astype(float)
    Z = np.column_stack([np.ones(len(zl)), neg, neg * zl, (1 - neg) * zl])
    with np.errstate(all="ignore"):              # Apple Accelerate raises spurious FP flags in matmul
        b, *_ = np.linalg.lstsq(Z, z2, rcond=None)
        u = z2 - Z @ b
        r2 = 1.0 - (u @ u) / np.sum((z2 - z2.mean()) ** 2)
    stat = len(z2) * r2
    return float(stat), float(stats.chi2.sf(stat, 3))


def _flag(code, level, message, fallback):
    return dict(code=code, level=level, message=message, fallback=fallback)


class CompareResults:
    """Output of compare(): .table (one row per model), .tests, .flags, .recommendation, .models,
    .backtest (None unless requested) and summary()."""

    def __init__(self, table, tests, flags, recommendation, models, backtest):
        self.table, self.tests, self.flags = table, tests, flags
        self.recommendation, self.models, self.backtest = recommendation, models, backtest

    def summary(self):
        lines = ["SARIMAX vs SARIMAX-GARCH comparison", "",
                 self.table.to_string(float_format=lambda v: f"{v:.4g}"), "",
                 "Log-likelihood / AIC / BIC are comparable only within one 'compare_group' (L4).", "",
                 self.tests.to_string(float_format=lambda v: f"{v:.4g}"), ""]
        rec = self.recommendation
        lines += [f"RECOMMENDATION: {rec['model']}  ({rec['reason']})"]
        lines += [f"  use it for: {rec['use_for']}"]
        for c in rec["caveats"]:
            lines.append(f"  caveat: {c}")
        if self.flags:
            lines += ["", "Flags (codes = GUIDE.md limitations):"]
            for f in self.flags:
                lines.append(f"  [{f['code']} {f['level']}] {f['message']}")
                lines.append(f"        fallback: {f['fallback']}")
        if self.backtest is not None:
            lines += ["", "Backtest (rolling origin, realised exog path):"]
            for name, bt in self.backtest.items():
                lines.append(f"  {name}: CRPS by h " + " ".join(f"{v:.4g}" for v in bt["crps"].values)
                             + f" | {bt.columns[-1]} " + " ".join(f"{v:.2f}" for v in bt.iloc[:, -1].values))
        return "\n".join(lines)

    def __repr__(self):
        return f"<CompareResults: recommend {self.recommendation['model']}>"


def compare(endog, exog=None, order=(1, 0, 0), trend="c", dists=("normal", "t"), arch_lags=(4, 12), alpha=0.05,
            backtest=None, holdout=None):
    """Fit and compare, on the same data:
        SARIMAX                 statsmodels (exact likelihood, all T observations) -- the reference model
        engine, const. variance garch=(1,1) with alpha = beta = 0 fixed            -- the null for GARCH
        engine, GARCH(1,1)      for each dist in `dists` ("normal", "t"); with "t" also t without GARCH
    and run: ARCH-LM on the SARIMAX residuals (lags `arch_lags`), likelihood-ratio tests GARCH vs constant
    variance within the engine, the Engle-Ng sign-bias test on the GARCH standardized residuals, and the
    t-vs-normal intercept check (E3).  backtest: None, or dict(horizon=..., start=..., step=...) for a
    rolling-origin CRPS / coverage comparison of SARIMAX and the recommended GARCH model.
    holdout: None, or the number (int) / share (float < 1) of final observations for the prequential one-step
    holdout log-likelihood of each model (holdout_logscore, parameters fitted on the rest), with
    Diebold-Mariano tests -- the out-of-sample comparison that is valid across backends.
    seasonal_order is not supported (GARCH or seasonal: pick one, L1) -- put seasonality in exog.
    Returns CompareResults; print(res.summary())."""
    p, d, q = (int(v) for v in order)
    flags, models = [], {}
    kw = dict(exog=exog, order=order, trend=trend)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sm = SARIMAXGARCH(endog, **kw).fit()
        models["SARIMAX"] = sm
        null = SARIMAXGARCH(endog, garch=(1, 1), **kw).fit(fixed_params={"alpha.L1": 0.0, "beta.L1": 0.0})
        models["engine const. variance (normal)"] = null
        garch, tnull = {}, None
        for dist in dists:
            garch[dist] = _fit_garch(endog, kw, dist, sm)
            models[f"engine GARCH(1,1) ({dist})"] = garch[dist]
        if "t" in dists:
            tnull = SARIMAXGARCH(endog, dist="t", **kw).fit()
            models["engine const. variance (t)"] = tnull
    nobs = null.nobs

    rows = []
    for name, r in models.items():
        engine = name != "SARIMAX"
        rp = r.params
        rows.append(dict(model=name, backend="engine (conditional)" if engine else "statsmodels (exact)",
                         compare_group="engine" if engine else "SARIMAX alone", nobs=int(r.nobs),
                         k=int(r.df_model) if engine else len(rp), llf=float(r.llf), aic=float(r.aic),
                         bic=float(r.bic),
                         intercept=float(rp["intercept"]) if "intercept" in rp.index else np.nan,
                         persistence=float(r.persistence) if engine and "garch" in name.lower() else np.nan,
                         nu=float(rp["nu"]) if "nu" in rp.index else np.nan))
    table = pd.DataFrame(rows).set_index("model")

    ho = {}
    if holdout is not None:
        specs = {"SARIMAX": dict(garch=None, dist="normal")}
        specs.update({f"engine GARCH(1,1) ({dist})": dict(garch=(1, 1), dist=dist) for dist in dists})
        if "t" in dists:
            specs["engine const. variance (t)"] = dict(garch=None, dist="t")
        for name, sp in specs.items():
            ho[name] = holdout_logscore(endog, exog=exog, order=order, trend=trend, holdout=holdout, **sp)
        table["holdout_logscore"] = [float(np.mean(ho[n])) if n in ho else np.nan for n in table.index]

    tests = []
    resid = np.asarray(sm.resid, float)[d:]
    resid = resid[np.isfinite(resid)]
    lm_p = []
    for lag in arch_lags:
        st, pv = _core.arch_lm_test(resid, lag)
        lm_p.append(pv)
        tests.append(dict(test=f"ARCH-LM({lag}) on SARIMAX residuals", stat=st, df=lag, p_value=pv,
                          note="H0: no ARCH; valid without fitting GARCH"))
    lr = {}
    for dist in dists:
        base = null if dist == "normal" else tnull
        st = max(0.0, 2 * (garch[dist].llf - base.llf))
        lr[dist] = float(stats.chi2.sf(st, 2))
        tests.append(dict(test=f"LR GARCH vs constant variance ({dist}, within engine)", stat=st, df=2,
                          p_value=lr[dist], note="chi2(2) is conservative here (alpha = 0 on the boundary, L5)"))
    ref = "normal" if "normal" in dists else dists[0]
    sb_st, sb_p = sign_bias_test(garch[ref].std_resid.dropna())
    tests.append(dict(test=f"Engle-Ng joint sign bias (GARCH {ref} std. residuals)", stat=sb_st, df=3, p_value=sb_p,
                      note="H0: no asymmetric volatility response"))
    ho_dm = None
    if ho:
        n_ho = len(ho["SARIMAX"])
        pairs = [(f"engine GARCH(1,1) ({ref})", "SARIMAX")]
        if "t" in dists and "normal" in dists:
            pairs.append(("engine GARCH(1,1) (t)", "engine GARCH(1,1) (normal)"))
        for a, b in pairs:
            dd = ho[a] - ho[b]
            st, pv = dm_test(dd)
            tests.append(dict(test=f"holdout log score: {a} - {b} (DM)", stat=st, df=np.nan, p_value=pv,
                              note=f"{dd.mean():+.4f} nats/obs over {n_ho} held-out obs; + = first model better"))
            if b == "SARIMAX":
                ho_dm = (float(dd.mean()), st, pv, n_ho)
    tests = pd.DataFrame(tests).set_index("test")

    # --- flags ---------------------------------------------------------------------------
    flags.append(_flag("L4", "info", "the SARIMAX row uses the exact likelihood on all T observations, the engine "
                       f"rows the conditional likelihood on T - d - p = {nobs}: their llf/AIC/BIC are not comparable",
                       "decide GARCH-vs-not within the engine (done here) or out of sample (backtest=)"))
    if nobs < 100:
        flags.append(_flag("L3", "stop", f"only {nobs} observations: GARCH is not identifiable this short",
                           "keep SARIMAX; use higher-frequency data if volatility matters"))
    elif nobs < 250:
        flags.append(_flag("L3", "warn", f"{nobs} observations: GARCH parameters are weakly identified below ~250 "
                           "(half the fits end on a boundary at 120)",
                           "dist='normal'; report res.bootstrap() CIs for omega/alpha/beta; Wald for the mean"))
    if min(lm_p) >= alpha and nobs < 360:
        flags.append(_flag("L6", "info", "ARCH-LM does not reject, but its power is low at this size (10-20% for "
                           "typical monthly ARCH at 10-15 years)",
                           "when it does not reject, GARCH rarely pays off at this size: keep SARIMAX"))
    g_ref = garch[ref]
    if g_ref._res.at_boundary:
        flags.append(_flag("L5", "warn", f"GARCH ({ref}) ends on a boundary (alpha or beta ~ 0): no usable volatility "
                           "dynamics; beta is unidentified when alpha = 0", "treat as no GARCH: SARIMAX"))
    if g_ref._res.at_persistence_cap:
        flags.append(_flag("L5", "warn", "alpha + beta is at the stationarity cap (integrated volatility)",
                           "SEs, half-life and unconditional variance are not meaningful; check for breaks"))
    if not g_ref._res.fourth_moment_exists:
        flags.append(_flag("L5", "info", "fourth moment of the errors is infinite: mean-parameter SEs are not backed "
                           "by standard QMLE theory", "prefer bootstrap CIs; treat p-values as indicative"))
    if "t" in dists and "normal" in dists and "intercept" in g_ref.params.index:
        dc = (garch["t"].params["intercept"] - garch["normal"].params["intercept"]) / garch["normal"].bse["intercept"]
        if abs(dc) > 1:
            flags.append(_flag("L7", "warn", f"the Student-t intercept differs from the Gaussian one by {dc:+.2f} SE: "
                               "skewed errors pull the t centre toward the mode (2% false alarms when symmetric)",
                               "use dist='normal' for the mean; conf_int(method='simulated') for bands"))
    if "t" in dists and garch["t"]._res.nu_at_bound:
        flags.append(_flag("L7", "info", "nu is at its upper bound: no fat tails beyond what GARCH explains",
                           "dist='normal'"))
    if sb_p < alpha:
        flags.append(_flag("L8", "warn", f"sign-bias test rejects (p = {sb_p:.3g}): volatility responds asymmetrically "
                           "(leverage); a symmetric GARCH fit then biases the intercept upward (E4)",
                           "take the intercept / long-run mean from the SARIMAX fit; GJR is not in the package yet"))
    for name, r in models.items():
        if name == "SARIMAX":
            continue
        rr = r._res
        if not rr.invertible or rr.ma_near_unit_root:
            flags.append(_flag("L9", "warn", f"{name}: MA root on or near the unit circle (conditional likelihood "
                               "unreliable there; often over-differencing)",
                               "compare ma.L* with the SARIMAX fit (exact likelihood, invertibility enforced); try d-1"))
            break
    if any(r._res.arma_cancellation for n, r in models.items() if n != "SARIMAX"):
        flags.append(_flag("L9", "warn", "AR and MA roots nearly cancel: orders too high", "lower p and q"))
    if any(r._res.near_unit_root for n, r in models.items() if n != "SARIMAX"):
        flags.append(_flag("L9", "warn", "AR root near the unit circle", "consider d+1; decide with backtest()"))
    if not all(r.converged for n, r in models.items() if n != "SARIMAX"):
        flags.append(_flag("L10", "info", "an engine fit reports converged=False (often spurious: the flag is "
                           "L-BFGS-B's only)", "compare() already refits GARCH from a second start and keeps the best"))

    # --- recommendation ---------------------------------------------------------------------
    # One decision test, not a union (a union of tests over-rejects): the LR test within the engine (E6: 3-4%
    # false positives, the most power of the correctly sized rules), except near an MA unit root, where the
    # conditional likelihood's start-up inflates it (11% at theta = .9) and ARCH-LM on SARIMAX's residuals
    # (exact likelihood) decides instead.
    ma_r = g_ref.maroots
    near_ma = bool(ma_r.size and np.min(np.abs(ma_r)) < 1.25)
    if near_ma:
        flags.append(_flag("L9", "info", "an MA root lies within 1.25 of the unit circle: the engine's LR test for GARCH "
                           "over-rejects there (11% at theta = .9, E6)", "the GARCH decision uses ARCH-LM on the SARIMAX "
                           "residuals instead (done here)"))
    decide_p = lm_p[0] if near_ma else lr[ref]
    decide_by = f"ARCH-LM({arch_lags[0]})" if near_ma else "LR within the engine"
    arch_signal = decide_p < alpha
    caveats = []
    if nobs < 100:
        model, reason = "SARIMAX", f"{nobs} observations are too few for GARCH (L3)"
    elif not arch_signal:
        model = "SARIMAX"
        reason = (f"no ARCH evidence ({decide_by} p = {decide_p:.3g}; ARCH-LM p = "
                  f"{', '.join(f'{v:.3g}' for v in lm_p)}); a GARCH fit would only add noise (E6)")
    elif g_ref._res.at_boundary:
        model, reason = "SARIMAX", "the GARCH fit ends on a boundary: no usable volatility dynamics (L5)"
    else:
        dist = "normal"
        if ("t" in dists and table.loc["engine GARCH(1,1) (t)", "aic"] < table.loc["engine GARCH(1,1) (normal)", "aic"]
                and not garch["t"]._res.nu_at_bound and not any(f["code"] == "L7" and f["level"] == "warn" for f in flags)):
            dist = "t"
        model = f"SARIMAX + GARCH(1,1), dist='{dist}'"
        reason = (f"ARCH evidence ({decide_by} p = {decide_p:.3g}; ARCH-LM p = {', '.join(f'{v:.3g}' for v in lm_p)})"
                  + ("; t preferred by AIC within the engine" if dist == "t" else ""))
        if nobs < 250:
            caveats.append("weakly identified at this size: bootstrap() CIs for omega/alpha/beta (L3)")
        if any(f["code"] == "L8" for f in flags):
            caveats.append("leverage detected: take the intercept from SARIMAX (L8)")
        caveats.append("bands: conf_int(method='simulated'); they ignore parameter uncertainty (L12, L13)")
        caveats.append("if you keep SARIMAX anyway, fit it with cov_type='robust': its default 'opg' SEs cover the "
                       "AR/MA coefficients only 0.83-0.92 of the time under GARCH (E6)")
    use_for = ("short-horizon intervals / VaR and efficient mean estimates; point forecasts and long-horizon bands "
               "change little (L11)" if model != "SARIMAX" else
               "point forecasts and intervals (normal bands are exact for this model)")
    recommendation = dict(model=model, reason=reason, caveats=caveats, use_for=use_for)
    if ho_dm is not None:
        diff, st, pv, n_ho = ho_dm
        recommendation["holdout_logscore_diff"] = diff
        caveats.append(f"holdout ({n_ho} obs, parameters fitted on the rest): GARCH({ref}) - SARIMAX = {diff:+.4f} "
                       f"nats/obs, DM p = {pv:.3g}")
        if model != "SARIMAX" and pv < alpha and st < 0:
            caveats.append("out of sample SARIMAX scores significantly better: prefer it for forecasting")
        elif model == "SARIMAX" and pv < alpha and st > 0:
            caveats.append("out of sample GARCH scores significantly better: consider it for forecasting")

    bt = None
    if backtest is not None:
        bkw = dict(horizon=12, start=None, step=1, n_sims=1000)
        bkw.update(backtest)
        bt = {"SARIMAX": SARIMAXGARCH(endog, **kw).backtest(**{k: v for k, v in bkw.items() if k != "n_sims"})}
        dist = "t" if "dist='t'" in model else "normal"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            bt[f"GARCH ({dist})"] = SARIMAXGARCH(endog, garch=(1, 1), dist=dist, **kw).backtest(**bkw)
        better = float(bt[f"GARCH ({dist})"]["crps"].mean() / bt["SARIMAX"]["crps"].mean())
        recommendation["backtest_crps_ratio"] = better
        recommendation["caveats"].append(f"backtest: GARCH/SARIMAX mean CRPS = {better:.3f} (< 1: GARCH forecasts "
                                         "better at these horizons)")
    return CompareResults(table, tests, flags, recommendation, models, bt)


def _fit_garch(endog, kw, dist, sm):
    """GARCH fit from the default start and from SARIMAX's mean estimates; keep the higher likelihood (L10)."""
    mod = SARIMAXGARCH(endog, garch=(1, 1), dist=dist, **kw)
    best = mod.fit()
    try:
        start = {n: float(v) for n, v in sm.params.items() if n in mod.param_names}
        s2 = float(sm.params.get("sigma2", np.var(np.asarray(sm.resid)[kw["order"][1]:])))
        start.update({"omega": s2 * 0.05, "alpha.L1": 0.05, "beta.L1": 0.90})
        if dist == "t":
            start["nu"] = 8.0
        alt = mod.fit(start_params=pd.Series(start)[mod.param_names])
        if alt.llf > best.llf + 1e-8:
            best = alt
    except (ValueError, KeyError, FloatingPointError, np.linalg.LinAlgError):
        pass
    return best
