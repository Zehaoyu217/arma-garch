---
name: sarimax-garch
description: Fit, forecast and validate regression-with-ARIMA-errors models with optional GARCH(1,1) volatility and Student-t errors using the sarimax_garch package (class SARIMAXGARCH), whose API mirrors statsmodels SARIMAX. Use this skill whenever the user mentions sarimax_garch, SARIMAXGARCH, joint_armax_garch or the arma-garch repo; asks for ARIMA-GARCH, ARMA-GARCH(-t), ARIMAX or SARIMAX "with GARCH", time-varying volatility or fat-tailed errors together with exogenous / macro regressors; wants volatility (sigma) forecasts from such a model; builds macro- or scenario-conditioned forecasts where the exog path is given and must choose levels vs differences or GARCH vs constant variance; or wants to add GARCH to existing statsmodels SARIMAX code — even if they only say "add GARCH to my SARIMAX" or "fit a GARCH-t with regressors".
---

# sarimax-garch

`SARIMAXGARCH` is statsmodels' `SARIMAX` with one extra switch: `garch=(1, 1)` (and/or `dist="t"`) turns on
GARCH(1,1) errors estimated jointly with the mean. The model is **regression with ARIMA errors** — the
SARIMAX formulation, so β is the effect of x on the level of y:

```
y_t = x_t' beta + u_t
(1 - L)^d u_t = c + phi_1 (1-L)^d u_{t-1} + ... + theta_1 eps_{t-1} + ... + eps_t
eps_t = sigma_t z_t,   sigma_t^2 = omega + alpha eps_{t-1}^2 + beta sigma_{t-1}^2,   z ~ N(0,1) or standardized t(nu)
```

The estimator is validated 1:1 against R rugarch, statsmodels SARIMAX and Python `arch` and by Monte Carlo
(see `references/evidence.md`), so put your effort into the modelling decisions below, not into re-checking
the numerics.

## Setup

```bash
python -c "import sarimax_garch; print(sarimax_garch.__version__)"          # installed already?
pip install -e /path/to/arma-garch/sarimax_garch                              # from a local clone
pip install "git+https://github.com/Zehaoyu217/arma-garch.git#subdirectory=sarimax_garch"   # private repo: needs GitHub access
```
Needs Python ≥ 3.9, numpy, scipy, pandas, statsmodels (matplotlib only for `plot_diagnostics`).

## Two backends — always know which one you are on

| You pass | Backend | `fit()` returns |
|---|---|---|
| `garch=None`, `dist="normal"` (default) | **statsmodels SARIMAX** (exact Kalman likelihood) | statsmodels' own results object: every SARIMAX feature works, incl. `seasonal_order` and missing values |
| `garch=(1, 1)` and/or `dist="t"` | joint conditional MLE (the package's engine) | `SARIMAXGARCHResults`: the statsmodels names (`params`, `bse`, `summary()`, `get_forecast()`, …) plus GARCH extras |

Consequences you will hit:
- **GARCH or seasonal ARMA: pick one.** `seasonal_order` raises with GARCH; model seasonality through exog
  instead (Fourier terms or month dummies — recipe below).
- statsmodels-only keyword arguments (`simple_differencing`, `measurement_error`, …) raise with GARCH, and
  the GARCH backend needs a gap-free series (no NaN).
- `trend` follows SARIMAX exactly. `trend=None` is **no constant** (SARIMAX's default — you almost always want
  `trend="c"`). `"c"` is the constant of the (differenced) equation, named `intercept` for every d; with
  d ≥ 1 it is the drift of the level. SARIMAX's `"t"`/`"ct"` (a time trend *inside* the differenced
  equation) work only without GARCH; with GARCH add a time column to exog.
- Log-likelihoods, AIC and BIC are comparable only within one backend and one d: the statsmodels backend uses
  the exact likelihood on all T observations, the GARCH backend the conditional likelihood on T − d − p.

## Workflow

Work through these in order; each step's reason is what to tell the user when you report.

### 1. Prepare the data
- `y`: a pandas Series with a `DatetimeIndex` that has a frequency (`y = y.asfreq("MS")` for monthly
  start-of-month data) — forecasts then come back with real dates.
- `X`: a DataFrame with named columns. Future exog for forecasting must have the same columns and exactly
  `steps` rows. With d ≥ 1 give X (and future X) in the same units as when fitting (levels); the model
  differences them itself, as SARIMAX does.
- Keep the scale of y moderate (std roughly 0.1–10; returns in percent, not decimals). The optimiser is tuned
  for that range, and the fit warns outside it.
- For GARCH: drop or fill NaN first.

### 2. Mean model: levels or differences
Fit the levels model with constant variance first and look at the AR root:
```python
import numpy as np, pandas as pd
from sarimax_garch import SARIMAXGARCH, arch_lm_test

res0 = SARIMAXGARCH(y, exog=X, order=(1, 0, 0), trend="c").fit(disp=False)   # statsmodels backend
np.abs(res0.arroots).min()        # close to 1 (below ~1.02): the level relation is spurious -> difference
```
Why it matters: with AR errors the levels model *is* an error-correction model — y is pulled back towards
β'x at speed 1 − φ. If y − β'x is really a random walk, a levels model is overconfident (in simulation its
90% bands covered only 55–70% from a year out); if it is stationary, differencing throws the anchor away
(35–75% worse CRPS at three years). Confirm the choice out of sample with `backtest()` (step 7). Never compare
AIC/BIC between different d — they are fitted to different data.

### 3. GARCH or constant variance
```python
stat, p = arch_lm_test(res0.resid.iloc[res0.loglikelihood_burn:], lags=12)   # ARCH-LM on the residuals
```
- p > 0.05: keep `garch=None`. The test has little power in short samples, so when it does not reject, a
  GARCH fit would mostly add noise.
- Fewer than ~100 observations after differencing: do not fit GARCH. 100–250: the GARCH parameters are
  weakly identified (at T = 120 about half the fits end with α or β on the boundary and the α/β Wald CIs
  cover ~60%); fit only if ARCH-LM rejects, use `dist="normal"` and `bootstrap()` CIs. The fit itself
  warns below 250.
- Ask what the model is for. Volatility / VaR over the next 1–6 periods: GARCH pays. A multi-year mean path
  under a given macro scenario: GARCH changes the forecast distribution of y very little (CRPS within ~0.02 of
  constant variance at every horizon in simulation) — skip it unless volatility is highly persistent.
- Monthly data: a GARCH variance forecast beats the sample variance from about 12–15 years with a typical ARCH
  effect (α ≈ .10), ~20 years if weak (α ≈ .05), under 10 if strong (α ≈ .15) or highly persistent (α+β ≈ .98).

### 4. Error distribution
- `dist="normal"` (default) is the safe choice: it is a QMLE, and the default robust standard errors stay valid
  when the true errors are fat-tailed.
- `dist="t"` is more efficient when the tails really are t-like and the sample is large (several hundred
  observations or more). If `summary()` warns that ν is at its upper bound (200), the data show no fat tails —
  refit with `dist="normal"`. In small samples ν is not identified: use `"normal"`.

### 5. Fit and read
```python
mod = SARIMAXGARCH(y, exog=X, order=(1, 0, 0), trend="c", garch=(1, 1))
res = mod.fit()                                   # robust (sandwich) SEs; cov_type="oim" or "opg" also available
print(res.summary())                              # read the WARNING lines at the bottom
res.params, res.bse, res.pvalues, res.conf_int(alpha=0.05)
res.persistence, res.half_life, res.unconditional_variance
res.conditional_volatility                        # in-sample sigma_t; the first d + p values are NaN
```
Check convergence before reading anything: `res.mle_retvals["converged"]` on the statsmodels backend,
`res.converged` on the GARCH backend. statsmodels' own default (50 L-BFGS iterations) silently stops early
on regressions with trending exog — in testing it returned φ = 0.895 where the MLE is 0.733 — so the package
gives it 1000 iterations and a Nelder-Mead polish unless you pass `maxiter` or `method`. If a fit still has
not converged, centre the exog (subtract column means: the same model, better conditioned) and refit.
Parameter names follow statsmodels: `intercept`, the exog column names, `ar.L1…`, `ma.L1…`, then `omega`,
`alpha.L1`, `beta.L1`, `nu` (with `dist="t"` but no GARCH: `sigma2`, `nu`). What the summary warnings mean:

| Warning | Meaning | What to do |
|---|---|---|
| α+β at the 0.9999 cap | volatility is (near-)integrated | half-life / unconditional variance are meaningless; SEs unreliable — say so, consider a longer sample |
| GARCH coefficient at the boundary | α or β ≈ 0 | there may be no GARCH effect; β is unidentified when α = 0 — prefer `garch=None` |
| AR root near the unit circle | φ ≈ 1 | difference (d+1) and compare with `backtest()` |
| MA root near the unit circle | θ ≈ −1 | likely over-differenced: try d−1 |
| AR and MA roots nearly cancel | orders too high | lower p and q |
| ν at its upper bound | no fat tails | `dist="normal"` |
| fourth moment infinite | heavy tails + high persistence | mean-parameter SEs lack standard theory; treat with care |
| fewer than 250 observations | GARCH weakly identified | see step 3 |

### 6. Forecast with a given exog path
```python
fc = res.get_forecast(steps=24, exog=X_future)            # X_future: 24 rows, the same columns as X
fc.summary_frame(alpha=0.10)                              # mean, mean_se, mean_ci_lower, mean_ci_upper (90%)
fc.conf_int(alpha=0.10, method="simulated")               # exact bands under GARCH / t errors
fc.conditional_volatility                                 # GARCH sigma_{T+h}: "the volatility forecast"
```
- `se_mean` / `var_pred_mean` is the forecast-error variance of **y** — it accumulates the ARMA dynamics (and
  the integration when d ≥ 1). `conditional_volatility` is the GARCH σₜ₊ₕ path of the innovation. They are
  equal at h = 1 only. When someone asks for a volatility forecast, give `conditional_volatility`.
- The default bands are normal (mean ± z·se, as in statsmodels). Under GARCH or t errors the multi-step
  predictive distribution is not normal; use `method="simulated"` (quantiles of `fc.sims`) for the bands you
  report beyond one step.
- One call per scenario: for stress tests, build each X_future path and call `get_forecast` on each.
- On the statsmodels backend the same calls work, but there is no `conditional_volatility` (the innovation
  variance is the constant `params["sigma2"]`).

### 7. Decide specifications out of sample
```python
specs = {
    "levels":         dict(order=(1, 0, 0), trend="c"),
    "levels + GARCH": dict(order=(1, 0, 0), trend="c", garch=(1, 1)),
    "diffs":          dict(order=(1, 1, 1), trend="c"),
    "diffs + GARCH":  dict(order=(1, 1, 1), trend="c", garch=(1, 1)),
}
bt = {name: SARIMAXGARCH(y, exog=X, **kw).backtest(horizon=12, start=120, step=6) for name, kw in specs.items()}
crps = pd.concat({k: v["crps"] for k, v in bt.items()}, axis=1)              # lower is better, rows = horizon
cover = pd.concat({k: v["coverage90"] for k, v in bt.items()}, axis=1)       # want ~0.90
```
`backtest()` refits at each origin `start, start+step, …` on the data so far and forecasts with the *realised*
exog path (the "macro given" setting), returning RMSE, MAE, CRPS and interval coverage by horizon. CRPS is the
proper score of the whole predictive distribution, so it rewards both accuracy and honest bands; coverage
shows calibration. Pick the specification with the lowest CRPS at the horizons the user cares about, unless its
coverage is badly off. Keep `start` at 100+ observations when a spec has GARCH. Runtime is about (number of
origins) × (fit time), so widen `step` on long series. `bt.attrs["n_not_converged"]` counts refits that did
not converge (it also warns); if it is not 0, centre the exog and rerun before trusting the scores.

### 8. Diagnostics and small-sample inference
```python
res.test_serial_correlation(lags=12)     # Ljung-Box on the standardized residuals: remaining autocorrelation
res.test_arch(lags=12)                   # ARCH-LM on the standardized residuals: remaining ARCH
res.test_normality()                     # Jarque-Bera
res.plot_diagnostics()                   # 2x2 panel (matplotlib)
res.bootstrap(n_boot=199, seed=0)        # se, ci_lower, ci_upper per parameter
```
Below ~250 observations, report `bootstrap()` CIs for ω, α, β (Wald CIs badly under-cover there) but keep
the Wald CIs for the mean parameters — the percentile bootstrap inherits the AR small-sample bias and
under-covers for them.

## Recipes

**Seasonality with GARCH** (seasonal ARMA is not available with GARCH):
```python
def fourier(start, n, period=12, K=2, index=None):
    t = np.arange(start, start + n)
    return pd.DataFrame({f"{f}{k}": fn(2 * np.pi * k * t / period)
                         for k in range(1, K + 1) for f, fn in (("sin", np.sin), ("cos", np.cos))}, index=index)

future_index = pd.date_range(y.index[-1], periods=25, freq=y.index.freq)[1:]
X_season = fourier(0, len(y), index=y.index)                      # continue the time counter into the future
X_season_future = fourier(len(y), 24, index=future_index)
res_season = SARIMAXGARCH(y, exog=pd.concat([X, X_season], axis=1), order=(1, 0, 0), trend="c", garch=(1, 1)).fit()
fc_season = res_season.get_forecast(24, exog=pd.concat([X_future, X_season_future], axis=1))
```
Month dummies work too: `pd.get_dummies(y.index.month, prefix="m", drop_first=True, dtype=float)` with
`index=y.index`. Drop one month when `trend="c"`, otherwise the dummies duplicate the constant (the package
raises a rank-deficiency error).

**Porting statsmodels SARIMAX code**: the call is the same — add `garch=(1, 1)` and, if there is a
`seasonal_order`, replace it by seasonal exog as above. Keep `trend="c"`, rename nothing: parameter names and
`get_forecast(...).summary_frame()` columns match. Then compare the two in `backtest()` rather than by AIC.

**Fixed parameters and filtering new data**:
```python
with mod.fix_params({"ar.L1": 0.5}):                          # or mod.fit(fixed_params={...})
    res_fixed = mod.fit()
new = SARIMAXGARCH(y_new, exog=X_new, order=(1, 0, 0), trend="c", garch=(1, 1)).fit(
    fixed_params=dict(res.params))                            # every parameter fixed = run the filter only
```
`fit(start_params=res.params)` warm-starts from an earlier fit; fixing `{"alpha.L1": 0, "beta.L1": 0}` gives
the constant-variance model inside the GARCH backend (same likelihood family as the GARCH fit).

## References
- `references/api.md` — every argument, attribute and method with exact names and return types. Read it when
  you need a signature, an attribute that is not shown above, or the meaning of an error message.
- `references/evidence.md` — what has been verified and the simulation evidence behind the recommendations
  (small samples, GARCH vs constant variance, levels vs differences, multi-step scores). Read it when the user
  asks how reliable the package is, how much data GARCH needs, or why you recommend something.
