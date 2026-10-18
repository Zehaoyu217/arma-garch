# sarimax_garch API reference (v0.2)

Contents: 1 `SARIMAXGARCH` · 2 `fit` · 3 `fix_params` · 4 `backtest` · 5 `SARIMAXGARCHResults` ·
6 `GARCHPredictionResults` · 7 module functions · 8 errors and warnings · 9 the engine

```python
from sarimax_garch import SARIMAXGARCH, SARIMAXGARCHResults, GARCHPredictionResults, arch_lm_test, ljung_box
```

## 1. `SARIMAXGARCH(endog, exog=None, order=(1, 0, 0), seasonal_order=(0, 0, 0, 0), trend=None, garch=None, dist="normal", enforce_stationarity=True, enforce_invertibility=True, **sm_kwargs)`

| Argument | Meaning |
|---|---|
| `endog` | Series, 1-column DataFrame or array. A pandas index is kept for `resid`, `fittedvalues` and forecasts; the Series name becomes the dependent-variable name (default `"y"`). |
| `exog` | DataFrame (column names kept), Series (its name) or array (named `x1, x2, …`). With d ≥ 1, give it in levels: it is differenced together with y. |
| `order` | `(p, d, q)`, any non-negative integers. |
| `seasonal_order` | statsmodels backend only. Non-trivial with GARCH → `NotImplementedError` ("GARCH or seasonal: pick one"). |
| `trend` | SARIMAX semantics. `None`/`"n"`: no constant (default). `"c"`: constant of the (differenced) error equation, named `intercept` for every d. `"t"`, `"ct"`, iterables: statsmodels backend only. |
| `garch` | `None` (off), `(1, 1)` or `True`. Other orders → `NotImplementedError`. |
| `dist` | `"normal"` or `"t"` (standardized Student-t, ν estimated). `"t"` alone selects the GARCH backend with GARCH off. |
| `enforce_stationarity`, `enforce_invertibility` | passed to statsmodels; with the GARCH backend they only warn after the fit. |
| `**sm_kwargs` | any other SARIMAX argument (statsmodels backend only; raises with GARCH). |

Model attributes: `backend` (`"statsmodels"` or `"garch"`), `param_names`, `endog_name`, `exog_names`, `order`,
`seasonal_order`, `trend`, `garch`, `dist`.

## 2. `fit(cov_type=None, fixed_params=None, start_params=None, disp=False, polish=True, init="backcast", **kwargs)`

- **statsmodels backend**: returns statsmodels' `SARIMAXResultsWrapper`. `cov_type`, `start_params`, `disp`
  and `**kwargs` go to `SARIMAX.fit`; `fixed_params` is applied with statsmodels' `fix_params`. Unless you
  pass `maxiter` or `method`, the fit uses up to 1000 L-BFGS iterations (statsmodels' default is 50, which
  stops early on trending exog) and, if that still has not converged, a Nelder-Mead polish followed by an
  L-BFGS restart, keeping the best fit; a fit that still fails raises a `ConvergenceWarning`. Check
  `res.mle_retvals["converged"]`.
- **GARCH backend**: returns `SARIMAXGARCHResults`.
  - `cov_type`: `"robust"` (default; QMLE sandwich H⁻¹S′SH⁻¹), `"oim"` / `"approx"` (inverse Hessian), `"opg"`.
    statsmodels' own default is `"opg"` — mention this when comparing SEs across backends.
  - `fixed_params`: `{display name: value}`; fixed parameters get NaN SEs and are excluded from `df_model`.
    Fixing every parameter skips estimation (a filter: new data, old parameters).
  - `start_params`: Series/dict keyed by display names, or an array in `param_names` order.
  - `polish=False` skips the Nelder-Mead / SLSQP refinement stages (faster in loops, slightly less robust).
  - `init`: pre-sample variance rule, `"backcast"` (arch's) or `"sample"` (FCP 1996 / fGarch).
  - other `**kwargs` → `TypeError`.

## 3. `fix_params(params)` — context manager
```python
with mod.fix_params({"ar.L1": 0.5}):
    res = mod.fit()
```

## 4. `backtest(horizon=12, start=None, step=1, alpha=0.1, n_sims=2000, seed=0, fixed_params=None)`
Rolling-origin evaluation of this specification on its own data. For origins `t0 = start, start+step, …`
with `t0 + horizon ≤ T`: refit on observations `[0, t0)`, forecast `horizon` steps using the realised exog rows
`[t0, t0+horizon)`, and score against the realised y. `start` defaults to `max(T // 2, 100 + p + d)`.
Returns a DataFrame indexed by `h = 1..horizon` with columns `rmse`, `mae`, `crps` and
`coverage{100(1-alpha)}` (e.g. `coverage90`). CRPS is closed-form Gaussian on the statsmodels backend and
computed from `n_sims` simulated paths on the GARCH backend. Successive fits are warm-started.
`result.attrs["n_origins"]` and `result.attrs["n_not_converged"]` record the number of refits and how many
did not converge; a warning is raised when any did.

## 5. `SARIMAXGARCHResults` (GARCH / t backend)

| Kind | Names |
|---|---|
| estimates (Series, statsmodels order) | `params`, `bse`, `zvalues` (= `tvalues`), `pvalues` |
| fit statistics | `llf`, `nobs` (= T − d − p: conditional likelihood), `df_model` (free parameters), `aic`, `bic`, `hqic`, `converged`, `cov_type`, `fixed_params`, `param_names`, `model` |
| series on the original index (first d + p values NaN) | `resid`, `fittedvalues`, `conditional_variance`, `conditional_volatility`, `std_resid` |
| GARCH / ARMA summaries | `persistence` (α+β), `half_life` (periods), `unconditional_variance`, `stationary`, `invertible`, `arroots`, `maroots` |
| inference | `cov_params()` (DataFrame), `conf_int(alpha=0.05)` (columns 0, 1), `summary(alpha=0.05)` (statsmodels `Summary`), `bootstrap(n_boot=199, method="residual"|"parametric", seed=None)` (DataFrame `se`, `ci_lower`, `ci_upper`) |
| diagnostics (on `std_resid`) | `test_serial_correlation(lags=None)` (Ljung-Box DataFrame `lb_stat`, `lb_pvalue`), `test_arch(lags=12)` (Series `lm_stat`, `lm_pvalue`, `f_stat`, `f_pvalue`), `test_normality()` (Series `jb_stat`, `jb_pvalue`, `skew`, `kurtosis`), `plot_diagnostics(lags=24, figsize=(10, 8))` |
| forecasting | `get_forecast(steps=1, exog=None, n_sims=10000, seed=None)` → `GARCHPredictionResults`; `forecast(steps, exog)` → Series of means; `get_prediction()` → in-sample one-step predictions; `simulate(nsimulations, repetitions=1, exog=None, seed=None)` → DataFrame (steps × repetitions) of future paths |

Parameter names: `intercept` (when `trend="c"`), exog column names, `ar.L1…`, `ma.L1…`, then `omega`,
`alpha.L1`, `beta.L1` (GARCH on) or `sigma2` (GARCH off with `dist="t"`), then `nu` (`dist="t"`).
`summary()` adds Ljung-Box, ARCH-LM and Jarque-Bera on the standardized residuals, persistence, half-life and
the estimator's warnings.

## 6. `GARCHPredictionResults` (from `get_forecast` / `get_prediction`)

| Name | Meaning |
|---|---|
| `predicted_mean` | E[y_{T+h}] (levels when d ≥ 1), Series with the forecast index |
| `var_pred_mean`, `se_mean` | Var(y_{T+h}) and its square root: accumulates ARMA dynamics and integration |
| `conditional_variance`, `conditional_volatility` | E[σ²_{T+h}] and its square root: the GARCH volatility path (equal to `var_pred_mean` at h = 1) |
| `sims` | ndarray (n_sims × steps) of simulated paths, or None when `n_sims=0` |
| `conf_int(alpha=0.05, method="normal")` | DataFrame `lower {y}`, `upper {y}`; `method="simulated"` uses quantiles of `sims` |
| `summary_frame(alpha=0.05, method="normal")` | DataFrame `mean`, `mean_se`, `mean_ci_lower`, `mean_ci_upper` (statsmodels column names) |
| `quantile(q)` | Series of simulated quantiles |

The forecast index continues the data index: dates if `endog` has a `DatetimeIndex` with a known or
inferable frequency (or a `PeriodIndex`), otherwise integers.

## 7. Module functions
- `arch_lm_test(resid, lags=12)` → `(statistic, p_value)`: Engle's ARCH-LM (n·R² of e²ₜ on its lags; equals
  statsmodels' `het_arch`). Use it on constant-variance residuals before deciding on GARCH.
- `ljung_box(x, lags=12, df_adjust=0)` → `(Q, p_value)`; set `df_adjust = p + q` for ARMA residuals.

## 8. Errors and warnings

| Message (start) | Cause | Fix |
|---|---|---|
| `GARCH or seasonal: pick one` | `seasonal_order` with GARCH / t | seasonal Fourier terms or month dummies in exog |
| `trend='t': with GARCH / t errors only None or 'c'` | SARIMAX time trend with GARCH | `trend="c"`, plus a time column in exog if a deterministic trend is needed |
| `[...]: statsmodels-only options` | SARIMAX keyword with GARCH | drop it or use `garch=None` |
| `only GARCH(1,1) is implemented` | `garch=(p, q)` other than (1, 1) | `garch=(1, 1)` |
| `unknown parameter 'x'` | wrong name in `fixed_params` / `start_params` | use `mod.param_names` |
| `the model has exog (...); pass exog with N rows` / `exog for the forecast must have N rows` | missing or wrong-length future exog | give exactly `steps` rows, same columns |
| `y has N NaN/inf value(s)` | gaps (GARCH backend) | drop or fill; or `garch=None`, which handles NaN |
| `the series is constant` | nothing to fit | — |
| `X is rank-deficient (... duplicates const)` | constant column, collinear columns, or all month dummies with `trend="c"` | drop the column / `drop_first=True` |
| `no forecast origins` | `start + horizon > T` in `backtest` | smaller `start` or `horizon` |
| `ConvergenceWarning: statsmodels SARIMAX did not converge` | optimiser failed even after the polish | centre / rescale exog; simplify the model |
| warning `backtest: k of n refits did not converge` | some rolling refits failed | centre exog, larger `step`, or check those origins |
| warning `std(series)=...; rescale` | data scale outside 0.05–50 | rescale (e.g. returns × 100) |
| warning `... observations: GARCH ... unreliable` / `weakly identified below ~250` | short sample | see SKILL.md step 3 |

## 9. The engine
`sarimax_garch._core` is a byte-identical copy of the repo's `fixed/joint_armax_garch.py` (a single numpy/scipy
file) and can be used directly: `fit_joint`, `forecast`, `simulate`, `bootstrap`, `backtest`, `arch_lm_test`,
`ljung_box`. Its self-test: `python -c "from sarimax_garch import _core; _core._selftest()"` (the published-
benchmark check runs only when `dem2gbp_benchmark.csv` is in the working directory). The engine's own
parameter names are `const`, `phi1`, `theta1`, `gamma1`, `omega`, `alpha`, `beta`, `nu`; the package
translates them to the statsmodels names.
