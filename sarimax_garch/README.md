# sarimax-garch

statsmodels-`SARIMAX`-style regression with ARIMA errors, with **GARCH(1,1) and Student-t errors as an opt-in**.

- `garch=None` (default): the model **is** `statsmodels.tsa.statespace.sarimax.SARIMAX` — `fit()` returns its
  results object, nothing is re-implemented, every SARIMAX option (including `seasonal_order`) passes through.
- `garch=(1, 1)` or `dist="t"`: the joint conditional MLE of regression with ARIMA(p,d,q)-GARCH(1,1) errors
  (`sarimax_garch._core`, the verified engine), returning a results object that keeps the SARIMAX attribute
  names and output style.
- **GARCH or seasonal: pick one.** The joint estimator has no seasonal ARMA terms; with `garch=(1, 1)` a
  non-trivial `seasonal_order` raises. Model seasonality through `exog` (month dummies or sin/cos Fourier
  terms) instead.

```python
from sarimax_garch import SARIMAXGARCH, arch_lm_test

res = SARIMAXGARCH(y, exog=X, order=(1, 0, 1), trend="c").fit()                  # == statsmodels SARIMAX
res = SARIMAXGARCH(y, exog=X, order=(1, 0, 1), trend="c", garch=(1, 1)).fit()    # ARMA-GARCH, normal
res = SARIMAXGARCH(y, exog=X, order=(1, 1, 0), trend="c", garch=(1, 1), dist="t").fit()   # ARIMA-GARCH-t with drift

res.params, res.bse, res.pvalues, res.llf, res.aic, res.bic, res.resid, res.fittedvalues
print(res.summary())
fc = res.get_forecast(12, exog=X_future)
fc.predicted_mean, fc.se_mean, fc.conf_int(), fc.summary_frame()      # statsmodels names
fc.conf_int(method="simulated")                                         # exact multi-step bands under GARCH / t
```

## Model

```
y_t = x_t' beta + u_t                                   (regression; exog in levels, differenced with y when d > 0)
(1 - L)^d u_t = c + phi(L) ... + theta(L) eps_t          (ARIMA(p, d, q) errors; c = "intercept": with d >= 1 the level's drift)
eps_t = sigma_t z_t,  sigma_t^2 = omega + alpha eps_{t-1}^2 + beta sigma_{t-1}^2,   z ~ N(0, 1) or t(nu)
```
With `garch=None` this is exactly SARIMAX's "regression with ARIMA errors"; with GARCH the mean and variance
parameters are estimated in one likelihood (so the mean estimates are implicitly GLS-weighted).

## API parity with statsmodels SARIMAX

| statsmodels | here (GARCH backend) | notes |
|---|---|---|
| `SARIMAX(endog, exog, order, seasonal_order, trend, ...)` | `SARIMAXGARCH(..., garch=None/(1,1), dist="normal"/"t")` | `seasonal_order` only with `garch=None` |
| `trend=None` | same: **no constant** (statsmodels' default) | `"c"`: the constant of the (differenced) equation, named `intercept` for every d — with d ≥ 1 it is the drift of the level, exactly as in SARIMAX. `"t"`/`"ct"` (SARIMAX's time trend *inside* the differenced equation, its `drift` parameter) work only with `garch=None`; with GARCH put a time column in `exog` |
| `fit(cov_type=..., disp=...)` | `fit(cov_type="robust"|"oim"|"opg", fixed_params=..., start_params=...)` | default cov is the QMLE sandwich (statsmodels' default is `opg`). With GARCH off, the statsmodels fit gets 1000 L-BFGS iterations plus a Nelder-Mead polish if needed (statsmodels' default of 50 iterations silently stops early on trending exog); pass `maxiter`/`method` for statsmodels' exact defaults |
| `with mod.fix_params({...}): mod.fit()` | same | names as in `params` |
| `params, bse, tvalues, pvalues, llf, aic, bic, hqic, nobs` | same names | `intercept`/`drift`, exog names, `ar.L1`…, `ma.L1`…, then `omega`, `alpha.L1`, `beta.L1`, `nu`; with `dist="t"` and no GARCH: `sigma2`, `nu` |
| `conf_int(alpha)`, `cov_params()` | same | |
| `resid`, `fittedvalues` | same, on the original index | first `d + p` values are NaN (conditional likelihood) |
| `summary()` | same (statsmodels `Summary` object) | adds ARCH-LM, persistence, half-life, and the estimator's warnings |
| `get_forecast(steps, exog)` → `predicted_mean, se_mean, var_pred_mean, conf_int(), summary_frame()` | same | plus `.conditional_variance` / `.conditional_volatility` (the GARCH σ²ₜ₊ₕ path — the volatility forecast; `var_pred_mean` is Var(yₜ₊ₕ)), `.sims`, `.quantile(q)`, `conf_int(method="simulated")` |
| `forecast(steps, exog)` | same | |
| `get_prediction()` | in-sample one-step predictions, se = conditional volatility | no dynamic prediction |
| `simulate(nsimulations, repetitions, exog)` | same (future paths) | |
| `test_serial_correlation()` | Ljung-Box on standardized residuals (DataFrame) | |
| `test_heteroskedasticity()` | `test_arch()` (ARCH-LM on standardized residuals) | |
| `test_normality()` | same (Jarque-Bera) | |
| `plot_diagnostics()` | same 2×2 panel | needs matplotlib |
| — | `conditional_variance`, `conditional_volatility`, `std_resid`, `persistence`, `half_life`, `unconditional_variance` | GARCH extras |
| — | `bootstrap(n_boot)` | small-sample CIs for the GARCH parameters |
| — | `SARIMAXGARCH.backtest(horizon, start, step)` | rolling-origin RMSE / CRPS / coverage by horizon, both backends |

Not available with GARCH / t: seasonal ARMA, `time_varying_regression`, `measurement_error`,
`concentrate_scale`, `simple_differencing`, missing values, exact (Kalman) likelihood (the engine uses the
conditional likelihood; estimates agree with SARIMAX to O(1/T)).

## Small samples

Below ~250 observations the GARCH parameters are weakly identified (the fit warns). Test for ARCH first on the
constant-variance fit — `arch_lm_test(res.resid.dropna())` or `res.test_arch()` — and prefer `dist="normal"`;
use `res.bootstrap()` for the GARCH confidence intervals. For multi-year forecasts GARCH changes little; the
levels-vs-differences choice matters far more — compare `d=0` and `d=1` with `backtest()` by CRPS and coverage.

## Install / test

```
pip install -e .            # needs pip >= 21.3 (PEP 660); or `pip install .`
pytest                      # 16 tests incl. parity with statsmodels, sync with the standalone engine, and the skill's code examples
```
`src/sarimax_garch/_core.py` is a verbatim copy of `../fixed/joint_armax_garch.py` (v3.2), the engine validated
1:1 against rugarch, statsmodels and arch in `../verification/`; a test keeps the two in sync.
