# Verification of `joint_armax_garch.py` against R rugarch, statsmodels and arch — 2026-10-17

**Verdict.** The current version is `../fixed/joint_armax_garch.py` **v3.1**: regression with
ARIMA(p,d,q)-GARCH(1,1) errors (the SARIMAX formulation, now the default), normal or Student-t errors.
- **Exact agreement:** its likelihood, recursions and forecasts match R **rugarch 1.5.6**, statsmodels **SARIMAX**
  and Python **arch** 1:1 to machine precision, across every ARMA order up to (2,2), d = 0/1/2 and five GARCH parameter sets.
- **Benchmark:** it reproduces the published FCP (1996) GARCH benchmark.
- **Monte Carlo:** 13,000 fits show consistent, asymptotically normal estimates with correctly sized robust SEs.

Version history:
- v2.2 fixed three numerical bugs in v2.1 (standard errors, and the optimiser at the α+β cap).
- v2.3 added edge-case handling, `fixed=` parameters and the regression-errors mode.
- v3.0 made the regression-errors mode the default.
- v3.1 added the small-sample tools (`arch_lm_test`, `ljung_box`, `bootstrap`, a warning below 250 obs) and
  capped ν in every optimiser stage (see "Small samples").
- v3.2 added `backtest()` (rolling-origin CRPS / RMSE / coverage by horizon with the macro path given) and the
  multi-horizon levels-vs-differences study.

The original `../joint_armax_garch.py` (v2.1) is untouched.

## v3.0 — SARIMAX formulation by default, and the full validation grid

The default is now `xreg="regression"`: u_t = y_t − x_t'γ follows the ARMA-GARCH, i.e.
statsmodels SARIMAX(exog=), rugarch `external.regressors`, R `arima(xreg=)`. ARMAX is still
available as `xreg="armax"`. The grid lives in `grid/`; run it with `GRID=1 ./run_all.sh`
(`MC_QUICK=1` for a quarter-size Monte Carlo).

| Check | Cases | Result |
|---|---|---|
| vs statsmodels SARIMAX, GARCH off: residuals and per-obs loglik at identical params | 7 (p,q) × d=0,1,2 × X/no X = 42 | ≤ 1.3e-15 / 5.3e-15 |
| vs SARIMAX: **level forecasts** (mean / variance) at identical params, d = 0, 1, 2 | 42 | mean ≤ 2e-9 on levels up to ~1e5 (≈1e-14 relative); variance ≤ 2.4e-15 relative |
| vs SARIMAX: free fits (exact vs conditional likelihood) | 42 | ≤ 0.2 SE; ARMA(2,2) up to 0.7 SE, and each package's estimate is best under its own likelihood |
| fit with d on levels vs d = 0 on the differences | 14 | identical (0.0) |
| GARCH-on forecasts vs 200k simulated paths, d = 0, 1, 2 | 9 | means within noise (z ~ N(0,1)); variances ≤ 1.1% |
| vs rugarch: filter and forecasts at identical params | 28 (all orders, normal/t, d = 0/1/2, 5 GARCH sets) | filter ≤ 4.6e-14 (t ≥ 1000); forecasts ≤ 5e-15 |
| vs rugarch: free fits | 28 | ≤ 0.2 SE; the script's optimum was never beaten under its own likelihood |
| vs Python `arch` (AR(p)-GARCH, ARMAX): σ_t and residuals at identical params | 26 | ≤ 4e-9 / 1.8e-15 |
| vs `arch`: estimates, robust SEs, Hessian SEs | 26 | ≤ 0.015 SE; SE ratios 0.992–1.002 / 0.996–1.001 |
| Monte Carlo: 48 designs (orders × d, 5 GARCH sets × normal/t5/t10, QMLE, T = 500–16000) | 13,000 fits | 0 failures; bias ≤ 0.44 SD; median coverage 0.947 |

**Asymptotics** (ARMA(1,1)+X, typical GARCH, t6):
- **Consistency:** SD·√T is flat across T, and bias/SD falls (ω 0.35 → 0.14, β −0.19 → −0.02).
- **SE calibration:** robust SE / true SD goes from 0.29–1.05 at T = 500 to 0.97–1.05 at T = 8000.
- **Coverage:** α coverage rises from 0.87 to 0.965.
- **Normality:** at T = 8000 the z-scores are N(0,1) (sd 0.987, 1.00% beyond 2.58, 0.11% beyond 3.29).
- **Finite samples:** at T ≤ 1000, GARCH-parameter intervals under-cover (0.86–0.90 where the GARCH effect is weak, i.e. low persistence or ARCH-like designs). This is the usual finite-sample behaviour of Wald intervals, and it shrinks with T.

**QMLE robustness** (t5 data fitted with `dist="normal"`): robust-SE coverage is 0.90, Hessian-SE coverage
0.74. Use `.se` (robust, the default) when you are not sure of the error distribution.

**rugarch problems found by the grid** (not script problems):
- **Wrong local optimum:** on ARMA(2,2) ARCH-like data `ugarchfit` stopped at α = 0, β = 0.985 and reported
  convergence. Under rugarch's own likelihood, the script's estimate is 50 log-likelihood units better.
- **Under-stated Hessian SEs near α+β = 1:** with persistence ≈ 0.993, rugarch's `matcoef` SEs for ω, α, β are about half
  the right size. Its default `numderiv.control$hess.d = 0.1` starts numDeriv at a 10% step, which
  crosses α+β = 1. With `hess.d = 1e-3` rugarch matches the script within 1% (`grid/rugarch_hessian_step.R`),
  and so does `arch`.

## Small samples — monthly data with 5–30 years (v3.1)

`grid/small_sample.py` (36,000 fits) and `grid/small_sample_extra.py`. Monthly-realistic designs: AR(1) or
ARMA(1,1) mean with a seasonal sin/cos pair as regressors, GARCH persistence 0.85 (ω .15, α .10, β .75; monthly
volatility is less persistent than daily), normal or t(6) errors; plus a no-GARCH design and a daily-like
high-persistence one (α .08, β .90). 600 replications per cell.

**Estimation, full ARMA-GARCH fit** (coverage = 95% Wald-robust CI; ranges across the three GARCH designs):

| T (years) | mean part: φ₁ cov / γ₁ cov | boundary (α or β = 0) | α coverage | β coverage (RMSE; truth .75) | ν at its bound (t fits) |
|---|---|---|---|---|---|
| 60 (5y) | 0.87 / 0.89 | 71–77% | 0.55–0.61 | 0.50–0.56 (0.27–0.33) | 29–34% |
| 120 (10y) | 0.92–0.94 / 0.91–0.93 | 46–51% | 0.59–0.67 | 0.56–0.58 (0.26–0.30) | 14–16% |
| 180 (15y) | 0.93–0.94 / 0.93–0.94 | 29–32% | 0.75–0.77 | 0.67–0.68 (0.25–0.28) | 5–6% |
| 360 (30y) | 0.94–0.95 / 0.94–0.95 | 9–13% | 0.80–0.85 | 0.74–0.77 (0.21–0.23) | 0% |

φ₁ has the usual small-sample bias (−0.03 at T = 120, −0.08 at 60), the same as any ARMA package.

**Forecasting** (QLIKE loss of the 1-step variance forecast against the true σ²_{T+1}; 0 = perfect; "no-GARCH"
= a fit with α = β = 0, i.e. constant variance):

| T | GARCH DGPs, persistence .85: full / no-GARCH | high persistence .98: full / no-GARCH | no-GARCH DGP: full / no-GARCH |
|---|---|---|---|
| 60 | 0.10–0.25 / 0.06–0.10 | 0.073 / 0.066 | 0.069 / 0.027 |
| 120 | 0.04–0.08 / 0.04–0.07 | 0.040 / 0.089 | 0.029 / 0.010 |
| 180 | 0.02–0.05 / 0.04–0.07 | 0.031 / 0.094 | 0.016 / 0.006 |
| 360 | 0.01–0.02 / 0.04–0.06 | 0.015 / 0.134 | 0.007 / 0.003 |

Point forecasts at h = 1 and 12 are within 1–8% of the oracle's RMSE for every fit variant: the GARCH choice
does not matter for them. `dist="t"` never beats `dist="normal"` on forecasts at T ≤ 180.

**When does the GARCH variance forecast beat the sample variance?** It depends on the strength of the ARCH
effect as much as on the years (`grid/small_sample_threshold.py`, AR(1) + seasonal X, normal errors, GARCH fit
with `dist="normal"` vs the constant-variance fit, 600 reps). Each cell: loss ratio GARCH / constant (< 1 =
GARCH better) · share of samples in which GARCH wins · share in which the ARCH-LM pre-test rejects at 5%.

| ARCH strength (α, β; persistence) | 10 years | 15 years | 20 years | 25 years | 30 years |
|---|---|---|---|---|---|
| weak (.05, .85; .90) | 1.58 · 45% · 7% | 1.25 · 46% · 11% | 0.89 · 54% · 15% | 0.75 · 56% · 18% | 0.71 · 60% · 22% |
| typical (.10, .75; .85) | 1.09 · 49% · 10% | 0.59 · 59% · 20% | 0.48 · 64% · 28% | 0.37 · 69% · 38% | 0.33 · 73% · 43% |
| strong (.15, .75; .90) | 0.38 · 66% · 22% | 0.24 · 74% · 46% | 0.20 · 71% · 63% | 0.16 · 81% · 73% | 0.13 · 82% · 82% |
| daily-like (.08, .90; .98) | 0.51 · 62% · 16% | 0.34 · 71% · 30% | 0.22 · 77% · 47% | 0.13 · 81% · 58% | 0.12 · 84% · 69% |

So the break-even point is ~20 years for weak ARCH, ~12–15 for typical, and under 10 for strong or highly
persistent ARCH. Even where GARCH wins on average, it loses in 25–45% of individual samples at 10–15 years.
The ARCH-LM pre-test has low power at these sizes (10–20% for typical ARCH), so when it rejects, the effect in
that sample is strong enough to be worth modelling; when it does not, little is lost by keeping the constant
variance.

**Simulator verification** (`grid/simulator_check.py`, T = 500,000, no fitting code involved): the simulated
series match the model's theoretical unconditional variance, kurtosis, ACF of ε² at lags 1–2, shock moments and
independence (Ljung-Box on z and z²), the ARMA part's mean/variance/ACF, and the regression coefficients by OLS;
the σ² recursion rebuilt from ε is exact and σ²_{T+1} depends only on data up to T. Together with the oracle rows
above (QLIKE = 0.000 with the true parameters) and the large-T recovery in the main grid, this pins down that the
small-sample results are properties of the model, not of the code.

**Follow-ups** (`small_sample_extra.py`):
- *Bootstrap vs Wald CIs* (150 reps × 199 residual-bootstrap refits). At T = 120 the coverage for ω / α / β is
  0.53 / 0.57 / 0.49 (Wald robust) vs 0.85 / 0.73 / 0.93 (bootstrap percentile); at 180, 0.65 / 0.73 / 0.66 vs
  0.94 / 0.77 / 0.91 (normal DGP; the t(6) DGP is the same). For the mean parameters the percentile bootstrap
  *under*-covers (0.81–0.84 at 120 vs 0.90–0.95 Wald) because it inherits the AR small-sample bias.
- *Same 150 series in rugarch* (T = 120/180): RMSE of φ₁ and α identical to the script's; rugarch ends on a
  boundary more often (66% / 51% vs 47% / 31%); ν ends at its upper bound in 7% / 4% (script) and 6% / 1%
  (rugarch) of these fits, and in 14–16% of the main grid's t fits at T = 120 (another ~5% end with ν > 50:
  no detectable fat tails although the truth is t(6)). The small-sample behaviour is in the data, not the
  implementation.
- *ARMA part only (no GARCH)*, script's conditional likelihood vs SARIMAX's exact likelihood at T = 60/120/180:
  RMSEs equal to within 0.012 (T = 60) and 0.002 (T ≥ 120); no MA pile-up in either.

**Guidance for 10–15 years of monthly data** (the fit-time warning below 250 obs says the same):
1. Fit the mean model without GARCH first: `fit_joint(y, X=X, p=p, q=q, dist="normal", fixed={"alpha": 0, "beta": 0})`.
   The ARMA and regression estimates are fine at this size (coverage 0.91–0.95, same RMSE as exact MLE).
2. `arch_lm_test(base.eps, 12)`: if p > 0.05 keep the constant variance — a GARCH fit would only add noise
   (QLIKE 0.029 vs 0.010 on homoskedastic data at T = 120).
3. If ARCH is present, fit `dist="normal"` (ν is not identified at this size), expect a boundary solution in
   about half the samples, and report `bootstrap()` CIs for ω/α/β and Wald CIs for the mean parameters.
4. Expect the GARCH variance forecast to beat the sample variance only from ~15 years on (30–40% lower loss at
   180, 60–70% at 360), or sooner when volatility is highly persistent (α+β ≈ .98: already at 10 years).
5. Below ~8 years (100 obs) do not fit GARCH at all.

**v3.1 fix found by this study:** ν had no upper wall in the Nelder-Mead stage, so on data without fat tails
it could run off to 10⁶ (the ν RMSE in the rugarch comparison was 381,248 before the fix and 58 after; rugarch's
own ν runs to its bound of 100 in the same samples). It is now capped at 200 like the other stages, and a
30-sample self-test guards it (the pre-fix code fails it: 23 of 30 fits past 200, max 5×10⁶). Only the reported
ν and its SE were affected — the likelihood is flat in ν there, so the other estimates are identical before and
after.

## Multi-step, macro-conditioned forecasts: levels vs differences, GARCH or not (v3.2)

`grid/horizon_study.py`. The macro path X (two random walks) is *given* over a 36-month horizon. DGP **L**:
y is cointegrated with X (y = c + β'X + u, u AR(1) φ = .8). DGP **D**: Δy = c + β'ΔX + u (φ = .3; y is I(1)
and its level is not tied to X). Every sample is fitted as a levels model (d = 0) and as a differences model
(ARIMA(1,1,1)), each with and without GARCH; 300 reps per cell; scored against an oracle that knows the true
form and parameters. Scores: RMSE of the point forecast, coverage of the 90% interval (simulated quantiles),
and CRPS — the proper score of the whole predictive distribution (lower is better; it is the absolute error
for a point forecast). "Better" = lower CRPS and coverage nearer 0.90.

CRPS / oracle (90% coverage in brackets) at 1, 12 and 36 months, typical GARCH persistence, GARCH fitted:

| truth | T | levels model (d = 0) | differences model (d = 1) |
|---|---|---|---|
| L (cointegrated) | 120 | 1.02 (.86) · 1.12 (.83) · 1.18 (.79) | 1.03 (.86) · 1.44 (.84) · 1.76 (.86) |
| L | 180 | 1.03 (.88) · 1.09 (.85) · 1.10 (.83) | 1.05 (.89) · 1.24 (.85) · 1.41 (.88) |
| L | 360 | 1.01 (.90) · 1.04 (.85) · 1.08 (.86) | 1.02 (.88) · 1.16 (.84) · 1.38 (.89) |
| D (differences) | 120 | 1.09 (.89) · 1.20 (.68) · 1.49 (.55) | 1.03 (.87) · 1.10 (.83) · 1.22 (.83) |
| D | 180 | 1.06 (.85) · 1.12 (.71) · 1.30 (.56) | 1.01 (.84) · 1.07 (.85) · 1.09 (.82) |
| D | 360 | 1.03 (.89) · 1.11 (.71) · 1.22 (.70) | 0.99 (.87) · 1.01 (.92) · 1.07 (.90) |

Findings:
- **GARCH vs constant variance makes no difference to multi-step forecasts of y.** For the correctly
  specified form, the GARCH and constant-variance rows agree to within 0.02–0.03 in CRPS, RMSE and coverage at
  every horizon and sample size. The only systematic edge is 2–4% in CRPS through h ≈ 12 when volatility
  persistence is 0.98. The 40% one-step QLIKE gain does not carry over because the location error dominates
  the score of y and E[σ²] reverts to its mean within a few months.
- **Levels vs differences is what matters.** Differencing a cointegrated system costs 35–75% in CRPS at three
  years (its intervals stay honest, it just forecasts worse, because it throws away the level relation).
  Fitting levels to a non-cointegrated I(1) system costs 20–50% in CRPS and, worse, its 90% intervals cover
  only 55–70% from one year out at every sample size: φ̂ ends near 1, the model still mean-reverts towards β'X
  and treats u as stationary, so it is overconfident.
- Even the right form pays an estimation cost at 10 years: 12–20% in CRPS at 1–3 years and coverage ≈ 0.8
  instead of 0.9 (φ̂ is biased down, so the long-run variance is understated). At 30 years, 4–8%.

Guidance:
1. The level-vs-difference decision is whether y − β'X is stationary (cointegration). Fit the levels model
   first: an AR root near 1 (`res.near_unit_root`, warning in `summary()`) means the level relation is
   spurious — difference.
2. Confirm out of sample with `backtest()` on both forms at the horizons you care about, comparing CRPS and
   coverage. AIC/BIC cannot compare d = 0 with d = 1.
3. GARCH is a short-horizon volatility tool (next 1–6 months, VaR-type questions) and worth having when
   volatility is highly persistent. For a 1–3-year macro-conditioned path it changes nothing; fix
   α = β = 0 unless `arch_lm_test` rejects and T ≥ 180.

## Package: `../sarimax_garch/` (statsmodels-SARIMAX-style front end)

`from sarimax_garch import SARIMAXGARCH`. `garch=None` delegates to statsmodels SARIMAX and returns its results
object; `garch=(1, 1)` or `dist="t"` uses the joint estimator (`_core.py`, a verbatim copy of
`fixed/joint_armax_garch.py` kept in sync by a test) behind the SARIMAX attribute names: `params`/`bse`/
`pvalues` with statsmodels naming (`intercept`, exog names, `ar.L1`, `ma.L1`, `omega`, `alpha.L1`,
`beta.L1`, `nu`), `summary()`, `conf_int()`, `cov_params()`, `fix_params()`, `resid`/`fittedvalues` on the
original index, `get_forecast()` with `predicted_mean`/`se_mean`/`conf_int()`/`summary_frame()` and a
proper date index, `test_serial_correlation()`, `test_arch()`, `plot_diagnostics()`, `bootstrap()`,
`backtest()`. GARCH or seasonal: pick one — `seasonal_order` is accepted only with `garch=None`. 12 pytest
tests, including parity with statsmodels when GARCH is off and agreement with SARIMAX when GARCH is fixed off.

## Evidence (v2.2; v2.1 gives the same numbers except where noted)

| Check | Result |
|---|---|
| Built-in self-test (7 sections, incl. Python `arch` cross-check) | ALL PASS |
| Vectorised likelihood vs a naive loop written from the equations, 60 random configs | max rel. diff 2e-15 |
| FCP (1996) published benchmark, DEM/GBP (data from fGarch) | coef 1e-7, loglik exact; SE rel. err 3e-5 (v2.1: 1.7e-3) |
| **Filter 1:1 vs `ugarchfilter`** at identical params, 6 models, t ≥ 300 | residuals, σ_t, per-obs loglik equal to ≤ 9e-14 |
| **Forecast 1:1 vs `ugarchforecast`**, h = 1..10 | mean and σ_{T+h} equal to ≤ 5e-15 |
| Free fits vs `ugarchfit`, 6 comparable models | estimates within 0.01–0.14 SE; Hessian SEs within ~1% |
| Per-observation scores vs rugarch `fit@fit$scores` | identical (rugarch stores −loglik, so the sign is flipped) |
| Monte Carlo, 400 × T=2000 ARMA(1,1)-GARCH-t | unbiased (≤0.27 SD); 95% CI coverage 92–97% |

**Why the log-likelihoods differ by 1–3 units from rugarch.** The two packages handle the first
observations differently. rugarch keeps the first max(p,q,1) observations in the likelihood with
h = mean(res²); this script conditions on the first p observations. The effect dies out geometrically:
σ differs by 0.2 at t=0, 1e-5 at t=50 and 1e-12 at t=150.

**Robust SEs differ 5–10% from rugarch's.** The script's are right. Inside R, the textbook sandwich
H⁻¹S'SH⁻¹ built from rugarch's *own* Hessian and *own* scores reproduces the script's values
(`rugarch_robust_se_check.R`). rugarch's reported `robust.matcoef` does not match that formula.

## Model-definition differences (not bugs; they are now documented in the v2.2 docstring)

- rugarch `mu` is the unconditional mean: `const = mu * (1 - sum(phi))`.
- With exogenous X and p > 0, rugarch (and R `arima(xreg=)`) fit **regression with ARMA errors**;
  this script fits **ARMAX**. They are different models: γ estimates differed by 16 SE in case E. The
  rugarch model equals this ARMAX with `[x_t, x_{t-1}]` and `γ_lag = −φγ`, verified to 2e-14
  (`xreg_reparam_check.py`). With p = 0 the two coincide (case D matches).
- rugarch caps α+β at 0.999; this script caps it at 0.9999.

## Bugs found in v2.1 (fixed in v2.2)

1. **Wrong standard errors for small-scale data.** Finite-difference steps were `1e-4·max(1,|θ|)`, i.e. absolute
   for ω. At std ≈ 0.2 (inside the script's own "OK" range) the ω SE came out as **0** and the β SE was 15% off; at
   std ≈ 0.02 (daily returns in decimals) `fit_joint` crashed with `TypeError` in `_num_scores`.
   Fix: data-scaled typical sizes (`_fd_typ`), a purely relative ω step, a penalty-free Hessian and one-sided scores
   when a step is invalid. SEs are now identical at scales 1, 0.1, 0.05 and 0.01.
2. **No warning when α+β is pinned at the 0.9999 cap**, and the SEs there were penalty artifacts (SE(α) = SE(β) to
   6 digits). Fix: `at_persistence_cap` property plus a `summary()` warning.
3. **The optimiser stalled against the cap**: DEM/GBP ARMA(1,1)-t stopped 0.24 loglik short (multi-start spread
   1.4 units) while still reporting `converged=True`. Fix: an SLSQP stage with α+β ≤ cap as a linear constraint,
   kept only if it is strictly better. The fit now reaches −986.7904, the constrained max.

## v2.3 — edge cases and rugarch/statsmodels parity

| Edge case (v2.2 behaviour) | v2.3 |
|---|---|
| NaN/inf in y or X → `LinAlgError: SVD did not converge` plus LAPACK noise | `ValueError` naming the count and the first index |
| Constant series → "fit" with loglik +162,367 | `ValueError` |
| X column constant (e.g. ones, or a time trend after d=1) → NaN SEs, no warning | `ValueError` (duplicates const) |
| Collinear X columns → effect split arbitrarily, finite-looking SEs | `ValueError` (rank-deficient) |
| `start` of the wrong length → cryptic scipy error | `ValueError` listing the parameter names |
| Fewer than 100 observations → silent | warning (rugarch refuses < 100) |
| `nu` at its 200 bound, AR root near 1, MA root near 1, AR/MA roots cancelling → silent | `summary()` warnings and properties (`nu_at_bound`, `near_unit_root`, `ma_near_unit_root`, `arma_cancellation`) |

No false positives on the 6 well-specified verification cases. DEM/GBP ARMA(1,1) is correctly flagged as
cancelling (its φ, θ SEs are 0.25–0.3).

New options:
- `xreg="regression"`: regression with ARMA-GARCH errors, the convention of rugarch
  `external.regressors`, statsmodels `ARIMA(exog=)` and R `arima(xreg=)`. Against rugarch: filter
  1:1 to 2e-14, forecasts to 5e-15, free-fit estimates within 0.2 SE (γ within 0.03 SE). Against
  statsmodels (GARCH off): φ, θ, γ within 0.0004 at d=0, where the SE is 0.013. With d=1 and X in
  levels, the drift and 5-step level forecasts agree. ARMAX on the same data gives γ₁ = 0.63 vs 1.01.
  It is statsmodels SARIMAX(exog=) exactly: at identical parameters, SARIMAX's one-step errors equal
  the script's residuals to 1.4e-15 from t ≥ 50 (`sarimax_check.py`). SARIMAX `trend='c'` is the
  script's `const` (an intercept inside the ARMA); ARIMA `trend='c'` is `uncond_mean`.
- `fixed={name: value}`: rugarch `fixed.pars` / statsmodels `fix_params`. `fixed={"const": 0}` is
  rugarch `include.mean=FALSE` (1:1: filter 6e-15, estimates within 0.04 SE). Fixing every parameter
  is rugarch `ugarchfilter`, i.e. evaluating fitted parameters on new data. AIC/BIC count only the
  estimated parameters.
- `uncond_mean` property (= rugarch `mu`), `ar_roots` and `ma_roots`.

## Not implemented (feature gaps, not bugs)

- **vs rugarch:**
  - variance models: GJR/eGARCH/apARCH/fiGARCH/csGARCH/iGARCH, and GARCH(p,q) orders other than (1,1);
  - distributions: skew-t, GED, skew-GED, NIG, JSU, GHYP;
  - model extensions: ARCH-in-mean, regressors in the variance equation, ARFIMA, variance targeting;
  - workflow tools: `ugarchroll`/`ugarchboot`;
  - diagnostic tests: weighted Ljung-Box, ARCH-LM, Nyblom, sign bias, Pearson GOF.
- **vs statsmodels ARIMA:**
  - seasonal (P,D,Q,s) terms and lag subsets (`order=([1,4],0,0)`);
  - missing values and the exact likelihood (Kalman filter; the script uses the conditional likelihood);
  - enforced stationarity/invertibility (here they are reported, not enforced);
  - in-sample `get_prediction`/dynamic prediction and `plot_diagnostics`;
  - pandas date index, HQIC/AICc.
  - A deterministic time trend (statsmodels `trend='ct'` at d=0) can be passed as an X column with
    `xreg="regression"`.

The `RuntimeWarning: divide by zero encountered in matmul` lines are a numpy 2.0.2 + Apple Accelerate quirk:
a plain `A @ b` on clean data raises them too, and the results are exact. They are harmless.

## Re-run

```bash
brew install r nlopt pkgconf && Rscript install_r_packages.R   # one-time (already done on this Mac)
./run_all.sh        # verifies ../fixed/joint_armax_garch.py  -> cmp_fixed/
./run_all.sh ..     # verifies the original v2.1            -> cmp_arma-garch/
```
Set `PY=/path/to/python` to use a Python that has `arch` (optional, for self-test 2).
