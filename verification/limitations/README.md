# Methodology limitations of `sarimax_garch` 0.2.0 / engine v3.2 — 2026-11-14

What the package delegates to statsmodels, how it forecasts under GARCH-t, what asymmetric residuals do to
the mean, and where the method stops. Every number below was measured by a script in this folder and is
printed by `report.py` from `out/*.json`.

```bash
cd verification/limitations
python3 e1_backends.py && python3 e2_bands.py && python3 e3_skew.py && python3 e4_leverage.py \
  && python3 e5_converged.py && python3 e6_sarimax_vs_garch.py && python3 e8_holdout_logscore.py \
  && python3 e9_dm_null.py && python3 e10_sample_size.py && python3 e11_valid_inference.py \
  && python3 e11_size_check.py && python3 e12_power_source.py && python3 e13_identification.py abcd 400 12 300 \
  && python3 e13_identification.py ef 300 12 && python3 report.py   # about 90 minutes on 12 cores
python3 e7_arch_mamean.py      # needs a Python with arch >= 9.0 (not a package dependency)
```
The results of E1–E5 are identical to the last digit (all 2,267 numbers in `out/`) on Python 3.9.6 and 3.11.15,
both with numpy 2.0.2, scipy 1.13.1, pandas 2.3.3 and statsmodels 0.14.6.

**Follow-ups added the same day:**
- E6 (SARIMAX vs SARIMAX-GARCH, both ways round) and E7 (MA-GARCH against arch 9.0).
- `compare()` in package 0.3.1, which runs the decision workflow.
- The user guide [`sarimax_garch/GUIDE.md`](../../sarimax_garch/GUIDE.md), with limitation codes L1–L17 and fallbacks.
- The literature review in [`reports/`](reports/), with its notes in `research_notes/`.
`common.py` holds the Hansen (1994) skew-t (ppf, cdf, density; checked: mass 1, mean 0, variance 1), a
simulator that takes any standardised shocks and an optional GJR leverage term, and two prototypes the
package does not have: a skew-t likelihood (`fit_skewt`) and a GJR variance (`fit_gjr`), both on the
engine's own mean filter `_core._filter`.

## Answers in brief

**1. Does it use statsmodels for GARCH?** It cannot. statsmodels has no GARCH model (0.14.6 has neither a
GARCH class nor the old sandbox one). Which backend does the work depends on the arguments:
- `garch=None, dist="normal"`: the package is statsmodels SARIMAX. You get its results object and its
  exact Kalman likelihood.
- `garch=(1, 1)` or `dist="t"`: the package runs its own numpy/scipy engine (`_core.py`), a conditional
  MLE with numerical derivatives. statsmodels supplies only the `Summary` table, `acorr_ljungbox`,
  `het_arch`, `jarque_bera` and `plot_acf`.

The engine's results object copies statsmodels' attribute names, but it is not a statsmodels results
class. It has no `t_test`, `wald_test`, `append`/`apply`, dynamic `get_prediction` or `save`/`load`.

Switching `garch` on therefore changes three things at once: the estimator (exact → conditional
likelihood), the scored sample (T → T − d − p), and the variance model. On the same model, the engine's
log-likelihood is about 1.5 units above statsmodels' (E1). So an AIC comparison made by switching `garch`
on picks GARCH for about half of all series that have no ARCH at all.

**2. GARCH-t forecasts.**
- **Point forecast** (`predicted_mean`): the conditional mean, computed with future shocks set to zero.
  GARCH and ν do not enter it, because there is no GARCH-in-mean term. They move it only through the
  estimated mean parameters. Under symmetric shocks it is also the median at every horizon.
- **Forecast-error SD** (`se_mean`): exact. It is the sum of squared ψ-weights times the GARCH variance
  path. ν drops out because the standardised t has unit variance.
- **Default band** (`conf_int()`, `summary_frame()`): mean ± z·se, a normal shape. Under t(5) it covers
  0.84 at nominal 80% and 0.978 at nominal 99%. The 99% band is breached 2.2% of the time instead of 1%,
  about twice as often in each tail. At 95% it is right by coincidence.
- **Simulated band** (`conf_int(method="simulated")`): the quantiles of 10,000 simulated paths with t
  shocks. It is at nominal when T = 1000.
- **Both bands are plug-in.** They ignore estimation error. At T = 250 with a persistent mean (φ = 0.9),
  the simulated 90% band covers 0.86 at h = 20 (E2).

**3. Asymmetric residuals and the mean.** Two different asymmetries, both measured:
- **Skewed shocks (E3).**
  - `dist="normal"` (Gaussian QMLE) keeps every mean parameter consistent, and its robust CIs cover
    0.92–0.96.
  - `dist="t"` does not. A symmetric-t likelihood puts the centre of a skewed distribution between its
    mean and its mode, so the **intercept is biased toward the mode**, and with it the long-run mean
    μ = c/(1 − φ) and every forecast level. On a true c = 0.1, the bias is +0.031, +0.076 and +0.151 at
    shock skewness −0.35, −0.84 and −1.47 (T = 2000). The 95% CI coverage of c falls to 0.79, 0.26 and
    0.00.
  - The measured shift matches the theory: a location-scale t fitted to these shocks centres them at
    +0.024, +0.062 and +0.122 SD.
  - The AR coefficient, the regression slope and the GARCH parameters are essentially unaffected.
  - Every symmetric band misplaces the tails. At skew −0.84 the lower 1% quantile is breached 1.8–2.3% of
    the time, the upper one 0.4–1.0%.
- **Asymmetric volatility, i.e. leverage (E4).** The shocks are symmetric but the variance is GJR. The
  package's symmetric GARCH fit biases the intercept upward by +0.6 SD (leverage 0.06) to +1.4 SD
  (leverage 0.14 with α = 0, typical of equity indices), and its CI coverage falls to 0.74. The cause is
  the joint likelihood: the variance equation depends on ε_{t−1} = f(c, φ, g), so the mean absorbs part of
  the asymmetry. statsmodels SARIMAX without GARCH stays unbiased.
- **What fixes them.**
  - *Skew-t:* a skew-t likelihood removes the E3 bias. It is also the most efficient estimator: at skew
    −1.47 the SD of the slope is 0.027, against 0.041 for the Gaussian QMLE. It calibrates the tails, and
    its LR test detects skewness of −0.35 in 88% of samples at T = 2000.
  - *GJR:* a GJR variance removes the E4 bias.
  - *Empirical shocks:* resampling the standardised residuals (filtered historical simulation, "FHS")
    calibrates the bands without any new model.

## Evidence

### E1 — two backends, one model (homoskedastic data, 200 series per design)

`sm` = `garch=None` (statsmodels), `e0` = the engine with α = β = 0 fixed (the right null), `e1` = the engine
with GARCH. "Within" compares e1 with e0, "across" compares e1 with sm (what you get by toggling `garch`).
The truth has no ARCH, so every "GARCH" verdict is a false positive.

| design | nobs statsmodels / engine | llf gap engine − statsmodels | LR > 5.99: within / across | AIC picks GARCH: within / across | median \|Δ\| intercept · x1 · ar.L1 (SE) |
|---|---|---|---|---|---|
| T=120,d=0 | 120 / 119 | +1.51 (sd 0.61) | 1.5% / **16.5%** | 6.0% / **46.0%** | 0.09 · 0.04 · 0.04 |
| T=500,d=0 | 500 / 499 | +1.57 (sd 0.62) | 5.0% / **26.5%** | 10.5% / **51.5%** | 0.05 · 0.02 · 0.02 |
| T=120,d=1 | 120 / 118 | +1.66 (sd 0.98) | 1.5% / **18.5%** | 6.5% / **45.0%** | 0.10 · 0.04 · 0.04 |
| T=500,d=1 | 500 / 498 | +1.52 (sd 0.55) | 4.0% / **21.5%** | 11.5% / **51.5%** | 0.05 · 0.02 · 0.02 |

The gap is the log-density of the p unscored observations (≈ −1.5 for one N(0, σ²/(1−φ²)) value). The
parameter estimates of the two backends agree closely; only the likelihood level differs.

**The sign of the gap depends on the units of y** (`python3 e1_backends.py scale`, AR(1), 100 series per scale):

| sd(y) | 11.5 | 1.15 | 0.28 | 0.115 | 0.012 |
|---|---|---|---|---|---|
| gap engine − statsmodels | +3.77 | +1.47 | +0.04 | −0.83 | −3.07 |

At unit scale the cross-backend comparison favours GARCH. On small-scale data, such as returns in decimals, it
is biased against GARCH. It is wrong at every scale; only the direction changes.

The 5.99 cut-off is the χ²(2) convention. Under the null the LR statistic is not χ²(2): α = 0 is on the
boundary and β is not identified there, which is why the within-engine rate stays at or below 5%. The point
is the gap between the two columns.

### E2 — bands under GARCH-t (400 origins per design, each scored against 20,000 true future paths)

DGP AR(1)-GARCH(1,1) (c 0.05, φ 0.5, ω 0.05, α 0.08, β 0.90). "default" = `conf_int()`, "simulated" =
`conf_int(method="simulated")`, "oracle" = the same bands at the true parameters. Cells: coverage at h = 1 · h = 20.

**D1 t5 T=1000** (median ν̂ 5.1; point-forecast RMSE vs the true conditional mean 0.04 predictive SD)

| band | 80% | 90% | 95% | 99% | 99% band: below / above at h1 (nominal 0.5% each) |
|---|---|---|---|---|---|
| fit-t default | 0.839 · 0.843 | 0.911 · 0.911 | 0.946 · 0.944 | 0.978 · 0.976 | 1.08% / 1.09% |
| fit-t simulated | 0.798 · 0.795 | 0.898 · 0.895 | 0.949 · 0.946 | 0.989 · 0.988 | 0.54% / 0.55% |
| fit-n default | 0.838 · 0.842 | 0.910 · 0.910 | 0.946 · 0.944 | 0.978 · 0.976 | 1.09% / 1.10% |
| fit-n simulated | 0.838 · 0.831 | 0.910 · 0.907 | 0.946 · 0.945 | 0.978 · 0.980 | 1.10% / 1.11% |
| oracle normal | 0.841 · 0.849 | 0.913 · 0.916 | 0.947 · 0.948 | 0.979 · 0.978 | 1.04% / 1.05% |
| oracle simulated | 0.800 · 0.800 | 0.900 · 0.900 | 0.950 · 0.950 | 0.990 · 0.990 | 0.51% / 0.51% |

**D2 t5 T=250** (median ν̂ 5.8; point RMSE 0.09 SD)

| band | 80% | 90% | 95% | 99% | 99% band: below / above at h1 |
|---|---|---|---|---|---|
| fit-t default | 0.832 · 0.828 | 0.905 · 0.899 | 0.941 · 0.934 | 0.976 · 0.970 | 1.21% / 1.22% |
| fit-t simulated | 0.796 · 0.788 | 0.893 · 0.885 | 0.943 · 0.936 | 0.986 · 0.981 | 0.72% / 0.72% |
| fit-n default | 0.828 · 0.823 | 0.902 · 0.895 | 0.939 · 0.932 | 0.974 · 0.969 | 1.27% / 1.28% |

**D3 normal shocks T=1000**: the simulated bands are at nominal; the default normal band drifts at long
horizons (80% → 0.810 and 99% → 0.983 at h = 20), because GARCH makes the multi-step predictive fat-tailed
(excess kurtosis 0.8 at h = 20) even with normal shocks. The effect is small.

**D4 t5 T=250 φ=0.9** (median ν̂ 5.7; point RMSE 0.08 SD at h1, 0.19 SD at h20)

| band | 80% | 90% | 95% | 99% | 99% band: below / above at h1 |
|---|---|---|---|---|---|
| fit-t default | 0.828 · 0.787 | 0.902 · 0.871 | 0.939 · 0.916 | 0.974 · 0.963 | 1.27% / 1.29% |
| fit-t simulated | 0.791 · 0.757 | 0.890 · 0.861 | 0.941 · 0.918 | 0.985 · 0.973 | 0.76% / 0.77% |
| oracle simulated | 0.800 · 0.800 | 0.900 · 0.900 | 0.950 · 0.950 | 0.990 · 0.990 | 0.50% / 0.52% |

The t(5) designs have α .08 and β .90, so κα² + 2αβ + β² = 1.012 > 1 with κ = 9. The observed series then
has no finite fourth moment, which puts these designs (and E5's) outside the Francq & Zakoian (2004) conditions
for asymptotic normality. That does not affect the band results, which are scored against simulated truth.

Reading: the shape error of the default band is the same at every T, and it disappears with
`method="simulated"`. The estimation error grows as T falls and as the horizon and the persistence grow,
and neither band accounts for it. In D4 the default band's normal shape partly offsets the estimation
error at 80–90%; that is a coincidence, not calibration.

### E3 — skewed shocks (Hansen skew-t, ν = 6; 400 series per design)

DGP: y = 0.5·x + u, u_t = 0.1 + 0.5 u_{t−1} + ε_t, GARCH (0.05, 0.08, 0.90), x exogenous AR(1). SM =
statsmodels SARIMAX (no GARCH), GN = engine GARCH normal, GT = engine GARCH-t, GS = skew-t prototype.

| design | z skew | t-MLE centre of z (theory) | estimator | c bias (bias/SD) | μ bias | φ bias/SD | g bias/SD | SD(g) | 95% CI cover c |
|---|---|---|---|---|---|---|---|---|---|
| T=2000 λ=0 | +0.00 | −0.001 SD | SM | +0.0014 (+0.0) | +0.0015 | −0.1 | −0.0 | 0.0503 | |
| | | | GN | +0.0015 (+0.1) | +0.0023 | −0.1 | −0.1 | 0.0402 | 0.94 |
| | | | GT | +0.0011 (+0.0) | +0.0016 | −0.1 | −0.0 | 0.0363 | 0.94 |
| | | | GS | +0.0010 (+0.0) | +0.0013 | −0.1 | −0.0 | 0.0365 | |
| T=2000 λ=−0.10 | −0.35 | +0.024 SD | SM | +0.0006 (+0.0) | +0.0013 | +0.0 | −0.1 | 0.0485 | |
| | | | GN | +0.0014 (+0.1) | +0.0035 | +0.1 | −0.1 | 0.0406 | 0.95 |
| | | | GT | **+0.0309 (+1.2)** | **+0.0627** | +0.1 | −0.0 | 0.0356 | **0.79** |
| | | | GS | +0.0007 (+0.0) | +0.0021 | +0.1 | −0.1 | 0.0352 | |
| T=2000 λ=−0.25 | −0.84 | +0.062 SD | SM | −0.0008 (−0.0) | −0.0024 | −0.0 | −0.1 | 0.0505 | |
| | | | GN | −0.0011 (−0.0) | −0.0019 | +0.0 | −0.0 | 0.0423 | 0.95 |
| | | | GT | **+0.0755 (+2.6)** | **+0.1511** | −0.0 | −0.0 | 0.0364 | **0.26** |
| | | | GS | −0.0020 (−0.1) | −0.0037 | +0.0 | −0.0 | 0.0353 | |
| T=2000 λ=−0.50 | −1.47 | +0.122 SD | SM | +0.0009 (+0.0) | +0.0006 | −0.1 | −0.0 | 0.0486 | |
| | | | GN | +0.0032 (+0.1) | +0.0059 | −0.1 | +0.0 | 0.0409 | 0.92 |
| | | | GT | **+0.1505 (+4.9)** | **+0.2975** | −0.2 | −0.0 | 0.0330 | **0.00** |
| | | | GS | +0.0022 (+0.1) | +0.0043 | −0.0 | −0.1 | 0.0272 | |
| T=500 λ=−0.25 | −0.84 | +0.062 SD | SM | +0.0019 (+0.0) | −0.0007 | −0.1 | −0.0 | 0.1079 | |
| | | | GN | +0.0087 (+0.1) | +0.0149 | −0.1 | +0.0 | 0.0863 | 0.94 |
| | | | GT | **+0.0815 (+1.4)** | **+0.1589** | −0.1 | −0.0 | 0.0763 | **0.69** |
| | | | GS | +0.0053 (+0.1) | +0.0077 | −0.1 | +0.0 | 0.0723 | |

The true μ is 0.2, so at skew −0.84 the GARCH-t long-run mean is 75% too high. The GARCH parameters are
nearly unbiased under all four estimators (largest: α −0.4 SD for GT at skew −1.47).

1-step forecasts: the location error against the true conditional mean (in units of the true σ_{T+1}),
and the true probability of falling outside each model's quantiles (nominal 1% / 1% / 5% / 5%):

| design | model | location error / σ | P(y < q01) | P(y > q99) | P(y < q05) | P(y > q95) |
|---|---|---|---|---|---|---|
| T=2000 λ=0 | GN normal | −0.002 | 1.51% | 1.52% | 4.6% | 4.7% |
| | GT t | −0.001 | 1.03% | 1.04% | 5.1% | 5.1% |
| T=2000 λ=−0.25 | SM normal | −0.003 | 2.24% | 0.99% | 5.1% | 3.1% |
| | GN normal | −0.003 | 2.25% | 0.66% | 5.7% | 2.9% |
| | GN + empirical z (FHS) | −0.003 | 1.06% | 1.07% | 5.0% | 5.1% |
| | GT t | **+0.055** | 1.76% | 0.36% | 6.8% | 3.0% |
| | GS skew-t | −0.004 | 1.03% | 1.05% | 5.0% | 5.1% |
| T=2000 λ=−0.50 | GN normal | +0.001 | 2.79% | 0.13% | 6.3% | 1.2% |
| | GN + empirical z (FHS) | +0.001 | 1.08% | 1.06% | 5.1% | 5.1% |
| | GT t | **+0.112** | 2.28% | 0.04% | 8.1% | 1.1% |
| | GS skew-t | +0.002 | 1.02% | 1.03% | 5.1% | 5.0% |
| T=500 λ=−0.25 | GN + empirical z (FHS) | +0.012 | 1.34% | 1.27% | 5.4% | 5.3% |
| | GS skew-t | +0.009 | 1.17% | 1.16% | 5.3% | 5.2% |

The skew-t prototype recovers λ: mean λ̂ −0.099, −0.252 and −0.500 for true λ of −0.10, −0.25 and −0.50.
Its LR test of λ = 0 rejects in 5% of the symmetric samples (the nominal size) and in 88%, 100% and 100%
of the skewed ones (97% at T = 500, λ = −0.25).

A check that works with today's code: fit both `dist="t"` and `dist="normal"`, and compare the intercepts
in units of the normal fit's robust SE.

| design | mean of (c_t − c_normal) / SE(c_normal) | its SD | share with \|·\| > 1 | sample skewness of std_resid: mean (SD) |
|---|---|---|---|---|
| T=2000 symmetric | −0.01 | 0.42 | 2.2% (false alarms) | −0.01 (0.23) |
| T=2000 skew −0.35 | +0.98 | 0.42 | 47% | −0.32 (0.26) |
| T=2000 skew −0.84 | +2.58 | 0.47 | 100% | −0.79 (0.21) |
| T=2000 skew −1.47 | +4.97 | 0.65 | 100% | −1.41 (0.33) |
| T=500 skew −0.84 | +1.17 | 0.50 | 62% | −0.74 (0.42) |

The sample skewness of the residuals is a weak signal under fat tails. Under t6 its SD is 0.23 at T = 2000,
four times the textbook √(6/T) = 0.055, because the sixth moment is infinite.

Why the t-MLE moves the intercept: its score for the location is Σ w_t ε_t/σ_t² with weights
w_t = (ν+1)/(ν−2+z_t²), which fall for large |z|. With a long left tail, the large negative shocks are
down-weighted more, so the weighted mean of ε is zero only when the fitted centre sits above the true mean.
The Gaussian QMLE weights every standardised shock equally, so the mean of ε stays at zero.

### E4 — leverage (GJR variance, normal shocks, persistence α + γ/2 + β = 0.97; 400 series per design)

| design | estimator | c bias (bias/SD) | μ bias | φ bias/SD | g bias/SD |
|---|---|---|---|---|---|
| lev 0 (α 0.07) | SM | +0.0007 (+0.03) | +0.0012 | −0.04 | −0.00 |
| | GN | +0.0006 (+0.03) | +0.0011 | −0.02 | +0.02 |
| | GJR | +0.0005 (+0.02) | +0.0010 | −0.01 | +0.02 |
| lev 0.06 (α 0.04) | SM | −0.0000 (−0.00) | −0.0009 | −0.07 | −0.02 |
| | GN | **+0.0122 (+0.59)** | +0.0231 | −0.13 | −0.04 |
| | GT | +0.0119 (+0.57) | +0.0225 | −0.13 | −0.05 |
| | GJR | −0.0001 (−0.00) | −0.0010 | −0.09 | −0.04 |
| lev 0.14 (α 0) | SM | +0.0013 (+0.05) | +0.0012 | −0.12 | −0.05 |
| | GN | **+0.0281 (+1.39)** | **+0.0534** | −0.30 | −0.02 |
| | GT | +0.0265 (+1.31) | +0.0503 | −0.28 | −0.03 |
| | GJR | +0.0008 (+0.04) | +0.0003 | −0.17 | −0.02 |

GN 95% robust-Wald coverage of c: 0.96, 0.91 and 0.745. The GJR prototype's LR test of lev = 0 rejects in 6%,
88% and 100% of samples. The mechanism is the second term of the Gaussian score for c,
½ Σ (ε_t²/σ_t² − 1) σ_t^{-2} ∂σ_t²/∂c, with ∂σ_t²/∂c = −2α Σ β^j ε_{t−1−j}. When the true variance after
a negative shock is higher than the symmetric model allows, this term has a non-zero mean, and the
intercept moves until the first term offsets it.

### E5 — the `converged` flag and local optima (AR(1)-GARCH-t5, T = 250, 400 fits, 9 starts each)

`converged` is L-BFGS-B's flag, although the answer returned may come from the Nelder-Mead or SLSQP stage that
follows. The flag is unreliable in both directions:
- 12 fits were flagged False, and 83% of them are at the best optimum found.
- 13 fits are below the best of 9 starts by more than 1e-4 log-likelihood units: 9 by more than 0.1,
  5 by more than 0.5, 3 by more than 1.0. Only 2 of the 13 were flagged.
- The worst fit marked `converged=True` is 1.87 units below the best.

### E6 — SARIMAX vs SARIMAX-GARCH, both ways round (28 designs, 300 series each)

**Design:**
- Regression y = 0.5x + u, with u AR(1) (φ .5), MA(1) (θ .5) or MA(1) with θ .9, and normal shocks.
- GARCH: none, typical monthly (α .10, β .75) or persistent (α .08, β .90).
- T = 120, 250, 500 and 1000.
- Every band is scored against the true predictive distribution.
- The full tables are printed by `report.py`; the decision workflow built on them is in `sarimax_garch/GUIDE.md`.

| finding | numbers |
|---|---|
| GARCH where there is none costs little, but the GARCH part is empty | RMSE of φ/θ/slope within +4% (typically ≤ 2%); CRPS +0.0–0.3%; α or β on a boundary in 64–87% of fits |
| GARCH's real edge: intervals that hold in both regimes | SARIMAX's 90% one-step band covers 0.91–0.96 in calm periods and 0.81–0.86 in turbulent ones (bottom/top third of the true σ²_{T+1}); GARCH covers 0.86–0.90 in both; average coverage is 0.88–0.90 for both |
| CRPS barely moves | GARCH/SARIMAX 0.996–1.000 (typical) and 0.984–0.994 (persistent) at h = 1; 0.989–1.003 at h = 12 |
| Mean estimates gain with persistent volatility | slope RMSE −3% to −17%, AR/MA coefficient −4% to −16% (persistent); slope 0 to −7% (typical) |
| SARIMAX's default SEs fail for AR/MA under GARCH | `opg` 95% CIs cover φ/θ 0.83–0.92; `cov_params_robust` 0.89–0.94; the GARCH fit's robust SEs 0.91–0.97; slope and intercept CIs fine (0.91–0.97) for all three |
| Detection, false positives (no GARCH; φ/θ = .5) | LR within the engine 3–4%, ARCH-LM(4) 3–7%, AIC within 7–11%, BIC 0–1%; AIC across backends 48–55% for AR(1) but only 8–10% for MA(1): the conditional likelihood drops one observation per AR lag, so E1's gap scales with p |
| Detection, power | LR within: 0.22–0.32 (T = 120), 0.46–0.78 (250), 0.79–0.97 (500), 0.99–1.00 (1000); ARCH-LM(4) somewhat lower |
| MA near the unit circle | at θ = .9 with no GARCH the engine's LR picks GARCH 11% (T = 1000) and AIC 21%, while ARCH-LM on SARIMAX's residuals stays at 4–6%: the conditional likelihood's start-up transient looks like a variance burst |

### E7 — MA-GARCH against Python arch 9.0 (`MAMean`, released 2026-11-11)

**Design:** MA(1), MA(2), MA(1) with θ = .9, and MA(1) with t shocks; T = 3000.

**Result:** every parameter matches `MAMean` within **0.008 SE** (largest absolute difference 3.7e-4). The
log-likelihoods differ by 0.003–0.012 because the two packages start the variance recursion differently.
Under the engine's likelihood, arch's estimate is never better than the engine's.

arch ≥ 9.0 has pure MA(q) means, but still no joint ARMA(p,q) and no regression-with-ARMA-errors form.

### E8 — holdout log-likelihood against CRPS (12 designs, 300 series each)

The prequential one-step holdout log score is the metric microprediction's
[`skaters`](https://github.com/microprediction/skaters) (Peter Cotton) ranks forecasters by.

**Design:**
- y = 0.5x + u, with u AR(1) (φ .5).
- GARCH: none, typical (α .10, β .75) or persistent (α .08, β .90), with normal shocks; plus persistent GARCH with t(5) shocks.
- Train / holdout sizes: 250/60, 500/120 and 1000/250.
- Each model is fitted on the training part and its parameters frozen: SM = SARIMAX, GN = GARCH normal, GT = GARCH t.
- Each held-out observation is scored one step ahead, by the log score (`holdout_logscore`) and by the exact CRPS of the same predictive distribution (normal or t closed form).
- Per series, a Diebold–Mariano test (Newey–West) decides "significantly better" at 5%.
- The package's `holdout_logscore` reproduces E8's own computation exactly.

| design | in-sample LR picks GARCH | DM picks GN over SM: log score / CRPS | Δ log score GN − SM (nats/obs) · CRPS ratio | DM picks GT over GN: log score / CRPS | Δ log score GT − GN · CRPS ratio |
|---|---|---|---|---|---|
| no GARCH | 0.02–0.04 | 0.02–0.04 / 0.02–0.04 | −0.0028 to −0.0005 · 1.000–1.001 | 0.01–0.02 / 0.02–0.05 | ≈ 0 · 1.000 |
| typical GARCH | 0.52–0.98 | 0.10–0.21 / 0.07–0.18 | +0.012 to +0.017 · 0.996 | 0.01–0.02 / 0.05–0.06 | ≈ 0 · 1.000 |
| persistent GARCH | 0.76–1.00 | 0.26–0.58 / 0.24–0.56 | +0.049 to +0.092 · 0.981–0.987 | 0.00–0.04 / 0.04–0.08 | ≈ 0 · 1.000 |
| persistent GARCH, t(5) shocks | 0.77–1.00 | 0.25–0.52 / 0.30–0.62 | +0.111 to +0.150 · 0.965–0.977 | 0.14–0.31 / 0.21–0.32 | +0.048 to +0.082 · 0.991–0.994 |

(The Diebold–Mariano decisions use the shipped `dm_test`: Newey–West variance, the Harvey–Leybourne–Newbold
correction, and t(n − 1) critical values; see E9.)

Findings:
- **Valid across backends.** On homoskedastic data the SARIMAX and GARCH holdout log scores agree within 0.003
  nats per observation. In-sample log-likelihoods of the same models differ by about 1.5 (E1).
- **The log score measures density quality that CRPS hardly registers.** Getting the tails right (t vs normal
  under t(5) shocks) is worth +0.05 to +0.08 nats per observation, while CRPS moves only 0.6–0.9%.
- **The two scores have similar power per series.** The log score is 2–4 points ahead with normal shocks.
  CRPS is 5–10 points ahead with t(5) shocks: log-score differences are heavy-tailed there, which weakens the
  Diebold–Mariano test.
- **Neither replaces the in-sample LR test at these holdout sizes.** The holdout is the out-of-sample check
  across model classes (backends, d, dist), not the GARCH detector. Report both scores, as `skaters` does.
- `backtest()` now reports `logscore` by horizon on both backends. For GARCH beyond one step the density is an
  exact mixture over the simulated paths: it integrates to 1, matches the closed form at h = 1, and needs no
  kernel bandwidth.

### E9 — is the Diebold–Mariano statistic N(0, 1) here? (`e9_dm_null.py`)

The test (`dm_test`):
- **Difference series:** d_t = S(F_A,t, y_t) − S(F_B,t, y_t) over the n held-out observations; for the log
  score, d_t = log f_A,t(y_t) − log f_B,t(y_t).
- **Hypotheses:** H0: E[d_t] = 0 against H1: E[d_t] ≠ 0. For the log score, E[d_t] = KL(g‖f_B) − KL(g‖f_A),
  where g is the true predictive density.
- **Statistic:** DM = d̄ / √(σ̂²/n). σ̂² is the Newey–West (Bartlett) long-run variance of d_t, with lag
  L = max(h − 1, ⌊4(n/100)^{2/9}⌋). The statistic is multiplied by the Harvey–Leybourne–Newbold factor
  √((n + 1 − 2h + h(h−1)/n)/n).
- **Distribution:** N(0, 1) asymptotically; t(n − 1) in finite samples with the correction.
- **Rejection rule:** reject H0 at 5% when |DM| > t₀.₉₇₅(n − 1), which is 2.00 at n = 60 and 1.97 at 250.
  DM > 0 means A is better.

**Part A: the null is exactly true, by construction.** Two forecasters are shifted ±0.25 SD from the true
conditional mean, on persistent-GARCH data with normal or t(5) shocks; 4,000 series per cell. Rejection rate at
nominal 5%:

| d_t | n | naive paired t (γ₀ only) | Newey–West, N(0, 1) | + lag ≥ h − 1 + HLN, t(n − 1) (shipped) | shipped statistic: SD, 2.5/97.5% |
|---|---|---|---|---|---|
| one-step, constant-variance forecasts (heteroskedastic martingale difference) | 60 | 0.055–0.058 | 0.067–0.073 | 0.061–0.065 | 1.07, ±2.1–2.2 |
| | 120 | 0.048 | 0.058–0.059 | 0.054–0.056 | 1.04 |
| | 250–1000 | 0.048–0.052 | 0.048–0.055 | 0.048–0.053 | 0.99–1.01 |
| 6-step, overlapping (MA(5)) | 60–250 | **0.25–0.27** | 0.09–0.13 | 0.08–0.09 | 1.12–1.16 |
| | 1000 | 0.25 | 0.07–0.08 | 0.07 | 1.09–1.10 |

Heavy tails (t(5) shocks, so ε has no fourth moment) do not change the one-step result. Overlapping multi-step
differences need the HAC variance: without it, 25% false rejections. Even with it, size is 7–9%, so treat
multi-step p-values as approximate.

**Part B: the nested population null.** This is E8 without GARCH, where SARIMAX is the true model nested in
GARCH (α = β = 0). The shipped statistic for GARCH − SARIMAX is *not* N(0, 1) here:
- **Log score:** mean −0.19 to −0.33 and SD 1.15–1.26. It backs GARCH in 1.7–4.3% of series and SARIMAX in
  6.7–8.0%.
- **CRPS:** mean −0.11 to −0.25 and SD 1.10–1.16.

The estimated larger model really is slightly worse out of sample, because it carries estimation noise. That is
Clark & McCracken's (2001) nested-model result. As a test of "the estimated models score equally"
(Giacomini & White 2006), the SARIMAX-better rejections are correct. As a test of "is there GARCH?", the test is
conservative and the wrong tool: use the in-sample LR or LM test (Diebold 2015: DM tests compare forecasts, not
models).

### E10 — sample size by model type, one stage vs two, p-value sizes (45 cells, 400 series each)

**Design:**
- **Model types:** ARIMA(1,1,1) with drift; regression with ARIMA(1,1,1) errors and a random-walk regressor;
  the airline model SARIMA(0,1,1)(0,1,1)₁₂ (θ −0.4, Θ −0.6).
- **GARCH:** none, typical monthly (α .10, β .75) or persistent (α .08, β .90).
- **Sample sizes:** T = 60, 120, 240, 360 and 600.
- **Fits:** SARIMAX; one-stage ARIMA-GARCH (not available for the airline model); and two-stage, a zero-mean
  GARCH(1,1) on SARIMAX's Kalman-standardized residuals after `loglikelihood_burn`.
- `report.py` prints the full tables; the guide's section 4 has the summary and the decision rules.

Findings:
- **The model type barely matters; T and the ARCH strength decide.**
  - The LR test finds GARCH in 4–8% of samples at T = 60.
  - It finds it in 19–34% at 120, 43–74% at 240, 65–90% at 360 and 86–99% at 600.
  - The GARCH fit sits on a boundary in 74–86% of fits at 60, 47–59% at 120, 16–21% at 240 and at most 11%
    from 360.
- **The one-step variance-forecast loss, GARCH / constant:**
  - It falls below 1 from T ≈ 120 for typical ARCH (0.8–1.0) and from about 60–120 for persistent ARCH.
  - At 240 it is 0.40–0.56 (typical) and 0.17–0.23 (persistent).
  - Where there is no GARCH it is 1.8–2.8, i.e. worse.
- **Two stages match one stage.** Detection is 0–3 points lower, with the same boundary rate and persistence
  RMSE. The variance forecasts are 6–26% better at T ≤ 120 and equal from 240. So the airline model plus GARCH
  works in two stages, which the guide's L1 fallback now gives as a recipe.
- **p-value sizes at nominal 5%:**
  - The GARCH tests are never oversized: ARCH-LM(12) 0–5%, LR one-stage 1–5%, LR two-stage 0–4%.
  - Under GARCH, the AR/MA Wald tests with SARIMAX's default `opg` SEs are oversized at **6–17%**, and the
    problem does not shrink with T. They are 3–11% with `cov_type="robust"` and 4–10% with the one-stage
    GARCH fit, from T = 120.
  - The slope and intercept tests are 1–8% with any method.
  - At T = 60, the AR/MA tests are 6–15% for every method even without GARCH: small-sample ARMA.

### E11 — valid inference when the textbook p-values fail (`e11_valid_inference.py`, `e11_size_check.py`)

**(a) Testing "no GARCH".** Under the null, α = 0 is on the boundary and β is not identified, so the LR test has
no χ² law. `garch_test` uses the restricted bootstrap (Cavaliere, Nielsen & Rahbek 2017):
- fit the model under H0;
- resample its standardized residuals;
- refit both models to each bootstrap series;
- reject when p* ≤ α, with α(B+1) a whole number (here B = 99).

Share rejecting at 5%:

| truth | T | χ²(2) LR | restricted bootstrap LR | ARCH-LM(4) |
|---|---|---|---|---|
| no GARCH (size) | 120 | 1.4% | **4.8%** | — |
| | 250 | 2.3% | 3.0% | 3.7% |
| typical GARCH | 120 / 250 | 25% / 52% | 36% / 57% | 20% / 41% |
| persistent GARCH | 120 / 250 | 34% / 75% | 46% / 79% | 21% / 50% |
| MA θ .9, no GARCH (size) | 250 / 1000 | 7.7% / 4.7% | 6.0% / 3.3% | 4.3% / 7.3% |

The T = 120 size row comes from 1,000 fresh series.

- **The restricted bootstrap is correctly sized.** Its null LR distribution matches the true one: 95% points 4.50
  and 4.49, KS p = 0.96.
- **It gains power from the right critical value,** not from over-rejecting: 10–11 points at T = 120 and 4–5 at
  T = 250 over the conservative χ²(2) cut-off.
- **It also mostly corrects the near-unit-root MA over-rejection,** from 7.7% to 6.0% at T = 250.
- **Two lessons from building it:**
  - A bootstrap must repeat the original procedure exactly, including optimizer starts. Refits that started
    from the original estimates were too small.
  - A size check that reuses the same 300 series for every variant can mislead. Those series showed 8.7% for
    the bootstrap and 2.7% for χ²(2), and were simply unrepresentative.

**(b) Post-selection inference for the mean.** The workflow selects GARCH when the LR test rejects, then
reports that model's 95% intervals for the slope and AR coefficient (1,000 series per cell; tables from
`report.py`):
- **Unconditional coverage, reporting robust SEs in both branches: 0.91–0.96.** This equals never selecting:
  always SARIMAX with robust SEs covers 0.92–0.96, and always GARCH 0.92–0.96.
- **Conditional on the selection outcome**, which is what selective inference asks for: 0.91–0.95 given that
  GARCH was selected, and 0.91–0.96 given that SARIMAX was selected. The exceptions are the no-GARCH cells,
  where only 20–30 series select GARCH.
- **The one shortfall is not caused by selection.** With SARIMAX's default `opg` SEs in the SARIMAX branch, the
  AR coefficient covers 0.91–0.93 when GARCH is present but missed. That is the `opg` problem, and robust SEs
  fix it.
- **Why selection does not distort the mean:**
  - The selection statistic, the GARCH LR, is built from squared residuals.
  - It is asymptotically independent of the mean estimators: the information matrix is block-diagonal
    (Engle 1982).
  - Conditioning on an event independent of an estimator leaves the estimator's distribution unchanged.
  - Selection does matter for quantities that exist in only one model (α, β, the persistence), and wherever
    that independence fails: skewed shocks (E3) and leverage (E4).

### E12 — is the low power due to GARCH or to the SARIMAX in front of it? (9 cells, 500 series each)

**Design:**
- **One set of shocks for every model:** each GARCH(1,1) shock series drives eight mean models, so the
  comparison is paired.
- **The eight models:**
  - an oracle that sees the true shocks;
  - a constant mean;
  - OLS on an exogenous regressor;
  - AR(1);
  - ARMA(1,1);
  - MA(1) with θ .9;
  - ARIMA(1,1,1);
  - the airline model, fitted in two stages only.
- **GARCH:** none (to measure size), typical or persistent; T = 120, 240 and 480.
- `report.py` prints every test.

LR test (one-stage; two-stage for the airline model), share choosing GARCH at 5%:

| truth | T | oracle | mean / OLS / AR(1) / ARMA(1,1) / ARIMA(1,1,1) | airline (two-stage) | MA(1) θ .9 (one-stage) |
|---|---|---|---|---|---|
| typical GARCH | 120 / 240 / 480 | 0.24 / 0.48 / 0.81 | 0.23–0.25 / 0.44–0.48 / 0.79–0.80 | 0.17 / 0.41 / 0.75 | 0.27 / 0.53 / 0.81 |
| persistent GARCH | 120 / 240 / 480 | 0.39 / 0.72 / 0.96 | 0.37–0.39 / 0.72–0.73 / 0.96 | 0.28 / 0.67 / 0.94 | 0.40 / 0.75 / 0.96 |
| no GARCH (size) | 120 / 240 / 480 | 0.01 / 0.03 / 0.05 | 0.01–0.02 / 0.02–0.03 / 0.04–0.05 | 0.01 / 0.02 / 0.04 | **0.05 / 0.06 / 0.10** |

- **The low power is GARCH's own.** With the shocks known exactly, typical monthly GARCH is found 24% of the
  time at T = 120 and 48% at 240.
  - The whole signal is a lag-1 autocorrelation in the squared shocks: 0.13 for typical GARCH and 0.21 for
    persistent GARCH, decaying at the rate α + β.
  - Squared shocks are heavy-tailed, which makes that autocorrelation noisy to estimate.
- **The mean model costs little.** OLS, AR, ARMA and ARIMA means cost 0–4 points of power.
- **Seasonal MA estimation costs the most in short samples.** The airline model, fitted in two stages, loses
  7–11 points at T = 120 and 1–7 from 240.
- **A near-unit-root MA costs no power but breaks the one-stage test's size,** which rises to 10% at T = 480.
  The two-stage LR on the Kalman-standardized residuals (1–5%) and ARCH-LM (2–4%) are fine on the same series.
- **The test matters more than the mean model.** ARCH-LM(4) has 3–25 points less power than the LR test, and
  ARCH-LM(12) less again at T = 120 (0.13 against 0.24 for typical GARCH).

### E13 — can the mean model, or fat tails, manufacture GARCH? (`e13_identification.py`)

E12 showed that the low power is GARCH's own. E13 asks two things:
- whether identification problems of the **mean** model reach the residuals that GARCH is tested on;
- whether fat tails alone fake GARCH.

Cells have 400 series each (300 for the bootstrap and for (e) and (f)); `report.py` prints every table.

**(a) Unidentified AR/MA coefficients do not mean unidentified residuals.** One shock series per replication
drives every design. Share choosing GARCH at 5%, T = 120 / 240 / 480:

| design | false GARCH, one-stage LR | false GARCH, two-stage LR | power at T = 120, one-stage (oracle 0.25) | AR/MA estimate, 5–95% | residual error / sd, T = 120 → 480 |
|---|---|---|---|---|---|
| white noise, constant only | 0.025 / 0.015 / 0.028 | 0.025 / 0.015 / 0.028 | 0.25 | — | 0.06 → 0.03 |
| white noise fitted as ARMA(1,1) (common factor) | 0.040 / 0.020 / 0.037 | 0.025 / 0.013 / 0.028 | 0.21 | φ̂ about −0.9 … 0.9 at every T (5–95%) | 0.17 → 0.08 |
| ARMA(1,1), roots .5 and .4 (near-cancelling) | 0.040 / 0.028 / 0.037 | 0.022 / 0.015 / 0.037 | 0.22 | φ̂ −0.82 … 0.84 → −0.57 … 0.78 | 0.16 → 0.07 |
| MA(1), θ .9 | 0.048 / 0.045 / **0.087** | 0.028 / 0.015 / 0.040 | 0.29 (inflated) | θ̂ 0.78 … 0.96 → 0.86 … 0.93 | 0.19 → 0.09 |
| MA(1), θ .9, pre-sample shock set to 0 | 0.035 / 0.018 / 0.035 | 0.028 / 0.015 / 0.037 | 0.23 | θ̂ 0.83 … 1.02 → 0.87 … 0.94 | 0.12 → 0.05 |
| white noise fitted as ARIMA(0,1,1) (over-differenced) | 0.062 / 0.083 / **0.168** | 0.033 / 0.018 / 0.040 | 0.28 (inflated) | θ̂ piles up at −1 (5% quantile −1.0 at every T) | 0.24 → 0.18 |

- **A common factor leaves the residuals identified.** The AR and MA coefficients are not identified: φ̂ lands
  anywhere in about [−0.9, 0.9] (5–95%), and that range does not shrink with T. But the residuals depend only on the net
  filter (1 − φL)/(1 + θL), which is identified, and their error shrinks like 1/√T.
- **So the GARCH test behaves as with any two-parameter mean model.** False alarms run 1–4%, and power is 3–4
  points below the oracle at T = 120 and 1–3 from 240. E12's well-identified ARMA(1,1) cost the same.
- **The residuals themselves are not identified when an MA root is on or near the unit circle.**
  - The conditional likelihood sets the pre-sample shock to zero. The error this leaves in the residuals decays
    like |θ|^t, or not at all at θ = −1: there the residual error stops shrinking like 1/√T.
  - The one-stage LR mistakes that start-up error for a volatility burst: 8.7% false GARCH at T = 480 with
    θ = .9, and 6–17% when white noise is over-differenced.
  - Setting the pre-sample shock to zero removes the θ = .9 excess (3.5%), which proves the cause.
  - The exact likelihood (two-stage, Kalman) is unaffected: 1–4%. `compare()` decides with ARCH-LM on
    SARIMAX's residuals there (L9). The MA(1) pile-up at the unit circle is a known result
    (Davis & Dunsmuir 1996).

**(b) Non-fundamental shocks: the one case where the mean model manufactures GARCH.**
y = 0.1 + ε_t + 2ε_{t−1} has its MA root inside the unit circle. Every fit returns its invertible twin
(θ̂ = 0.50, with the same autocorrelations), whose residuals are an all-pass filter of the shocks. Those
residuals are uncorrelated, but dependent unless the shocks are normal (Breidt, Davis & Trindade 2001). The data
have no GARCH. Share rejecting "no ARCH" at 5%, T = 120 / 240 / 480:

| shocks | ARCH-LM(4) on the true shocks | ARCH-LM(4) on the residuals | t LR, one-stage | lag-1 autocorrelation of squared residuals |
|---|---|---|---|---|
| normal, θ = 2 | 0.040 / 0.048 / 0.048 | 0.025 / 0.050 / 0.048 | 0.022 / 0.028 / 0.030 | −0.02 / −0.01 / −0.01 |
| t(8), θ = 2 | 0.050 / 0.037 / 0.043 | **0.155 / 0.235 / 0.407** | 0.120 / 0.175 / 0.378 | 0.07 / 0.08 / 0.09 |
| t(5), θ = 2 | 0.033 / 0.062 / 0.055 | **0.225 / 0.557 / 0.775** | 0.210 / 0.495 / 0.767 | 0.12 / 0.18 / 0.18 |
| t(5), θ = .5 (the invertible twin) | 0.050 / 0.070 / 0.060 | 0.040 / 0.065 / 0.062 | 0.020 / 0.030 / 0.033 | −0.03 / −0.02 / −0.01 |

- **With normal shocks nothing happens.** An all-pass filter of iid normal shocks is again iid normal.
- **With fat tails the "GARCH" is real dependence, but not in the shocks.** The one-step errors given the
  observed past really are conditionally heteroskedastic. The shocks themselves are not: their own ARCH-LM stays
  at 3–6%.
- **No test on the residuals can tell the two apart.** Locating the root needs a non-Gaussian likelihood
  without the invertibility constraint (Lii & Rosenblatt 1992), which the package does not have. The
  noncausal-AR version of the same problem is in Lanne & Saikkonen (2011).

**(c) Fat tails alone fake GARCH in the Gaussian LR.** Under iid shocks with kurtosis κ, the Gaussian quasi-LR
of a variance parameter is (κ − 1)/2 times its normal-theory value, because the Gaussian likelihood assumes
Var(z²) = 2. For t(5) shocks, κ = 9, which makes it four-fold. AR(1) mean; share rejecting "no GARCH" at 5%,
T = 120 / 240 / 480:

| shocks | Gaussian LR (`compare()` ≤ 0.3.0) | Gaussian LR ÷ (κ̂ − 1)/2, null residuals | Gaussian LR ÷ max(1, (κ̂ − 1)/2), GARCH std. residuals | **t LR** | ARCH-LM(4) | restricted bootstrap, T = 240 |
|---|---|---|---|---|---|---|
| normal | 0.015 / 0.015 / 0.030 | 0.018 / 0.010 / 0.035 | 0.013 / 0.010 / 0.030 | 0.015 / 0.007 / 0.030 | 0.030 / 0.028 / 0.043 | — |
| t(8) | 0.080 / 0.090 / 0.098 | 0.020 / 0.035 / 0.028 | 0.028 / 0.048 / 0.033 | 0.013 / 0.025 / 0.033 | 0.037 / 0.055 / 0.043 | — |
| t(5) | 0.138 / 0.188 / 0.175 | 0.013 / 0.035 / 0.022 | 0.035 / 0.052 / 0.035 | 0.028 / 0.037 / 0.018 | 0.065 / 0.045 / 0.043 | 0.050 |
| contaminated normal (κ 8.3) | **0.172 / 0.260 / 0.345** | 0.013 / 0.028 / 0.037 | 0.052 / 0.058 / 0.070 | 0.015 / 0.037 / 0.020 | 0.020 / 0.035 / 0.077 | 0.050 |
| skew-t (ν 8, skewness −1) | 0.065 / 0.147 / 0.135 | 0.013 / 0.025 / 0.028 | 0.028 / 0.045 / 0.040 | 0.020 / 0.045 / 0.028 | 0.035 / 0.043 / 0.058 | 0.057 |

Power with typical GARCH:

| shocks | Gaussian LR | ÷ null κ̂ | ÷ GARCH κ̂ | **t LR** | ARCH-LM(4) |
|---|---|---|---|---|---|
| normal | 0.22 / 0.47 / 0.78 | 0.20 / 0.47 / 0.77 | 0.21 / 0.46 / 0.77 | 0.20 / 0.44 / 0.77 | 0.18 / 0.41 / 0.68 |
| t(8) | (0.29 / 0.56 / 0.84) | 0.18 / 0.41 / 0.72 | 0.22 / 0.45 / 0.72 | 0.20 / 0.48 / 0.79 | 0.20 / 0.37 / 0.63 |
| t(5) | (0.35 / 0.61 / 0.84) | 0.14 / 0.35 / 0.61 | 0.22 / 0.42 / 0.64 | 0.26 / 0.50 / 0.78 | 0.19 / 0.36 / 0.57 |
| contaminated normal | (0.47 / 0.68 / 0.88) | 0.14 / 0.31 / 0.54 | 0.23 / 0.38 / 0.58 | 0.27 / 0.52 / 0.86 | 0.18 / 0.28 / 0.47 |
| skew-t | (0.34 / 0.56 / 0.88) | 0.14 / 0.41 / 0.69 | 0.22 / 0.44 / 0.72 | 0.23 / 0.47 / 0.82 | 0.16 / 0.35 / 0.62 |

The Gaussian LR's power is in brackets because its size is wrong.

- **The t LR is the test to use.** Its false-alarm rate is 1–4.5% under all five shapes, including the two that
  are not t, and it has the most power of the valid cheap tests.
- **It matches the exactly sized restricted bootstrap:** 0.50 against 0.49 with t(5) shocks at T = 240. The
  bootstrap itself holds 5.0–5.7% on every shape, because it resamples the real residuals.
- **With normal shocks the t LR gives up 0–3 points** to the Gaussian LR.
- **Scaling by the null residuals' kurtosis restores size but costs power,** because GARCH itself fattens the
  unconditional tails.
- **Scaling by the GARCH fit's kurtosis keeps more power,** but runs 5–7% under the contaminated normal.

**(d) GARCH's weak identification does not leak into the mean.** Under symmetric shocks the information matrix
is block-diagonal between the mean parameters (constant, regression, AR, MA) and the GARCH parameters. The
reason: each cross term is an odd function of a single past shock (Engle 1982 for regression with ARCH; the same
argument goes through for ARMA means), so it averages to zero when the shocks are symmetric.
- **Measured on AR(1)-GARCH and MA(1)-GARCH at T = 50,000 (robust sandwich):** with normal or t(8) shocks, every
  correlation between a mean estimate and α̂ or β̂ is within ±0.03.
- **With skew-t shocks the separation fails:** correlations reach 0.05–0.14. That is also where E3 and E4 found
  the mean biased.
- **E11(b) shows the practical side:** fitting GARCH where there is none, so that β is unidentified, still
  covers the AR coefficient 0.92–0.96.

**(e) `compare()` end to end.** The raw tests in (c) overstate the damage. `compare()` also demotes a GARCH fit
that ends on a boundary to SARIMAX (L5), and fat-tail false alarms often do. Share of series for which
`compare()` recommends GARCH, AR(1) mean, T = 120 / 240 / 480:

| shocks | GARCH | Gaussian LR alone | `compare()` ≤ 0.3.0 (Gaussian LR) | `compare()` 0.3.1 (t LR) | 0.3.1 with `dists=("normal",)` (kurtosis-scaled) |
|---|---|---|---|---|---|
| normal | none | 0.013 / 0.023 / 0.040 | 0.003 / 0.013 / 0.007 | 0.000 / 0.013 / 0.007 | 0.000 / 0.013 / 0.007 |
| t(8) | none | 0.077 / 0.087 / 0.123 | 0.010 / 0.033 / 0.043 | 0.003 / 0.017 / 0.010 | 0.010 / 0.007 / 0.010 |
| t(5) | none | 0.120 / 0.197 / 0.183 | 0.030 / 0.053 / 0.057 | 0.013 / 0.013 / 0.010 | 0.017 / 0.017 / 0.020 |
| contaminated normal | none | 0.207 / 0.230 / 0.263 | 0.030 / 0.073 / **0.123** | 0.010 / 0.010 / 0.037 | 0.017 / 0.027 / 0.020 |
| normal | typical | 0.22 / 0.49 / 0.78 | 0.18 / 0.45 / 0.77 | 0.16 / 0.44 / 0.76 | 0.16 / 0.44 / 0.75 |
| t(8) | typical | 0.28 / 0.66 / 0.82 | 0.18 / 0.58 / 0.81 | 0.14 / 0.47 / 0.77 | 0.14 / 0.47 / 0.72 |
| t(5) | typical | 0.40 / 0.63 / 0.84 | 0.27 / 0.53 / 0.81 | 0.19 / 0.43 / 0.75 | 0.17 / 0.32 / 0.63 |
| contaminated normal | typical | 0.48 / 0.67 / 0.85 | 0.27 / 0.57 / 0.81 | 0.17 / 0.49 / 0.81 | 0.15 / 0.31 / 0.57 |

- **The old rule's protection fades as T grows.** False GARCH reaches 4–6% with t shocks and 12% with
  contaminated-normal shocks at T = 480, and it keeps rising. The new rule stays at or below 3.7% in every cell.
- **The price is power under fat-tailed GARCH:** 4–11 points (t(8) at T = 240: 0.47 against 0.58), and about one
  point with normal shocks.
  - Part of that gap is the old rule's extra size.
  - The rest is the t LR's conservatism: 1–4.5% at a nominal 5% under fat tails, against 3–5% for the Gaussian
    LR under normal shocks. A likely reason, not measured here, is that the joint ν estimate absorbs outliers
    that would otherwise look like volatility clustering.
- **`compare(bootstrap_lr=199)` recovers the power at exact size.** In (c), the restricted bootstrap found
  t(5) GARCH in 49% of series at T = 240, against 50% for the t LR and 5.0% false alarms.
- **Keep `"t"` in `dists`.** The kurtosis-scaled rule that `dists=("normal",)` falls back on loses up to 24
  points of power under heavy tails (contaminated normal at T = 480: 0.57 against 0.81).
- **E13 also caught a crash.** On one series in 300 (t(5) shocks with GARCH, T = 120), `compare()` raised
  `LinAlgError`.
  - The constant-variance t fit ran towards ν = 2.
  - The simplex polish slid below the optimiser's bound of 2.05, and the finite-difference scores stepped below
    ν = 2.
  - Fixed in 0.3.1: the objective now enforces the same bound, and a non-finite covariance gives NaN standard
    errors with a warning instead of raising. A regression test covers it.

**(f) Does GARCH still help when the shocks are non-fundamental?** Holdout log score over the last 25% (parameters
fitted on the rest) of MA(1)-GARCH-t minus MA(1)-t, 300 series per cell:

| design | T = 240: mean gain (nats/obs) | T = 480: mean gain | T = 480: share with GARCH better |
|---|---|---|---|
| θ = 2, normal shocks | −0.006 (±0.002) | −0.002 (±0.001) | 0.41 |
| θ = 2, t(5) shocks | +0.005 (±0.002) | **+0.010** (±0.002) | 0.69 |
| θ = .5, t(5) shocks (the invertible twin) | −0.006 (±0.001) | −0.004 (±0.001) | 0.38 |

So in the non-fundamental case GARCH is not a false positive for forecasting. It captures a real, if small,
feature of the one-step errors. Where GARCH is merely overfitting, in the two control rows, it costs 0.002–0.006
nats per observation. What it gets wrong is the attribution: the shocks themselves have no volatility
clustering.

## Literature: what is known, what is new

The full review is [`reports/ARMA GARCH limitations literature review.md`](reports/ARMA%20GARCH%20limitations%20literature%20review.md).
It is built from eight research notes in `research_notes/`, with every citation checked against the publisher,
Crossref or RePEc; most results were read at abstract level.

| finding | verdict | anchor in the literature |
|---|---|---|
| E1 | known rule; the size is package-specific | AIC and likelihoods compare only on the same observations under one definition (R `arima` help; Hyndman & Athanasopoulos, FPP3) |
| E2 | known; the ARMA-GARCH numbers are new | standardized-t quantile arithmetic; Baillie & Bollerslev (1992); plug-in bands too narrow: Kim (2001), Pascual, Romo & Ruiz (2006) |
| E3 | known; the magnitudes are new | Newey & Steigerwald (1997): a symmetric non-Gaussian quasi-likelihood identifies the mean only under symmetry; Bollerslev & Wooldridge (1992) |
| E4 | extension: sign and size are new | theory allows the bias (Engle 1982, Theorem 4, needs a symmetric variance; Sin 2003); GJR (1993) showed the variance specification flips GARCH-M conclusions |
| E5, E7 | package-specific | — |
| E6 | known pieces; quantifying them together is new | Engle (1982); weak ARMA: Francq & Zakoian (1998), Francq, Roy & Zakoian (2005); Christoffersen (1998); Lee (1991); Demos & Sentana (1998); Ma, Nelson & Startz (2007); Hughes, King & Kwek (2004) |

What the review added to the study:
- **E1's sign:** the gap depends on the units of y (verified above).
- **E6's ordering is predictable.** Under GARCH the outer product of the AR/MA scores exceeds the Hessian.
  So `opg` is the most anti-conservative choice, then the Hessian, and only the sandwich is consistent. E6
  measured exactly that order. For a regressor independent of the volatility, all three agree, which is why
  slope coverage is fine.
- **E4's reading:** a sandwich fixes the spread of an estimator, not its centre. That is why E4's robust
  intercept CI covers 0.745 under a misspecified variance.
- **Corrections, now applied:**
  - arch 9.0 has `MAMean`.
  - The fourth-moment condition is sufficient, not necessary.
  - Ljung–Box degrees of freedom are an iid-ARMA convention.
  - Only `ugarchboot` "Full" bootstraps parameter uncertainty.
  - GJR persistence is α + γ/2 + β.
  - The χ²(2) LR cut-off is conservative.

## Limitations, ranked by consequence

The guide and `compare()` number these limitations L1–L17 in workflow order. L16 (the Gaussian LR is not robust to
fat tails; fixed in `compare()` 0.3.1) and L17 (non-fundamental MA shocks) come from E13 and are described there.
Mapping from this section's numbers to the guide's codes:
- #1 → L7
- #2 → L8
- #3 and #4 → L12
- #5 → L4
- #6 → L13
- #7 → L10
- #9 → L1
- #10 → L2 and L9
- #11 → L14
- #13–15 → L15

#8 (GARCH(1,1) only) and #12 (the results class) have no code in the guide.

**Wrong answers without a warning**

| # | limitation | evidence | fix | size |
|---|---|---|---|---|
| 1 | `dist="t"` under skewed shocks biases the intercept, the long-run mean and every forecast level toward the mode, and its CIs collapse | E3 | `dist="skewt"` (Hansen 1994, the parameterisation `arch` uses; rugarch's `sstd` is Fernández–Steel). Until then, have `summary()` warn when a `dist="t"` intercept differs from the Gaussian-QMLE intercept by more than one SE (2% false alarms, E3) | medium; the prototype is `common.fit_skewt` |
| 2 | symmetric GARCH under leverage biases the intercept (even with `dist="normal"`) | E4 | GJR variance (arch's `o=1`, rugarch's `gjrGARCH`) and an Engle–Ng sign-bias test in `summary()` | medium; the prototype is `common.fit_gjr` |
| 3 | the default band is normal under t / GARCH: 80% → 0.84, 99% → 0.978 | E2 | make `method="simulated"` the default on the GARCH/t backend (keep `"normal"` for statsmodels parity on request); the exact t quantile at h = 1 | small |
| 4 | symmetric bands under skew: lower 1% tail breached 2–3× too often | E3 | `get_forecast(..., shocks="empirical")`: resample `std_resid` (filtered historical simulation) | small |
| 5 | toggling `garch` compares two different likelihoods: AIC picks GARCH 45–52% of the time, LR false positives 17–27%, on data with no ARCH | E1 | a GARCH-vs-not helper that fits the null on the engine (`fixed α = β = 0`), or `backend="engine"` for `garch=None`; a warning in `summary()` | small |
| 6 | plug-in bands ignore estimation error: 90% → 0.86 at h = 20 when T = 250 and φ = 0.9 | E2 D4 | bootstrap predictive intervals (Pascual, Romo & Ruiz 2006 = rugarch `ugarchboot(method="Full")`; "Partial" and arch's bootstrap forecast ignore parameter uncertainty), reusing `bootstrap()`'s refits | medium; costs `n_boot` refits |
| 7 | a single optimiser start: 2% of fits at T = 250 end more than 0.1 log-lik units below a better optimum, and `converged` misses them while flagging good fits | E5 | 3–5 restarts (two-step estimates plus perturbed starts), keep the best; report convergence from the final candidate (gradient norm, step) | small |

**Missing features (documented, and they limit what can be modelled)**

| # | limitation | note |
|---|---|---|
| 8 | GARCH(1,1) only: no GARCH(p,q), no GARCH-in-mean, no variance regressors, no variance targeting | GARCH-in-mean is the only way for volatility to move the point forecast |
| 9 | no seasonal ARMA with GARCH ("GARCH or seasonal: pick one") | multiplicative SARIMA dynamics (e.g. the airline model) cannot be combined with GARCH; seasonality only through exog |
| 10 | conditional likelihood only | no missing values; the first d + p observations are not scored; unreliable near MA unit roots (already flagged in `summary()`) |
| 11 | the future exog path is taken as known | scenario uncertainty is not in any band; one `get_forecast` per scenario |
| 12 | the results object is a lookalike, not a statsmodels class | no `t_test` / `wald_test` / `f_test`, `append` / `extend` / `apply`, dynamic prediction, `save` / `load`. Wrapping the engine as a statsmodels `GenericLikelihoodModel` would give the real results class |

**Diagnostics**

| # | limitation | fix |
|---|---|---|
| 13 | `summary()` runs Ljung–Box without the p + q degrees-of-freedom adjustment | `model_df=p+q` is Box–Pierce's iid-ARMA convention; under GARCH errors neither version is χ². The corrected laws are Li–Mak (1994), Wong–Ling (2005) and Francq–Roy–Zakoian (2005). On z², count the GARCH parameters |
| 14 | ARCH-LM on standardised residuals uses χ²(lags), but GARCH estimation changes its null distribution | the Li–Mak (1994) test. rugarch prints a weighted Ljung–Box on z and z² and a weighted Li–Mak ARCH-LM; these are different statistics, both with Fisher & Gallagher (2012) gamma approximations |
| 15 | Jarque–Bera tests normality even when `dist="t"` (it rejects by design); nothing tests skewness, sign bias or the fitted density | skewness test, Engle–Ng sign bias, PIT/KS test against the fitted density |

## What to do today

- **Start with `compare(y, exog=X, order=..., trend="c")`** (package 0.3.1). It runs the checks below and prints a
  recommendation with flagged limitations and fallbacks; `sarimax_garch/GUIDE.md` is the workflow behind it.
- **If you keep SARIMAX while GARCH may be present**, fit it with `cov_type="robust"` (E6, E10).
- **For seasonal models such as the airline model, fit GARCH in two stages** (`sarimax_garch/GUIDE.md`,
  section 4). Do not fit GARCH below about 100 observations, and expect it to be identifiable from about 240 (E10).
- **To compare models across backends, d or distributions, use the holdout log-likelihood.** Call
  `compare(..., holdout=0.2)`, or `holdout_logscore()` with `dm_test()`, and report CRPS alongside (E8). Never
  compare in-sample llf or AIC across backends.
- **For the mean** (regression effects, long-run level, point forecasts), use `garch=(1, 1), dist="normal"`.
  It is consistent under skew; under leverage, cross-check the intercept against `garch=None`. Use
  `dist="t"` for the mean only when its intercept agrees with the normal fit's, within one robust SE of
  the normal fit's intercept. That check has a 2% false-alarm rate and 47–100% power (E3). With symmetric
  fat tails, t is the more efficient choice (E3, λ = 0: SD of the slope 0.036 against 0.040). Do not rely
  on the residual skewness printed by `summary()`: it is too noisy under fat tails.
- **For bands**, use `conf_int(method="simulated")` with a fixed `seed`. At small T, widen them, or
  backtest their coverage with `SARIMAXGARCH.backtest()`, because they ignore estimation error.
- **To decide on GARCH**, compare `garch=(1, 1)` with the same model fitted with
  `fixed_params={"alpha.L1": 0, "beta.L1": 0}`. Never compare it with the `garch=None` fit.
- **Compute that LR with `dist="t"`** (what `compare()` 0.3.1 does). The Gaussian LR is (κ − 1)/2 times too
  large under fat tails and picks GARCH for 8–35% of homoskedastic fat-tailed series (E13). Then estimate the
  mean with `dist="normal"`.
- **If GARCH is found while an MA estimate sits near ±1,** trust ARCH-LM on SARIMAX's residuals or the
  two-stage test, not the one-stage LR (E13).
