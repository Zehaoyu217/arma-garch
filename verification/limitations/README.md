# Methodology limitations of `sarimax_garch` 0.2.0 / engine v3.2 — 2026-11-14

What the package delegates to statsmodels, how it forecasts under GARCH-t, what asymmetric residuals do to
the mean, and where the method stops. Every number below was measured by a script in this folder and is
printed by `report.py` from `out/*.json`.

```bash
cd verification/limitations
python3 e1_backends.py && python3 e2_bands.py && python3 e3_skew.py && python3 e4_leverage.py \
  && python3 e5_converged.py && python3 e6_sarimax_vs_garch.py && python3 report.py   # about 7 minutes on 12 cores
python3 e7_arch_mamean.py      # needs a Python with arch >= 9.0 (not a package dependency)
```
The results of E1–E5 are identical to the last digit (all 2,267 numbers in `out/`) on Python 3.9.6 and 3.11.15,
both with numpy 2.0.2, scipy 1.13.1, pandas 2.3.3 and statsmodels 0.14.6.

**Follow-ups added the same day:**
- E6 (SARIMAX vs SARIMAX-GARCH, both ways round) and E7 (MA-GARCH against arch 9.0).
- `compare()` in package 0.3.0, which runs the decision workflow.
- The user guide [`sarimax_garch/GUIDE.md`](../../sarimax_garch/GUIDE.md), with limitation codes L1–L15 and fallbacks.
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

The guide and `compare()` number these limitations L1–L15 in workflow order. Mapping from this section's
numbers to the guide's codes:
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

- **Start with `compare(y, exog=X, order=..., trend="c")`** (package 0.3.0). It runs the checks below and prints a
  recommendation with flagged limitations and fallbacks; `sarimax_garch/GUIDE.md` is the workflow behind it.
- **If you keep SARIMAX while GARCH may be present**, fit it with `cov_type="robust"` (E6).
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
