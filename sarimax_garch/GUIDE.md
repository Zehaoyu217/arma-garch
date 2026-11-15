# SARIMAX or SARIMAX-GARCH? A compare workflow with evidence, limitations and fallbacks

This guide answers four questions with measured evidence:
- How does the package compare with plain SARIMAX?
- When does GARCH gain an edge?
- Can MA models carry GARCH errors?
- How do you decide, and what goes wrong if you decide wrong?

`compare()` runs the workflow below as one function call.

**Where the evidence comes from:**
- **E1–E7:** Monte Carlo experiments in [`../verification/limitations/`](../verification/limitations/README.md). Each one is a script you can rerun; E6 is the SARIMAX-vs-GARCH grid, with 28 designs and 300 series each.
- **"Small-sample study" and "horizon study":** sections of [`../verification/README.md`](../verification/README.md).
- **Literature:** [the literature review](../verification/limitations/reports/ARMA%20GARCH%20limitations%20literature%20review.md); the key references are at the end of this guide.

**Limitation codes L1–L15** are the codes `compare()` prints in its flags.

## 1. What changes when you switch GARCH on

| | `garch=None` (SARIMAX) | `garch=(1, 1)` (SARIMAX-GARCH) |
|---|---|---|
| Implementation | statsmodels SARIMAX itself, results object unchanged | the package's engine (numpy/scipy), with statsmodels names and output |
| Mean model | regression with SARIMA errors | regression with ARIMA(p, d, q) errors, **including MA terms**; no seasonal ARMA (L1) |
| Error variance | constant σ² | GARCH(1,1) σ²ₜ = ω + α ε²ₜ₋₁ + β σ²ₜ₋₁ |
| Shocks | normal | normal or Student-t |
| Likelihood | exact (Kalman filter), all T observations, missing values allowed | conditional, T − d − p observations, no missing values (L2, L4) |
| Point forecast | conditional mean | the same formula; it differs only through the estimated parameters |
| Bands | normal, width constant over time | width follows σₜ; use `method="simulated"` (L12) |
| Default standard errors | `opg` | robust sandwich (Bollerslev–Wooldridge) |
| Comparing fits | llf/AIC only within this backend | llf/AIC only within the engine. The holdout log score (`holdout_logscore`, `compare(holdout=)`) compares models across both backends, across d and across distributions (E8). |

## 2. Where GARCH gains an edge, and where it does not

**It gains an edge:**

| You need | What GARCH buys | When | Evidence |
|---|---|---|---|
| Intervals that hold in calm **and** turbulent periods (1–6 steps) | SARIMAX's 90% one-step band covers **0.91–0.96 when volatility is low and 0.81–0.86 when it is high**. GARCH covers **0.86–0.90 in both**. Average coverage is the same (0.88–0.90), so only a regime-by-regime check reveals the gap. | whenever GARCH is present; from T ≈ 250 the GARCH band is within 0.02 of nominal | E6; Christoffersen (1998) |
| Volatility or VaR forecasts | the one-step variance-forecast loss falls 30–70% | from ~15 years of monthly data for typical ARCH; under 10 years for strong or persistent ARCH | small-sample study |
| More precise regression and AR/MA estimates | with persistent volatility (α .08, β .90), slope RMSE falls 3–17% and AR/MA coefficient RMSE 4–16%. With typical monthly GARCH (α .10, β .75), slope RMSE falls 0–7%, and at T = 120 the AR coefficient is 4% worse. | grows with persistence and T; at T = 120 only with persistent volatility | E6; Engle (1982), Weiss (1986) |
| Valid AR/MA standard errors | under GARCH, SARIMAX's default `opg` intervals cover the AR/MA coefficient only **0.83–0.92** of the time; the GARCH fit covers 0.91–0.97 | any T with GARCH present. SARIMAX `cov_type="robust"` fixes it too (0.89–0.94). | E6; Francq, Roy & Zakoian (2005), Romano & Thombs (1996) |

**It does not gain an edge:**

| Use | Why not | Evidence |
|---|---|---|
| Point forecasts | the point error against the true conditional mean shrinks by up to 13%, but that error is only 5–17% of the predictive SD, so CRPS moves 0–1.6% | E6 |
| Bands more than ~6–12 steps ahead, or macro-conditioned paths | the GARCH variance reverts to its mean within a few periods; CRPS within 0.02–0.03 (2–4% only when persistence is 0.98) | horizon study; E6 at h = 12 |
| Short samples | GARCH is weakly identified below ~250 observations (half the fits end on a boundary at 120); below ~100, do not fit it | small-sample study; L3 |
| Data without ARCH | GARCH costs ≤ 4% RMSE and ≤ 0.3% CRPS, and its α or β lands on a boundary in 64–87% of fits | E6; Ma, Nelson & Startz (2007) |
| Seasonal ARMA dynamics | not available with GARCH | L1 |

## 3. Can I fit MA models with GARCH?

**Yes.** Use `order=(p, d, q)` with `q > 0` and `garch=(1, 1)`. Exog enters in the regression-with-ARMA-errors form, as in SARIMAX.

```python
from sarimax_garch import SARIMAXGARCH

ma_garch = SARIMAXGARCH(y, exog=X, order=(0, 0, 1), trend="c", garch=(1, 1)).fit()
ma_garch.params["ma.L1"], ma_garch.maroots, ma_garch.invertible
```

**Evidence that it is right:**
- **vs R rugarch:** the filter and forecasts match to 1e-14 across all ARMA orders up to (2,2), and free fits agree within 0.2 SE (`verification/README.md`).
- **vs Python arch 9.0's new `MAMean`:** MA(1), MA(2) and θ = 0.9 fits, normal and t, agree within **0.008 SE** (E7).
  - arch ≥ 9.0, released 2026-11-11, now has pure MA(q) means.
  - It still has no joint ARMA(p,q) and no regression-error form.
- **In E6's MA(1) designs:** without GARCH, θ's RMSE equals SARIMAX's. With persistent GARCH it is 4–13% better (θ = .5 and .9). CI coverage is 0.91–0.97.

**Limits (L9):**
- **Invertibility:** the conditional likelihood starts the recursion with pre-sample errors at 0, and invertibility is only warned about, not enforced. Near-non-invertible MA (|θ| ≳ 0.8) is the weak spot. At θ = 0.9 with no GARCH, the engine's LR test picks GARCH 11% of the time instead of ≤ 5%. `compare()` therefore switches the decision to ARCH-LM on the SARIMAX residuals there.
- **Over-differencing:** an MA root on the unit circle usually means d is one too high.
- **Cancelling roots:** AR and MA roots that nearly cancel mean the orders are too high. `summary()` and `compare()` flag both.

## 4. How many observations, one stage or two, and are the p-values right?

**E10 design** (400 series per cell, normal shocks; full tables from `report.py`):
- **Three SARIMAX types:**
  - ARIMA(1,1,1) with drift;
  - regression with ARIMA(1,1,1) errors and a random-walk regressor;
  - the airline model SARIMA(0,1,1)(0,1,1)₁₂, with θ = −0.4 and Θ = −0.6.
- **GARCH:** none, typical monthly (α .10, β .75) or persistent (α .08, β .90).
- **Sample sizes:** T = 60 to 600.

**The model type barely matters; T and the strength of the ARCH effect decide.** The table gives ranges
across the three types:

| T (years of monthly data) | LR test finds GARCH: typical / persistent | GARCH fit on a boundary | one-step variance-forecast loss, GARCH / constant (< 1: GARCH better): typical / persistent |
|---|---|---|---|
| 60 (5) | 4–8% / 4–8% | 74–86% | 1.0–1.7 / 0.7–1.1 |
| 120 (10) | 19–26% / 28–34% | 47–59% | 0.8–1.0 / 0.34–0.47 |
| 240 (20) | 43–54% / 69–74% | 16–21% | 0.40–0.56 / 0.17–0.23 |
| 360 (30) | 65–68% / 88–90% | 4–11% | 0.31–0.40 / 0.10–0.13 |
| 600 (50) | 86–89% / 97–99% | 1–4% | 0.17–0.22 / 0.05–0.06 |

The loss is QLIKE against the true σ²_{T+1}. Fitting GARCH where there is none makes the variance forecast
1.8–2.8 times worse.

Rules of thumb:
- **Below ~100 observations: do not fit GARCH.** It is found in at most 8% of the samples where it exists, the
  fit sits on a boundary three times in four, and its variance forecast is no better than a constant (loss
  ratio 0.7–1.7).
- **Around 120 observations (10 years monthly): fit it only when the LR test rejects.** Expect a boundary fit
  about half the time, a variance-forecast gain of 0–22% for typical ARCH and about 60% for persistent ARCH,
  and report `bootstrap()` CIs.
- **From about 240 observations (20 years) GARCH is usually identifiable.** It roughly halves the
  variance-forecast loss for typical ARCH and cuts it by about 80% for persistent ARCH.
- **The persistence α + β stays imprecise.** Its RMSE is 0.13–0.20 for typical monthly GARCH, and still
  0.13–0.15 at 600 observations, so do not over-read half-lives.

**Why the power is low: GARCH itself, not the SARIMAX in front of it (E12).** E12 drives several mean models
from the same GARCH shocks:
- **Even with the shocks known exactly** (an oracle with no mean model), the LR test finds typical monthly GARCH
  in only 24% of samples at T = 120, 48% at 240 and 81% at 480. For persistent GARCH it is 39%, 72% and 96%.
- **Ordinary mean models cost little.** A constant mean, OLS, AR(1), ARMA(1,1) and ARIMA(1,1,1) cost 0–4 points.
- **The airline model's seasonal MA costs the most in short samples,** fitted in two stages: 7–11 points at
  T = 120 and 1–7 from 240.
- **A near-unit-root MA costs no power,** but it pushes the one-stage LR's false-alarm rate to 5–10% (L9).
- **Why the signal is weak:** it is only the autocorrelation of the squared shocks, about 0.13 at lag 1 for
  typical GARCH and 0.21 for persistent GARCH. Squared shocks are heavy-tailed, so that correlation is noisy to
  estimate.
- **The choice of test matters more than the mean model.** ARCH-LM(4) has 3–25 points less power than the LR
  test, and ARCH-LM(12) less again: 13% against 24% for typical GARCH at T = 120.

**One stage or two?**
- **Non-seasonal SARIMAX, any (p, d, q), with or without X:** the package estimates the mean and GARCH together
  in one likelihood. The only two-stage element is the ARCH-LM pre-test on the SARIMAX residuals.
- **Seasonal ARMA, such as the airline model:** joint fitting is not available (L1), but two stages work. Fit
  the seasonal SARIMAX, then a zero-mean GARCH(1,1) on its standardized residuals. In E10 it detects GARCH as
  often as one-stage fitting (0–3 points lower), with the same boundary rate and persistence accuracy.
  - At T ≤ 120 its variance forecasts are better: the loss is 6–26% lower than one-stage's (0.78 against 0.93
    for typical GARCH on ARIMA(1,1,1) at T = 120), because it estimates fewer parameters at once. From
    T = 240 the two are equal.
- **Why that is safe:** with symmetric shocks and a symmetric variance equation, the information matrix is
  block-diagonal between the mean and variance parameters (Engle 1982, Theorem 4). So estimating the variance
  from consistent residuals loses nothing asymptotically.
- **What two stages give up:** the mean loses the efficiency gain of joint estimation (E6: −3% to −17% in slope
  RMSE), so its SEs must be robust. Under leverage, though, the two-stage mean is the safer one (E4).

```python
import numpy as np
import pandas as pd

sm2 = SARIMAXGARCH(y, order=(0, 1, 1), seasonal_order=(0, 1, 1, 12)).fit(cov_type="robust")   # stage 1: airline
burn = sm2.loglikelihood_burn                                       # d + D*s diffuse start-up observations
resid = sm2.standardized_forecasts_error[0, burn:] * np.sqrt(sm2.params["sigma2"])   # Kalman-standardized
r = pd.Series(resid, index=y.index[burn:])
stage2 = SARIMAXGARCH(r, order=(0, 0, 0), trend=None, garch=(1, 1))   # stage 2: zero-mean GARCH(1,1)
g2, g2_null = stage2.fit(), stage2.fit(fixed_params={"alpha.L1": 0.0, "beta.L1": 0.0})
lr2 = 2 * (g2.llf - g2_null.llf)                                    # > 5.99 -> GARCH in the residuals
var_next = g2.get_forecast(1).conditional_variance                   # next period's innovation variance
```

Multi-step two-stage bands are not built. They would combine SARIMAX's ψ-weights
(`sm2.impulse_responses(h - 1)`) with `g2.get_forecast(h).conditional_variance`, using the variance formula of
section 1.

**Are the p-values the right size under the null?**

| p-value | its null | rejection rate at nominal 5% | evidence |
|---|---|---|---|
| ARCH-LM(12) on SARIMAX's residuals | no ARCH | 0–5% (conservative at T ≤ 120) | E10 |
| LR test, GARCH vs constant variance, one stage / two stages | no GARCH | 1–5% / 0–4% | E6, E10 |
| LR test with restricted-bootstrap p-value (`garch_test`) | no GARCH | 4.8% at T = 120, 3.0% at 250 (exact level by construction) | E11 |
| Wald test of an AR/MA coefficient, SARIMAX's default `opg` | its true value, GARCH present | **6–17%**: oversized, and it does not shrink with T | E6, E10 |
| the same, SARIMAX `cov_type="robust"` | | 3–11% from T = 120 (4–15% at 60) | E10 |
| the same, one-stage GARCH fit (robust) | | 4–10% from T = 120 (7–17% at 60) | E10 |
| Wald test of the slope or intercept, any of the three | its true value | 1–8% with `opg`; robust and one-stage similar from T = 120 | E10 |
| Wald test of an AR/MA coefficient at T = 60, without GARCH | its true value | 6–15% for every method (small-sample ARMA) | E10 |
| Wald test of α or β | α = 0 lies on the boundary | not valid: non-standard there (Andrews 2001). Wald CIs cover 0.5–0.7 at 5–10 years, so use `bootstrap()` | small-sample study |
| Diebold–Mariano, one-step holdout | equal expected score | 5.4–6.5% at n ≤ 120, about 5% from 250 | E9 |

**When a p-value is not valid: what to do instead**

| problem | why the textbook p-value fails | what to do | in the package | evidence |
|---|---|---|---|---|
| testing "no GARCH" | α = 0 lies on the boundary and β is not identified there. The LR has no χ² law, and the χ²(2) cut-off is conservative (size 1–3%), which costs power. | **Restricted bootstrap of the LR:** fit under H0, resample its residuals, refit both models, and reject when p* ≤ α, choosing B so that α(B + 1) is a whole number (Cavaliere, Nielsen & Rahbek 2017). Or the LM test, whose χ² law holds because it is computed at the null (Lee 1991). | `garch_test(..., B=199)`, `compare(bootstrap_lr=199)`; `arch_lm_test` | E11: size 4.8% at T = 120, and 10–11 points more power than χ²(2) |
| intervals for α, β or the persistence when α may be near 0 | Wald fails at the boundary, and so does the ordinary bootstrap of the estimator (Andrews 2000). | Test first (row above). Report intervals only when GARCH is clearly present (α̂ well above 0, T ≥ 240), using `bootstrap()`. Near the boundary, use subsampling or the m-out-of-n bootstrap (Andrews 2000), or invert the restricted-bootstrap test (not built). | `res.bootstrap()` | small-sample study: at T = 120–180 the bootstrap covers ω and β 0.85–0.95 and α 0.73–0.85, where Wald covers 0.5–0.7 |
| AR/MA p-values at T ≈ 60 | the t-ratio is still far from normal (6–15% size) | a parametric bootstrap-t (the t-ratio is asymptotically pivotal at interior parameters), or profile-likelihood intervals | not built | E10 |
| SARIMAX's default p-values under GARCH | `opg` ignores the GARCH (6–17% size) | the robust sandwich; no bootstrap is needed | `fit(cov_type="robust")` | E6, E10 |
| p-values after choosing GARCH-or-not on the same data | the reported model was selected by the data | **Mean parameters:** nothing beyond robust SEs in every branch (below). **Quantities that exist only in the chosen model:** bootstrap the whole pipeline, or split the sample. | `compare()` | E11 (b) |

**Post-selection: do we need the conditional distribution?** Not for the mean parameters. E11(b) measures it:
the workflow selects GARCH when the LR test rejects, then reports that model's 95% intervals for the slope and
the AR coefficient (1,000 series per cell).
- **Unconditional coverage, with robust SEs in both branches, is 0.91–0.96.** That is the same as never
  selecting: always SARIMAX with robust SEs covers 0.92–0.96, and always GARCH 0.92–0.96. The shortfall at
  T = 120 is a small-sample effect, not a selection effect.
- **Coverage conditional on the selection outcome is 0.91–0.95 given that GARCH was selected, and 0.91–0.96
  given that SARIMAX was.** That conditional coverage is what selective inference requires: an answer "valid
  given that the question was asked" (Fithian, Sun & Taylor).
- **Why conditioning changes nothing here:** the selection statistic (the GARCH LR, built from squared residuals)
  is asymptotically independent of the mean estimators. That is the block-diagonal information matrix of
  Engle (1982) again, and conditioning on an event independent of an estimator leaves its distribution
  unchanged.
- **The shortfall that looks like a selection effect is the `opg` problem.** With SARIMAX's default `opg` SEs in
  the SARIMAX branch, the AR coefficient covers only 0.91–0.93 when GARCH is present but missed. Robust SEs fix
  it.

**When selection does matter.** It matters for α, β, the persistence and the GARCH band width, which exist only
in the chosen model. It also matters when the independence fails, as with skewed shocks (E3) or leverage (E4).
The options then:
- **Bootstrap the whole pipeline:** rerun the selection and the estimation on every bootstrap series. That is
  valid for the procedure as a whole, unconditionally (Efron 2014's bootstrap smoothing).
- **Split the sample in time:** choose the model on the first part and test on the rest. That is valid
  conditionally, at the cost of power.
- **Exact selective inference**, the conditional law given the selection event, exists only for Gaussian linear
  models with polyhedral selection events (Lee, Sun, Sun & Taylor 2016). A likelihood-ratio pre-test on a
  nonlinear MLE has no closed form. A Monte Carlo version keeps only the bootstrap draws that reproduce the same
  selection.
- **Simultaneous intervals over the small set of candidate models** (PoSI: Berk et al. 2013) are valid for any
  selection rule, but conservative.

Leeb & Pötscher (2005) show why the naive bootstrap of a post-selection estimator can't be trusted uniformly.

**Other things to check before trusting GARCH:**
- **Outliers.** A few additive outliers make the ARCH-LM test reject homoskedasticity when it is true; apparent
  GARCH effects are sometimes just outliers (van Dijk, Franses & Lucas 1999). Re-test with dummies for known
  shocks, such as 2020 in monthly macro data.
- **Breaks.** A shift in the variance level looks like highly persistent GARCH (Lamoureux & Lastrapes 1990). If
  α + β is near 1, fit the halves of the sample separately before believing it.
- **Seasonal heteroskedasticity.** A month-specific variance is not GARCH, and GARCH(1,1) cannot represent it;
  it shows up as ARCH at lag 12. Rescale the residuals by month before stage 2.
- **Seasonal residuals.** For seasonal models use the Kalman-standardized residuals after `loglikelihood_burn`,
  as above. The raw residuals include the diffuse start-up values, which look like a variance burst.
- **Test choice.** ARCH-LM(12) has 10–20 points less power than the LR test (E10), and ARCH-LM(4) is closer
  (E6). For GARCH(1,1) the LM test is the same as the ARCH(1) LM test (Lee 1991).

## 5. The compare workflow

One call runs steps 1–4, and step 6 if you pass `backtest=`:

```python
from sarimax_garch import compare

cmp = compare(y, exog=X, order=(1, 0, 0), trend="c")      # SARIMAX, the engine's constant-variance null, GARCH-normal, GARCH-t
print(cmp.summary())                                        # table, tests, recommendation, flags with L-codes and fallbacks
cmp.recommendation["model"], [f["code"] for f in cmp.flags]
```

The steps, with what each one guards against:

**Step 0 — Prepare the data.**
- **Gaps:** the GARCH backend needs a gap-free series (L2); SARIMAX handles missing values.
- **Seasonality:** put it in `exog` as month dummies or Fourier terms (L1).
- **Scale:** keep the series' standard deviation roughly between 0.1 and 10.

**Step 1 — Fit SARIMAX as the reference, with robust standard errors.** Under GARCH its default `opg` intervals cover the AR/MA coefficients only 0.83–0.92 of the time (E6; this is the weak-ARMA problem, Francq, Roy & Zakoian 2005).

```python
from sarimax_garch import SARIMAXGARCH, arch_lm_test

sm = SARIMAXGARCH(y, exog=X, order=(1, 0, 0), trend="c").fit(cov_type="robust")
```

**Step 2 — Test for ARCH, with one likelihood per comparison.**
- **ARCH-LM on the SARIMAX residuals.** It needs no GARCH fit, and for GARCH(1,1) it is the same LM test as for ARCH(1) (Lee 1991).
- **The LR test of GARCH against the engine's constant-variance null.** That null is the same engine with α = β = 0, so both fits share one likelihood and one sample.
- **Never compare `sm.llf` or `sm.aic` with a GARCH fit.** The two backends use different likelihoods (L4). On homoskedastic data that comparison picks GARCH for 45–55% of AR(1) series and 8–10% of MA(1) series (E1, E6). Each AR lag the engine drops shifts its log-likelihood by that observation's log-density. At unit scale that is +1.5, which favours GARCH. At sd 0.1 it is −0.8, and at sd 0.01 (returns in decimals) −3.1, which works against GARCH (E1). The comparison is wrong at every scale.

```python
lm_stat, lm_p = arch_lm_test(sm.resid, lags=4)
mod = SARIMAXGARCH(y, exog=X, order=(1, 0, 0), trend="c", garch=(1, 1))
null = mod.fit(fixed_params={"alpha.L1": 0.0, "beta.L1": 0.0})   # constant variance, same likelihood as the GARCH fit
g = mod.fit()
lr = 2 * (g.llf - null.llf)                                       # > 5.99 -> GARCH
```

How often each rule picks GARCH (E6, 300 series per cell, normal shocks, AR(1) φ .5 and MA(1) θ .5 means):

| truth | T | LR within the engine | ARCH-LM(4) | AIC within the engine | BIC within the engine | AIC across backends (wrong) |
|---|---|---|---|---|---|---|
| no GARCH | 120–1000 | 0.03–0.04 | 0.03–0.07 | 0.07–0.11 | 0.00–0.01 | 0.48–0.55 (AR) / 0.08–0.10 (MA) |
| typical monthly GARCH (α .10, β .75) | 120 / 250 / 500 / 1000 | 0.22–0.23 / 0.46–0.53 / 0.79–0.83 / 0.99 | 0.20–0.21 / 0.40–0.46 / 0.70 / 0.95 | 0.35–0.36 / 0.62–0.70 / 0.89 / 0.99–1.00 | 0.10–0.11 / 0.23–0.27 / 0.51–0.55 / 0.83–0.87 | — |
| persistent GARCH (α .08, β .90) | 120 / 250 / 500 / 1000 | 0.31–0.32 / 0.71–0.78 / 0.97 / 1.00 | 0.20–0.23 / 0.49–0.52 / 0.83–0.85 / 0.99 | 0.44–0.45 / 0.82–0.88 / 0.98–1.00 / 1.00 | 0.19–0.22 / 0.54–0.61 / 0.87–0.90 / 0.99–1.00 | — |

With an MA root near the unit circle (θ = .9, no GARCH), the engine-based rules false-alarm more: LR 0.06–0.11
and AIC 0.12–0.21, against 0.04–0.06 for ARCH-LM. That is why `compare()` decides with ARCH-LM there (L9).

**Use the LR test within the engine.** It has the right size (the χ²(2) cut-off is conservative because α = 0 lies on the boundary; Andrews 2001, Demos & Sentana 1998) and the most power of the correctly sized rules. A parametric bootstrap of the LR statistic, or Demos & Sentana's one-sided LM test, would give exact size. AIC is liberal and BIC conservative (Hughes, King & Kwek 2004).

`compare()` decides on that single test and does not combine tests, because a union of tests over-rejects. When an MA root is near the unit circle, it uses ARCH-LM on the SARIMAX residuals instead (L9).

**In small samples:** no rejection is weak evidence (ARCH-LM has 10–20% power for typical monthly ARCH at 10–15 years, L6). But it is also when GARCH rarely pays off (small-sample study).

**Step 3 — If ARCH is there, fit GARCH with `dist="normal"`, then read the flags.** The Gaussian QMLE is consistent for the mean even when the shocks are skewed or fat-tailed (E3; Bollerslev & Wooldridge 1992, Newey & Steigerwald 1997).

```python
print(g.summary())          # warnings: boundary, persistence cap, MA/AR roots, fourth moment, sample size
g.persistence, g.half_life
```

What to check before using the fit:
- **α or β on a boundary (L5):** there are no usable volatility dynamics, and β and its standard error are spurious (Ma, Nelson & Startz 2007). Go back to SARIMAX.
- **Fewer than 250 observations (L3):** report `g.bootstrap()` CIs for ω, α and β, and Wald CIs for the mean.
- **Sign-bias test rejects (L8, Engle & Ng 1993):** volatility reacts more to bad news than to good. A symmetric GARCH then biases the intercept upward by 0.6–1.4 SD (E4). Take the intercept or long-run mean from SARIMAX; GJR is not in the package yet.
- **Local optima (L10):** `compare()` refits GARCH from SARIMAX's estimates and keeps the better fit. By hand, pass `start_params=` and compare `llf`.

**Step 4 — Student-t only if three checks pass (L7).**
- AIC within the engine prefers t.
- ν is not at its upper bound.
- The t intercept stays within one SE of the Gaussian intercept.

If the intercept moves more than that, skewed shocks are pulling the t centre toward the mode. That bias reaches 2.6–4.9 SD and CI coverage 0.26–0.00 at skewness −0.84 to −1.47 (E3; Newey & Steigerwald 1997). The check false-alarms 2% of the time when the shocks are symmetric.

**Step 5 — Forecast with simulated bands.**

```python
fc = g.get_forecast(12, exog=X_future.iloc[:12], seed=0)
fc.conf_int(alpha=0.10, method="simulated")     # quantiles of simulated paths (L12)
fc.conditional_volatility                       # the volatility forecast itself
```

Four things to know about these bands:
- **The default band is normal-shaped (L12).** `conf_int()` with no `method` gives mean ± z·se. Under t(5) shocks its 80% band covers 0.84 and its 99% band 0.978 (E2).
- **Both bands are plug-in (L13).** They ignore estimation error, so with T = 250 and φ = 0.9 the simulated 90% band covers 0.86 at h = 20, and the default band 0.86–0.87 (E2). Widen them, or check coverage with `backtest()`. rugarch's `ugarchboot(method="Full")` implements the Pascual, Romo & Ruiz (2006) bootstrap that would close this gap.
- **The future exog path is taken as known (L14).** Run one forecast per scenario.
- **Long horizons (L11):** beyond ~6–12 steps, GARCH and SARIMAX bands converge.

**Step 6 — Confirm out of sample with the holdout log-likelihood and CRPS.**

Score every model on the same held-out observations. This is the comparison in-sample llf and AIC cannot make: it is valid across the two backends, across d, GARCH or not, and normal or t. On homoskedastic data the two backends' holdout log scores agree within 0.003 nats per observation (E8), while their in-sample log-likelihoods differ by about 1.5 (E1).

Both scores are strictly proper (Gneiting & Raftery 2007):
- **Holdout log-likelihood (log score).** Fit on all but the last observations, freeze the parameters, and score each held-out observation by the log of its one-step predictive density given everything before it. This is the metric microprediction's [`skaters`](https://github.com/microprediction/skaters) (Peter Cotton) ranks forecasters by, "because it ranks reusable densities". It rewards a right density, tails included.
- **CRPS.** The integrated squared distance between the predictive CDF and the outcome, in the units of y. It is more forgiving in the tails.

```python
from sarimax_garch import dm_test, holdout_logscore

ls_sm = holdout_logscore(y, exog=X, order=(1, 0, 0), trend="c", holdout=50)                # SARIMAX
ls_g = holdout_logscore(y, exog=X, order=(1, 0, 0), trend="c", garch=(1, 1), holdout=50)   # GARCH
gain, (dm_stat, dm_p) = (ls_g - ls_sm).mean(), dm_test(ls_g - ls_sm)    # nats per held-out obs; Diebold-Mariano
cmp_ho = compare(y, exog=X, order=(1, 0, 0), trend="c", holdout=50)     # the same comparison inside compare()
```

For multi-step horizons, `backtest()` refits at each origin and reports `logscore` next to `crps` and coverage. The GARCH density beyond one step is an exact mixture over the simulated paths, with no kernel bandwidth.

```python
bt_sm = SARIMAXGARCH(y, exog=X, order=(1, 0, 0), trend="c").backtest(horizon=6, start=150, step=6)
bt_g = SARIMAXGARCH(y, exog=X, order=(1, 0, 0), trend="c", garch=(1, 1)).backtest(horizon=6, start=150, step=6)
(bt_g["crps"] / bt_sm["crps"]).round(3), (bt_g["logscore"] - bt_sm["logscore"]).round(3)   # CRPS < 1, logscore > 0: GARCH better
```

What E8 measured (300 series per cell; a Diebold–Mariano test at 5% on the holdout):
- **The log score sees density gains CRPS barely registers.**
  - t vs normal under t(5) shocks: +0.05 to +0.08 nats per observation, against a CRPS gain of only 0.6–0.9%.
  - GARCH vs SARIMAX: +0.01 to +0.02 nats (typical GARCH), +0.05 to +0.09 (persistent) and +0.11 to +0.15 (persistent with t(5) shocks). The CRPS gains are 0.4%, 1.3–1.9% and 2.3–3.5%.
- **Power is similar for the two scores.**
  - The test picks GARCH over SARIMAX in 10–21% (typical) and 26–58% (persistent) of series by log score, and 7–18% and 24–56% by CRPS.
  - With t(5) shocks CRPS is ahead (30–62% against 25–52%), because log-score differences are heavy-tailed.
  - False alarms when there is no GARCH: 2–4% for both.
- **Neither replaces the in-sample LR test.** On the training data it picks GARCH in 52–100% of the same series. Use the holdout to confirm the choice across model classes and to rank densities, and report both scores, as `skaters` does.

**Why the Diebold–Mariano test, and exactly what it tests (`dm_test`).**

*The question.* Will model A's density forecasts score better than model B's on future data? That is a
question about two forecasts scored on the same observations, which is the question Diebold–Mariano answers.
Diebold (2015) is explicit that it compares forecasts, not models.

*Hypothesis.* Let d_t = log f_A,t(y_t) − log f_B,t(y_t) for the held-out observations t = 1, …, n. The null is
H0: E[d_t] = 0, against H1: E[d_t] ≠ 0. For the log score, E[d_t] = KL(g‖f_B) − KL(g‖f_A), where g is the true
predictive density. So the null says both forecasts are equally far from the truth, in Kullback–Leibler terms.

*Statistic.* DM = d̄ / √(σ̂²/n). Here σ̂² is the Newey–West long-run variance of d_t, with Bartlett weights and
lag L = max(h − 1, ⌊4(n/100)^{2/9}⌋). The statistic is then multiplied by the Harvey–Leybourne–Newbold
small-sample factor √((n + 1 − 2h + h(h − 1)/n)/n).

*Distribution under H0.* N(0, 1) as n → ∞, when d_t is stationary and weakly dependent. With the
small-sample factor, the reference is t(n − 1).

*Rejection rule at 5%.* Reject when |DM| > t₀.₉₇₅(n − 1), which is 2.00 at n = 60 and 1.97 at n = 250.
DM > 0 means A is better. `dm_test(d)` returns (DM, p); reject when p < 0.05.

*Why it fits here:*
- The comparison is paired on the same observations, so a volatility shock that hits both models cancels.
- d_t is heteroskedastic under GARCH, and autocorrelated for overlapping multi-step forecasts. The HAC variance
  handles both; a naive paired t-test does not. In E9 it rejects a true null 25–26% of the time for 6-step
  forecasts.
- It needs no likelihood theory, so it works across backends, d, normal vs t, and nested or non-nested models.
- On the log score it is the Amisano & Giacomini (2007) weighted likelihood-ratio test with uniform weights.
- `skaters` uses the same per-series test.

*Size when the null is exactly true (E9).*
- One-step holdouts: 6.1–6.5% at n = 60, 5.4–5.6% at n = 120, and about 5% from n = 250.
- Overlapping 6-step forecasts: 7–9% even with the correction, so treat multi-step p-values as approximate.

*When it is the wrong test: deciding whether GARCH exists at all.* SARIMAX is GARCH with α = β = 0. When the
smaller model is true, the estimated larger model is slightly worse out of sample, so the null distribution
shifts. In E9 its mean is −0.19 to −0.33 and its SD 1.15–1.26. It then backs GARCH only 1.7–4.3% of the time
and SARIMAX 6.7–8.0% (Clark & McCracken 2001).

With parameters frozen at the training fit, the null the test checks is "the estimated models score equally"
(Giacomini & White 2006). That is the forecasting question. The population question, "is there GARCH?",
belongs to the in-sample LR test.

## 6. What goes wrong when the choice is wrong

| You fit | The truth | What it costs (measured) | How you notice | Fallback |
|---|---|---|---|---|
| GARCH (incl. MA-GARCH) | no GARCH | RMSE within +4% (typically ≤ 2%), CRPS +0.0–0.3%; α or β on a boundary in 64–87% of fits; β and persistence then meaningless | `summary()` / `compare()` boundary flag (L5); LR p-value large | SARIMAX |
| MA-GARCH with θ near 1 | MA, no GARCH | the engine's LR test picks GARCH 11% of the time (θ = .9, T = 1000) | `compare()` L9 flag | decide with ARCH-LM on SARIMAX's residuals (`compare()` does) |
| SARIMAX | GARCH | the 90% band covers 0.91–0.96 in calm periods and 0.81–0.86 in turbulent ones; `opg` CIs for AR/MA cover 0.83–0.92; slope RMSE up to 21% higher than GARCH's | ARCH-LM on the residuals | GARCH, or at least `cov_type="robust"` |
| GARCH-t | skewed shocks | intercept and long-run mean biased toward the mode by +0.02 to +0.12 shock-SD; CI coverage 0.79 → 0.00 as skew grows | `compare()` L7 (t vs normal intercept) | `dist="normal"` with simulated bands |
| symmetric GARCH | leverage (GJR) | intercept biased up by 0.6–1.4 SD; long-run mean up to 27% too high | `compare()` L8 (sign-bias test) | take the intercept from SARIMAX; GJR not yet built |
| the default normal band | t shocks | 80% → 0.84, 99% → 0.978 | — | `method="simulated"` |
| any plug-in band | small T, persistent mean | simulated 90% band → 0.86 at h = 20 | `backtest()` coverage | widen; bootstrap intervals (not built) |
| SARIMAX llf/AIC vs a GARCH fit | either | at unit scale, AIC picks GARCH for 45–55% of homoskedastic AR(1) series; on small-scale data the bias reverses | — | compare within the engine (`compare()`), or by the holdout log score, which is valid across backends (E8) |

## 7. Limitations and fallbacks

| Code | Limitation | Consequence | `compare()` flag | Fallback |
|---|---|---|---|---|
| L1 | no seasonal ARMA with GARCH | multiplicative SARIMA dynamics cannot carry GARCH in one stage | (`compare()` has no `seasonal_order`) | two stages: seasonal SARIMAX, then GARCH on its standardized residuals (section 4 recipe; E10: as good as one stage); or month dummies / Fourier terms in exog |
| L2 | GARCH backend needs a gap-free series | NaN raises | — | SARIMAX (Kalman handles gaps) or impute |
| L3 | GARCH weakly identified below ~250 observations | boundary fits; Wald CIs cover α/β only 0.5–0.7 at 5–10 years of monthly data | warn < 250, stop < 100 | `dist="normal"` + `bootstrap()`; keep SARIMAX below 100 |
| L4 | the two backends' likelihoods differ (exact vs conditional) | biased GARCH choices across backends: towards GARCH at unit scale, against it below sd ≈ 0.25 | always shown | compare within the engine, or out of sample by the holdout log score (`compare(holdout=)`, `holdout_logscore`) or `backtest()` |
| L5 | α = 0 is a boundary; β unidentified there; χ²(2) LR conservative; fourth-moment condition | spurious β/persistence; non-standard inference | boundary, cap, fourth moment | treat boundary fits as no GARCH; bootstrap CIs |
| L6 | ARCH tests have low power in short samples | GARCH missed at 10–15 years | info when ARCH-LM does not reject and nobs < 360 | accept SARIMAX there (GARCH rarely pays off) |
| L7 | symmetric t under skewed shocks biases the mean | intercept toward the mode | t-vs-normal intercept, ν at bound | `dist="normal"` |
| L8 | no asymmetric (GJR) variance | intercept biased when leverage is present | sign-bias test | intercept from SARIMAX |
| L9 | conditional likelihood near MA/AR unit roots; invertibility not enforced | MA θ near 1 unreliable; LR over-rejects | MA/AR roots, cancellation | compare with SARIMAX (exact, enforces invertibility); try d − 1; decide with ARCH-LM |
| L10 | single-start optimiser; `converged` reflects L-BFGS-B only | 2% of T = 250 t-fits at a worse optimum (E5) | converged = False | `compare()` refits from SARIMAX's estimates; `start_params=` |
| L11 | GARCH's edge is short-horizon | no gain beyond ~6–12 steps | in the recommendation | SARIMAX for long-horizon paths |
| L12 | default band is normal-shaped | wrong tails under t / GARCH | in the caveats | `conf_int(method="simulated")` |
| L13 | bands ignore parameter uncertainty | under-coverage at small T, long h | in the caveats | widen; `backtest()` coverage; bootstrap intervals (not built) |
| L14 | future exog taken as known | scenario risk missing from bands | — | one forecast per scenario |
| L15 | residual diagnostics' p-values are approximate after GARCH (Li & Mak 1994; Francq, Roy & Zakoian 2005) | Ljung-Box / ARCH-LM on z indicative only | — | rely on `backtest()`; weighted tests (rugarch) for formal checks |

## 8. Evidence and references

**Experiments**, all in `../verification/limitations/`, reproducible with the listed scripts:
- **E1:** backend likelihood gap and false GARCH choices.
- **E2:** band coverage under GARCH-t.
- **E3:** skewed shocks and the mean.
- **E4:** leverage and the mean.
- **E5:** convergence and local optima.
- **E6:** SARIMAX vs SARIMAX-GARCH both ways (`e6_sarimax_vs_garch.py`).
- **E7:** MA-GARCH against arch 9.0 (`e7_arch_mamean.py`, needs `pip install "arch>=9.0"`).
- **E8:** the holdout log-likelihood against CRPS (`e8_holdout_logscore.py`).
- **E9:** the Diebold–Mariano statistic's null distribution, exact and nested (`e9_dm_null.py`).
- **E10:** sample size by model type (ARIMA(1,1,1), regression with ARIMA errors, airline), one stage vs two, and p-value sizes (`e10_sample_size.py`).
- **E11:** valid inference when the textbook p-values fail: the restricted bootstrap for the GARCH test, and post-selection coverage of the mean (`e11_valid_inference.py`).
- **E12:** where the low power comes from, with the same GARCH shocks behind eight mean models and an oracle (`e12_power_source.py`).

**Earlier studies**, in `../verification/README.md`:
- 1:1 checks against rugarch, statsmodels and arch;
- the Monte Carlo grid (13,000 fits);
- the small-sample study (monthly, 5–30 years);
- the horizon study (levels vs differences; GARCH or not).

**References.** These were checked against the publisher, Crossref or RePEc; the full list, with what each paper shows and its relation to E1–E7, is in the literature review.
- Andrews (2001), *Econometrica* 69(3), 683–734 — testing with a parameter on the boundary.
- Bollerslev & Wooldridge (1992), *Econometric Reviews* 11(2), 143–172 — QMLE and robust standard errors.
- Christoffersen (1998), *International Economic Review* 39(4), 841–862 — conditional coverage of interval forecasts.
- Demos & Sentana (1998), *Journal of Econometrics* 86(1), 97–127 — one-sided tests for GARCH; boundary mixtures.
- Engle (1982), *Econometrica* 50(4), 987–1007 — ARCH-LM; ML vs least squares under ARCH.
- Amisano & Giacomini (2007), *JBES* 25(2), 177–190 — comparing density forecasts by weighted likelihood ratio tests.
- Andrews (2000), *Econometrica* 68(2), 399–405 — the bootstrap is inconsistent when a parameter is on the boundary.
- Berk, Brown, Buja, Zhang & Zhao (2013), *Annals of Statistics* 41(2), 802–837 — valid post-selection inference (PoSI).
- Cavaliere, Nielsen & Rahbek (2017), *Journal of Time Series Analysis* 38(4), 513–534 — the restricted bootstrap is consistent for testing no ARCH.
- Clark & McCracken (2001), *Journal of Econometrics* 105(1), 85–110 — equal-accuracy tests for nested models.
- Diebold (2015), *JBES* 33(1), doi:10.1080/07350015.2014.983236 — Diebold–Mariano tests compare forecasts, not models.
- Diebold & Mariano (1995), *Journal of Business & Economic Statistics* 13(3), 253–263 — comparing predictive accuracy.
- Efron (2014), *JASA* — estimation and accuracy after model selection (bootstrap smoothing).
- Engle & Ng (1993), *Journal of Finance* 48(5), 1749–1778 — sign-bias tests.
- Fithian, Sun & Taylor, "Optimal inference after model selection", arXiv:1410.2597 — selective inference.
- Francq & Zakoian (2004), *Bernoulli* 10(4), 605–637 — QMLE of ARMA-GARCH.
- Francq, Roy & Zakoian (2005), *JASA* 100(470), 532–544 — ARMA diagnostics with uncorrelated, dependent errors.
- Giacomini & White (2006), *Econometrica* 74(6), 1545–1578 — tests of (conditional) predictive ability for estimated, possibly nested models.
- Gneiting & Raftery (2007), *JASA* 102(477), 359–378 — proper scoring rules, CRPS.
- Harvey, Leybourne & Newbold (1997), *International Journal of Forecasting* 13(2), 281–291 — the small-sample correction.
- Glosten, Jagannathan & Runkle (1993), *Journal of Finance* 48(5), 1779–1801 — GJR asymmetric GARCH.
- Hughes, King & Kwek (2004), *Economics Letters* 83(2), 269–275 — information criteria for ARCH.
- Lamoureux & Lastrapes (1990), *JBES* 8, 225–235 — structural change and spurious persistence in GARCH.
- Lee, Sun, Sun & Taylor (2016), *Annals of Statistics* 44(3), 907–927 — exact post-selection inference (polyhedral lemma).
- Leeb & Pötscher (2005), *Econometric Theory* 21, 21–59 — model selection and inference: facts and fiction.
- Lee (1991), *Economics Letters* 37(3), 265–271 — the LM test for GARCH.
- Ma, Nelson & Startz (2007), *Studies in Nonlinear Dynamics & Econometrics* 11(1) — weak identification in GARCH(1,1).
- Newey & Steigerwald (1997), *Econometrica* 65(3), 587–599 — non-Gaussian QMLE under asymmetry.
- Pascual, Romo & Ruiz (2006), *Computational Statistics & Data Analysis* 50(9), 2293–2312 — bootstrap prediction for GARCH.
- Romano & Thombs (1996), *JASA* 91(434), 590–600 — autocorrelation inference under weak assumptions.
- van Dijk, Franses & Lucas (1999), *Journal of Applied Econometrics* 14(5), 539–562 — additive outliers and spurious ARCH.
- Weiss (1986), *Econometric Theory* 2(1), 107–131 — ARMA models with ARCH errors.
- microprediction `skaters` (Cotton), https://github.com/microprediction/skaters — ranks forecasters by the prequential one-step holdout log-likelihood and reports CRPS alongside.
