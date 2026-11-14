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

## 4. The compare workflow

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

## 5. What goes wrong when the choice is wrong

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

## 6. Limitations and fallbacks

| Code | Limitation | Consequence | `compare()` flag | Fallback |
|---|---|---|---|---|
| L1 | no seasonal ARMA with GARCH | multiplicative SARIMA dynamics cannot carry GARCH | (`compare()` has no `seasonal_order`) | month dummies or Fourier terms in exog; SARIMAX when seasonality dominates and ARCH is weak |
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

## 7. Evidence and references

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
- Clark & McCracken (2001), *Journal of Econometrics* 105(1), 85–110 — equal-accuracy tests for nested models.
- Diebold (2015), *JBES* 33(1), doi:10.1080/07350015.2014.983236 — Diebold–Mariano tests compare forecasts, not models.
- Diebold & Mariano (1995), *Journal of Business & Economic Statistics* 13(3), 253–263 — comparing predictive accuracy.
- Engle & Ng (1993), *Journal of Finance* 48(5), 1749–1778 — sign-bias tests.
- Francq & Zakoian (2004), *Bernoulli* 10(4), 605–637 — QMLE of ARMA-GARCH.
- Francq, Roy & Zakoian (2005), *JASA* 100(470), 532–544 — ARMA diagnostics with uncorrelated, dependent errors.
- Giacomini & White (2006), *Econometrica* 74(6), 1545–1578 — tests of (conditional) predictive ability for estimated, possibly nested models.
- Gneiting & Raftery (2007), *JASA* 102(477), 359–378 — proper scoring rules, CRPS.
- Harvey, Leybourne & Newbold (1997), *International Journal of Forecasting* 13(2), 281–291 — the small-sample correction.
- Glosten, Jagannathan & Runkle (1993), *Journal of Finance* 48(5), 1779–1801 — GJR asymmetric GARCH.
- Hughes, King & Kwek (2004), *Economics Letters* 83(2), 269–275 — information criteria for ARCH.
- Lee (1991), *Economics Letters* 37(3), 265–271 — the LM test for GARCH.
- Ma, Nelson & Startz (2007), *Studies in Nonlinear Dynamics & Econometrics* 11(1) — weak identification in GARCH(1,1).
- Newey & Steigerwald (1997), *Econometrica* 65(3), 587–599 — non-Gaussian QMLE under asymmetry.
- Pascual, Romo & Ruiz (2006), *Computational Statistics & Data Analysis* 50(9), 2293–2312 — bootstrap prediction for GARCH.
- Romano & Thombs (1996), *JASA* 91(434), 590–600 — autocorrelation inference under weak assumptions.
- Weiss (1986), *Econometric Theory* 2(1), 107–131 — ARMA models with ARCH errors.
- microprediction `skaters` (Cotton), https://github.com/microprediction/skaters — ranks forecasters by the prequential one-step holdout log-likelihood and reports CRPS alongside.
