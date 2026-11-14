# Conditional vs exact ARMA likelihood, comparability of likelihood/AIC, and ARMA-GARCH software capabilities (checked 2026-11-14)

Versions checked: statsmodels 0.14.6 (released 2025-12-05) and 0.15.0 (released 2026-08-27, current on PyPI); arch 9.0.0 (released 2026-11-11, current on PyPI; 8.0.0 was 2025-10-21); rugarch 1.5-6 (CRAN, 2026-07-18). Dates from the PyPI JSON API and the CRAN package page. Anything not read from a primary source is marked UNVERIFIED.

## Verdict on the claims listed in the study context

| Claim | Verdict |
|---|---|
| statsmodels 0.14.6 has no GARCH model (no GARCH class, no sandbox one) | CORRECT. The 0.14.6 macOS wheel has no `garch` module. `sandbox/tsa/` holds only movstat, fftarma, varma and similar files. The old sandbox GARCH was deprecated and moved to an archive in 2019. Still true for 0.15.0 (see Q3 caveat). |
| Python arch has no MA terms in the mean | OUTDATED / WRONG as of arch 9.0.0 (2026-11-11). It was true for 8.0.0 and earlier. arch 9.0.0 adds `MAMean`, a pure MA(q) mean model. There is still no joint ARMA(p,q) mean. |
| arch's SkewStudent is Hansen (1994) | CORRECT. The class docstring cites Hansen (1994). |
| rugarch's sstd is Fernandez-Steel | Supported but not stated in one sentence. The rugarch manual cites Fernandez & Steel (1998) under the distribution functions. See Q5. |
| rugarch ugarchboot implements Pascual-Romo-Ruiz | CORRECT for method "full". "partial" is a simpler variant with no parameter uncertainty. See Q5. |
| ugarchfit prints weighted Ljung-Box and weighted ARCH-LM (Fisher-Gallagher), Nyblom, sign bias, adjusted Pearson | Largely correct. The manual documents Nyblom, sign bias and the adjusted goodness-of-fit method, and cites Fisher & Gallagher. The printed headings were verified only from user-posted output. See Q5. |
| arch's GJR is o=1 | CORRECT. |
| statsmodels SARIMAX default cov_type is 'opg' | CORRECT. |

---

## 1. Conditional vs exact Gaussian ARMA likelihood, and comparability of log-likelihoods / AIC

### Takeaway
Conditional (CSS / conditional ML) and exact (Kalman-filter) Gaussian ARMA likelihoods are different objective functions. They are asymptotically equivalent for invertible/stationary models but differ in finite samples. The conditional version drops the density of the initial observations and sets pre-sample errors to zero. Likelihood-based comparisons (AIC, BIC, LR) are valid only for the same response data scored by the same likelihood definition. The E1 offset of about +1.5 is consistent with this rule.

### Cited Findings
- Ansley & Newbold (1980), "Finite sample properties of estimators for autoregressive moving average models", Journal of Econometrics 13(2), 159-183, DOI 10.1016/0304-4076(80)90013-5. Simulation comparison of three estimators (maximum likelihood, least squares, conditional least squares). Estimators that are asymptotically equivalent can differ substantially in moderate samples. Interval estimates can be too narrow in mixed models with parameter redundancy. Relation to study: extends (finite-sample backdrop for the conditional/exact gap). Source: [bibliographic record](https://www.econbiz.de/Record/finite-sample-properties-of-estimators-for-autoregressive-moving-average-models/10001835242). UNVERIFIED: full text not read. A secondary citation says a nominal-5% t-test of a zero MA coefficient had actual size 45.7%, which I did not confirm in the paper.
- Hamilton (1994), Time Series Analysis, Princeton University Press (ISBN of the 2020 reprint: 9780691218632). Chapter 5 "Maximum Likelihood Estimation" (pp. 117-151) has the Gaussian AR, MA and ARMA likelihood sections. Chapter 13 "The Kalman Filter" (pp. 372-408), section 13.4, gives ML of state-space models, which is the route to the exact ARMA likelihood. Relation to study: E1 known. Sources: [table of contents](https://oeaw.ac.at/resources/Record/9780691218632/TOC), [Project MUSE](https://muse.jhu.edu/book/77442). UNVERIFIED: I did not find a source that labels these sections as "conditional vs exact". Check the text. The 2020 reprint's pagination differs from 1994, so use 1994 pages.
- Harvey (1989), Forecasting, Structural Time Series Models and the Kalman Filter, Cambridge University Press, xvi+554 pp. DOI 10.1017/CBO9781107049994. Treats exact likelihood by prediction-error decomposition from the Kalman filter. Relation to study: known. Source: [Cambridge Core](https://www.cambridge.org/core/books/forecasting-structural-time-series-models-and-the-kalman-filter/univariate-time-series-models/6CF957B9D6D6CD7055172EDF82F8768C), [library catalogue](https://find.library.upenn.edu/catalog/9979239295203681). UNVERIFIED: chapter or section for the ARMA likelihood.
- Box & Jenkins, Time Series Analysis: Forecasting and Control. The current edition is Box, Jenkins, Reinsel & Ljung, 5th ed., Wiley, 2015. UNVERIFIED: my search returned nothing for it, so publisher, year and ISBN come from memory. The book's treatment of conditional least squares versus exact likelihood (back-forecasting) was not checked.
- R `stats::arima` help: methods "CSS-ML" (default, CSS for starting values then ML), "ML" (exact likelihood via a state-space representation) and "CSS" (conditional sum of squares from observation `n.cond`, earlier innovations set to zero, "mainly for expositional purposes"). The `aic` component is "only valid for method = ML fits". For CSS the reported "part log-likelihood" is half the log of the estimated mean square. `n.cond` "can be used to allow comparability between different fits". Relation to study: known; direct documentary support for the E1 rule. Source: [R stats::arima](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/arima.html).
- Hyndman & Athanasopoulos, Forecasting: Principles and Practice, 3rd ed., section "ARIMA modelling in R": AIC/AICc values are not comparable across different orders of differencing because "differencing changes the data on which the likelihood is computed". AICc compares models only within the same class (not ETS vs ARIMA). The text does not discuss conditional vs exact likelihood. Relation to study: extends (same-data rule). Source: [FPP3 ch. 9.5](https://otexts.com/fpp3/arima-r.html).
- Burnham & Anderson (2002), Model Selection and Multimodel Inference: A Practical Information-Theoretic Approach, 2nd ed., Springer, DOI 10.1007/b97636. A secondary source attributes to it the rule that all candidate models must be fitted to the same observations. Source: [Springer DOI](https://link.springer.com/doi/10.1007/b97636). UNVERIFIED: I did not read the book. The attribution and page number are not confirmed.
- statsmodels SARIMAX likelihood handling, from source: with the default approximate-diffuse initialization, the likelihood burn is set to the number of non-stationary (diffuse) states, i.e. `k_states` minus the ARMA order when `enforce_stationarity` is True. So d=0 gives burn 0 and all T observations are scored with the exact stationary-initialized likelihood. With d>0 the first d+sD diffuse terms are dropped. `use_exact_diffuse=True` switches to exact diffuse initialization. `simple_differencing=True` differences the data before estimation (discards the first sD+d rows) and so changes the sample being scored. `concentrate_scale=True` concentrates the scale out of the likelihood. Relation to study: package-specific (it shows why statsmodels scores T observations while the package scores T-d-p). Sources: [SARIMAX source, release branch main](https://raw.githubusercontent.com/statsmodels/statsmodels/main/statsmodels/tsa/statespace/sarimax.py), [SARIMAX docs 0.15.0](https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html).
- statsmodels `fit` default `cov_type` is `'opg'` (outer product of gradients), or `'approx'` when memory conservation is used. Relation to study: the claim is correct. Source: [MLEModel.fit docs](https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.mlemodel.MLEModel.fit.html).
- statsmodels AIC definitions: a maintainer discussion states that SARIMAX, ARIMA and AutoReg include the scale in the AIC while VAR does not, and that AutoReg/VAR follow Lutkepohl, dropping the constant and dividing by nobs. The constant dropped differs from the one used by R's `forecast`. Relation to study: extends (constants and parameter counts also differ across backends, not only the burn-in). Source: [statsmodels issue 6577](https://github.com/statsmodels/statsmodels/issues/6577).
- arch `MAMean` documents its MA likelihood as conditional: innovations before the estimation sample are set to 0, and "This is the conditional likelihood, and its effect vanishes when the process is invertible". Relation to study: contradicts the claim that arch cannot do MA; confirms the same conditional-likelihood design as the package. Source: arch source (see Q4).

### Inferences
- The E1 offset (the package's conditional log-likelihood about 1.5 above the exact one) is explained by definition. The exact log-likelihood includes the log-density of the first observation(s), which the conditional likelihood omits. With unit-scale errors the log-density of one observation is about -1.4 (-0.5*ln(2*pi*sigma^2) - 0.5). Under that reading the offset is not an estimation difference. This arithmetic is mine, not from a source.
- Documented practice in R (`arima` AIC valid only for ML; `n.cond` for comparability) and FPP3 (same data) match the study's conclusion that GARCH-engine versus no-GARCH-statsmodels comparisons need a common likelihood definition, for example by rescoring both on the same T-d-p observations or by computing both with the same engine.
- Near MA unit roots the Gaussian ML and CSS estimators are expected to differ more and the pile-up problem arises. I found no source I read that confirms this for the conditional/exact comparison, so treat it as a gap.

### Gaps
- Box & Jenkins and Hamilton: no source I could read states the "conditional vs exact" wording or page numbers. Chapter and section names are secondary.
- Burnham & Anderson: exact page for the same-data requirement not located.
- Behaviour near MA unit roots (Ansley & Newbold 1980 full text, Davis & Dunsmuir style pile-up results) was not verified.
- Hyndman's explicit comments on CSS versus ML AIC were not located. FPP3 text read does not discuss the distinction.

---

## 2. R `stats::arima` and forecast/fable guidance

### Takeaway
R documents that only ML-method fits have a valid AIC, and `n.cond` exists to align CSS fits. FPP3 says AIC across different differencing is invalid because the data differ.

### Cited Findings
- See Q1 entries for `arima` and FPP3. Relation to study: E1 known / extends.
- fable ARIMA uses maximum likelihood and reports the log-likelihood. The FPP3 text does not say conditional or exact. Source: [FPP3 ch. 9.5](https://otexts.com/fpp3/arima-r.html).

### Inferences
- The R default "CSS-ML" ends in an exact ML fit, so its reported AIC is on the exact likelihood. The package's conditional likelihood is closer to R's method="CSS" objective, whose AIC R labels invalid.

### Gaps
- Behaviour of `forecast::auto.arima` with CSS (e.g. `approximation=TRUE`) was not checked.

---

## 3. statsmodels: SARIMAX and any GARCH model

### Takeaway
statsmodels 0.14.6 and 0.15.0 contain no supported GARCH model. SARIMAX is an exact (Kalman-filter) likelihood with approximate diffuse initialization by default.

### Cited Findings
- Version: statsmodels 0.14.6 released 2025-12-05 and 0.15.0 released 2026-08-27 (requires Python >= 3.10). Source: [PyPI JSON](https://pypi.org/pypi/statsmodels/json).
- No GARCH module in the 0.14.6 wheel (file listing: no `garch` file; `sandbox/tsa` has only movstat, fftarma, varma, try_* and example files). The sandbox docs list only movstat functions. Sources: [0.14.6 wheel on PyPI](https://pypi.org/project/statsmodels/0.14.6/#files), [sandbox docs](https://www.statsmodels.org/stable/sandbox.html). Relation to study: the claim is correct.
- History: PR 6000 "DOC: Clarify that GARCH models are deprecated" merged 2019-07-23 (milestone 0.11). PR 6059 "REF: move garch to archive/" merged 2019-08-03. Issue 5718 "DISC: Make arch usage official?" closed 2019-07-23. Issue 5441 "availability of ARCH and GARCH" closed 2019-07-15. Sources: [PR 6000](https://github.com/statsmodels/statsmodels/pull/6000), [issue search](https://github.com/statsmodels/statsmodels/issues?q=GARCH). Relation to study: the "old sandbox one" was removed from the package in 2019 (moved to archive/); the "removed from 0.14.6" fact itself is confirmed by the wheel listing.
- statsmodels 0.14.0 release notes do not mention GARCH. Source: [0.14.0 notes](https://www.statsmodels.org/stable/release/version0.14.0.html).
- Issue 6400 "FAQ: related and complementary packages" is open and the usual place users are pointed to arch. Source: same issue search. Not read in detail.

### Inferences
- The package's statement "garch=None and normal errors returns statsmodels SARIMAX results" is consistent with statsmodels providing the only ARMA exact-likelihood path in the stack.

### Gaps
- For 0.15.0, I checked the GitHub tree listing (no path containing "garch" matched) but did not check whether the listing was truncated, and I did not inspect the 0.15.0 wheel or release notes. Treat "0.15.0 has no GARCH" as strongly likely, not file-verified.
- The contents of `sandbox/archive/tsa.py` in the 0.14.6 wheel were not inspected.

---

## 4. Python arch (version 9.0.0)

### Takeaway
arch offers ZeroMean, ConstantMean, ARX, HARX, LS, ARCHInMean, and, new in 9.0.0, a pure MA(q) mean `MAMean` estimated by conditional likelihood with all volatility processes and distributions. It still has no joint ARMA(p,q) mean.

### Cited Findings
- Version: arch 9.0.0 released 2026-11-11 (the release page says last release supporting Python 3.10 and NumPy 1.x). Sources: [PyPI JSON](https://pypi.org/pypi/arch/json), [releases](https://github.com/bashtage/arch/releases). The 9.0 changes page returned HTTP 404; the release page entry was the source.
- Mean models listed in the docs (latest, a pre-9.0 snapshot at fetch time): ZeroMean, ConstantMean, ARX, HARX, LS, ARCHInMean; the page does not mention MA. Source: [mean docs](https://arch.readthedocs.io/en/latest/univariate/mean.html).
- `MAMean` added: issue 147 "Add a univariate MA model" (opened 2017-01-05) closed "completed" on 2026-10-18; PR 778 (opened 2025-06-16 by a contributor) was closed 2026-10-12 and superseded by PR 903 "ENH: Add MA(q) mean model with GARCH support and tests", merged 2026-10-12. The 9.0.0 release notes list MAMean as a moving average mean that works with any volatility process and distribution. Sources: [issue 147](https://github.com/bashtage/arch/issues/147), [PR 778](https://github.com/bashtage/arch/pull/778), [PR 903](https://github.com/bashtage/arch/pull/903), [releases](https://github.com/bashtage/arch/releases). Relation to study: contradicts "arch has no MA terms" for >= 9.0.0.
- `MAMean` docstring (main branch source): y_t = mu + e_t + theta_1 e_{t-1} + ... + theta_q e_{t-q}, `lags` = q >= 1, optional constant, `hold_back`. Innovations are computed recursively with pre-sample innovations 0, "the conditional likelihood". Invertibility is enforced for q=1 (bounds) and q=2 (linear constraints) only. Starting values are invertible. It is MA only: no AR terms can be combined. Source: [arch mean.py](https://raw.githubusercontent.com/bashtage/arch/main/arch/univariate/mean.py). Relation to study: contradicts the claim; confirms the same zero-pre-sample conditional design as the package; the package's niche shrinks to ARMA(p,q)+GARCH jointly and to numpy/scipy-only dependencies. PR 903 notes explosive residuals are capped for larger q rather than prevented, and that the zero-pre-sample assumption means the recursion only matches the exact Kalman filter after the initial effect decays. Not yet in a published docs page I could read.
- Maintainer's 2017 remark in issue 147: reliable estimators for full ARFIMA models with ARCH errors are hard to build. Source: [issue 147](https://github.com/bashtage/arch/issues/147).
- Volatility processes: ConstantVariance, GARCH(p,o,q,power), FIGARCH(p,q,power,truncation), EGARCH(p,o,q), HARCH(lags), MIDASHyperbolic, ARCH(p), APARCH(p,o,q,delta,common_asym); fixed-parameter EWMAVariance and RiskMetrics2006; FixedVariance. 9.0.0 release notes also list FIAPARCH and NGARCH (not independently checked in the docs). Source: [volatility docs](https://arch.readthedocs.io/en/latest/univariate/volatility.html), [releases](https://github.com/bashtage/arch/releases).
- GJR: `GARCH(p=1, o=1, q=1)` is GJR-GARCH; with `power=1.0` it is TARCH. The variance recursion is sigma^lambda = omega + sum alpha |e|^lambda + sum gamma |e|^lambda I[e<0] + sum beta sigma^lambda. Source: [GARCH docs](https://arch.readthedocs.io/en/latest/univariate/generated/arch.univariate.GARCH.html). Relation to study: the claim is correct.
- Distributions: Normal, StudentsT (standardized), SkewStudent (standardized, two parameters eta (tail) and lambda (skew); lambda = 0 gives standardized Student-t; bounds eta in [2.05, 300], lambda in [-1, 1]), GeneralizedError (one shape parameter, bounds [1.01, 500]). SkewStudent cites Hansen, B. E. (1994), "Autoregressive conditional density estimation", International Economic Review 35(3), 705-730. Sources: [distribution docs](https://arch.readthedocs.io/en/latest/univariate/distribution.html), [SkewStudent docs](https://arch.readthedocs.io/en/latest/univariate/generated/arch.univariate.SkewStudent.html), [source](https://raw.githubusercontent.com/bashtage/arch/main/arch/univariate/distribution.py). Relation to study: the claim is correct. UNVERIFIED: the GED scale convention (unit-variance standardization) was not read in the docs.
- Forecast methods: "analytic" (only for models linear in squared residuals such as GARCH, HARCH), "simulation" and "bootstrap". Bootstrap forecasts are nearly identical to simulation, but shocks are drawn with replacement from the standardized residuals e_t = (r_t - mu)/sigma_t. The docs never use the term filtered historical simulation, but the described method is that. Both need horizon > 1 to be useful. The page does not discuss parameter uncertainty: forecasts use the fitted (or supplied) parameters and vary only the shocks. Source: [forecasting docs](https://arch.readthedocs.io/en/latest/univariate/forecasting.html). Relation to study: package-specific (differs from rugarch's "full" bootstrap, which refits parameters).
- Mean model example in the docs and a Stack Overflow thread both say arch has no ARMA mean, only ARX. Source: [search result](https://kiwix.lema.org/stackoverflow.com_en_all_2023-11/questions/41477162/how-to-fit-a-arma-garch-model-in-python). Dated, pre-9.0.

### Inferences
- "arch has no MA terms" should be rewritten as: "arch added a pure MA(q) mean (MAMean) in 9.0.0 (2026-11-11); there is no ARMA(p,q) mean". Anything dated before 2026-11-11 is correct under the old wording.
- arch estimates mean and volatility jointly, so MAMean+GARCH is joint conditional ML with zero pre-sample errors, the same definition as the package. A MAMean fit and the package's fit should agree on log-likelihood definition (not checked numerically). A cross-check of the package against `MAMean` for q>0 is a natural test.

### Gaps
- MAMean docs page on arch.readthedocs.io was not found as a fetched page (404 for the generated page), so docs coverage is unverified. Source code was read instead.
- Whether arch scores T or T - hold_back observations for MAMean by default was not checked.
- Parameter uncertainty in arch forecasts: absent from docs, absence not proven.

---

## 5. R rugarch (version 1.5-6, CRAN 2026-07-18)

### Takeaway
rugarch supports ARFIMAX mean models with MA terms and ARCH-in-mean, many GARCH variants, nine distributions, a two-method bootstrap based on Pascual-Romo-Ruiz, and a battery of diagnostics.

### Cited Findings
- Version and description: rugarch 1.5-6, published 2026-07-18, "univariate GARCH modeling" with ARFIMA, in-mean and external regressor models. Source: [CRAN page](https://cran.r-project.org/web/packages/rugarch/index.html), [reference manual PDF](https://cran.r-project.org/web/packages/rugarch/rugarch.pdf).
- `mean.model`: `armaOrder` (AR and MA orders), `include.mean`, `archm`, `archpow` (1 = standard deviation, 2 = variance in the mean), `arfima`, `external.regressors`, `archex` (multiply last `archex` regressors by the conditional standard deviation). Parameter names: mu, ar1, ma1, archm, mxreg1, arfima. Default `armaOrder = c(1,1)`. Source: reference manual, ugarchspec section. Relation to study: package-specific (rugarch is the reference joint ARMA+GARCH implementation).
- `variance.model`: model in sGARCH, fGARCH, eGARCH, gjrGARCH, apARCH, iGARCH, csGARCH (the manual also mentions mcsGARCH and realGARCH elsewhere); fGARCH submodels GARCH, TGARCH, AVGARCH, NGARCH, NAGARCH, APARCH, GJRGARCH, ALLGARCH; `garchOrder`, `external.regressors`, `variance.targeting`. Source: reference manual, ugarchspec.
- `distribution.model`: norm, snorm, std, sstd, ged, sged, nig, ghyp, jsu. Some are taken from fBasics and reimplemented locally; jsu is the gamlss reparametrization. Source: reference manual, ugarchspec and ugarchdistribution sections.
- Skewing method: the distribution help page lists Fernandez & Steel (1998), "On Bayesian modelling of fat tails and skewness", JASA 93(441), 359-371, DOI 10.1080/01621459.1998.10474117 as a reference. Wuertz (Rmetrics) is credited for norm, snorm, std, sstd, ged, sged, nig. Relation to study: the claim "rugarch sstd is Fernandez-Steel" is supported but not stated in one explicit sentence in the text I read. Sources: reference manual; [Fernandez-Steel record](https://research.tilburguniversity.edu/en/publications/on-bayesian-modeling-of-fat-tails-and-skewness/) (the Tilburg record lists Steel first; the DOI comes from Scilit and should be checked with the publisher).
- `ugarchboot`: resamples innovations from the fitted model. Method "full" follows Pascual, Romo & Ruiz (2006) and accounts for parameter uncertainty by simulating and refitting. Method "partial" considers only distribution uncertainty and gives no prediction intervals for the 1-step sigma. Innovation resampling schemes: "raw" (as in the original paper), "kernel" (Gaussian kernel from the ks package), "spd" (semi-parametric). References cited: Pascual, Romo & Ruiz (2004), "Bootstrap predictive inference for ARIMA processes", Journal of Time Series Analysis; and Pascual, Romo & Ruiz (2006), "Bootstrap prediction for returns and volatilities in GARCH models", Computational Statistics & Data Analysis 50(9), 2293-2312, DOI 10.1016/j.csda.2004.12.008. Relation to study: claim correct for "full"; "partial" is not a full PRR bootstrap. Sources: reference manual, ugarchboot section; [paper record](https://portal.mardi4nfdi.de/wiki/Publication:959315). UNVERIFIED: the 2004 paper's volume/pages were not checked.
- Tests: the fit-class methods documented in the manual are `infocriteria` (AIC, BIC etc), `nyblom` (Hansen-Nyblom joint and individual stability), `signbias` (Engle & Ng 1993), `gof` (adjusted goodness-of-fit with grouped bins, after Vlaar & Palm 1993) and a Berkowitz density forecast test. The manual's reference list includes Fisher & Gallagher (2012), "New weighted portmanteau statistics for time series goodness of fit testing", JASA 107(498), 777-787, DOI 10.1080/01621459.2012.688465. The printed headings "Weighted Ljung-Box Test on Standardized Residuals", "Weighted ARCH LM Tests", "Nyblom stability test", "Sign Bias Test" and "Adjusted Pearson Goodness-of-Fit Test" are seen in user-posted ugarchfit output, not in the manual text I read. Sources: reference manual; [Fisher-Gallagher record](https://econpapers.repec.org/RePEc:taf:jnlasa:v:107:y:2012:i:498:p:777-787); [example output](https://rstudio-pubs-static.s3.amazonaws.com/1223107_8f945cae51ce489d845713662446d404.html). Relation to study: the claim is largely correct. UNVERIFIED: that the weighted ARCH-LM is the Fisher-Gallagher statistic, the manual text I extracted does not say so explicitly (I did not locate the show method source; the raw source request returned 404).
- Likelihood definition: rugarch's ugarchfit/arfimafit likelihood is computed in C with feasible starting points; the manual does not state how pre-sample MA residuals or the first observations are treated. `likelihood()` extracts it, `infocriteria` reports AIC from it. UNVERIFIED: conditional or exact. Relation to study: whether rugarch's log-likelihood is on T or T - p observations, and with what pre-sample errors, is not documented in what I read. Keep any cross-engine comparison with rugarch explicitly labelled.

### Inferences
- rugarch is the only mainstream package I confirmed with joint ARMA(p,q) (including MA) + GARCH and nine conditional densities, so it is the natural external benchmark for the package.
- The "full" vs "partial" bootstrap split matters for the study if it compares forecast intervals: arch's bootstrap resembles rugarch "partial".

### Gaps
- Exact initialization of MA recursion and first-observation treatment in rugarch likelihood: not found in documentation; would need the C source.
- The printed-output claim rests on example output rather than package documentation.

---

## 6. Other Python packages with ARMA(p,q)+GARCH, and the package's niche

### Takeaway
No maintained mainstream Python package I confirmed offers joint ARMA(p,q) with MA terms plus GARCH with numpy/scipy only. A few small packages claim it but were not verified.

### Cited Findings
- statsforecast (Nixtla): has GARCH(p,q) and ARCH(p) models. Its docs describe a mean equation plus volatility equation and say that for autocorrelated series you should first fit an ARMA model and then model its residuals with GARCH, i.e. a two-step approach. The estimator is not named, and the page says errors are assumed normal. Sources: [GARCH model page](https://nixtlaverse.nixtla.io/statsforecast/docs/models/garch.html.md), [search results](https://nixtlaverse.nixtla.io/statsforecast/docs/models/arch.html). Relation to study: package-specific (niche remains).
- `armagarch` (PyPI): self-described as a library for flexible ARMA-GARCH mean and volatility models, with an ARMA mean given by AR and MA orders. Whether the estimation is joint is not stated on the page I could see. Source: [PyPI listing](https://pypi.org/project/armagarch/1.0.3/). UNVERIFIED: its PyPI page failed to load when fetched, version, maintenance and estimator unknown.
- `fynance` has an `ARMA_GARCH` function with a helper that splits ML parameters into AR, MA and GARCH groups. Source: [fynance source](https://fynance.readthedocs.io/en/master/_modules/fynance/models/econometric_models.html). UNVERIFIED: not read in detail.
- `generalized-timeseries` (ARIMA+GARCH): its PyPI page says it is archived. Source: [PyPI](https://pypi.org/project/generalized-timeseries/0.1.21/).
- A common tutorial claim: Python has no package that combines ARIMA and GARCH as rugarch does, and two-stage fits are inconsistent when MA terms are present (consistent but inefficient for pure AR). A Stack Overflow answer disputes part of this, saying the mean and variance parts are separable. Sources: [tutorial](https://medium.com/analytics-vidhya/arima-garch-forecasting-with-python-7a3f797de3ff), [Stack Overflow thread](https://kiwix.lema.org/stackoverflow.com_en_all_2023-11/questions/41477162/how-to-fit-a-arma-garch-model-in-python). Opinion sources, not primary.
- Feature requests: arch issue 147 asked for a univariate MA model (opened 2017, done in 9.0.0 as pure MA). See Q4.

### Inferences
- With arch 9.0.0 the "no MA in Python arch" part of the niche is gone. The remaining niche is joint ARMA(p,q) (AR and MA together) + GARCH in one engine, with a SARIMAX-style API, exact SARIMAX pass-through when garch=None, and numpy/scipy only.

### Gaps
- pmdarima, sktime, PyFlux: not checked in this pass (PyFlux's unmaintained status and pmdarima's lack of GARCH are from memory only, UNVERIFIED).
- I did not run a clean install of armagarch or fynance to test their estimators.
- statsforecast's estimation method and whether its GARCH includes ARMA terms: docs do not say.

---

## Corrections for the study text (summary)
1. Replace "Python arch has no MA terms in the mean" with: arch 9.0.0 (2026-11-11) adds MAMean (pure MA(q), conditional likelihood, pre-sample errors 0, any volatility and distribution); no ARMA(p,q) mean, ARX/HARX/LS remain AR-only.
2. Keep "statsmodels 0.14.6 has no GARCH model", and add that the old sandbox GARCH was deprecated in 2019 and moved to an archive directory.
3. "ugarchboot implements Pascual-Romo-Ruiz" applies to method "full"; "partial" is an innovation-only bootstrap.
4. For the printed diagnostics, cite the manual for Nyblom, sign bias and the adjusted goodness-of-fit, and mark the weighted Ljung-Box/ARCH-LM headings as seen in example output.
5. Add: R's own documentation states `arima`'s AIC is valid only for method "ML", and FPP3 states AIC is not comparable across data differenced differently. These are the closest documentary support for the E1 rule.
