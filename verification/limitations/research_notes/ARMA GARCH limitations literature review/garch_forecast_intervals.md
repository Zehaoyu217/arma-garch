# Forecast intervals for ARMA-GARCH models: literature notes

Verification method. Every bibliographic record below was checked against the Crossref API (and, where noted, OpenAlex or the publisher/IDEAS record). "Result" lines paraphrase abstracts or records that were actually read. Where the full text was not accessible (publisher pages returned 403), the result is limited to what the abstract supports and the rest is labelled UNVERIFIED or INFERENCE. Page ranges whose end page could not be confirmed are flagged.

## (a) Multi-step forecast-error variance of ARMA-GARCH and non-normality of multi-step predictive distributions

### Takeaway
Baillie & Bollerslev (1992) is the primary reference. Per its abstract it derives the MMSE predictor and its conditional MSE for an ARMA mean with GARCH innovations. It also gives moments of the prediction-error distribution for GARCH(1,1) and builds ex ante intervals with a Cornish-Fisher expansion. That is consistent with the study's finding that normal-innovation GARCH gives a fat-tailed multi-step predictive distribution and a normal band that drifts slightly. The full text was not readable, so exact formulas and the size of the Cornish-Fisher correction are UNVERIFIED.

### Cited Findings

- **Baillie, R.T. and Bollerslev, T. (1992). "Prediction in dynamic models with time-dependent conditional variances." *Journal of Econometrics* 52(1-2), 91-113. DOI: 10.1016/0304-4076(92)90066-Z** (Crossref and OpenAlex both confirm; [Crossref record](https://doi.org/10.1016/0304-4076(92)90066-Z)).
  - Result: Setting is a single-equation dynamic model with ARMA disturbances whose innovations follow a linear GARCH process. Per the abstract (as relayed by search snippets of the Duke repository record), the paper derives the minimum-MSE predictor and its conditional MSE, gives theoretical moments of the prediction-error distribution for GARCH(1,1) innovations, and uses the Cornish-Fisher asymptotic expansion to construct ex ante prediction confidence intervals. The empirical application is expected depreciation of foreign exchange rates. Sources: [Duke record](https://dukespace.lib.duke.edu/items/08261c94-189e-43ff-acd5-3d66e177dcf4), [EconPapers](https://econpapers.repec.org/RePEc:eee:econom:v:52:y:1992:i:1-2:p:91-113) (bibliographic data only). The ScienceDirect page returned 403.
  - Conditions and caveat: the abstract does not say the word "multi-step" and I could not read the formulas. The statement that the multi-step error is non-normal even with Gaussian innovations is INFERENCE from "moments of the prediction-error distribution" plus the use of Cornish-Fisher (a device for non-normal distributions). It is not a quoted claim. Marked UNVERIFIED at full-text level. Also note the Duke metadata lists the author names garbled; cite the journal record.
  - Relation to study: E2 known. The study's variance formula sum psi_j^2 E[sigma^2_{T+h-j}] and its excess kurtosis of 0.75 at h = 20 fit this framework. The Cornish-Fisher route is an analytic alternative to the package's simulated band, and the package does not offer it (package-specific).

- **Engle, R.F. and Bollerslev, T. (1986). "Modelling the persistence of conditional variances." *Econometric Reviews* 5(1), 1-50. DOI: 10.1080/07474938608800095** (Crossref confirmed).
  - Result: Not read. The title concerns persistence (IGARCH) of conditional variances. The full text was not accessed, so any statement about its GARCH variance-forecast formulas is UNVERIFIED.
  - Flag on the brief: "Engle & Bollerslev (1986) on GARCH variance forecasts" is a loose description. The paper is about persistence and IGARCH. GARCH itself is Bollerslev (1986), *J. Econometrics* 31(3), 307-327, which I recall as the source of the GARCH(1,1) variance-forecast recursion but did not verify in this session (UNVERIFIED).
  - Relation to study: E2 package-specific (the E[sigma^2] recursion the package uses).

### Inferences
- With normal shocks, a GARCH(1,1) conditional variance is stochastic, so the h-step error is a variance mixture of normals (sum of psi_j times shocks scaled by random sigma). That gives positive excess kurtosis, which matches the study's 0.75 at h = 20. This is a standard mixture argument, not a quote from a verified source here.
- Fat tails at long horizons for normal innovations explains why the normal band over-covers slightly at 80% (0.81) and under-covers at 99% only mildly in the opposite sense in the study (0.983 vs 0.99). Treat the direction as consistent with, but not proven by, Baillie & Bollerslev.

### Gaps
- Exact multi-step MSE formulas and the size of the Cornish-Fisher correction: full text not accessible.
- Whether the paper reports empirical coverage of the Cornish-Fisher intervals: unknown.

## (b) Bootstrap prediction intervals including parameter uncertainty

### Takeaway
Pascual, Romo & Ruiz (2004, ARIMA; 2006, GARCH) proposed bootstrapping that re-estimates parameters in each replicate, so estimation variability enters the interval, and for the ARIMA case without needing a backward representation. rugarch's `ugarchboot` documents method "full" as based on the 2006 paper, which accounts for parameter uncertainty, and method "partial" as distribution uncertainty only.

### Cited Findings

- **Pascual, L., Romo, J. and Ruiz, E. (2006). "Bootstrap prediction for returns and volatilities in GARCH models." *Computational Statistics & Data Analysis* 50(9), 2293-2312. DOI: 10.1016/j.csda.2004.12.008** (Crossref; online 2005 per OpenAlex; issue date May 2006 per [IDEAS](https://ideas.repec.org/a/eee/csdana/v50y2006i9p2293-2312.html)).
  - Result: Proposes a bootstrap resampling scheme giving prediction densities of both returns and volatilities of GARCH processes. Unlike earlier GARCH bootstraps, it incorporates parameter-estimation uncertainty, which is what makes bootstrap densities for the volatility process possible. It does not depend on a distributional assumption for the innovations. Asymptotic properties are derived. Finite-sample Monte Carlo shows "good behaviour" versus alternative procedures. It is applied to IBEX-35. Sources: working-paper abstract at [EconPapers](https://econpapers.repec.org/RePEc:cte:wsrepe:10059) (Universidad Carlos III de Madrid, 2000, same authors) and search-result fragments of the published abstract. The working paper describes the method as a generalization of Pascual et al. (2004).
  - UNVERIFIED: the specific steps the brief lists (resample standardized residuals, re-estimate, condition on last observations) match the general design of this literature and of the ARIMA predecessor, but I could not read the paper. Numerical coverage results (nominal vs empirical, sample sizes, horizons, comparison to normal plug-in intervals) were not obtainable (ScienceDirect and the working-paper PDF returned 403). I therefore cannot quote any coverage numbers.
  - Relation to study: README recommendation is supported in kind (the method does include parameter uncertainty). E2 extends: the study's own plug-in under-coverage at T = 250 (simulated 90% band 0.86 at h = 20 with phi = 0.9) is the gap this method targets. Whether the Pascual et al. bootstrap closes that gap in this design is untested by the study and unverified here.

- **Pascual, L., Romo, J. and Ruiz, E. (2004). "Bootstrap predictive inference for ARIMA processes." *Journal of Time Series Analysis* 25(4), 449-465. DOI: 10.1111/j.1467-9892.2004.01713.x** (Crossref and OpenAlex; abstract read via OpenAlex).
  - Result: A new bootstrap strategy for ARIMA prediction intervals. Its stated advantage over earlier bootstraps for autoregressive integrated processes is that estimation variability enters the intervals without a backward representation, so it extends to processes with MA components. Asymptotic properties of the bootstrap prediction densities are derived. Monte Carlo finite-sample experiments show it equals or outperforms alternatives in most cases.
  - Relation to study: E2 extends (the mean-equation analogue of the GARCH method; relevant to the package's ARMA mean).

- **rugarch `ugarchboot` documentation (R package, reference manual).** [rdrr.io man page](https://rdrr.io/cran/rugarch/man/ugarchboot-methods.html) (the CRAN PDF was not machine-readable).
  - Result (documentation text, paraphrased): method "full" accounts for parameter uncertainty by building a simulated distribution of parameters through simulation and refitting, and is more accurate but very time consuming. Method "partial" considers only distribution uncertainty, is faster, and will not give prediction intervals for the 1-ahead sigma forecast (that forecast depends only on parameter uncertainty). `n.bootfit` is the number of simulation-based refits used for the parameter distribution (not relevant to "partial"); `n.bootpred` is the number of bootstrap replications per parameter draw per horizon (for "partial", the number of draws from the empirical distribution). References listed: Pascual, Romo & Ruiz (2004, JTSA) and (2006, CSDA); the 2006 paper is the basis for the "full" method.
  - Answer to the brief: yes, `ugarchboot` is based on Pascual et al., with "Full" = parameter-uncertainty version and "Partial" = empirical-residual resampling without refitting.
  - Relation to study: README citation is consistent with the rugarch docs. Note that the package's own "simulated band" is not the Pascual method; it is parametric-shock simulation with plug-in parameters. The README should not imply the package implements the bootstrap (package-specific).

### Inferences
- rugarch "partial" (resample empirical standardized residuals, no refit) is operationally the same idea as filtered historical simulation in section (c) plus path simulation. That equivalence is my inference, not stated in the docs read.
- E3's result (resampling standardized residuals restores tail calibration under skewness) corresponds to the "partial" mechanism; E2's T = 250 under-coverage corresponds to what "full" is designed to address.

### Gaps
- Coverage numbers from Pascual et al. (2006) and their comparison against plug-in/normal intervals: paper not accessible.
- Exact algorithm steps for the GARCH case: not read.

## (c) Filtered historical simulation (FHS)

### Takeaway
Barone-Adesi, Giannopoulos & Vosper (1999) filter returns by GARCH, resample the standardized residuals, and simulate forward; Hull & White (1998) is a close precursor scaling historical returns by volatility ratios.

### Cited Findings

- **Barone-Adesi, G., Giannopoulos, K. and Vosper, L. (1999). "VaR without correlations for portfolios of derivative securities." *Journal of Futures Markets* 19(5), 583-602. DOI: 10.1002/(SICI)1096-9934(199908)19:5<583::AID-FUT5>3.0.CO;2-S** (Crossref and OpenAlex).
  - Result (abstract): "filtering historical simulation by GARCH processes" to model the future distribution of asset and swap values; options repriced by full revaluation; correlations handled implicitly; the method is a refinement of plain historical simulation, which gives equal weight to past returns and so ignores current market conditions. Conditions: portfolio VaR setting, not forecast intervals for an ARMA mean as such.
  - Relation to study: E3 known (the study's resampling of standardized residuals is the FHS idea applied to forecast bands). The abstract does not report tail-calibration evidence under skewness, so E3's 1.06% / 1.07% result is the study's own evidence.
  - Flag: an earlier FHS source (Barone-Adesi, Bourgoin & Giannopoulos 1998, Risk magazine, "Don't look back") exists in my memory but was not verified here: UNVERIFIED.

- **Hull, J. and White, A. (1998). "Incorporating volatility updating into the historical simulation method for value-at-risk." *Journal of Risk* 1(1), 5-19. DOI: 10.21314/jor.1998.001** (Crossref).
  - Result: Only the bibliographic record was verified; the abstract was not read. That the method scales historical returns by the ratio of current to historical volatility is general knowledge, UNVERIFIED in this session.
  - Relation to study: E3 known (background only).

### Inferences
- FHS makes the shock distribution empirical while the dynamics stay parametric, so it fixes tail shape but not parameter uncertainty; this matches E3 (restores calibration at T where parameters are well estimated).

### Gaps
- Published coverage tests of FHS for multi-step intervals: none found in this session.

## (d) Plug-in under-coverage: parameter uncertainty, small samples, persistence, horizons

### Takeaway
The AR bootstrap-interval literature documents that plug-in intervals are too narrow in small samples, worst for unit-root and near-unit-root processes. This is consistent with E2's phi = 0.9, T = 250 under-coverage. I found no quantitative result specific to ARMA-GARCH plug-in coverage.

### Cited Findings

- **Thombs, L.A. and Schucany, W.R. (1990). "Bootstrap prediction intervals for autoregression." *Journal of the American Statistical Association* 85(410), 486-492. DOI: 10.1080/01621459.1990.10476225** (Crossref; abstract read via OpenAlex).
  - Result: Nonparametric bootstrap for AR(p) of known order. Residuals are resampled, using a backward representation so that every replicate shares the same final observed values (conditioning on the sample's end). Simulations compare it with the standard Gaussian-based technique in low-order Gaussian and non-Gaussian AR models and "demonstrate the potential" of the bootstrap. The abstract gives no numerical coverage values.
  - Relation to study: E2 known (conditioning-on-last-observations design that Pascual et al. later avoided needing a backward form for).

- **Kim, J.H. (2001). "Bootstrap-after-bootstrap prediction intervals for autoregressive models." *Journal of Business & Economic Statistics* 19(1), 117-128. DOI: 10.1198/07350010152472670** (Crossref).
  - Result (abstract, from the T&F page via fetch, paraphrased): a Bonferroni prediction interval based on bootstrap-after-bootstrap for AR models; Monte Carlo over stationary, unit-root and near-unit-root cases; it is a better small-sample alternative to asymptotic and standard bootstrap intervals, which are often too narrow, especially with unit or near-unit roots; the new intervals are accurate and conservative in nearly all scenarios.
  - Flag: the brief cites "Kim 2001" without a title; this is the 2001 JBES paper. Kim (2004), *Int. J. Forecasting* 20(1), 85-97, DOI 10.1016/S0169-2070(03)00007-4 ("Bootstrap prediction intervals for autoregression using asymptotically mean-unbiased estimators") is a different paper (Crossref verified; abstract not read).
  - Relation to study: E2 extends. Directly supports the study's near-unit-root (phi = 0.9) under-coverage finding, though for AR not ARMA-GARCH.

- **Chatfield, C. (1993). "Calculating interval forecasts." *Journal of Business & Economic Statistics* 11(2), 121-135 (pages: Crossref/OpenAlex give start page 121 only; end page 135 from memory, UNVERIFIED). DOI: 10.1080/07350015.1993.10509938** (Crossref/OpenAlex; abstract read).
  - Result (abstract, paraphrased): reviews approaches (theoretical formulas from a fitted model with or without a parameter-uncertainty correction, "approximate" formulas that should be avoided, empirical/simulation/resampling procedures); gives general reasons why prediction intervals tend to be too narrow in practice; an example shows the importance of model specification, with the stationary-vs-nonstationary choice critical for nearly nonstationary data.
  - Relation to study: E2 known (qualitative support for plug-in narrowness).

- **Christoffersen, P. and Goncalves, S. (2005). "Estimation risk in financial risk management." *Journal of Risk* 7(3), 1-28 (volume/pages from memory, UNVERIFIED; Crossref lookup hit a rate limit).** Abstract seen via [risk.net](https://www.risk.net/journal-risk/2161126/estimation-risk-financial-risk-management) search snippet: builds confidence intervals around VaR and ES forecasts from dynamic variance models using a resampling technique that accounts for parameter estimation error; working paper (CIRANO 2004s-15) says estimation risk is substantial even in tightly parameterized models. A correction exists (Giamouridis 2006). Relation to study: E2 extends in spirit (estimation risk with GARCH variance), but concerns VaR/ES not forecast bands.

- **Clements & Hendry** (books/papers on forecast intervals and parameter uncertainty): UNVERIFIED. No specific reference was verified; do not cite without a further check.

### Inferences
- Magnitude evidence in this section is for AR models; the study's ARMA-GARCH under-coverage at T = 250, h = 20 (0.86 vs 0.90, 0.76 vs 0.80) is the study's own contribution, and I found no published ARMA-GARCH quantification to compare against.
- Long horizons matter because the forecast error for persistent mean processes is dominated by accumulated parameter error in phi^h; this is an inference, not a verified claim.

### Gaps
- Quantitative plug-in coverage for GARCH specifically (apart from the Pascual et al. paper I could not read).
- Clements & Hendry reference; Christoffersen & Goncalves volume/pages.

## (e) Evaluation tools for interval and density forecasts

### Takeaway
Kupiec tests unconditional coverage, Christoffersen adds independence and conditional coverage, Diebold-Gunther-Tay evaluates whole densities by PIT, and Gneiting & Raftery give proper scoring rules including CRPS. All bibliographic records verified.

### Cited Findings

- **Kupiec, P.H. (1995). "Techniques for verifying the accuracy of risk measurement models." *The Journal of Derivatives* 3(2), 73-84. DOI: 10.3905/jod.1995.407942** (Crossref). Result: proportion-of-failures (likelihood-ratio) test of whether the observed exceedance rate equals the nominal rate; abstract not read, description from general knowledge, UNVERIFIED. Relation to study: E2/E3 known (the study's coverage and 1%-tail breach rates are this kind of unconditional check).

- **Christoffersen, P.F. (1998). "Evaluating interval forecasts." *International Economic Review* 39(4), 841-862 (start page verified by Crossref; end page inferred as the article before DGT's 863, UNVERIFIED). DOI: 10.2307/2527341.** Result: likelihood-ratio tests of unconditional coverage, independence of the hit sequence, and their joint conditional coverage; abstract not read, UNVERIFIED in detail. Relation to study: E2 extends (independence across origins is not tested in the study's 400-origin design as described).

- **Diebold, F.X., Gunther, T.A. and Tay, A.S. (1998). "Evaluating density forecasts with applications to financial risk management." *International Economic Review* 39(4), 863-883 (start page verified; end page UNVERIFIED). DOI: 10.2307/2527342.** Result: the probability integral transform of the realization under the forecast density should be i.i.d. uniform if the density is correct; abstract not read, UNVERIFIED in detail. The NBER working paper version is DOI 10.3386/t0215 (1997). Relation to study: E3 extends (PIT histograms would show the skewness misplacement directly).

- **Gneiting, T. and Raftery, A.E. (2007). "Strictly proper scoring rules, prediction, and estimation." *Journal of the American Statistical Association* 102(477), 359-378. DOI: 10.1198/016214506000001437** (Crossref/OpenAlex; abstract read). Result: reviews and develops proper scoring rules; the continuous ranked probability score applies to predictive CDFs, generalizes absolute error, and is a special case of the energy score. Proper rules reward honest forecasts. Relation to study: extends (CRPS or interval score would let the study rank normal vs simulated bands beyond coverage).

- **Gneiting, T., Balabdaoui, F. and Raftery, A.E. (2007). "Probabilistic forecasts, calibration and sharpness." *J. Royal Statistical Society Series B* 69(2), 243-268. DOI: 10.1111/j.1467-9868.2007.00587.x** (Crossref). Optional: calibration-versus-sharpness paradigm; abstract not read.

### Inferences
- Coverage alone cannot separate a wide, calibrated band from a sharp, calibrated one; the study's use of true future paths allows exact calibration but not sharpness comparisons without a score.

### Gaps
- Abstracts for Kupiec, Christoffersen and DGT were not read this session; confirm the exact statistics before quoting.

## Normal band with t innovations: crossing near 95%

### Takeaway
I found no paper stating this. It follows from t(5) arithmetic.

### Cited Findings
- None found in the literature during this session.

### Inferences
- Computed from standard t(5) quantiles with unit-variance scaling by sqrt(3/5) = 0.7746: the standardized t(5) 90% two-sided quantile is 2.015 x 0.7746 = 1.56 (< 1.645, so the normal band over-covers), the 95% quantile is 2.571 x 0.7746 = 1.99 (> 1.96, so a slight under-cover, one-step coverage about 0.947), and the 99% quantile is 4.032 x 0.7746 = 3.12 (> 2.576, so under-cover). These match the study's 0.91, 0.946 and 0.978 (and 0.84 at 80%). The crossing is therefore a known property of the t(5) distribution against the normal, not a result needing a citation. These quantile values are my arithmetic from standard t tables, not from a fetched source.

### Gaps
- No reference found that states the crossover explicitly.

## Flags on the brief

- The README's "Pascual, Romo & Ruiz 2006, rugarch ugarchboot" is accurate: rugarch cites Pascual et al. (2004, 2006) and ties method "full" to the 2006 paper. "Partial" does not account for parameter uncertainty.
- Engle & Bollerslev (1986) concerns persistence/IGARCH, not variance forecasting per se; the GARCH definition is Bollerslev (1986).
- "Kim 2001" is the bootstrap-after-bootstrap paper in JBES.
- Pascual et al. (2006) was published online in 2005 (DOI contains 2004); the journal year is 2006.
- Not verified: Clements & Hendry reference; Christoffersen & Goncalves volume and pages; end pages of Chatfield, Christoffersen (1998) and DGT.
