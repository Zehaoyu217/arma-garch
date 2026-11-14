# Residual diagnostics after ARMA-GARCH estimation: which tests have the right null, and the corrected alternatives

Method note. Bibliographic details were checked against Crossref (API), RePEc/IDEAS/EconPapers, EconBiz, or the publisher DOI where the entry says "verified". Result summaries come from abstracts and catalogue records, not full texts (paywalled), unless stated. Items from the author's background knowledge, or not checked, are marked UNVERIFIED. Nothing here is a verbatim copy of the papers.

Tags: "Relation to study" uses E3 (skewness SD 0.23 vs sqrt(6/T) = 0.055), plus the study's own claims in the README and summary() output.

## 1. Portmanteau tests on standardized residuals: degrees of freedom and the null distribution

### Takeaway
The m - p - q degrees-of-freedom correction (Box-Pierce, Ljung-Box) is derived for ARMA residuals with iid innovations. Once the errors are conditionally heteroskedastic, neither the plain Q on z_t nor the Q on z_t^2 is chi2(m) or chi2(m-p-q). The limiting law is a weighted sum of chi2(1) variables with parameter-dependent weights (Li-Mak 1994; later Wong-Ling 2005 and Carbon-Francq). Corrected statistics exist (Li-Mak, Wong-Ling mixed test, Fisher-Gallagher weighted tests). Passing model_df = p+q to acorr_ljungbox is therefore a heuristic, not a fix, on z_t from an ARMA-GARCH fit.

### Cited Findings

- **Box & Pierce (1970)**. G. E. P. Box and D. A. Pierce, "Distribution of Residual Autocorrelations in Autoregressive-Integrated Moving Average Time Series Models," *Journal of the American Statistical Association* 65(332), 1509-1526. DOI 10.1080/01621459.1970.10481180 (verified via Crossref).
  Result: for a fitted ARMA(p,q) with iid innovations, the Q statistic on the first m residual autocorrelations is approximately chi2(m - p - q) for large m, because estimating p+q parameters reduces the dimension of the residual autocorrelation vector. Conditions: correct ARMA specification, iid errors, m large relative to p+q. The paper is the origin of the m - p - q correction.
  Relation to study: known. The README's model_df = p+q follows this paper. The conditions (iid errors) fail for GARCH errors.

- **Ljung & Box (1978)**. G. M. Ljung and G. E. P. Box, "On a measure of lack of fit in time series models," *Biometrika* 65(2), 297-303. DOI 10.1093/biomet/65.2.297 (verified via Crossref).
  Result: the finite-sample correction Q = T(T+2) sum r_k^2/(T-k) has a null distribution closer to chi2(m - p - q) than Box-Pierce. Same iid and correct-model conditions.
  Relation to study: known. This is the statistic the study computes. It omits p+q, so it is too conservative in the ARMA sense, since the true df is smaller. Even with df = p+q the null is not chi2 under GARCH.

- **McLeod & Li (1983)**. A. I. McLeod and W. K. Li, "Diagnostic Checking ARMA Time Series Models Using Squared-Residual Autocorrelations," *Journal of Time Series Analysis* 4(4), 269-273. DOI 10.1111/j.1467-9892.1983.tb00373.x (verified via Crossref).
  Result (from memory of the paper, not checked against full text; UNVERIFIED in detail): a portmanteau statistic on autocorrelations of squared ARMA residuals, treated as chi2(m), to detect nonlinearity/ARCH in the data. It is derived for ARMA residuals, not for squared standardized residuals of a fitted GARCH.
  Relation to study: extends. McLeod-Li is the plain Q on squared residuals. After GARCH fitting its null differs (see Li-Mak). The chi2(m) form is only valid for squared ARMA residuals before GARCH estimation.

- **Li & Mak (1994)**. W. K. Li and T. K. Mak, "On the squared residual autocorrelations in non-linear time series with conditional heteroskedasticity," *Journal of Time Series Analysis* 15(6), 627-636. DOI 10.1111/j.1467-9892.1994.tb00217.x (verified via IDEAS and Crossref-style record).
  Result (abstract): defines a general class of squared standardized residual autocorrelations for conditional heteroskedastic models and derives their asymptotic distribution, which is used to build diagnostic tests. Secondary sources (see the Carbon-Francq lineage below) report that the Box-Pierce-type statistic on the squared residuals does not follow a chi-square law after estimation, and that Li-Mak gives the corrected asymptotic covariance of the autocorrelations (it accounts for the estimation of the conditional-variance parameters). The formula for the corrected covariance was not retrieved; UNVERIFIED in detail.
  Relation to study: extends. It is the right test for "no remaining ARCH after GARCH(1,1)". The README cites it correctly (year, journal). The package does not implement it.

- **Wong & Ling (2005)**. H. Wong and S. Ling, "Mixed Portmanteau Tests for Time-Series Models," *Journal of Time Series Analysis* 26(4), 569-579. DOI 10.1111/j.1467-9892.2005.00420.x (verified via EconPapers/IDEAS records).
  Result (abstract-level): derives the joint limiting distribution of residual and squared-residual autocorrelations for a general time-series model and builds a mixed portmanteau statistic that checks first and second moments together. Later literature reports higher power than Ljung-Box and McLeod-Li when the first and second moments are both misspecified. Under conditions of a correct model the joint limit is non-chi2 in general; the simplification to a sum of known portmanteau statistics holds only in some cases.
  Relation to study: extends. A candidate alternative to Q(z) plus Q(z^2) printed separately, which have a joint dependence.

- **Carbon & Francq (2011)**. M. Carbon and C. Francq, "Portmanteau goodness-of-fit test for asymmetric power GARCH models," *Austrian Journal of Statistics* 40(1&2), 55-64 (volume/pages from memory; UNVERIFIED). Found as MPRA paper 27686 (https://mpra.ub.uni-muenchen.de/27686/).
  Result (abstract-level): derives the asymptotic distribution of a vector of squared-residual autocorrelations for a wide class of asymmetric power GARCH models and deduces portmanteau tests, i.e. the limit is a quadratic form in a Gaussian vector with a non-identity covariance. The df formula of the resulting test was not retrieved.
  Relation to study: extends. Evidence that the null is a GARCH-specific weighted chi2 and not chi2(m - p - q).

- **Francq & Zakoian, GARCH Models (book)**. C. Francq and J.-M. Zakoian, *GARCH Models: Structure, Statistical Inference and Financial Applications*, Wiley (2nd ed. 2019; 1st ed. 2010). Chapter on diagnostics/portmanteau tests (chapter number and theorem not retrieved; UNVERIFIED). ISBN not checked.
  Result: not retrieved here. Based on the Carbon-Francq lineage above, the book's portmanteau result for squared residuals is the same Li-Mak-type quadratic form. Treat chapter-level claims as UNVERIFIED.
  Relation to study: not verified; use Carbon-Francq (above) or Li-Mak as the primary citation.

- **Francq, Roy & Zakoian (2005)** (UNVERIFIED, from memory): C. Francq, R. Roy, J.-M. Zakoian, "Diagnostic checking in ARMA models with uncorrelated errors," *Journal of the American Statistical Association* 100(470), 532-544. I recall the result: when the ARMA innovations are uncorrelated but not independent (e.g. GARCH), the standard Box-Pierce/Ljung-Box test on residuals has the wrong null distribution. Not retrieved in this session; verify before citing.
  Relation to study: this would be the direct source for "Q on z_t with m - p - q df is mis-sized under GARCH errors", for the mean-equation part. If confirmed, it extends the study's design question.

- **Li (2004) book**. W. K. Li, *Diagnostic Checks in Time Series*, Chapman & Hall/CRC Monographs on Statistics and Applied Probability, vol. 102. Print ISBN 978-1-58488-337-1; e-book ISBN 978-0-203-48560-6. Catalogue records disagree on the year (2003 vs 2004); British Library lists copyright 2004 (verified via library catalogue records).
  Result: book-length treatment of residual autocorrelation portmanteau tests, including squared-residual tests for heteroskedastic models. Chapter-level content was not retrieved; UNVERIFIED.
  Relation to study: not verified; general reference.

- **Fisher & Gallagher (2012)**. T. J. Fisher and C. M. Gallagher, "New Weighted Portmanteau Statistics for Time Series Goodness of Fit Testing," *Journal of the American Statistical Association* 107(498), 777-787. DOI 10.1080/01621459.2012.688465 (verified via EconPapers; some catalogues give 777-788, the publisher/RePEc range is 777-787).
  Result (abstract-level): weighted sums of squared sample autocorrelations (weights decreasing in lag), approximately gamma-distributed, with stable behaviour even for large lag counts, and with extensions to test for nonlinearity and fitted nonlinear-model adequacy (squared residuals; the "weighted Li-Mak" variant). Higher simulated power than Ljung-Box and Li-Mak in the long-memory nonlinear settings they study. The exact df/centering for GARCH squared residuals was not retrieved; UNVERIFIED.
  Relation to study: known. The README is correct on the citation. The package does not implement it.

- **rugarch uses Fisher-Gallagher (confirmed in source)**. The rugarch test code (R/rugarch-tests.R, viewed through rdrr.io) states it directly imports the weighted tests from the WeightedPortTest package (Fisher & Gallagher). The weighted Ljung-Box helper takes a fitted-df argument; lags are run at 1, max(2, 2*df + df - 1) and max(5, 4*df + df - 1) with fitdf = df, and it errors when lag < 2*fitdf + fitdf - 1. The weighted ARCH-LM helper is labelled "Weighted Li-Mak test on autocorrelations (Gamma Approximation)" with fitdf defaulting to 2 (the GARCH(1,1) p+q), requiring lags > fitdf. R-SIG-Finance posts report the same lag formula. Which fitdf ugarchfit passes for the standardized-residual test (ARMA order for z, GARCH order for z^2) was NOT seen in the code excerpt; UNVERIFIED.
  Sources: https://rdrr.io/cran/rugarch/src/R/rugarch-tests.R ; https://stat.ethz.ch/pipermail/r-sig-finance/2018q4/014677.html
  Relation to study: known. README claim "rugarch's default" is supported for both weighted tests. Note the README says "weighted Ljung-Box on z^2": in rugarch the squared-residual test is called the weighted Ljung-Box on standardized squared residuals, and the ARCH LM is the weighted Li-Mak, a different object.

### Inferences
- The README advice to pass model_df = p+q to acorr_ljungbox is correct as the ARMA-iid convention but is not a valid correction for z_t from an ARMA-GARCH fit. It should be described as an approximation. On z_t^2, df = p+q of the mean model is wrong in any case; the relevant count is GARCH parameters, and even that is heuristic (rugarch uses it inside a weighted statistic).
- "Li-Mak (1994) test, or the weighted Ljung-Box on z^2": Li-Mak is a corrected test of the squared autocorrelations; the Fisher-Gallagher weighted tests are what rugarch actually reports, and its ARCH LM version is the weighted Li-Mak.

### Gaps
- Exact Li-Mak asymptotic covariance and df formula; Francq-Zakoian book theorem; Francq-Roy-Zakoian 2005 details; the Fisher-Gallagher df rule for GARCH squared residuals; rugarch's actual fitdf argument inside ugarchfit. Full texts are paywalled; only abstracts/catalogue records and code were read.

## 2. Remaining ARCH after estimation and misspecification tests for GARCH

### Takeaway
The plain Engle ARCH-LM with chi2(lags) on z_t from an estimated GARCH is not asymptotically exact; Lundbergh-Terasvirta (2002) give score-based tests of no remaining ARCH, parameter constancy and asymmetry whose nulls account for estimation, with robust versions for non-normal errors.

### Cited Findings
- **Engle (1982)**. R. F. Engle, "Autoregressive Conditional Heteroscedasticity with Estimates of the Variance of United Kingdom Inflation," *Econometrica* 50(4), 987-1007. DOI 10.2307/1912773 (DOI, volume, issue, first page verified via Crossref; end page 1007 from memory, UNVERIFIED).
  Result: LM test for ARCH: T*R^2 from regressing squared residuals on their lags, asymptotically chi2(q) under iid, homoskedastic null, with residuals from a model not itself containing ARCH terms.
  Relation to study: known. Using it on z_t after GARCH estimation reuses a statistic whose null assumed no ARCH was estimated, and chi2(lags) ignores that GARCH parameters were estimated. The correct reference law is different (Li-Mak; Lundbergh-Terasvirta).

- **Lundbergh & Terasvirta (2002)**. S. Lundbergh and T. Terasvirta, "Evaluating GARCH models," *Journal of Econometrics* 110(2), 417-435. DOI 10.1016/S0304-4076(02)00096-9 (verified via EconPapers; working-paper origin SSE/EFI No. 292, 1998, revised 2001).
  Result (abstract-level): a unified framework of Lagrange-multiplier-type tests of an estimated GARCH model: no ARCH in standardized errors, parameter constancy, and linearity/asymmetry; null distributions are standard after allowing for estimation. Robust versions for non-normal errors outperform the non-robust ones in simulation when errors are non-normal. A later critique (not identified here) argues that estimation effects from the conditional mean can be asymptotically non-negligible for their nonlinearity test; UNVERIFIED which paper.
  Relation to study: extends. This is the framework for "no remaining ARCH, asymmetry, constancy" that the package lacks.

- **Engle & Ng (1993)**. R. F. Engle and V. K. Ng, "Measuring and Testing the Impact of News on Volatility," *Journal of Finance* 48(5), 1749-1778. DOI not retrieved (UNVERIFIED; check Wiley/JSTOR). Pages 1749-1778 per the R documentation and Scilit; EconPapers shows 1749-78.
  Result: sign bias, negative size bias and positive size bias tests, based on a regression of squared standardized residuals on a sign dummy and signed-size terms, plus a joint test. They detect asymmetric response to shocks not captured by a symmetric GARCH.
  Relation to study: extends. rugarch prints these; the README names the sign bias test. Note rugarch's implementation uses a joint chi-square from a linear-hypothesis test with no adjustment for estimated parameters (per the code), so the nulls are approximate.

- **Nyblom (1989)**. J. Nyblom, "Testing for the Constancy of Parameters over Time," *Journal of the American Statistical Association* 84(405), 223-230 (journal, volume, pages per secondary sources; DOI NOT verified. The DOI 10.1080/01621459.1989.10478752 offered by one search summary is a recollection and should be treated as UNVERIFIED).
  Result: LM-type test locally most powerful against parameters following a martingale; asymptotic critical values are non-standard and obtained by simulation, depending on the number of parameters. rugarch's code attributes its parameter-constancy implementation to Hansen's 1990 mimeo (joint and individual statistics, critical value tables up to 20 parameters), which is a Hansen (1990)-style extension of Nyblom's test; citation details of the Hansen paper are UNVERIFIED.
  Relation to study: extends. Optional diagnostic for parameter stability.

### Inferences
- Mapping to the package: ARCH-LM on z_t with chi2(lags) is a convenience statistic. The README should call it approximate, point to Li-Mak/weighted version, and mention that the p-values can be mis-sized in either direction depending on the model (direction not established here).

### Gaps
- No source retrieved giving the size distortion of ARCH-LM on GARCH standardized residuals (direction/magnitude). Not checked.

## 3. Skewness and symmetry under fat tails and dependence

### Takeaway
The textbook skewness test with variance 6/T needs iid normality. For fat-tailed or dependent data the sampling variance depends on higher moments and long-run covariances (Bai-Ng 2005). The skewness SD finding in E3 is consistent with this theory: the sample skewness variance requires a finite sixth moment, which a t(6) does not have.

### Cited Findings
- **Bai & Ng (2005)**. J. Bai and S. Ng, "Tests for Skewness, Kurtosis, and Normality for Time Series Data," *Journal of Business & Economic Statistics* 23(1), 49-60. DOI 10.1198/073500104000000271 (verified via IDEAS/EconPapers; issue number from memory, UNVERIFIED; working paper Boston College WP 501, 2001).
  Result (abstract-level): gives sampling distributions of skewness and kurtosis and a joint normality test for time series, using long-run (HAC) covariance of the sample moments instead of the iid/normal variances 6/T and 24/T. Simulations: symmetry and normality tests have good size and power; kurtosis tests have severe size distortion except for thin-tailed distributions. A Jarque-Bera-type test remains usable if its variance estimate accounts for serial correlation. The moment conditions required (I recall finite moments of order higher than 6 for skewness; UNVERIFIED) were not retrieved.
  Relation to study: extends E3. The paper's message (variance depends on higher moments and dependence, the kurtosis test is almost uninformative under fat tails) supports E3's conclusion that skewness of standardized residuals is noisy; a HAC/robust skewness test is the corrected alternative.

- **Premaratne (and Bera) (2005)**. G. Premaratne [and A. K. Bera], "A Test for Symmetry with Leptokurtic Financial Data," *Journal of Financial Econometrics* 3(2), 169-187. DOI 10.1093/jjfinec/nbi009 (verified via Crossref and IDEAS).
  CITATION FLAG: Crossref and IDEAS list a single author (Premaratne) for the published article, while EconBiz and later papers list Premaratne and Bera. Bera's co-authorship is certain for the 2001 working papers ("A Test for Asymmetry with Leptokurtic Financial Data", SSRN 278228; "Modeling Asymmetry and Excess Kurtosis in Stock Return Data", SSRN 259009), and the journal version's author list could not be settled. Journal and year are right. Cite as Premaratne (2005) (or "Premaratne and Bera" only after checking the OUP page).
  Result (abstract-level): most symmetry tests are built under an implicit or explicit normal null; the paper proposes a symmetry test based on the Pearson type IV family and a bounded function of the data, which reduces sensitivity to outliers. Monte Carlo evidence: good finite-sample performance; illustration on individual stock returns. The explicit size numbers of the sqrt(6/T) test were not retrieved; UNVERIFIED.
  Relation to study: extends. The fitted density in the package is Student-t or skewed-t (a different family); this test is not a drop-in for z_t.

### Inferences
- E3 (SD 0.23 vs 0.055 at T = 2000): the asymptotic variance of the sample skewness involves E[z^6], which is infinite for a t(nu) with nu <= 6, so the skewness converges slower than root-T and the textbook SD is far off. This is mathematically consistent with the cited literature but I did not find a paper that states the t(6) value specifically.
- If a skewness check is kept, a HAC-variance (Bai-Ng) or a bounded-function (Premaratne) version is the supported alternative.

### Gaps
- No paper found quantifying the size of the sqrt(6/T) test under t(6) and GARCH-dependence.

## 4. Normality: Jarque-Bera

### Takeaway
JB is an LM test for normality of iid or regression-error data; it has no valid null when the error is dependent, and by construction it rejects Student-t innovations. Printing it with a t-fit is uninformative.

### Cited Findings
- **Jarque & Bera (1980)**. C. M. Jarque and A. K. Bera, "Efficient tests for normality, homoscedasticity and serial independence of regression residuals," *Economics Letters* 6(3), 255-259. DOI 10.1016/0165-1765(80)90024-5 (verified via Scilit/Wikipedia list; not Crossref).
  Result: LM tests for normality, homoscedasticity and serial independence of regression residuals; chi2 limits under the null of iid normal errors.
  Relation to study: known. Contradicts using JB as a diagnostic of a t(nu) model: the null is normality.

- **Jarque & Bera (1987)**. C. M. Jarque and A. K. Bera, "A Test for Normality of Observations and Regression Residuals," *International Statistical Review* 55(2), 163-172. DOI 10.2307/1403192 (verified via Wikipedia and a working paper; one source lists volume 50, treated as an error).
  Result: the JB statistic T[S^2/6 + (K-3)^2/24] ~ chi2(2) under iid normality (test for regression residuals using OLS residuals). Not robust to serial correlation or heteroskedasticity.
  Relation to study: extends E-findings. A cleaner fix is a Bai-Ng (HAC) JB variant; for a t-model, test the fitted density instead.

### Gaps
- No source retrieved on JB behaviour specifically under GARCH standardized residuals (the effect of estimated parameters on the null).

## 5. Tests of the fitted density

### Takeaway
PIT-based tests (DGT 1998, Berkowitz 2001) test the whole fitted density but assume known parameters; with estimated parameters the nulls need adjustment. rugarch's adjusted Pearson test uses chi2(g-1), with no subtraction for estimated parameters per the code viewed.

### Cited Findings
- **Diebold, Gunther & Tay (1998)**. F. X. Diebold, T. A. Gunther and A. S. Tay, "Evaluating Density Forecasts with Applications to Financial Risk Management," *International Economic Review* 39(4), 863-883. DOI 10.2307/2527342 (verified via secondary records).
  Result: if the density forecast is correct, the probability integral transform series is iid U(0,1); the test checks uniformity (histograms/KS) and independence (autocorrelation of PIT powers). Applies to out-of-sample-style forecasts with known (given) densities.
  Relation to study: extends. Supports the PIT/KS recommendation. Caveat (UNVERIFIED in detail): in-sample PITs from estimated parameters make standard KS critical values inexact; the paper itself is not an estimation-effect paper.

- **Berkowitz (2001)**. J. Berkowitz, "Testing Density Forecasts, With Applications to Risk Management," *Journal of Business & Economic Statistics* 19(4), 465-474. DOI 10.1198/07350010152596718 (verified via Scilit and library record).
  Result: transforms PITs to a standard normal via the inverse normal CDF and applies a likelihood-ratio test of zero mean, unit variance and no first-order autocorrelation; reported good power at sample sizes as small as 100 (EconPapers abstract).
  Relation to study: extends. Candidate for the fitted-density check; same estimation-effect caveat.

- **Palm & Vlaar (1997) / Vlaar & Palm (1993)**. F. C. Palm and P. J. G. Vlaar, "Simple Diagnostic Procedures for Modeling Financial Time Series," *Allgemeines Statistisches Archiv* 81(1), 85-101 (1997) (verified via EconBiz catalogue). The 1993 paper (P. J. G. Vlaar and F. C. Palm, "The message in weekly exchange rates in the European Monetary System: mean reversion, conditional heteroscedasticity, and jumps," *Journal of Business & Economic Statistics* 11(3), 351-360) is cited from memory and is UNVERIFIED; it was not searched.
  Result: not retrieved (no abstract). The adjusted Pearson goodness-of-fit test is the Chi-square test on binned standardized residuals against the fitted density, adjusted for estimated parameters; the standard Chernoff-Lehmann result puts the null between chi2(g-1-k) and chi2(g-1), which I recall as the rationale; UNVERIFIED.
  rugarch: per the code viewed, the p-value uses chi2(g-1) with expected counts from the fitted distribution (including skew and shape), labelled p-value(g-1), and has no subtraction of the estimated parameters in the code excerpt, so it is the conservative-direction upper bound if the Chernoff-Lehmann bounds apply (UNVERIFIED).
  Relation to study: package-specific. If the package adds a Pearson test, it should state df = g-1 as an upper bound, and note that the exact null lies between g-1-k and g-1.

### Gaps
- The Palm-Vlaar text itself; DGT/Berkowitz treatment of estimated parameters (not retrieved; other papers, e.g. Bai 2003 on estimated parameters in PIT tests, exist but were not checked).

## Cross-check of the study's README and printed diagnostics

| Study claim | Status |
|---|---|
| Ljung-Box on z_t without a p+q adjustment | Wrong with respect to ARMA convention (Box-Pierce 1970; Ljung-Box 1978). With GARCH errors even the adjusted chi2 is approximate (Li-Mak 1994 and successors). Francq-Roy-Zakoian 2005 is the probable direct source (UNVERIFIED). |
| ARCH-LM on z_t uses chi2(lags) | Approximate only; corrected tests: Li-Mak, Lundbergh-Terasvirta 2002, weighted Li-Mak (rugarch). |
| JB printed with t innovations | Rejects by design (JB 1980, 1987 are normality tests with iid null). |
| "Li-Mak (1994) test" | Citation correct (JTSA 15(6):627-636). |
| "Fisher & Gallagher 2012, rugarch's default" | Citation correct (JASA 107(498):777-787); rugarch imports the weighted tests (confirmed in source). Slight wording issue: rugarch's ARCH-LM version is the weighted Li-Mak. |
| Engle-Ng sign bias | Citation correct (JF 48(5):1749-1778). |
| "PIT/KS against the fitted density" | DGT 1998; Berkowitz 2001 are the sources; estimation-effect caveat. |
| Premaratne & Bera (2005) (task statement) | Journal/year/pages right (JFEC 3(2):169-187); author list conflicts across sources (see Section 3). |
