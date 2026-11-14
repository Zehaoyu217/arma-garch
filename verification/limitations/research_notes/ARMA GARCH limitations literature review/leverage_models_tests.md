# Asymmetric-volatility (leverage) GARCH models, tests, magnitudes, and consequences of ignoring them

Status key. VERIFIED = bibliographic record confirmed this session against Crossref and/or RePEc/IDEAS. CONTENT-VERIFIED = result confirmed from an abstract or landing page. UNVERIFIED = from memory or secondary snippet only; check before publishing. I could not read the full text of any paywalled paper, and the NBER/author PDFs I fetched were scanned images that could not be machine-read. Result paraphrases are therefore limited to abstract-level claims unless stated.

## 1. Model definitions and exact citations

### Takeaway
All four models let negative shocks move variance more than positive shocks. GJR and TGARCH add a threshold term. EGARCH works on log variance. APARCH nests several of them through a power parameter. The citations below are verified. Model equations are standard textbook forms, not quotes from the papers.

### Cited Findings
- **GJR.** Glosten, L. R., Jagannathan, R., Runkle, D. E. (1993). "On the Relation between the Expected Value and the Volatility of the Nominal Excess Return on Stocks." *The Journal of Finance* 48(5), 1779-1801. DOI 10.1111/j.1540-6261.1993.tb05128.x (VERIFIED, Crossref). Open working-paper version: Federal Reserve Bank of Minneapolis Staff Report 157 (1993), DOI 10.21034/sr.157. The paper uses a GARCH-M model modified in three ways: seasonal volatility patterns, different impact of positive and negative innovations on conditional volatility, and nominal interest rates as predictors of conditional variance. Positive unanticipated returns lead to a downward revision of conditional volatility and negative ones to an upward revision. With these changes the authors report a weak but significant negative relation between conditional expected monthly return and conditional variance. They attribute the earlier null or positive findings to misspecification of the standard GARCH-M. Little evidence of highly persistent volatility under the modified model. CONTENT-VERIFIED via the Minneapolis Fed and publisher abstracts; coefficient values not read. Model form: sigma_t^2 = omega + (alpha + gamma*1[eps_{t-1}<0]) eps_{t-1}^2 + beta sigma_{t-1}^2. **Relation to study: E4 known (the same specification the study uses as true DGP). The paper is also direct precedent that the variance specification changes the mean-equation (risk-return) estimates, in a GARCH-M setting that the study does not test.**
  - Citation note: Engle and Ng (1993) and many later papers cite this work as "Glosten, Jagannathan and Runkle (1989)" (working-paper year). The published year is 1993. The search tool echoing "(1989)" in the Engle-Ng abstract is the same work.
- **EGARCH.** Nelson, D. B. (1991). "Conditional Heteroskedasticity in Asset Returns: A New Approach." *Econometrica* 59(2), 347-370. DOI 10.2307/2938260 (VERIFIED, Crossref gives volume 59, issue 2, start page 347; end page 370 is from memory, UNVERIFIED). Models log sigma_t^2 as a function of lagged standardized innovations, with a sign term and a magnitude term, so no positivity constraints are needed and leverage is built in. Relation to study: package-specific (the package has no EGARCH). Not tested in E4.
- **TGARCH / threshold GARCH.** Zakoian, J.-M. (1994). "Threshold heteroskedastic models." *Journal of Economic Dynamics and Control* 18(5), 931-955. DOI 10.1016/0165-1889(94)90039-6 (VERIFIED, Crossref). Models the conditional standard deviation (not variance) as linear in the positive and negative parts of lagged shocks. Note the naming confusion: "TGARCH" in some software (e.g. GJR-type in EViews) is the variance-form threshold model, which is the GJR model; Zakoian's original is in standard deviations. Relation to study: package-specific.
- **APARCH.** Ding, Z., Granger, C. W. J., Engle, R. F. (1993). "A long memory property of stock market returns and a new model." *Journal of Empirical Finance* 1(1), 83-106. DOI 10.1016/0927-5398(93)90006-D (VERIFIED, Crossref). Model: sigma_t^delta = omega + alpha(|eps_{t-1}| - gamma eps_{t-1})^delta + beta sigma_{t-1}^delta, with a power delta that is estimated. It nests GARCH, GJR and TGARCH as special cases. Note Crossref lists the authors in the order Ding, Granger, Engle for the journal paper; a book-chapter reprint (Essays in Econometrics, Cambridge UP, DOI 10.1017/CBO9780511753978.020, pp. 349-372) lists Ding, Engle, Granger. Use the journal order. Relation to study: package-specific.
- **Software definitions.** Python `arch` implements GJR through the `o` argument (o=1) and EGARCH, APARCH-type power options; R `rugarch` has `gjrGARCH`, `eGARCH`, `apARCH`. I did not re-open the documentation in this session (UNVERIFIED this session; the study context already cites them).

### Inferences
- GJR is the simplest of the four to add to a package that already has GARCH(1,1), since it adds a single indicator term and keeps the variance recursion linear in parameters.

### Gaps
- Equation forms and pages for Nelson 1991 end page not read from the primary source.

## 2. Engle-Ng tests and the news impact curve

### Takeaway
Engle and Ng (1993) define the news impact curve and three residual-based diagnostics (sign bias, negative size bias, positive size bias) plus their joint test. On Japanese stock returns they found the GJR model the best parametric model.

### Cited Findings
- **Engle, R. F., Ng, V. K. (1993).** "Measuring and Testing the Impact of News on Volatility." *The Journal of Finance* 48(5), 1749-1778. DOI 10.1111/j.1540-6261.1993.tb05127.x (VERIFIED, Crossref; RePEc handle bla:jfinan:v:48:y:1993:i:5:p:1749-78). Earlier version: NBER Working Paper 3681 (1991), DOI 10.3386/w3681. Abstract-level content: defines the news impact curve, which measures how new information is incorporated into volatility estimates; compares ARCH-type models, including a partially nonparametric one, on daily Japanese stock returns; presents diagnostics that focus on asymmetric volatility response. Conclusion: GJR is the best parametric model; EGARCH captures most of the asymmetry but its implied conditional variance appears too variable. CONTENT-VERIFIED via IDEAS abstract.
- **Test construction (secondary, partly UNVERIFIED).** I could not read the paper's equations. Software documentation and the secondary sources I saw describe the tests as OLS regressions of the squared standardized residual from the fitted model on S-_{t-1} (indicator for a negative lagged shock; sign bias test), S-_{t-1}*eps_{t-1} (negative size bias test, where eps may be the raw or standardized shock) and S+_{t-1}*eps_{t-1} (positive size bias test, S+ = 1 - S-). The t-ratio on each coefficient is the single-test statistic. The joint test regresses on all three together and tests that all slopes are zero, giving a chi-square with 3 degrees of freedom (LM-style in Engle-Ng; the R `tstests::signbias_test` package labels its implementation as a Wald chi-square on the three coefficients). Exact choice of regressor scaling and the residual used should be checked against the paper before the package implements them. Sources: [R tstests signbias_test docs](https://search.r-project.org/CRAN/refmans/tstests/html/signbias_test.html) (confirms input is model residuals plus conditional standard deviations; does not give the regression), [EViews help on signbias](https://help.eviews.com/content/equationcmd-signbias.html) (not opened).
- **News impact curve.** For a variance model, the NIC plots sigma_t^2 against eps_{t-1} holding sigma_{t-1}^2 at its unconditional value. Symmetric GARCH gives a parabola centered at zero; GJR gives a parabola with a steeper left branch (extra slope gamma for eps<0); EGARCH is exponential, steeper on the left and with no cap. These forms are textbook; the Engle-Ng definition could not be read directly.
- **Japanese data.** The sample is daily Japanese stock-index returns (TOPIX; the exact sample period of the paper is from memory, 1980-1988, UNVERIFIED). Which of the three bias tests rejected for GARCH, and the GJR estimates, were not retrievable this session.

### Inferences
- The sign-bias family is a diagnostic on a fitted symmetric model, so it fits the package's summary() as a post-estimation check. Under the study's E4 design the LR test (88-100% rejection) is a more powerful check against the specific GJR alternative.

### Gaps
- Exact regression equations, residual scaling, per-test rejection pattern for GARCH, GJR parameter values on the Japanese data (scanned PDFs unreadable; JSTOR paywall).

## 3. Typical leverage magnitudes in equities

### Takeaway
The leverage effect is well documented for equity indices and individual stocks, and the best models in a 330-model comparison for IBM were leverage models. Specific alpha/gamma/beta values quoted in the task (alpha near 0, gamma 0.1-0.2, beta about 0.9) could not be confirmed from a fetched source this session.

### Cited Findings
- **Black, F. (1976).** "Studies of stock price volatility changes." *Proceedings of the 1976 Meetings of the American Statistical Association, Business and Economic Statistics Section*, 177-181. UNVERIFIED (not found in Crossref; citation from memory). Original leverage-effect argument: falling stock prices raise financial leverage and hence equity volatility.
- **Christie, A. A. (1982).** "The stochastic behavior of common stock variances: Value, leverage and interest rate effects." *Journal of Financial Economics* 10(4), 407-432. DOI 10.1016/0304-405X(82)90018-6 (VERIFIED, Crossref). Content of result (variance elasticity with respect to equity value) from memory, UNVERIFIED.
- **Bekaert, G., Wu, G. (2000).** "Asymmetric Volatility and Risk in Equity Markets." *The Review of Financial Studies* 13(1), 1-42. DOI 10.1093/rfs/13.1.1 (VERIFIED, Crossref). Studies Japanese market and portfolio data. Result from memory: finds volatility feedback dominates the leverage effect for asymmetry at the market level; not confirmed here, UNVERIFIED. Relation to study: extends (provides a risk-return/feedback rationale for why mean and asymmetric variance estimates interact).
- **Hansen, P. R., Lunde, A. (2005).** "A forecast comparison of volatility models: does anything beat a GARCH(1,1)?" *Journal of Applied Econometrics* 20(7), 873-889. DOI 10.1002/jae.800 (VERIFIED, Crossref). Compares 330 ARCH-type models. For DM-USD exchange rates no evidence that GARCH(1,1) is beaten; for IBM returns GARCH(1,1) is clearly inferior to models that accommodate a leverage effect. One summary identifies APARCH(2,2) as best overall (secondary source, not confirmed in the paper). CONTENT-VERIFIED via abstract and search summaries. **Relation to study: extends (supports using GJR for equity, and the near-symmetric exchange-rate result supports the symmetric default for FX).**
- **Engle, R. F., Patton, A. J. (2001).** "What good is a volatility model?" *Quantitative Finance* 1(2), 237-245. DOI 10.1088/1469-7688/1/2/305 (VERIFIED, Crossref). Content (asymmetric/leverage models fit DJIA better) from memory, UNVERIFIED.
- **French, K. R., Schwert, G. W., Stambaugh, R. F. (1987).** "Expected stock returns and volatility." *Journal of Financial Economics* 19(1), 3-29. DOI 10.1016/0304-405X(87)90026-2 (VERIFIED, Crossref). Content UNVERIFIED; included as the GARCH-M/risk-return precedent.
- **Campbell, J. Y., Hentschel, L. (1991).** "No news is good news: an asymmetric model of changing volatility in stock returns." NBER Working Paper 3742, DOI 10.3386/w3742 (VERIFIED as working paper; the Journal of Financial Economics version, 31(3), 281-318 (1992), is from memory, UNVERIFIED). Volatility-feedback model.

### Inferences
- Magnitudes: no fetched source gives numeric GJR estimates, so the study's "equity-like" parameter choices (alpha 0, lev 0.14, persistence 0.97) cannot be cited to a specific paper from this session. A follow-up pass should extract GJR parameter tables from Hansen-Lunde, Engle-Ng or published index studies.
- Persistence caveat: in the GJR variance the persistence is alpha + lev/2 + beta (under symmetric innovations), so the study's "persistence 0.97" convention should be stated in terms of that sum.

### Gaps
- Numeric alpha/gamma/beta from any single study; Black 1976 page range; Bekaert-Wu results.

## 4. Asymmetry in macro, rates, credit and FX

### Takeaway
There is a literature that applies TGARCH/EGARCH/GJR to inflation uncertainty and finds a significant leverage-type term in many countries, but I verified none of these papers against a primary source. Exchange rates are comparatively near-symmetric in the Hansen-Lunde comparison.

### Cited Findings
- Grier and Perry (1998), on G7 inflation and inflation uncertainty (positive effect of inflation on inflation uncertainty in all G7 countries), as reported by a secondary search summary; full citation UNVERIFIED. Bivariate AGARCH extensions: Grier et al. (2004), Shields et al. (2005), UNVERIFIED.
- Several applied papers on Asian economies and Pakistan report a significant asymmetry (leverage) parameter in inflation volatility, via secondary summaries only. UNVERIFIED, low quality, do not cite as evidence of effect size.
- Exchange rates: Hansen and Lunde (2005, above) found GARCH(1,1) not beaten for DM-USD.

### Inferences
- For macro series, sign conventions matter: a "negative shock" may mean unexpectedly low inflation or a credit-spread widening, so the GJR indicator has a different economic meaning than for equities. This would matter for the package's default assumption.

### Gaps
- Interest rates (e.g. level-dependent volatility), credit spreads, and primary-source inflation-uncertainty papers were not located and verified.

## 5. Consequences of ignoring asymmetry for mean-equation estimates

### Takeaway
I found no paper that simulates exactly the study's E4 (symmetric GARCH fit to GJR data, intercept bias in an ARMA-GARCH joint MLE). The closest results are GJR (1993) on risk-return sign changes, and theory showing QMLE consistency of the mean can fail in ARMA-GARCH because mean and variance parameters are linked. The E4 result is best described as consistent with that theory but not a direct replication of a published result.

### Cited Findings
- **Sin, C.-Y. (2003).** "Estimating a linear exponential density when the weighting matrix and mean parameter vector are functionally related." *Advances in Econometrics* 17 (Maximum Likelihood Estimation of Misspecified Models: Twenty Years Later), DOI 10.1016/S0731-9053(03)17009-0 (DOI and title confirmed via IDEAS/EconPapers search listing; author first name and pages not confirmed, UNVERIFIED). Abstract-level result: the usual QMLE-in-the-linear-exponential-family consistency of the conditional-mean parameters despite misspecified other moments fails when the weighting matrix and mean parameters are functionally related; ARMA-GARCH is the prominent example. **Relation to study: extends (supplies a theoretical reason a misspecified variance can bias the mean, supporting that the E4 intercept bias is an expected rather than a package-specific effect).** The study's finding that AR and regression slopes are unaffected but the intercept is biased is not predicted or explained by this abstract.
- **Newey, W. K., Steigerwald, D. G. (1997).** "Asymptotic bias for quasi-maximum-likelihood estimators in conditional heteroskedasticity models." *Econometrica* 65(3), 587-599. Entirely UNVERIFIED (from memory; not found in this session's results beyond being cited by others). Reported idea: with misspecified conditional variance, QML estimators of the mean are consistent only under symmetry-type conditions on the innovation density; asymmetric (non-symmetric) innovations produce asymptotic bias for location parameters. Check before using: E4 uses symmetric normal innovations, so the bias in E4 comes from the variance misspecification interacting with the mean, not from innovation asymmetry.
- **Iglesias, E. M., Phillips, G. D. A. (2011).** Small sample estimation bias in GARCH models with any number of exogenous variables in the mean equation (journal name UNVERIFIED in this session; search snippet only). Reports non-trivial finite-sample biases in some circumstances. Relation to study: extends (finite-sample bias in a related setting), but concerns correctly specified GARCH.
- **Conrad, C., Mammen, E. (2015).** Cited secondarily as stating that no general asymptotic theory exists for QMLE of GARCH-M; full citation UNVERIFIED.
- **GJR (1993), above:** variance specification (asymmetry, seasonal, interest-rate predictor) flips the GARCH-M risk-return sign from positive or null to negative. Supports the broader point that mean-equation conclusions can depend on the variance specification. Relation to study: extends (risk-return, not intercept).
- **Brooks, C., Burke, S. P., Persand, G. (2001).** "Benchmarks and the accuracy of GARCH model estimation." *International Journal of Forecasting* 17(1), 45-56. DOI 10.1016/S0169-2070(00)00070-4 (VERIFIED, Crossref). Software-comparison paper; content not read. Relevant to package-vs-reference validation, not to asymmetry.
- Consistency theory for correctly specified GARCH: Lumsdaine (1996), *Econometrica* 64(3), 575-596, DOI 10.2307/2171862 (VERIFIED start page via Crossref; end page from memory); Francq and Zakoian (2004), "Maximum likelihood estimation of pure GARCH and ARMA-GARCH processes," *Bernoulli* 10(4), 605-637, DOI 10.3150/bj/1093265632 (VERIFIED). These give consistency/normality of Gaussian QMLE when the model is correct; they do not cover the E4 misspecified-variance case. Relation to study: package-specific (background for E1-E3 style well-specified runs).

### Inferences
- Plausible mechanism (my reasoning, not a published result): the symmetric GARCH variance path is a poor weight for squared-residual information after large negative shocks, so the likelihood reweights observations differently from the true variance, shifting the intercept estimate. Not tested here.
- The "long-run mean +27% too high" and "coverage 0.745" are simulation results of the study; nothing found contradicts or reproduces them.

### Gaps
- No simulation paper directly measuring mean-parameter bias from fitting symmetric GARCH to GJR data was found. Newey-Steigerwald (1997) and the Sin (2003) text were not opened.
- Volatility-feedback literature (Campbell-Hentschel, Bekaert-Wu) relation to mean bias beyond the above was not verified.

## 6. Forecasting implications

### Takeaway
Asymmetry helps volatility forecasts for equities in the comparisons I could confirm, and not for exchange rates.

### Cited Findings
- Hansen and Lunde (2005): see Section 3. IBM: leverage models beat GARCH(1,1); DM-USD: GARCH(1,1) not beaten. Their reality-check test had low power to separate good and bad models (summary-level, CONTENT-VERIFIED).
- **Awartani, B. M. A., Corradi, V. (2005).** "Predicting the volatility of the S&P-500 stock index via GARCH models: the role of asymmetries." *International Journal of Forecasting* 21(1), 167-183. DOI 10.1016/j.ijforecast.2004.08.003 (VERIFIED, Crossref). Result from memory: asymmetric GARCH models beat GARCH(1,1) for one-step-ahead forecasts of S&P-500 volatility in the data used, with the evidence weaker at longer horizons; not confirmed from the paper, UNVERIFIED. Relation to study: extends (forecast, not mean estimation).
- Engle and Ng (1993) see Section 2: GJR preferred among parametric models for Japanese returns.

### Inferences
- No source found on interval-forecast coverage under a misspecified symmetric variance. The study's 95% CI coverage drop concerns parameter confidence intervals, not prediction intervals.

### Gaps
- Interval-forecast or density-forecast consequences of ignoring leverage were not found.

## Mis-citation and consistency checks on the study context

- The study text names "Engle-Ng sign bias test" correctly; the correct reference is *J. Finance* 48(5), 1749-1778 (1993), verified.
- GJR is correctly "1993, J. Finance", but note that the working-paper date 1989 appears in some downstream citations (including the Engle-Ng abstract text) and should not be used.
- Zakoian's TGARCH works on standard deviations; the study's equation is the GJR variance form, so calling it "TGARCH" would be inexact.
- The persistence of 0.97 for GJR depends on convention (alpha + lev/2 + beta under symmetric innovations); state it.
- The quoted leverage figures (gamma 0.1-0.2, beta about 0.9) are plausible but are not supported by a citation found in this session.
