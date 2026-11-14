# Testing for GARCH effects in regression / ARMA residuals: tests, selection rules, size and power

Verification method: bibliographic details were checked against Crossref (DOI records) and/or RePEc/EconPapers. Result statements come from abstracts, from full text where freely available (Demos & Sentana 1998 working-paper revision, Ma-Nelson-Startz 2007, Hughes-King-Kwek working paper), or are flagged as secondary. Items not checked against full text are marked UNVERIFIED.

Mis-citation flags for the study context:
- Lee (1991) is Economics Letters 37(3), 265-271. The nearby pages 305-308 belong to a different paper by K. Y. Lee ("Are the GARCH models best in out-of-sample performance?").
- "Hughes, King & Teng (2004)" has the third author's surname as Kwek (Kwek Kian Teng): Hughes, King & Kwek (2004).
- Lee & King (1993) is in J. Business & Economic Statistics 11(1), 17-27, and has a published Correction in 12(1), 139 (1994).
- Engle (1982) page range is 987-1007 in JSTOR/most indexes; the Econometric Society page gives 987-1008. Crossref lists only the start page, 987.

---

## (a) Lagrange-multiplier tests for ARCH/GARCH

### Takeaway
The ARCH-LM test is n times the R-squared (T*R^2) from regressing squared residuals on a constant and lagged squared residuals, and it is asymptotically chi-square with q degrees of freedom under homoskedasticity. Lee (1991) showed the LM test against GARCH is numerically the same statistic as the LM test against ARCH (for GARCH(1,1), the ARCH(1) test), so the ARCH-LM(4) run in the study is also the LM test for GARCH. Being a test against ARCH-type dependence, it can be run without estimating the GARCH model.

### Cited Findings
- **Engle (1982).** R. F. Engle, "Autoregressive conditional heteroscedasticity with estimates of the variance of United Kingdom inflation," *Econometrica* 50(4), 987-1007, July 1982. DOI 10.2307/1912773 (https://www.jstor.org/stable/1912773). Introduces the ARCH class (non-constant conditional variance, constant unconditional variance), ML estimation, and the LM test for ARCH effects, applied to UK inflation (abstract-level, EconBiz/ScienceOpen/Econometric Society records). The T*R^2 form of the statistic and its chi2(q) null distribution are standard (the T*R^2 equivalence is also stated in Demos & Sentana 1998, below); I did not read the 1982 full text. Relation to study: **known** (ARCH-LM(4) is this test).
- **Lee (1991).** J. H. H. Lee, "A Lagrange multiplier test for GARCH models," *Economics Letters* 37(3), 265-271, Nov 1991. DOI 10.1016/0165-1765(91)90221-6 (Crossref-verified). The abstract (EconPapers) says the LM test for white-noise disturbances against GARCH disturbances in the linear regression model is the same as the LM test against ARCH. Demos & Sentana (1998) state that Bollerslev (1986) had noted the usual LM derivation fails because the relevant information-matrix block is singular under the null, and that Lee showed the GARCH(1,1) LM test is numerically identical to the ARCH(1) LM test. The reason given is that the score for beta is identically zero under the null, so only the alpha score matters. The generalisation "GARCH(p,q) LM = ARCH(q) LM" is the form in the abstract and in the literature; I verified the (1,1) case via Demos & Sentana, not the general-order proof in Lee's full text (UNVERIFIED for general p,q). Relation to study: **known**. The study's ARCH-LM(4) is therefore a legitimate GARCH test without fitting GARCH, and it is a different test from the study's LR.
- **Lee & King (1993).** J. H. H. Lee and M. L. King, "A locally most mean powerful based score test for ARCH and GARCH regression disturbances," *J. Business & Economic Statistics* 11(1), 17-27, Jan 1993. DOI 10.1080/07350015.1993.10509930 (Crossref-verified). Correction: JBES 12(1), 139 (1994), DOI 10.2307/1391932. Builds a one-sided score test (sum of the scores) that exploits the non-negativity of the parameters, which the LM test ignores. A Monte Carlo under normal and leptokurtic errors found it typically more powerful than two versions of the LM test, and possibly with more accurate asymptotic critical values (abstract, EconPapers). Demos & Sentana report that Lee-King's GARCH(1,1) test coincides with their own one-sided ARCH(1) test, and that power falls with t5 innovations relative to normal. Relation to study: **extends** (one-sided alternative with higher power than the two-sided ARCH-LM the study uses; the study does not run it).
- **McLeod & Li (1983).** A. I. McLeod and W. K. Li, "Diagnostic checking ARMA time series models using squared-residual autocorrelations," *J. Time Series Analysis* 4(4), 269-273, 1983. DOI 10.1111/j.1467-9892.1983.tb00373.x (Crossref-verified). Shows normalised squared-residual autocorrelations are asymptotically unit multivariate normal after ARMA fitting and gives a portmanteau test (chi-square approximation); a simulation checked small-sample validity (abstract/secondary). No direct power comparison with ARCH-LM was obtained from the sources I could read (UNVERIFIED: any head-to-head power result). Relation to study: **known** as the ARMA-residual analogue of the study's ARCH-LM; no claim about its power is made.

### Inferences
- Because the LM test for GARCH(1,1) reduces to the ARCH(1) test, its power depends on the first autocorrelation of squared errors. Demos & Sentana give rho1 = alpha*(1 - beta^2 - alpha*beta)/(1 - beta^2 - 2*alpha*beta) for GARCH(1,1). For (alpha, beta) = (.10, .75) this is about 0.19 and for (.08, .90) about 0.34, so the persistent design has higher low-order autocorrelation of squared errors, matching the study's higher ARCH-LM power in that design at T >= 250. (Arithmetic done here from the cited formula.)
- Using q = 4 lags spends power relative to a well-targeted statistic; this is my inference, not a sourced finding.

### Gaps
- Lee (1991) full text (proof for general p,q) not accessed. McLeod-Li vs ARCH-LM power comparison not located.

---

## (b) LR and Wald tests with a boundary null and unidentified beta

### Takeaway
Under H0 (alpha = 0), alpha is on the boundary and beta is not identified, so the LR/Wald statistics are not chi-square. With only alpha restricted (beta fixed), the LR is a 50:50 mixture of chi2(0) and chi2(1), with 5% critical value 2.706. With beta free under the alternative, the supremum over beta needs the nonstandard theory of Andrews (2001) and Hansen (1996), and I could not find a published tabulated critical value for the GARCH(1,1) case. The LM test is unaffected.

### Cited Findings
- **Demos & Sentana (1998).** A. Demos and E. Sentana, "Testing for GARCH effects: a one-sided approach," *J. Econometrics* 86(1), 97-127, 1998. DOI 10.1016/S0304-4076(97)00110-3 (Crossref-verified). Read in the CEMFI working-paper revision (June 1997). Findings: (i) for ARCH(q), H0: alpha_i = 0 puts parameters on the boundary, so LR and Wald are a mixture of q+1 chi-squares with weights binomial(q, i)/2^q on chi2(i), i = 0..q (their eq. 4; Self & Liang 1987, Wolak 1989); the critical values are tabulated for q = 1..12, e.g. 2.706 at 5% for q = 1 versus the chi2(1) value 3.84. (ii) LM tests are not affected by the boundary. (iii) A one-sided LM (Kuhn-Tucker multiplier) test is more powerful than the two-sided LM test in their Monte Carlo. (iv) For GARCH(1,1) H0 is alpha = 0 with beta only identified under the alternative; they note Bollerslev's singular-information-block problem and Lee (1991)'s equivalence; their one-sided GARCH(1,1) LM equals the ARCH(1) one-sided LM, asymptotically a 50:50 chi2(0)/chi2(1) mixture whatever the fixed beta used. (v) They say Davies/Andrews-Ploberger/Hansen-type sup procedures for the unidentified beta are not valid in their one-sided setting because those assume a null value not on the boundary, and the distribution of sup-LR in a one-sided context had not been investigated. Monte Carlo (T up to 5000, reference T = 250, 15,000 replications): at nominal 5% the one-sided LM has reasonable size with normal errors, size distortions are larger with t5 innovations (largest for the two-sided LM and for the GARCH(1,1) design), the one-sided test's power exceeds the two-sided test's, and power falls under t5 innovations. Relation to study: **extends** the study's LR finding (the study's chi2(2)=5.99 cut-off is not the null law; the correct reference law is a boundary/sup law) and **known** for ARCH-LM size.
- **Andrews (2001).** D. W. K. Andrews, "Testing when a parameter is on the boundary of the maintained hypothesis," *Econometrica* 69(3), 683-734, May 2001. DOI 10.1111/1468-0262.00210 (Crossref-verified). Derives asymptotic null and local-alternative distributions of quasi-LR, rescaled QLR, Wald and score statistics when parameters are on the boundary and/or a nuisance parameter is present only under the alternative. One of two worked examples is testing for no conditional heteroskedasticity in a GARCH(1,1) regression (abstract at EconPapers/Econometric Society). I could not retrieve the full text, so the specific limit law and any numerical critical values for the GARCH example are UNVERIFIED. Relation to study: **extends** (the formal reason the chi2(2) LR is not exact).
- **Hansen (1996).** B. E. Hansen, "Inference when a nuisance parameter is not identified under the null hypothesis," *Econometrica* 64(2), 413-430, Mar 1996. DOI 10.2307/2171789 (Crossref-verified). Test statistics are functionals of chi-square processes with data-dependent covariance; he proposes a conditional-measure (simulation) approach for p-values, illustrated with threshold autoregressions (abstract, EconPapers/Econometric Society). Demos & Sentana point out its conditions exclude the one-sided boundary case. Relation to study: **extends** (a simulation route to valid p-values for sup-type GARCH LR if the boundary were handled; no sourced GARCH application).
- **Andrews & Ploberger (1994).** D. W. K. Andrews and W. Ploberger, "Optimal tests when a nuisance parameter is present only under the alternative," *Econometrica* 62(6), 1383-1414, Nov 1994. DOI 10.2307/2951753 (Crossref-verified). Weighted-average-power optimal tests (the exp-type and average statistics) for a nuisance parameter present only under the alternative; in these nonstandard cases the LR test is not optimal (abstract). Regular boundary conditions are not covered, as noted by Demos & Sentana. Relation to study: **known** background; nothing GARCH-specific verified.

### Inferences
- The study's LR rejects 3-4% at the 5.99 cut-off. This is consistent with chi2(2) being too large a cut-off. The chi2(2) cut-off overstates the 5% value of a boundary mixture (e.g. 2.706 for one restriction; Demos & Sentana tabulate values) but a sup over beta enlarges the null law, so the true 5% value lies between those; the magnitude is not known from sources. The observed 3-4% (conservative) pattern is therefore plausible, but I did not find a source quantifying the GARCH(1,1) sup-LR critical value. Flag as inference.
- Rejections of 3-4% mean the LR is slightly conservative, not liberal; this agrees in direction with the boundary-theory expectation.

### Gaps
- No tabulated GARCH(1,1) sup-LR critical values found; Andrews (2001) full text inaccessible here.

---

## (c) Small-sample size and power of ARCH tests

### Takeaway
Primary Monte Carlo evidence I could read directly (Demos & Sentana) shows ARCH/GARCH LM tests with reasonable 5% size at T = 250 and Gaussian errors, with larger distortions under fat tails, and power increasing in T and in the alpha parameter, against GARCH(1,1) essentially an ARCH(1) test. A secondary source reports that earlier studies found ARCH-LM over-rejecting in finite samples; the study's 3-7% rejection is consistent with mild size distortion either way.

### Cited Findings
- **Engle, Hendry & Trumble (1985).** R. F. Engle, D. F. Hendry and D. Trumble, "Small-sample properties of ARCH estimators and tests," *Canadian J. Economics* 18(1), 66-93, Feb 1985. DOI 10.2307/135114 (Crossref-verified). Abstract not retrievable. Secondary: another paper cites it as suggesting the square root of the LM statistic for testing ARCH(1) (one-sided, signed form), and groups it with Luukkonen-Saikkonen-Terasvirta (1988), Bollerslev-Wooldridge (1988/1992), Diebold-Pauly (1989), Gregory (1989) as studies whose "typical finding" is LM over-rejection in finite samples (search-result snippet; original page not read, so the quoted direction is UNVERIFIED). Relation to study: **known** (study's ARCH-LM(4) 3-7% rejection at nominal 5% is within that qualitative picture).
- **Demos & Sentana (1998)** (see b). Size and power Monte Carlo summarised above: power increases with T and alpha, rarely changes with beta except large beta, falls for t5 errors relative to normal, and for GARCH(1,1) the LM-type tests are essentially ARCH(1) tests. Their power tables are in the paper; I did not transcribe numbers. Relation to study: **known** (power rises with T). The study's higher ARCH-LM power in the persistent design is not directly established by this paper, which finds little beta-dependence except for large beta.
- **Lee & King (1993)** reports a Monte Carlo (normal and leptokurtic errors) where the one-sided score test generally has higher power than two LM versions. Sample sizes not retrieved (UNVERIFIED). Relation to study: **extends**.
- **Ma, Nelson & Startz (2007)** (see e) report LM and LR tests of beta in GARCH(1,1) at T = 500/1000/5000 (1000 replications); the LM test has size 4.3-6.3% across designs, the LR test is 5.6-13.0% (above nominal when alpha is small). These test beta given alpha>0, not the alpha = 0 null. Relation to study: **contradicts the assumption** that LR is conservative generally: LR for beta is oversized when alpha is small; the study's LR tests the different null alpha = 0.

### Inferences
- The study's power at T=120 (0.20) and T=250 (0.40-0.52) for ARCH-LM(4) is low for the monthly sample sizes; the literature I could read does not give comparable numbers for those exact (alpha, beta, T), so the study's numbers are **package-specific**.

### Gaps
- Engle-Hendry-Trumble and McLeod-Li (1983) actual results not seen; no comparative power numbers at T = 100-500 retrieved.

---

## (d) Information criteria for GARCH vs constant variance

### Takeaway
The standard AIC/BIC derivations assume regular parameter spaces; Brooks & Burke find unmodified criteria unreliable for GARCH order and Hughes-King-Kwek derive a boundary-adjusted AIC. Hughes-King-Kwek's simulations show AIC more liberal and BIC more conservative than the boundary-adjusted AIC. Compared likelihoods must be computed on the same observations.

### Cited Findings
- **Akaike (1974).** H. Akaike, "A new look at the statistical model identification," *IEEE Trans. Automatic Control* 19(6), 716-723, Dec 1974. DOI 10.1109/TAC.1974.1100705 (Crossref-verified). AIC = -2 log L + 2k. Relation to study: **known** background.
- **Schwarz (1978).** G. Schwarz, "Estimating the dimension of a model," *Annals of Statistics* 6(2), 461-464, 1978. DOI 10.1214/aos/1176344136 (Crossref-verified). BIC penalty k log n. Relation to study: **known**.
- **Hughes, King & Kwek (2004).** A. W. Hughes, M. L. King and K. K. Teng (Kwek Kian Teng), "Selecting the order of an ARCH model," *Economics Letters* 83(2), 269-275, May 2004. DOI 10.1016/j.econlet.2003.05.003 (Crossref-verified; Crossref lists the third author as Kwek). Read in the Adelaide working-paper version (WP 99-1). Derives a one-sided AIC (OSAIC) that uses the non-negativity constraint: the penalty is a probability-weighted sum, e.g. 0.5, 1.0, 1.5, 2.0 for ARCH(1)...ARCH(4) versus 1, 2, 3, 4 for AIC. Monte Carlo (2000 replications; T = 60 and 200, mean 0.2, constant 0.1, ARCH(1)-ARCH(3) truths; inequality-constrained ML) found OSAIC selecting the correct model most often at T = 60, AIC marginally better than OSAIC for ARCH(1) at T = 200, and BIC worse than both in correct-selection rate for every DGP. They excluded the constant-variance and ARCH(4) truths for that comparison because BIC (smallest-model truth) or OSAIC (largest-model truth) would then trivially win. In their tables at T = 60 with ARCH(1) truth (alpha1 = 0.2), AIC chose constant variance 0.48 of the time and BIC 0.67 (underfitting risk). Relation to study: **extends** (BIC conservative / AIC more liberal; gives the boundary-adjusted AIC alternative that the study does not use).
- **Brooks & Burke (1998).** C. Brooks and S. P. Burke, "Forecasting exchange rate volatility using conditional variance models selected by information criteria," *Economics Letters* 61(3), 273-278, Dec 1998. DOI 10.1016/S0165-1765(98)00178-5 (Crossref-verified). Per Reading repository/EconBiz summaries: modified information criteria rarely chose GARCH(1,1) (under 20% of the time), and out-of-sample the selected models gave lower MAE but higher MSE than GARCH(1,1) (secondary summary; abstract-level). Relation to study: **known** (information criteria and GARCH order selection are unstable).
- **Brooks & Burke (2002).** "Selecting from amongst non-nested conditional variance models: information criteria and portfolio determination," *The Manchester School* 70(6), 747-767. DOI 10.1111/1467-9957.00323 (Crossref-verified). Per summary: in simulations the criteria are largely ineffective at identifying the correct model (secondary; UNVERIFIED in detail).
- **Brooks & Burke (2003).** "Information criteria for GARCH model selection," *The European Journal of Finance* 9(6), 557-580. DOI 10.1080/1351847021000029188 (Crossref-verified). Abstract: unmodified or naively modified traditional criteria cannot be used to determine the order of conditionally heteroscedastic models; the derived modified criteria sometimes improved variance forecasts, not consistently. Relation to study: **known** background; their modification concerns lag-order selection, not GARCH vs constant variance, so the study's 7-12% (AIC) and 0-1% (BIC) false-GARCH rates are **package-specific** and are not tested by that paper.
- **Same data / same likelihood requirement.** I did not locate a single GARCH-specific citation stating it. Hughes et al. note all candidate models are estimated by the same inequality-constrained ML and compared with the same likelihood basis; this is the standard condition of AIC/BIC (all models explain the same data). Relation to study: the study's finding that AIC comparing a statsmodels exact-likelihood fit with the engine's conditional likelihood picks GARCH ~50% for AR(1) means (conditional likelihood drops p observations, so the likelihoods are on different samples and not comparable) is **package-specific** and a violation of this standard condition, not a literature result.

### Inferences
- GARCH(1,1) adds two parameters (alpha, beta) over constant variance, so AIC prefers GARCH when 2*LR exceeds 4; the null distribution of 2*LR is a boundary law not chi2(2) (Demos-Sentana), so the false-GARCH rate under AIC (7-12%) reflects the criterion's rule, not a test size. Arithmetic/inference only; not a sourced finding.
- Hughes et al. show BIC underfits relative to AIC when ARCH is present, matching the study's lower BIC power.

### Gaps
- No source on AIC/BIC for GARCH vs constant variance with a GARCH(1,1) alternative at T = 120-1000; no source on comparing conditional with exact likelihood.

---

## (e) Weak identification of GARCH

### Takeaway
When alpha is near zero, beta is weakly identified: standard errors for beta are too small, beta-hat is biased upward, and t-tests are badly oversized, while LM tests behave and LR tests are only moderately oversized. The same logic implies the quasi-Wald/t for beta should not be trusted when the ARCH-LM or LR test fails to reject.

### Cited Findings
- **Ma, Nelson & Startz (2007).** J. Ma, C. R. Nelson and R. Startz, "Spurious inference in the GARCH(1,1) model when it is weakly identified," *Studies in Nonlinear Dynamics & Econometrics* 11(1), Article 1, 1-27, 2007. DOI 10.2202/1558-3708.1434 (Crossref-verified; full text read). The Zero-Information-Limit Condition holds in GARCH(1,1): standard errors of beta-hat shrink too much when alpha is small even with very large samples, and combined with upward bias this gives spurious high persistence. Monte Carlo (1000 replications; omega = 1; alpha = 0.01, 0.05, 0.10; beta = 0 or 0.5; T = 500, 1000, 5000): the t-test for beta has 5% size 48.7%, 45.6%, 44.5% at alpha = 0.01 (T = 500/1000/5000) and 30.6%, 21.0%, 8.5% at alpha = 0.10; at alpha = 0.01, beta = 0, T = 1000 the median beta-hat is 0.32 and median omega-hat 0.69; the LR test for beta is 13.0%, 10.9%, 8.3% (alpha = 0.01) and 8.9%, 8.4%, 5.6% (alpha = 0.10); the LM test stays 4.3-6.3%. For (alpha+beta) the t-test size at T = 1000 and alpha = 0.01 is 45.6% (beta = 0) and 42.3% (beta = 0.5), with upward-biased alpha+beta. Their suggested strategy is a pure ARCH(q) approximation. Relation to study: **extends/contradicts**: reported beta estimates and beta standard errors are unreliable at small alpha, a limitation the study should report; the study's alpha = 0.08-0.10 sits in the moderately weak region of their tables; their LR/LM tests concern beta given alpha > 0, not the study's alpha = 0 null.
- **Andrews & Cheng (2012).** D. W. K. Andrews and X. Cheng, "Estimation and inference with weak, semi-strong, and strong identification," *Econometrica* 80(5), 2153-2211, Sept 2012. DOI 10.3982/ECTA9456 (Crossref-verified). Studies estimators, tests and confidence sets for parameters that are unidentified or weakly identified in parts of the parameter space (ML, LS, GMM, etc.), proposes identification-robust confidence sets, and illustrates with ARMA(1,1) by ML and nonlinear regression (abstract, Econometric Society). A GARCH application is not stated in the abstract; the general lesson for GARCH (standard Wald confidence sets under-cover for weakly identified nuisance parameters) is an inference. Relation to study: **extends** by analogy; GARCH-specific results UNVERIFIED.

### Inferences
- The study's reported beta estimates in designs with weak GARCH should be flagged as potentially biased upward with understated standard errors, per Ma-Nelson-Startz; its alpha = 0 null (beta unidentified) is the extreme of this case.

### Gaps
- Andrews & Cheng's treatment of GARCH, if any, not located.

---

## Summary of study-finding relations
| Study finding | Relation | Source |
|---|---|---|
| ARCH-LM(4) rejects 3-7% under no GARCH | known | Engle 1982; Demos-Sentana 1998; secondary on EHT 1985 |
| ARCH-LM is also the GARCH LM test | known | Lee 1991 (verified via Demos-Sentana for GARCH(1,1)) |
| LR at 5.99 rejects 3-4% | extends | Demos-Sentana 1998; Andrews 2001 (boundary + unidentified beta; no GARCH critical values found) |
| AIC picks GARCH 7-12%, BIC 0-1% | package-specific / known pattern | Hughes-King-Kwek 2004 (AIC liberal vs BIC conservative) |
| AIC ~50% (AR(1)) with mismatched likelihoods | package-specific | same-sample requirement |
| Power rises with T and persistence | known | Demos-Sentana 1998 |
| Beta estimates when alpha is small | extends | Ma-Nelson-Startz 2007 |
