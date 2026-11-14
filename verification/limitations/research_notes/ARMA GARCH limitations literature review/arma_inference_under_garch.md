# ARMA / regression inference under GARCH errors, and the cost of over-fitting GARCH

Scope: literature notes grounding a Monte Carlo study of SARIMAX (homoskedastic) versus joint ARMA-GARCH(1,1) Gaussian ML.
Verification status: bibliographic details were checked against publisher, RePEc/IDEAS, EconBiz, Crossref or institutional records (see per-entry "Checked against"). Paper theorem statements were mostly NOT read in full text (paywalls, unreadable PDFs); where a result below rests only on an abstract or a secondary description, it says so. Anything not verified is marked UNVERIFIED. Entries marked "Inference" are my reasoning, not a reported result.

## 1. Least squares under ARCH/GARCH errors: consistency and efficiency of ML

### Takeaway
Least squares stays consistent when the conditional mean is correctly specified and the errors are ARCH/GARCH, but it is not efficient; Engle's abstract says ML is more efficient and that the relative efficiency can be infinite. I could not retrieve a quantified efficiency formula or any statement of the efficiency gain under typical monthly-financial parameter values, so the study's 8-17% RMSE reduction has no directly comparable published number in what I verified.

### Cited Findings

- **Engle (1982).** R. F. Engle, "Autoregressive Conditional Heteroscedasticity with Estimates of the Variance of United Kingdom Inflation," *Econometrica* 50(4), July 1982, 987-1007 (the Econometric Society page and some listings give 987-1008; JSTOR/EconBiz records give 987-1007; treat the end page as inconsistent across sources). DOI 10.2307/1912773 (reported by ScienceOpen; the Econometric Society page itself shows no DOI). JSTOR: https://www.jstor.org/stable/1912773 . Checked against: Econometric Society abstract page, EconBiz, ScienceOpen.
  - Result: introduces ARCH and a regression model with ARCH disturbances; derives ML; gives an LM test based on the autocorrelation of squared OLS residuals. The abstract states that OLS keeps its optimality properties in this set-up but ML is more efficient, and that the relative efficiency is calculated and can be infinite (second-hand wording from a search summary; the efficiency calculation itself was NOT read). Engle's page-level proofs of OLS consistency were not read; I flag the claim "OLS is consistent" as the standard textbook reading, not verified in the abstract.
  - Relation to study: **known / consistent with** the finding that adding GARCH improves slope RMSE (8-17%). The "infinite" bound applies to extreme ARCH; the study's moderate designs sit far from it. Quantification under the study's parameters: UNVERIFIED in literature.

- **Weiss (1986).** A. A. Weiss, "Asymptotic Theory for ARCH Models: Estimation and Testing," *Econometric Theory* 2(1), 107-131. DOI 10.1017/S0266466600011397 (from Scilit; not seen on the Cambridge page). Checked against: RePEc/IDEAS and Cambridge listing for volume/pages. URL https://ideas.repec.org/a/cup/etheor/v2y1986i01p107-131_01.html .
  - Result (secondary): asymptotics of (quasi-)ML for ARCH models, including linear (AR/regression) mean equations. Ling and McAleer (2003) credit Weiss for QMLE asymptotic normality in ARCH models and note his proofs assume a fourth-moment condition. The theorem statements, and whether Weiss proves LS consistency specifically, were NOT read. A Bohrium abstract mentions only fractional-moment conditions; I could not confirm that, and it conflicts with the fourth-moment description from Ling-McAleer. Treat moment conditions as UNVERIFIED.
  - Relation to study: **known**; supports using Gaussian QML for the joint model when shocks are normal (study is correctly specified there).

- **Francq & Zakoian (2004).** C. Francq and J.-M. Zakoian, "Maximum likelihood estimation of pure GARCH and ARMA-GARCH processes," *Bernoulli* 10(4), 605-637. DOI 10.3150/bj/1093265632 (Project Euclid, open access: https://www.projecteuclid.org/euclid.bj/1093265632). Checked against: Project Euclid listing, IP Paris research portal.
  - Result: strong consistency and asymptotic normality of the Gaussian QMLE for GARCH and for ARMA-GARCH, with only fractional moment conditions on the observed process in the consistency part (secondary description). Gives the sandwich covariance structure for the joint ARMA-GARCH QMLE used in the study.
  - Relation to study: **known**; the robust sandwich SEs for the joint model are theoretically supported.

- **Ling & McAleer (2003).** S. Ling and M. McAleer, "Asymptotic theory for a vector ARMA-GARCH model," *Econometric Theory* 19(2), 280-310 (a PDF header shows 278-308; page range conflict, 280-310 per RePEc). Checked against: RePEc, Cambridge Core. URL https://econpapers.repec.org/RePEc:cup:etheor:v:19:y:2003:i:02:p:280-310_19 .
  - Result: consistency of the QMLE under a second-moment condition, asymptotic normality under a fourth-moment condition (second-moment claim per abstract; fourth-moment per secondary text). Extends Ling and Li (1997).
  - Relation to study: **known**.

- **Ling & Li (1997).** S. Ling and W. K. Li, "On fractionally integrated autoregressive moving-average time series models with conditional heteroscedasticity," *Journal of the American Statistical Association* 92(439), 1184-1194. DOI 10.2307/2965585. NOTE: the issue number 439 is from my memory and was not confirmed by the search (which gave pages 1184-1194 and the DOI). Checked against: search result only. Marked UNVERIFIED for issue. The title in the request ("Asymptotic inference for nonstationary ...") was a mis-citation of this paper.

- **Newey & Steigerwald (1997).** W. K. Newey and D. G. Steigerwald, "Asymptotic bias for quasi-maximum-likelihood estimators in conditional heteroskedasticity models," *Econometrica* 65(3), 587-599 (EconPapers lists 587-600; Econometric Society lists 587-599). Checked against: Econometric Society page. https://econometricsociety.org/publications/econometrica/1997/05/01/asymptotic-bias-quasi-maximum-likelihood-estimators-conditional
  - Result (from search summary, not read): non-Gaussian QMLE can be asymptotically biased for the location parameter when the innovation density is asymmetric (condition: the true standardised innovation distribution is not symmetric); no efficiency loss for conditional-variance parameters of GARCH under their conditions. The summary I received was sparse; exact statement UNVERIFIED.
  - Relation to study: **not directly tested** (study uses normal shocks and Gaussian ML, so no such bias arises).

- **Gonzalez-Rivera & Drost (1999).** G. Gonzalez-Rivera and F. C. Drost, "Efficiency comparisons of maximum-likelihood-based estimators in GARCH models," *Journal of Econometrics* 93(1), 93-111. Issue "(1)" from memory, volume/pages checked against a PDF header. Result: compares ML, QML and semiparametric estimators; shows that the normal is the only density for which all three have the same asymptotic efficiency. Concerns non-normal innovations.
  - Relation to study: **not directly tested**; marks where Gaussian QML would lose efficiency beyond the study's normal-shock design.

### Inferences
- Under normal innovations and a correctly specified joint model, the joint estimator is ML; SARIMAX is a quasi-ML estimator that ignores heteroskedasticity. The efficiency loss of the latter should grow with the persistence and variability of sigma_t^2, which matches the study (8% at alpha .10 / beta .75, T=500 vs 12-17% at alpha .08 / beta .90, T=1000). This direction is my reasoning, not a published number for ARMA regression.
- I did not find a published efficiency comparison of ARMA-GARCH ML versus ARMA LS for realistic financial GARCH parameters, or versus two-step estimators; this is a gap in what I could verify.

### Gaps
- Engle (1982) efficiency formula for the ARCH(1) regression; JSTOR full text not accessible here.
- Any paper quantifying ML versus LS RMSE for regression slope under GARCH(1,1) at monthly parameter values.
- Two-step versus joint ARMA-GARCH efficiency (no source found).

## 2. Weak ARMA theory: validity of standard errors and Ljung-Box under GARCH

### Takeaway
When the innovations are only uncorrelated (white noise, not iid or martingale difference with constant conditional variance), the usual ARMA LS asymptotic covariance and the Box-Pierce/Ljung-Box null distribution are not generally valid; robust sandwich estimators and modified portmanteau tests exist. Whether the bias is upward or downward depends on the model and on the moments of the noise; I found no source giving a general direction, and the study's SARIMAX opg coverage of 0.91-0.97 on the slope is not contradicted.

### Cited Findings

- **Francq & Zakoian (1998).** C. Francq and J.-M. Zakoian, "Estimating linear representations of nonlinear processes," *Journal of Statistical Planning and Inference* 68(1), 145-165, May 1998. DOI 10.1016/S0378-3758(97)00139-0. Checked against: EconPapers/HAL record, MaRDI/zbMATH (zbMATH 0942.62100; keywords ARMA, asymptotic normality, consistency, least-squares, nonlinear models).
  - Result: the abstract was NOT retrieved. From the keywords and the companion French notes, the paper concerns least-squares estimation of ARMA representations of nonlinear (uncorrelated, non-independent) processes, with consistency and asymptotic normality whose covariance differs from the iid-noise formula. Details UNVERIFIED.
  - Relation to study: **known** theoretically (SARIMAX opg SEs rest on iid/martingale assumptions; the sandwich is the robust fix).

- **Francq, Roy & Zakoian (2005).** C. Francq, R. Roy and J.-M. Zakoian, "Diagnostic checking in ARMA models with uncorrelated errors," *Journal of the American Statistical Association* 100(470), 532-544, June 2005. DOI 10.1198/016214504000001510. Checked against: IP Paris research portal record (lists keywords including GARCH and weak ARMA); consistent across three listings.
  - Result (abstract-level): derives the asymptotic covariance matrix of residual autocorrelations under weak noise assumptions; shows standard Box-Pierce and Ljung-Box portmanteau tests can perform poorly; gives a consistent estimator and a modified portmanteau test with valid significance limits; Monte Carlo and S&P 500 application. The direction of the size distortion was not stated in the abstract. A search for "Ljung-Box over-rejects under GARCH" returned no confirmation.
  - Relation to study: **extends** (the study did not examine residual-diagnostic tests; it did examine coefficient SEs, where opg coverage remained 0.91-0.97 for the slope).

- **Romano & Thombs (1996).** J. P. Romano and L. A. Thombs, "Inference for autocorrelations under weak assumptions," *Journal of the American Statistical Association* 91(434), 590-600. DOI 10.1080/01621459.1996.10476928 (Crossref). Checked against: Crossref; content from later citing papers, not the paper itself.
  - Result (secondary): derives the asymptotic distribution of sample autocorrelations without iid innovations, under mixing and moment conditions; the Bartlett formula can be invalid; proposes block bootstrap and subsampling. Their ARMA coefficient result was mentioned only in an abstract fragment.
  - Relation to study: **known** background; not tested by the study.

### Inferences
- Inference (mine): for a regression slope whose regressor x is independent of the conditional variance process, the HAC correction is small, which would explain slope opg coverage near nominal in the study. For AR coefficients, correlation between squared errors and squared lagged observations is the channel through which standard SEs are distorted; I did not verify its sign from a source.
- A natural extra check (not in the study): compare SARIMAX opg and robust SEs for phi/theta against the Monte Carlo SD.

### Gaps
- Abstract and theorems of Francq-Zakoian (1998) (paywalled).
- A published statement of the sign of the SE bias for AR(1)/MA(1) with GARCH errors.
- Li & Mak (1994, J. Time Series Analysis, portmanteau test for GARCH) was mentioned by a secondary source only; not verified.

## 3. Conditional coverage of constant-width intervals

### Cited Findings

- **Christoffersen (1998).** P. F. Christoffersen, "Evaluating Interval Forecasts," *International Economic Review* 39(4), 841-862, November 1998. DOI 10.2307/2527341 (one listing only; verify at JSTOR). Checked against: RePEc/IDEAS (volume, pages); DOI from one aggregator.
  - Result (abstract-level): prior work implicitly assumed homoskedastic errors and tested only unconditional coverage; proposes likelihood-ratio tests of unconditional coverage, independence of the hit sequence, and joint conditional coverage. A constant-width interval can have correct average coverage but clustered violations when volatility clusters (the stated motivation; the specific formal example in the paper was not read).
  - Relation to study: **known / consistent with** the calm (0.92-0.96) versus turbulent (0.81-0.86) coverage of the SARIMAX band; the study's result is a conditional-on-true-sigma^2 version of the same point.

### Gaps
- No Monte Carlo evidence of calm-versus-turbulent coverage for homoskedastic intervals located; the study's numbers appear to be the only quantification in my sources.
- Christoffersen's text was not read.

## 4. Over-fitting GARCH to homoskedastic data

### Takeaway
When alpha is at or near zero, beta is not identified and standard asymptotics fail; estimates pile up at boundaries, standard errors are too small, and persistence is spuriously high. This matches the 64-87% boundary rate and gives a reason that the cost to the mean parameters stays small.

### Cited Findings

- **Ma, Nelson & Startz (2007).** J. Ma, C. R. Nelson and R. Startz, "Spurious inference in the GARCH(1,1) model when it is weakly identified," *Studies in Nonlinear Dynamics & Econometrics* 11(1), Article 1 (pp. 1-27 per a secondary listing). DOI 10.2202/1558-3708.1434 (from a search summary; unconfirmed on publisher site). Checked against: University of Washington repository record (title, authors, year, volume/issue, abstract).
  - Result: the Zero-Information-Limit Condition holds in GARCH(1,1): when the ARCH parameter is small, the beta estimate's standard error is too small even in very large samples; combined with upward bias this can lead to spurious inference that volatility is highly persistent. The authors propose an empirical strategy based on a pure ARCH(q) approximation.
  - Relation to study: **known / consistent with** the boundary solutions and the near-zero cost to mean parameters; the study's SE issue for beta was not directly measured (UNVERIFIED relation).

- **Nelson & Startz (2007).** C. R. Nelson and R. Startz, "The zero-information-limit condition and spurious inference in weakly identified models," *Journal of Econometrics* 138(1), 47-62. Checked against: EconPapers record. Result: when a parameter's asymptotic variance depends on another parameter that approaches a critical value, standard errors tend to be underestimated and t-test size can be too large or too small. Relation: background for Ma-Nelson-Startz.

- **Andrews (1999).** D. W. K. Andrews, "Estimation when a parameter is on a boundary," *Econometrica* 67(6), 1341-1383 (RePEc lists 1341-1384). DOI 10.1111/1468-0262.00082 (reported by a search summary; not independently confirmed). Checked against: Econometric Society and RePEc records. Result: asymptotic distribution of extremum estimators when the true parameter is on the boundary (non-normal, a function of a Gaussian vector, in models without stochastic trends). Examples include QML of random coefficient regressions. Relation: **known** background for the boundary solutions.

- **Andrews (2001).** D. W. K. Andrews, "Testing when a parameter is on the boundary of the maintained hypothesis," *Econometrica* 69(3), 683-734. DOI 10.1111/1468-0262.00210. Checked against: Econometric Society, RePEc, Zendy (DOI). Result: asymptotic null distributions of QLR, Wald and score tests with boundary parameters and nuisance parameters unidentified under the null; examples include tests for conditional heteroskedasticity in GARCH(1,1). Relation: **known** background; relevant to LR-testing "GARCH vs no GARCH" before fitting.

- **Lumsdaine (1996).** R. L. Lumsdaine, "Consistency and asymptotic normality of the quasi-maximum likelihood estimator in IGARCH(1,1) and covariance stationary GARCH(1,1) models," *Econometrica* 64(3), 575-596. Checked against: Econometric Society, RePEc. Result: QML consistent and asymptotically normal for GARCH(1,1) including the IGARCH case; this is for interior parameters, so it does not cover the boundary case. Relation: **known**; context for the persistent design (beta .90).

- **Jensen & Rahbek (2004).** S. T. Jensen and A. Rahbek, "Asymptotic normality of the QMLE estimator of ARCH in the nonstationary case," *Econometrica* 72(2), 641-646. DOI 10.1111/j.1468-0262.2004.00504.x. Checked against: Econometric Society, RePEc. Result: QMLE of linear ARCH is consistent and asymptotically normal even where no stationary solution exists. Relevance to the boundary question: not directly (the issue there is alpha = 0, a different boundary); mention only as context.

### Gaps
- No source found quantifying the frequency of boundary solutions, so the 64-87% figure has no direct literature comparator.
- Full text of Ma-Nelson-Startz not read; the ARCH(q) approximation details and Monte Carlo numbers are UNVERIFIED.

## 5. Textbook workflow: fit the mean, test residuals for ARCH, then fit jointly

### Cited Findings

- **Engle (1982)** (above): the LM test on squared OLS residuals is the original ARCH-effect test; this is the basis for the "fit mean, test residuals, then model variance" workflow.
- **Tsay, Analysis of Financial Time Series** (Wiley; editions 2005 and 2010). The ARCH-effect test is a Ljung-Box test on squared mean-equation residuals or Engle's LM test, per forum and software documentation citing pp. 101-103 of the 2nd edition. The text itself and the claim that a joint-estimation step follows were NOT verified. DOI 10.1002/9780470644560 for the 3rd edition: UNVERIFIED.
- **Francq & Zakoian, GARCH Models: Structure, Statistical Inference and Financial Applications**, Wiley, 1st ed. 2010 (ISBN 9780470683910), 2nd ed. 2019 (ISBN 9781119313571, 504 pp.). Checked against: library catalogue and Wiley-VCH records. No DOI found. Chapters on ARMA-GARCH estimation not read.
- **Hamilton (1994), Time Series Analysis, ch. 21** (Princeton University Press): UNVERIFIED; no source retrieved.

### Relation to study
The recommended workflow is **known**; the study's finding that the SARIMAX-first approach loses only 8-17% RMSE on the slope and 0-1.6% CRPS says the first-step mean fit is mostly adequate, while the interval band is the main gain.

## Flags on the request's citations
- Ling & Li (1997): title in the request was wrong; correct title above.
- Engle (1982) end page 1007 vs 1008 differs across authoritative listings.
- Andrews (1999): 1341-1383 (request) vs 1341-1384 (RePEc).
- Ling & McAleer (2003): pages 280-310 vs 278-308 in different sources.
- Romano-Thombs and Francq-Roy-Zakoian: citations correct as given.
- "Hall & Yao" and "Jensen & Rahbek" appear only as pointers; the latter is verified above.

## Not found / honest limitations
Direction of SE bias under GARCH for ARMA coefficients, efficiency quantification for ARMA-GARCH ML vs LS, calm-versus-turbulent coverage MC evidence, boundary frequency, and textbook chapter contents were not verified from sources.
