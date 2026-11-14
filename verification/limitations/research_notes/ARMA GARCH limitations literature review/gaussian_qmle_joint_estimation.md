# Gaussian QMLE for ARMA-GARCH and joint mean-variance estimation under variance misspecification

Verification key. VERIFIED-META = bibliographic data checked against Crossref/publisher or RePEc records. VERIFIED-TEXT = the theorem or abstract wording was read in a primary or near-primary copy. SECONDARY = result is reported only through a later paper's description. UNVERIFIED = could not be confirmed. Paraphrases only; short quotes are marked.

## (a) When is the Gaussian QMLE of ARMA-GARCH consistent and asymptotically normal, and under which moments?

### Takeaway
For pure GARCH, Francq-Zakoian (2004) need only strict stationarity for consistency and a finite fourth moment of the innovation eta for asymptotic normality, with no moment on the observed process. For ARMA-GARCH they still need a finite fourth moment of the observed process for asymptotic normality. "E[eps^4] < inf" is therefore right for ARMA-GARCH, but only if eps means the observed process (equivalently the GARCH noise, with eta^4 also finite). It is not right for pure GARCH. Check that the package docstring says which eps it means. Ling (2007) removes the process-moment condition by self-weighting.

### Cited Findings

**Bollerslev & Wooldridge (1992)** — Bollerslev, T. and Wooldridge, J.M. (1992), "Quasi-maximum likelihood estimation and inference in dynamic models with time-varying covariances", *Econometric Reviews* 11(2), 143-172. DOI 10.1080/07474939208800229. VERIFIED-META (Crossref). The abstract wording was seen via search-result snippets, not the full text.
- Dynamic models with jointly parameterised conditional mean and conditional covariance, estimated by maximising a normal likelihood. Because the normal score keeps its martingale-difference property when the first two conditional moments are correct, the QMLE is generally consistent and asymptotically normal.
- They give computable standard errors valid under non-normality (the sandwich form A^-1 B A^-1, with A the Hessian-type matrix and B the outer product of scores) and robust LM tests from auxiliary regressions. A Monte Carlo and a stock-volatility application are included.
- Condition: both conditional mean and conditional variance correctly specified. Newey-Steigerwald (1997, footnote 1) add that B-W require a finite unconditional variance.
- Relation to study: E3 known (Gaussian QMLE is robust to non-normal and skewed innovations when mean and variance are right; robust-SE coverage). E4 extends (the theorem's premise, a correct variance, is exactly what E4 violates).

**Weiss (1986)** — Weiss, A.A. (1986), "Asymptotic theory for ARCH models: estimation and testing", *Econometric Theory* 2(1), 107-131. DOI 10.1017/S0266466600011397. VERIFIED-META.
- Treats ARMA models with ARCH errors (not GARCH). Meitz-Saikkonen (2008, CREATES RP 2008-30, below) report that Weiss, Pantula (1988) and Lange-Rahbek-Jensen (2006) cover ARCH but not GARCH errors. SECONDARY for content; the abstract was not read.
- Relation to study: package-specific (the ARCH-only precursor; Ling-McAleer and Francq-Zakoian extend it to GARCH).

**Lee & Hansen (1994)** — Lee, S.-W. and Hansen, B.E. (1994), "Asymptotic theory for the GARCH(1,1) quasi-maximum likelihood estimator", *Econometric Theory* 10(1), 29-52. DOI 10.1017/S0266466600008215. VERIFIED-META.
- Pure GARCH(1,1) QMLE consistency and asymptotic normality. Newey-Steigerwald (1997, fn. 1) say Lee-Hansen (as a 1991 working paper) and Lumsdaine do not require a finite unconditional variance for the Gaussian QMLE of GARCH(1,1). SECONDARY.
- Relation to study: package-specific (the GARCH(1,1) variance block).

**Lumsdaine (1996)** — Lumsdaine, R.L. (1996), "Consistency and asymptotic normality of the quasi-maximum likelihood estimator in IGARCH(1,1) and covariance stationary GARCH(1,1) models", *Econometrica* 64(3), 575-596. DOI 10.2307/2171862. VERIFIED-META for journal, volume, issue and first page 575. UNVERIFIED for end page 596, which is from memory.
- The Gaussian QMLE for GARCH(1,1) is consistent and asymptotically normal in IGARCH(1,1) and covariance-stationary GARCH(1,1), with no unconditional-variance moment needed (per SECONDARY report above). It is a pure GARCH result with no ARMA mean.
- Relation to study: package-specific.

**Ling & McAleer (2003)** — Ling, S. and McAleer, M. (2003), "Asymptotic theory for a vector ARMA-GARCH model", *Econometric Theory* 19(2), 280-310. DOI 10.1017/S0266466603192092. VERIFIED-META for journal, volume, issue, year and DOI. Crossref gives no page range, so the pages 280-310 are UNVERIFIED (from memory).
- Per Meitz-Saikkonen (2008): global QMLE proved weakly consistent under a second-moment condition and asymptotically normal under a sixth-moment condition. In the ARCH-only case a fourth moment suffices for normality. SECONDARY. A 2025 Statistica Sinica preprint, found only in a search snippet, notes the sixth-moment requirement is still the vector case's problem.
- Relation to study: E-none directly. Context: the 4th-moment condition of Francq-Zakoian is a weakening of this 6th-moment condition.

**Francq & Zakoian (2004)** — Francq, C. and Zakoian, J.-M. (2004), "Maximum likelihood estimation of pure GARCH and ARMA-GARCH processes", *Bernoulli* 10(4), 605-637. DOI 10.3150/bj/1093265632. VERIFIED-META (Crossref). The abstract was seen only in part. The conditions below are SECONDARY (Meitz-Saikkonen 2008, CREATES RP 2008-30, Intro and Sect. on linear AR-GARCH, and a second secondary summary).
- Strong consistency and asymptotic normality of the Gaussian QMLE for pure GARCH(p,q) and for ARMA(P,Q) with GARCH(p,q) noise. Consistency needs only strict stationarity (a log-moment condition) plus a fractional-order moment of the observed process.
- Pure GARCH: weak moment conditions suffice for normality (finite fourth moment of eta, no moment of the process).
- ARMA-GARCH: finite fourth moment of the observed process is still needed for normality, and eta has a finite fourth moment. Meitz-Saikkonen state that this remains the weakest known condition for the global (unweighted) QMLE, and call dropping the fourth moment hard "even in the linear case".
- Verdict on the docstring: "E[eps^4] < inf is the condition for asymptotic normality of the ARMA-GARCH QMLE" is correct if eps is the observed process and the sufficiency (not necessity) is implied. It does not describe the pure-GARCH case, and it is a condition on the process, not only on the standardised innovation. Primary theorem text not read; state as SECONDARY until checked in the paper or the book (Ch. 7).
- Relation to study: E3 known (fourth moment holds for skewed innovations used in the simulation, so Gaussian QMLE root-n normal; a theory basis for CI coverage 0.92-0.96). E4 package-specific (conditions assume the variance model is correct).

**Ling (2007)** — Ling, S. (2007), "Self-weighted and local quasi-maximum likelihood estimators for ARMA-GARCH/IGARCH models", *Journal of Econometrics* 140(2), 849-873. DOI 10.1016/j.jeconom.2006.07.016. VERIFIED-META (Crossref). Abstract content SECONDARY (search snippets of the abstract, not full text).
- Proposes a self-weighted QMLE that is consistent and asymptotically normal under only a fractional moment condition on the observed process (E|eps|^tau < inf for some tau > 0), keeping the fourth moment of eta. It addresses the open case of infinite fourth moment of the process. Also gives asymptotic normality of a local QMLE for stationary ARMA-GARCH and IGARCH, Wald tests for linear restrictions, and shows the IGARCH tail index is 2.
- Cost: the self-weighted estimator is less efficient than the unweighted QMLE (the weights discount large observations). That cost is not confirmed from the paper text; treat as UNVERIFIED.
- Relation to study: package-specific (the docstring's cited fallback when E eps^4 = inf; the package does not implement it).

**Francq & Zakoian (2007)** — Francq, C. and Zakoian, J.-M. (2007), "Quasi-maximum likelihood estimation in GARCH processes when some coefficients are equal to zero", *Stochastic Processes and their Applications* 117(9), 1265-1284. DOI 10.1016/j.spa.2007.01.001. VERIFIED-META. The abstract was seen via RePEc/EconPapers snippet.
- When true GARCH coefficients are zero (boundary), the QMLE limit is non-standard: the law of a normal vector projected on a convex cone. For an important subclass no moment condition is needed. Their companion (Francq-Zakoian, JASA 2009 per a search snippet; not checked) shows t, Wald and LR tests are invalid on the boundary and gives corrected critical values.
- The user's citation (SPA) is correct.
- Relation to study: package-specific (alpha = 0 boundary; not an E3/E4 issue).

**Andrews (2001)** — Andrews, D.W.K. (2001), "Testing when a parameter is on the boundary of the maintained hypothesis", *Econometrica* 69(3), 683-734. DOI 10.1111/1468-0262.00210. VERIFIED-META.
- Asymptotics of ML-type test statistics (including Wald, LR, LM) when the true value lies on the boundary of the parameter space; the limit is a function of a truncated normal, and the usual chi-squared critical values fail. Gives results for GARCH-type examples. Result content from memory and not checked in this pass: SECONDARY at best.
- Relation to study: package-specific.

### Inferences
- The docstring should say "E[(observed-process)^4] < inf and E[eta^4] < inf" for ARMA-GARCH, and say pure GARCH needs less.
- The skewed-innovation design in E3 only needs eta to have four moments; skewness does not break the normal-score argument.

### Gaps
- Primary theorem statements for Weiss, Lee-Hansen, Lumsdaine, Ling-McAleer, Francq-Zakoian 2004 and Ling 2007 were not read; publisher pages returned 403 or empty. Conditions come from secondary descriptions.
- End pages for Lumsdaine (596), Ling-McAleer (310), GMT (700) are UNVERIFIED.

## (b) Robust sandwich standard errors

### Takeaway
The B-W (1992) formula H^-1 S'S H^-1 is the right robust covariance for the Gaussian QMLE when mean and variance are correctly specified. Its validity needs enough moments for the score covariance to exist (fourth moment of the process in ARMA-GARCH).

### Cited Findings
- B-W (1992) give the robust asymptotic standard errors, valid under non-normality (see entry above). Wang, Wooldridge & Xu (2025), JTSA 46(3), 458-490, DOI 10.1111/jtsa.12770 (VERIFIED-META and abstract/intro read in the open-access PDF) restate it: if the first two conditional moments are correctly specified, the Gaussian QMLE consistently estimates mean and variance parameters, and B-W show how to estimate the asymptotic variance allowing for arbitrary departures from normality "subject to the existence of enough finite moments".
- Wang-Wooldridge-Xu relation to study: E3 known (Gaussian QMLE is consistent with skewed errors; their estimator improves efficiency under asymmetric and non-Gaussian errors; E3's symmetric-t result matches their statement that t-based mean and variance estimators are generally inconsistent when the t is misspecified, e.g. under asymmetry).
- Relation to study for the sandwich in general: E3 known; E4 contradicts any hope that the sandwich repairs bias. Sandwich SEs correct variance, not centring; under pseudo-true shift they stay valid around the pseudo-true value but not the true one, hence the coverage drop to 0.745 (inference, not a cited result).

### Gaps
- No source found that studies sandwich-SE coverage under variance misspecification specifically.

## (c) Engle's block-diagonality

### Takeaway
Engle (1982), Theorem 4: the information matrix between regression coefficients and ARCH parameters is block diagonal if the ARCH regression model is "symmetric and regular". Symmetric means the variance is an even function of lagged errors (each past error's sign can be flipped without changing h) and the conditional density is symmetric (normal). Leverage (GJR) breaks the evenness.

### Cited Findings
**Engle (1982)** — Engle, R.F. (1982), "Autoregressive conditional heteroscedasticity with estimates of the variance of United Kingdom inflation", *Econometrica* 50(4), 987-1007. DOI 10.2307/1912773; JSTOR 1912773. VERIFIED-META and VERIFIED-TEXT (JSTOR copy read; some sources print 987-1008, the JSTOR header says 987-1007).
- Definition (Sect. 5): an ARCH process is symmetric if h is unchanged when past errors flip sign, with the matching derivative conditions; the paper notes this is "the main distinction between mean and variance models".
- Theorem 4: if the ARCH regression model is symmetric and regular, the off-diagonal information block between the regression (beta) and ARCH (alpha) parameters is zero. The author notes beta and alpha can be estimated separately without asymptotic loss of efficiency and their variances calculated separately.
- Proof (Appendix lemma): the cross-term score is an anti-symmetric function of past errors, with symmetric conditional density, so its expectation vanishes.
- Implication: two-step (regression, then ARCH on residuals) is asymptotically efficient only in that symmetric, correctly specified setting. It is about the efficiency of separate estimation under the true model, not about consistency of the mean under a misspecified variance.
- Bollerslev (1986), *J. Econometrics* 31(3), 307-327, DOI 10.1016/0304-4076(86)90063-1 (VERIFIED-META) is reported in a search snippet to say the asymptotic independence of mean and variance estimates carries over to GARCH. SECONDARY; full text not read.
- Newey & Steigerwald (1997) (below) mention block diagonality between mean and variance parameters as the condition under which an added location parameter costs no efficiency for the variance parameters. SECONDARY from the working-paper text.
- Relation to study: E4 extends. In E4 the true variance is GJR, so it is not an even function of past errors; the cross block need not vanish, and the symmetric GARCH fit is also misspecified. Engle's theorem is silent on that case. E3 known (skewed innovations do not enter the symmetry condition as stated for the normal-score proof only if the innovation is symmetric; unverified how Engle's proof extends; B-W guarantees consistency regardless).

### Gaps
- Whether block diagonality is exact for GARCH(1,1) with ARMA mean in the package's parameterisation was not checked; only secondary support (Bollerslev 1986 snippet).

## (d) Mean-parameter consistency when the variance is misspecified but estimated jointly

### Takeaway
No source found that proves or disproves the E4 result directly. B-W 1992 and Wooldridge-type theory give consistency when mean AND variance are correct, a sufficient condition. Mean-only estimators (OLS, fixed-weight WLS, Gourieroux-Monfort-Trognon linear exponential family PML) are consistent when only the mean is correct. Joint estimation with a misspecified variance, where the variance depends on the mean parameters through lagged residuals, falls outside both results; the E4 mechanism below is an inference consistent with the theory.

### Cited Findings
**Gourieroux, Monfort & Trognon (1984)** — "Pseudo maximum likelihood methods: theory", *Econometrica* 52(3), 681-700. DOI 10.2307/1913471. VERIFIED-META for journal, volume, issue, first page 681 (end page 700 UNVERIFIED, from memory). Abstract not read in this pass.
- Standard result (from memory, not checked in the text): for the linear exponential family (normal with a fixed variance included), PML estimators of conditional-mean parameters are consistent whenever the conditional mean is correctly specified, whatever the true distribution. This is a statement about mean-only estimation with variance parameters not estimated jointly from the same likelihood.
- Relation to study: E4 known as the reason OLS/SARIMAX without GARCH is unbiased; does not cover jointly estimated variance parameters.

**Wooldridge (1994)** — "Estimation and inference for dependent processes", Chapter 45 in *Handbook of Econometrics*, pp. 2639-2738. DOI 10.1016/S1573-4412(05)80014-5. VERIFIED-META (Crossref). Volume number (4, eds. Engle and McFadden) from memory, UNVERIFIED. Content not read.
- Relation to study: UNVERIFIED. I did not find a quotable statement on mean-parameter consistency under variance misspecification in it.

**Newey & Steigerwald (1997)** — Newey, W.K. and Steigerwald, D.G. (1997), "Asymptotic bias for quasi-maximum-likelihood estimators in conditional heteroskedasticity models", *Econometrica* 65(3), 587-599. JSTOR 2171754 per a search result; DOI not checked here. Publisher and RePEc records found; abstract and introduction read in an author-hosted working-paper copy.
- Gaussian QMLE is known to be consistent under correct specification of both conditional mean and variance (their footnote credits B-W 1992, Lumsdaine, Lee, Lee-Hansen). For a non-Gaussian quasi-likelihood, correct mean and variance are not enough: the identification condition (the expected log-likelihood is maximised at the true mean and relative scale) holds if the conditional mean is identically zero or if a symmetry condition holds (both true and assumed innovation densities unimodal and symmetric about zero). Without symmetry, an extra location parameter for the innovation density is needed.
- Relation to study: E3 known (symmetric-t quasi-likelihood biases the intercept when innovations are skewed; Gaussian does not). E4 extends in spirit (symmetry plays the identification role) but the paper's setting has a correctly specified variance, so it does not decide E4.

**Wang, Wooldridge & Xu (2025)** — as above. Quotable points: the Gaussian QMLE needs only the first two conditional moments correct for consistency; t-likelihood mean and variance estimators are generally inconsistent when the t is misspecified (asymmetry, or fourth moment not proportional to variance squared); they cite Newey-Steigerwald for the role of symmetry; their setup allows mean and variance parameters to overlap (ARCH-in-mean, Engle et al. 1987).
- Relation to study: E3 known. Nothing found on variance misspecification.

**Other sources checked but not usable for (d)**: Meitz-Saikkonen (2008) is about nonlinear AR-GARCH asymptotics with a correct variance; Fiorentini-Sentana's consistent non-Gaussian PML paper (J. Econometrics 2019, 88-page working-paper copy opened, text not extracted in time) is UNVERIFIED for this question.

### Inferences (not cited findings)
- Sign of the E4 bias. With the Gaussian pseudo-score for c, dl/dc = eps/sigma2 + 0.5 (eps^2/sigma2 - 1) sigma2^-1 dsigma2/dc, and dsigma2/dc = -2 alpha sum beta^j eps_{t-1-j} (proposed mechanism). The first term has zero mean (martingale difference). Under GJR true variance, negative lagged errors raise the true variance above the fitted symmetric one, so (eps^2/sigma2_fit - 1) is positive on average exactly when dsigma2/dc is positive (negative lagged eps). The product has positive mean; the estimator shifts c up, matching E4 (upward, slopes unaffected because the regressors do not enter the lagged-error term the same way; the slope claim is not derived here). This is the same mechanism as the failure of Engle's anti-symmetry lemma: with a variance that is not even in past errors, E[(eps^2/h - 1) dh/dc] need not be zero.
- If the variance is symmetric but the innovation is skewed (E3), the cross-term still has zero mean at the true variance, so the consistency theory applies. That fits E3.
- The result "mean parameters stay consistent regardless of the variance" is true only for estimators that do not let the variance model depend on the mean parameters. For ARMA-GARCH it does (eps_t = y_t - mean), so the intuition from OLS or GMT does not carry over. No paper was found that states this explicitly; flag as the study's contribution to confirm with a small derivation or the GJR-GARCH pseudo-true value computed numerically.

### Gaps
- No paper found with a quotable statement on joint-estimation bias of mean parameters when the variance is misspecified (leverage ignored). GARCH-in-mean sensitivity to the variance specification was searched without a usable result.
- Wooldridge (1994), GMT (1984) and Bollerslev-Wooldridge (1992) full texts not read.

## (e) Textbook and benchmark references

**Francq & Zakoian book** — Francq, C. and Zakoian, J.-M., *GARCH Models: Structure, Statistical Inference and Financial Applications*, Wiley. 1st ed. 2010, DOI 10.1002/9780470670057. 2nd ed. 2019, DOI 10.1002/9781119313472. VERIFIED-META (Crossref; 2nd ed. issued 2019-03-25).
- Per publisher TOC listings found in search: in the 2nd ed. Ch. 7 is "Estimating GARCH Models by Quasi-Maximum Likelihood" (p. 175), Ch. 8 "Tests Based on the Likelihood", Ch. 5 "Identification" (includes identifying ARMA orders in an ARMA-GARCH). The 1st ed. Ch. 9 is "Optimal Inference and Alternatives to the QMLE"; the 2nd ed. Ch. 9 title not confirmed. The ARMA-GARCH QMLE theory is expected in Ch. 7 but I did not confirm the section number: UNVERIFIED.
- Relation to study: package-specific (recommended textbook source for the moment conditions in (a)).

**Fiorentini, Calzolari & Panattoni (1996)** — "Analytic derivatives and the computation of GARCH estimates", *Journal of Applied Econometrics* 11(4), 399-417. DOI 10.1002/(SICI)1099-1255(199607)11:4<399::AID-JAE401>3.0.CO;2-R. VERIFIED-META (Crossref). The user's citation is right.
- Relation to study: package-specific (GARCH benchmark dataset and results).

**McCullough & Renfro** — "Benchmarks and software standards: a case study of GARCH procedures", *Journal of Economic and Social Measurement* 25(2), 59-71. DOI 10.3233/JEM-1999-0160. VERIFIED-META. Crossref shows issued 1998-12-01 while the volume is the 1999 volume; cite as 1999 (or 1998 online). Flag: the user's list gives only the year, not the journal; the journal is JESM. Whether this paper tabulates the FCP results was not verified from the text: UNVERIFIED.

**Brooks, Burke & Persand (2001)** — "Benchmarks and the accuracy of GARCH model estimation", *International Journal of Forecasting* 17(1), 45-56. DOI 10.1016/S0169-2070(00)00070-4. VERIFIED-META. The user's citation (17:45-56) is correct. Whether it re-uses the FCP benchmark was not verified from the text.

## Citation flags

- Francq-Zakoian 2004, "E[eps^4] < inf is the condition": right only for ARMA-GARCH with eps the observed process, plus E[eta^4] < inf. For pure GARCH the condition is on eta alone.
- Ling-McAleer (2003): pages 280-310 UNVERIFIED (Crossref gave none).
- McCullough-Renfro: journal is *Journal of Economic and Social Measurement*, not named in the list.
- Engle (1982): pages 987-1007 per the JSTOR copy; some indexes say 987-1008.
- No other citation in the user's list was found to be wrong. Unchecked: Andrews (2001) content; Francq-Zakoian JASA 2009 companion.
