# Non-Gaussian (Student-t) QMLE under asymmetric true innovations, and skewed densities

Verification legend: VERIFIED = checked against an authoritative record (publisher, RePEc/IDEAS, package docs/source) during this research. UNVERIFIED = from general knowledge or an aggregator only; check before publishing. Full texts of the paywalled papers were NOT read; results are close paraphrases of abstracts and secondary statements, so theorem-level conditions are flagged where not seen.

## (a) When does a non-Gaussian QMLE give consistent location/scale estimates, and when not?

### Takeaway
A non-Gaussian (e.g. symmetric Student-t) QMLE needs an identification condition beyond a correct conditional-variance model. Newey and Steigerwald (1997) show it holds if the conditional mean is identically zero or a symmetry condition holds; otherwise the location of the innovation density must be an extra parameter. Fan-Qi-Xiu (2014) and Francq-Lepage-Zakoian (2011) show the scale is also not identified at its true value without an adjustment, and neither adding a shape parameter nor choosing t fixes this. This matches E3: only the intercept (location) is biased, because a symmetric-t fit centres on the density's pseudo-true centre, not its mean.

### Cited Findings

**Newey, W. K. and Steigerwald, D. G. (1997).** "Asymptotic bias for quasi-maximum-likelihood estimators in conditional heteroskedasticity models." *Econometrica* 65(3), 587-599 (May 1997).
- Pages: the Econometric Society page gives 587-599; RePEc/IDEAS/EconPapers give 587-600. Conflict; 587-599 used by the publisher. JSTOR stable id 2171754 ([JSTOR](https://www.jstor.org/stable/2171754)); DOI 10.2307/2171754 is the standard JSTOR-form DOI for that id (UNVERIFIED, not displayed on the pages checked).
- URLs: [Econometric Society](https://econometricsociety.org/publications/econometrica/1997/05/01/asymptotic-bias-quasi-maximum-likelihood-estimators-conditional), [IDEAS](https://ideas.repec.org/a/ecm/emetrp/v65y1997i3p587-600.html)
- Result (abstract level): for a non-Gaussian QMLE in conditional heteroskedasticity models, the identification condition (the quasi-log-likelihood peaks uniquely at the true conditional mean and scale parameters) holds if the conditional mean is identically zero or a symmetry condition holds. Without symmetry, an additional parameter for the location of the innovation density must be added. For GARCH variance parameters, adding that location parameter costs no efficiency.
- Not verified (full text unread): the exact form of the symmetry condition, and the exact list of parameters that become inconsistent. The Qi-Xiu-Fan preprint cites the paper for the point that the scale need not be identified at its true value.
- Relation to study: E3 known / extends. E3's intercept bias is the situation the paper's remedy (extra location parameter) addresses; E3 quantifies it for a skewed-t truth with symmetric-t fit.

**Fan, J., Qi, L. and Xiu, D. (2014).** "Quasi-maximum likelihood estimation of GARCH models with heavy-tailed likelihoods." *Journal of Business & Economic Statistics* 32(2), 178-191. DOI 10.1080/07350015.2013.840239.
- URLs: [IDEAS](https://ideas.repec.org/a/taf/jnlbes/v32y2014i2p178-191.html), [EconPapers](https://econpapers.repec.org/RePEc:taf:jnlbes:v:32:y:2014:i:2:p:178-191). (One listing shows 178-205, which appears to include the discussion/rejoinder; 178-191 is the article.)
- Result: with a non-Gaussian likelihood, QMLE is inconsistent unless the likelihood family contains the true density; the authors identify an unknown scale parameter eta_f critical to consistency and propose a three-step non-Gaussian QMLE (NGQMLE). It is consistent and asymptotically normal under weak moment conditions and more efficient than Gaussian QMLE, particularly with heavy-tailed innovations.
- From the working-paper version (arXiv 1001.3895, read via a mirror; may differ from the published text): eta_f is the maximiser of an expected quasi-log-likelihood; the unscaled QMLE recovers the volatility scale sigma_0 only if eta_f = 1; estimating a shape parameter does not cure the inconsistency unless the truth is in the family; the theory is said not to rely on symmetry or unimodality of the innovations; the two-step version (2SNG-QMLE) uses Gaussian QMLE residuals for eta_f and assumes a finite fourth moment. Source: [arXiv 1001.3895](https://ar5iv.arxiv.org/html/1001.3895).
- Caveat: their scale fix concerns the variance/GARCH parameters in a zero-location model; I found no statement that they treat the location bias from asymmetric truths (UNVERIFIED either way; their simulation tables are captioned as Student-t and generalized Gaussian innovations).
- Relation to study: E3 extends. E3's GARCH parameters are "essentially unaffected" by the symmetric-t fit under skewed truth, so the scale problem they correct did not materialise in the skewed-t design; the location problem is the Newey-Steigerwald one.

**Francq, C., Lepage, G. and Zakoian, J.-M. (2011).** "Two-stage non Gaussian QML estimation of GARCH models and testing the efficiency of the Gaussian QMLE." *Journal of Econometrics* 165(2), 246-257. DOI 10.1016/j.jeconom.2011.08.001.
- URL: [EconPapers](https://econpapers.repec.org/RePEc:eee:econom:v:165:y:2011:i:2:p:246-257). (EconBiz lists 246-258; some aggregators mislabel the journal. Use EconPapers/DOI.)
- Result (abstract level): a non-Gaussian QMLE with a reparameterisation, from replacing the unit-variance identifiability assumption with an alternative moment normalisation, can be more efficient than Gaussian QMLE for the original GARCH specification; a test indicates whether the reparameterisation (a non-Gaussian density) is needed, i.e. whether the Gaussian QMLE is efficient.
- Relation to study: E3 known / package-specific. Supports the two-stage idea; the package's t-MLE is the unadjusted one-stage version.

**Fiorentini, G. and Sentana, E. (2019).** "Consistent non-Gaussian pseudo maximum likelihood estimators." *Journal of Econometrics* 213(2), 321-358. [EconPapers](https://econpapers.repec.org/RePEc:eee:econom:v:213:y:2019:i:2:p:321-358). DOI not displayed in what I read (UNVERIFIED).
- Result (abstract level): characterises which mean and variance parameters a distributionally misspecified ML estimator estimates consistently in location-scale models and gives closed-form consistent estimators for the remainder; covers GARCH-M and multivariate regressions. A search snippet from a working version says variance and covariance are natural location/scale only for the normal, so the mean cannot generally be expected consistent under non-normal pseudo-densities (UNVERIFIED wording).
- Relation to study: E3 known / extends. The most directly relevant modern treatment of E3's intercept bias; its closed-form correction is a candidate remedy.

### Inferences
- E3's finding (intercept biased toward mode, slopes/GARCH unaffected, Gaussian QMLE unbiased) is consistent with this literature: with a symmetric pseudo-density the pseudo-true location is the fitted-t centre, which differs from the mean for skewed truths. A constant shift is absorbed only by the intercept, so slopes are unaffected. This is an interpretation, not a quote from any paper.
- Sanity check on E3 (my calculation, not from a source): with arch's constants for eta = 6, lambda = -0.5, a = -0.750, b = 1.090, so the Hansen mode is at -a/b = +0.69 SD from the mean; E3's reported +0.122 SD pseudo-true centre lies between mean (0) and mode (0.69), as expected.

### Gaps
- Theorem-level statements and exact symmetry condition of Newey-Steigerwald unread (paywalled).
- No published simulation found that reports intercept bias magnitudes for t-QMLE with skewed truth (see section d).

## (b) Remedies

### Takeaway
Three remedies exist: (1) add a location (and where needed scale) parameter to the quasi-density (Newey-Steigerwald); (2) rescale / two- or three-step estimation (Francq et al.; Fan et al.; closed-form adjustments in Fiorentini-Sentana); (3) use a flexible or skewed density so the truth is nested (Hansen skew-t, which E3 does) or semiparametric adaptive estimation.

### Cited Findings
- Newey-Steigerwald extra location parameter, Fan-Qi-Xiu three-step NGQMLE, Francq et al. two-stage, Fiorentini-Sentana closed-form adjustments: see section (a).
- **Engle, R. F. and Gonzalez-Rivera, G. (1991).** "Semiparametric ARCH models." *Journal of Business & Economic Statistics* 9(4), 345-359. DOI 10.2307/1391236 (per Scilit, aggregator). [EconPapers](https://econpapers.repec.org/RePEc:bes:jnlbes:v:9:y:1991:i:4:p:345-59). Result: ARMA/ARCH mean-variance with nonparametrically estimated conditional density; Monte Carlo compares efficiency gains of a fully adaptive procedure with those of the semiparametric estimator, which captures a substantial share. Relation: context only; not directly on location bias.
- Relation to study: E3 known. The E3 skew-t MLE (correct family) is the "nest the truth" remedy; its removal of bias and efficiency gain are expected. Not tested in E3: the extra-location-parameter and two/three-step fixes.

### Gaps
- No evidence found on how well the two/three-step schemes perform with skewed truths.

## (c) Skewed-t and related densities

### Takeaway
Hansen (1994) skew-t, Fernandez-Steel skew-t, Jones-Faddy, Theodossiou SGT, Johnson SU and the non-central t of Harvey-Siddique are different constructions; the symbols lambda, xi, skew mean different things. The README claim is correct: arch's SkewStudent is Hansen (1994); rugarch's "sstd" is Fernandez-Steel based.

### Cited Findings

**Hansen, B. E. (1994).** "Autoregressive conditional density estimation." *International Economic Review* 35(3), 705-730. DOI 10.2307/2527081.
- URLs: [IDEAS](https://ideas.repec.org/a/ier/iecrev/v35y1994i3p705-30.html), [author page](https://users.ssc.wisc.edu/~behansen/papers/ier_94.html).
- Result: introduces a skewed Student-t conditional density (eta = degrees of freedom, lambda in (-1,1) = skewness) whose parameters may vary with conditioning information (ARCD); applied to Treasury yields and dollar/Swiss franc.
- Parameterisation (from arch source code, VERIFIED as implemented there): c = Gamma((eta+1)/2) / (sqrt(pi (eta-2)) Gamma(eta/2)); a = 4 lambda c (eta-2)/(eta-1); b^2 = 1 + 3 lambda^2 - a^2. Density (standard form from memory, UNVERIFIED against the paper text): f(z) = b c (1 + ((b z + a)/(1 + lambda sign(b z + a)))^2 / (eta-2))^(-(eta+1)/2), mean 0, variance 1. lambda = 0 gives the standardized t; lambda < 0 gives negative skew with the mode above the mean. Bounds in arch: eta in [2.05, 300], lambda in (-1,1).
- Relation: E3 known (the generator and the fitted skew-t).

**arch package, SkewStudent.** [docs](https://bashtage.github.io/arch/univariate/generated/arch.univariate.SkewStudent.html), [source](https://raw.githubusercontent.com/bashtage/arch/main/arch/univariate/distribution.py). Docs cite Hansen (1994) IER 35(3) 705-730; parameter names `eta`, `lambda`; lambda = 0 reduces to the standardized Student-t. Relation: README claim confirmed (package-specific).

**Fernandez, C. and Steel, M. F. J. (1998).** "On Bayesian modeling of fat tails and skewness." *Journal of the American Statistical Association* 93(441), 359-371. DOI 10.1080/01621459.1998.10474117 (publisher link seen via search; the JSTOR-form 10.2307/2669632 also appears; neither resolved against doi.org by me).
- URL: [Tilburg record](https://research.tilburguniversity.edu/en/publications/on-bayesian-modeling-of-fat-tails-and-skewness/)
- Result: a general way to add skewness to symmetric unimodal distributions without changing tail behaviour, applied to the Student-t to get a "skewed Student" with separate tail and skewness parameters; posterior existence under improper priors, Gibbs sampling.
- Mechanism (rugarch vignette, VERIFIED): inverse scale factors on the two half-lines, f(z|xi) = 2/(xi + 1/xi) times f(xi z) for z<0 and f(z/xi) for z>=0; xi = 1 is symmetric.
- Use: the rugarch vignette says 'sstd', 'sged', 'snorm' use Fernandez-Steel (and Ferreira-Steel 2006) transformations, standardised ([rugarch vignette](https://cran.r-project.org/web/packages/rugarch/vignettes/Introduction_to_the_rugarch_package.pdf)). fGarch's `sstd` is documented as based on the Fernandez-Steel scheme with a Wuertz et al. parameterisation, parameters `nu`, `xi` ([fGarch docs](https://rdrr.io/cran/fGarch/man/dist-sstd.html)); that page cites "Fernandez and Steel (2000)" as an unpublished preprint, so cite the 1998 JASA paper. Relation: README claim confirmed (package-specific). Warning: Hansen's lambda (in (-1,1), 0 symmetric) and xi (>0, 1 symmetric) are not interchangeable; the two skew-ts differ in shape and mode location for the same nominal skew.

**Jones, M. C. and Faddy, M. J. (2003).** "A skew extension of the t-distribution, with applications." *Journal of the Royal Statistical Society, Series B* 65(1), 159-174. DOI 10.1111/1467-9868.00378. [EconPapers](https://econpapers.repec.org/RePEc:bla:jorssb:v:65:y:2003:i:1:p:159-174). (Issue is 1, not 2.)
- Result: two shape parameters a, b; a < b negative skew, a > b positive, a = b gives t with 2a df; likelihood inference developed. Relation: not used in E3 (context; also in SciPy as jf_skew_t).

**Theodossiou, P. (1998).** "Financial data and the skewed generalized t distribution." *Management Science* 44(12, part 1), 1650-1661. DOI 10.1287/mnsc.44.12.1650. [Repository record](https://ktisis.cut.ac.cy/handle/20.500.14279/1988). Result (abstract level): skewed extension of the McDonald-Newey generalized t, with moments; fits skewed/fat-tailed financial series; special cases include skewed GED and skewed t. Relation: context.

**Johnson SU in rugarch ('jsu').** Reparameterised Johnson SU of Rigby and Stasinopoulos (2005) from the GAMLSS package, with mean mu and sd sigma for all shape parameters nu (skew), tau (shape) ([rugarch vignette](https://cran.r-project.org/web/packages/rugarch/vignettes/Introduction_to_the_rugarch_package.pdf)). Full Rigby-Stasinopoulos citation not verified (UNVERIFIED). Relation: package-specific.

**Harvey, C. R. and Siddique, A. (1999).** "Autoregressive conditional skewness." *Journal of Financial and Quantitative Analysis* 34(4), 465-487. DOI 10.2307/2676230. [EconPapers](https://econpapers.repec.org/RePEc:cup:jfinqa:v:34:y:1999:i:04:p:465-487_00). Result: time-varying skewness via a non-central t within ML jointly with time-varying mean and variance; conditional skewness matters and including it changes estimated variance persistence. Relation: context; the non-central t is a third skew-t family.

**Bauwens, L. and Laurent, S. (2005).** "A new class of multivariate skew densities, with application to generalized autoregressive conditional heteroscedasticity models." *Journal of Business & Economic Statistics* 23(3), 346-354. DOI 10.1198/073500104000000523 (via UCLouvain record). [EconPapers](https://econpapers.repec.org/RePEc:bes:jnlbes:v:23:y:2005:p:346-354). Result: method to add skewness to multivariate symmetric densities (generalising Fernandez-Steel); multivariate skew-Student with marginal asymmetry coefficients, used for GARCH and portfolio VaR. One RePEc record shows pages 346-353; 346-354 is the majority. Relation: context.

### Inferences
- E3's skewness values (-0.35, -0.84, -1.47 for lambda = -0.1, -0.25, -0.5) refer to Hansen's lambda; fitting a Fernandez-Steel sstd would need xi < 1 and different numbers. Keep naming the parameterisation, as the README does.

### Gaps
- The exact Hansen density printed in the 1994 paper was not re-read; constants verified from arch code only. Rigby-Stasinopoulos 2005 citation not verified.

## (d) Evidence, magnitude of location bias, empirical skewness

### Takeaway
I found no published simulation quantifying the intercept bias of t-QMLE under skewed innovations; E3's magnitudes appear new relative to what I could verify. Robust-statistics theory supports the mechanism. Empirical papers support nonzero skewness and better VaR with skewed densities.

### Cited Findings
- **Huber, P. J. (1964).** "Robust estimation of a location parameter." *Annals of Mathematical Statistics* 35(1), 73-101. DOI 10.1214/aoms/1177703732 (VERIFIED via Project Euclid listings). [Project Euclid](https://www.projecteuclid.org/euclid.aoms/1177703732). Result: asymptotic theory of M-estimators of location (minimising a sum of rho of residuals), the sample mean being rho = x^2; for non-symmetric distributions the target parameter must be defined via an untranslated prototype, or one restricts to symmetric cases (per a search-result reading of the abstract). Relation: supports the E3 mechanism (a different rho, as in the t score, converges to a different functional than the mean when F is asymmetric). The "between mean and mode" statement is my inference, not a quoted result.
- **Huber, P. J. (1981).** *Robust Statistics.* Wiley, New York. UNVERIFIED: could not locate the text; search did not confirm what it says about asymmetric F. Do not cite for a specific claim without checking the book.
- **Giot, P. and Laurent, S. (2003).** "Value-at-risk for long and short trading positions." *Journal of Applied Econometrics* 18(6), 641-663 (some listings 641-664). DOI 10.1002/jae.710. [EconPapers](https://econpapers.repec.org/RePEc:jae:japmet:v:18:y:2003:i:6:p:641-663). Result: APARCH with skewed-Student innovations gave better daily VaR for long and short positions than normal or symmetric Student on three indices and three US stocks (secondary descriptions). Relation: context for why skew-t is used. Standardised-residual skewness estimates not extracted.
- Searches for a simulation with skewed truth and t pseudo-density reporting mean bias returned nothing matching.

### Gaps
- Typical skewness of standardised GARCH residuals for equity indices and macro series: no verified figures retrieved; do not quote numbers.
- Density-forecast gains of skew-t beyond Giot-Laurent not searched.
- Iglesias and Phillips (2011, Econometric Reviews 30(3), 303-336) on small-sample GARCH bias with mean regressors surfaced in search (IDEAS listing); not read, and it concerns finite-sample bias, not skewness.

## Flags against the study context
- README statement on parameterisations is CORRECT: arch SkewStudent is Hansen (1994) (eta, lambda); rugarch sstd is Fernandez-Steel (inverse scale factors); fGarch sstd likewise.
- Newey-Steigerwald page range differs by source (587-599 publisher vs 587-600 RePEc).
- Jones-Faddy is JRSS B 65(1), not 65(2).
- Fan-Qi-Xiu author order in the published article is Fan, Qi, Xiu (the arXiv preprint lists Qi, Xiu, Fan).
- Theodossiou is in Management Science (confirmed).
