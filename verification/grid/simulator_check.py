"""Is the simulator right?  Compare one long simulated series (T = 500,000) with the model's
theoretical moments -- no fitting code involved.  Exits non-zero on failure.
  E[eps^2] = omega/(1-a-b);  kurtosis(eps) = k_z (1-(a+b)^2) / (1-(a+b)^2-(k_z-1) a^2);
  ACF_1(eps^2) = a (1-ab-b^2)/(1-2ab-b^2), ACF_k = (a+b)^(k-1) ACF_1;  shocks z iid with mean 0, var 1,
  kurtosis k_z (3 normal, 3(nu-2)/(nu-4) for t);  ARMA part: mean c/(1-phi), var E[eps^2]/(1-phi^2), ACF_1 = phi;
  OLS on the seasonal regressors recovers gamma;  sigma^2 recursion rebuilt from eps is exact; sigma^2_{T+1}
  depends only on data up to T."""
import sys

import numpy as np

from common import J

T = 500_000
t = np.arange(T)
X = np.column_stack([np.sin(2 * np.pi * t / 12), np.cos(2 * np.pi * t / 12)])
ok = True


def check(name, got, want, tol):
    global ok
    good = abs(got - want) <= tol
    ok &= good
    print(f"  [{'PASS' if good else 'FAIL'}] {name:34s} {got:9.4f}  theory {want:9.4f}  (tol {tol})")


for lab, (om, al, be), dist, nu in (("typical (a .10, b .75), normal", (0.15, 0.10, 0.75), "normal", 6.0),
                                     ("strong  (a .15, b .75), normal", (0.10, 0.15, 0.75), "normal", 6.0),
                                     ("typical (a .10, b .75), t(6)", (0.15, 0.10, 0.75), "t", 6.0)):
    y, eps, sig2 = J.simulate(T, 0.1, [0.5], [], om, al, be, dist=dist, nu=nu, X=X, gamma=[1.0, -0.5], seed=1)
    per = al + be
    kz = 3.0 if dist == "normal" else 3 * (nu - 2) / (nu - 4)
    var_th = om / (1 - per)
    z = eps / np.sqrt(sig2)
    e2 = eps ** 2
    u = y - X @ np.array([1.0, -0.5])
    g = np.linalg.lstsq(np.column_stack([np.ones(T), X]), y, rcond=None)[0]
    s = np.empty(T); s[0] = sig2[0]
    for k in range(1, T):
        s[k] = om + al * eps[k - 1] ** 2 + be * s[k - 1]
    print(lab)
    check("E[eps^2]", e2.mean(), var_th, 0.03)
    if dist == "normal":                       # for t(6) the sample kurtosis has infinite variance
        check("kurtosis(eps)", (eps ** 4).mean() / e2.mean() ** 2, kz * (1 - per ** 2) / (1 - per ** 2 - (kz - 1) * al ** 2), 0.15)
    check("ACF_1(eps^2)", np.corrcoef(e2[1:], e2[:-1])[0, 1], al * (1 - al * be - be ** 2) / (1 - 2 * al * be - be ** 2), 0.015)
    check("shock mean", z.mean(), 0.0, 0.01)
    check("shock variance", z.var(), 1.0, 0.01)
    check("shock kurtosis", (z ** 4).mean(), kz, 0.1 if dist == "normal" else 0.6)
    check("Ljung-Box(12) p-value of z > 0.01", float(J.ljung_box(z, 12)[1] > 0.01), 1.0, 0)
    check("Ljung-Box(12) p-value of z^2 > 0.01", float(J.ljung_box(z ** 2, 12)[1] > 0.01), 1.0, 0)
    check("ARMA part mean", u.mean(), 0.2, 4 * np.sqrt(var_th / (1 - 0.5) ** 2 / T))   # 4 SE of the sample mean
    check("ARMA part variance", u.var(), var_th / 0.75, 0.04)
    check("ARMA part ACF_1", np.corrcoef(u[1:], u[:-1])[0, 1], 0.5, 0.01)
    check("OLS gamma1", g[1], 1.0, 0.01)
    check("OLS gamma2", g[2], -0.5, 0.01)
    check("sigma^2 recursion rebuilt from eps", np.abs(s - sig2).max(), 0.0, 1e-12)
    check("sigma^2_T+1 from data to T only", abs(sig2[-1] - (om + al * eps[-2] ** 2 + be * sig2[-2])), 0.0, 1e-12)
print("SIMULATOR:", "ALL PASS" if ok else "SOME FAILURES")
sys.exit(0 if ok else 1)
