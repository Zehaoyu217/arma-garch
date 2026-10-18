import sys, warnings, numpy as np
import os; sys.path.insert(0, os.environ.get("JAG_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fixed")))
import joint_armax_garch as J
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
warnings.filterwarnings("ignore")
np.set_printoptions(precision=4, suppress=True, linewidth=150)

rng = np.random.default_rng(31); T = 3000
X = np.column_stack([np.sin(2*np.pi*np.arange(T)/12), rng.normal(size=T)])
# DGP: u_t = y_t - X_t b follows ARMA(1,1) with intercept 0.3 (mean 0.3/(1-0.6)=0.75), homoskedastic
y, _, _ = J.simulate(T, 0.3, [0.6], [0.3], 1.0, 0.0, 0.0, dist="normal", X=X, gamma=[1.0, -0.5], seed=31, xreg="regression")
nog = {"alpha": 0.0, "beta": 0.0}            # GARCH off: SARIMAX has constant variance

print("(A) what each package calls the constant")
sx = SARIMAX(y, exog=X, order=(1, 0, 1), trend="c").fit(disp=False)
ar = ARIMA(y, exog=X, order=(1, 0, 1), trend="c").fit()
r = J.fit_joint(y, X=X, p=1, q=1, dist="normal", xreg="regression", fixed=nog)
S, A = dict(zip(sx.model.param_names, sx.params)), dict(zip(ar.param_names, ar.params))
print(f"  SARIMAX trend='c'  : intercept={S['intercept']:.4f}  phi={S['ar.L1']:.4f} theta={S['ma.L1']:.4f} "
      f"b=[{S['x1']:.4f},{S['x2']:.4f}] s2={S['sigma2']:.4f}  LL={sx.llf:.3f}")
print(f"  script regression  : const    ={r['const']:.4f}  phi={r['phi1']:.4f} theta={r['theta1']:.4f} "
      f"b=[{r['gamma1']:.4f},{r['gamma2']:.4f}] s2={r['omega']:.4f}  LL={r.loglik:.3f}")
print(f"  ARIMA trend='c'    : const={A['const']:.4f}  <- the MEAN of u;  script uncond_mean={r.uncond_mean:.4f}, "
      f"SARIMAX intercept/(1-phi)={S['intercept']/(1-S['ar.L1']):.4f}")

print("\n(B) identical parameters -> same residuals? (script params pushed into SARIMAX)")
mod = SARIMAX(y, exog=X, order=(1, 0, 1), trend="c")
par = np.r_[r["const"], r["gamma1"], r["gamma2"], r["phi1"], r["theta1"], r["omega"]]   # SARIMAX order
f = mod.filter(par)
err = f.forecasts_error[0][1:]                 # script conditions on y_0 (p=1)
llo = mod.loglikeobs(par)[1:]
lls = J._loglik_obs(r.params, y, X, 1, 1, "normal", r.init, "regression")
for t0 in (0, 10, 50, 100, 300):
    print(f"  t>={t0:<4d} max|SARIMAX one-step error - script eps|={np.abs(err[t0:] - r.eps[t0:]).max():.1e}"
          f"   max|loglik_t diff|={np.abs(llo[t0:] - lls[t0:]).max():.1e}")

print("\n(C) d=1, y and X in levels: SARIMAX differences inside the state space")
lvl, Xl = np.cumsum(y), np.cumsum(X, axis=0)
for simple in (False, True):
    m1 = SARIMAX(lvl, exog=Xl, order=(1, 1, 1), trend="c", simple_differencing=simple).fit(disp=False)
    P = dict(zip(m1.model.param_names, m1.params))
    print(f"  SARIMAX simple_differencing={simple!s:5}: intercept={P['intercept']:.4f} phi={P['ar.L1']:.4f} "
          f"theta={P['ma.L1']:.4f} b=[{P['x1']:.4f},{P['x2']:.4f}]")
r1 = J.fit_joint(lvl, X=Xl, p=1, q=1, d=1, dist="normal", xreg="regression", fixed=nog)
print(f"  script d=1 regression            : const    ={r1['const']:.4f} phi={r1['phi1']:.4f} "
      f"theta={r1['theta1']:.4f} b=[{r1['gamma1']:.4f},{r1['gamma2']:.4f}]")
