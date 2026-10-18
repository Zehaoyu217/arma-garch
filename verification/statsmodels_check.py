"""xreg="regression" vs statsmodels ARIMA(exog=): same model (regression with ARIMA errors).
GARCH switched off (alpha = beta = 0 fixed, normal errors) so the two are comparable; the
remaining gap is statsmodels' exact (Kalman) likelihood vs this script's conditional one."""
import os, sys, warnings, numpy as np
sys.path.insert(0, os.environ.get("JAG_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fixed")))
import joint_armax_garch as J
from statsmodels.tsa.arima.model import ARIMA
warnings.filterwarnings("ignore")

rng = np.random.default_rng(31)
T = 3000
X = np.column_stack([np.sin(2 * np.pi * np.arange(T) / 12), rng.normal(size=T)])
y, _, _ = J.simulate(T, 0.3, [0.6], [0.3], 1.0, 0.0, 0.0, dist="normal", X=X, gamma=[1.0, -0.5],
                     seed=31, xreg="regression")
nog = {"alpha": 0.0, "beta": 0.0}

print("d=0  y_t = mu + x_t'g + u_t, u ~ ARMA(1,1)   (truth: mu=0.75, phi=0.6, theta=0.3, g=[1,-0.5], s2=1)")
sm = ARIMA(y, exog=X, order=(1, 0, 1), trend="c").fit()
rr = J.fit_joint(y, X=X, p=1, q=1, dist="normal", xreg="regression", fixed=nog)
ra = J.fit_joint(y, X=X, p=1, q=1, dist="normal", xreg="armax", fixed=nog)
P = dict(zip(sm.param_names, sm.params))
print(f"  statsmodels ARIMA exog : mu={P['const']:.4f} phi={P['ar.L1']:.4f} theta={P['ma.L1']:.4f} "
      f"g=[{P['x1']:.4f},{P['x2']:.4f}] s2={P['sigma2']:.4f}")
print(f"  script xreg=regression : mu={rr.uncond_mean:.4f} phi={rr['phi1']:.4f} theta={rr['theta1']:.4f} "
      f"g=[{rr['gamma1']:.4f},{rr['gamma2']:.4f}] s2={rr['omega']:.4f}")
print(f"  script xreg=armax      : mu={ra.uncond_mean:.4f} phi={ra['phi1']:.4f} theta={ra['theta1']:.4f} "
      f"g=[{ra['gamma1']:.4f},{ra['gamma2']:.4f}] s2={ra['omega']:.4f}   <- a different model")
gap = np.max(np.abs(np.r_[rr["phi1"] - P["ar.L1"], rr["theta1"] - P["ma.L1"],
                          rr["gamma1"] - P["x1"], rr["gamma2"] - P["x2"]]))
print(f"  regression vs statsmodels max |diff| (phi, theta, gamma) = {gap:.4f}   "
      f"(statsmodels SE of gamma1 = {np.sqrt(np.diag(sm.cov_params()))[2]:.3f})")

lvl = np.cumsum(y - 0.5)
Xl = np.cumsum(X, axis=0)
sm1 = ARIMA(lvl, exog=Xl, order=(1, 1, 1), trend="t").fit()           # trend='t' = drift when d=1
r1 = J.fit_joint(lvl, X=Xl, p=1, q=1, d=1, dist="normal", xreg="regression", fixed=nog)
P1 = dict(zip(sm1.param_names, sm1.params))
print("d=1  levels with X in levels (both difference X):")
print(f"  statsmodels trend='t'  : drift={P1['x1']:.4f} phi={P1['ar.L1']:.4f} theta={P1['ma.L1']:.4f} "
      f"g=[{P1['x2']:.4f},{P1['x3']:.4f}]")
print(f"  script d=1 regression  : drift={r1.uncond_mean:.4f} phi={r1['phi1']:.4f} theta={r1['theta1']:.4f} "
      f"g=[{r1['gamma1']:.4f},{r1['gamma2']:.4f}]")
Xf = Xl[-1] + np.cumsum(np.column_stack([np.sin(2 * np.pi * np.arange(T, T + 5) / 12), rng.normal(size=5)]), axis=0)
f_sm = sm1.get_forecast(5, exog=Xf).predicted_mean
f_j = J.forecast(r1, 5, X_future=Xf, n_sims=0)["mean"]
print(f"  5-step level forecasts  statsmodels={np.round(f_sm, 3)}\n                          script     ={np.round(f_j, 3)}")
