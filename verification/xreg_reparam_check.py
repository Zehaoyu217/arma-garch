"""Case E (fitted by rugarch): rugarch's regression-with-ARMA-errors == this script's ARMAX with [x_t, x_{t-1}] and gamma=[g, -phi*g]."""
import os, sys, json, warnings, numpy as np
sys.path.insert(0, os.environ.get("JAG_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))
import joint_armax_garch as J
import inspect
MODE = ("backcast", "armax") if "xreg" in inspect.signature(J._filter).parameters else ()   # v2.1 is ARMAX
warnings.filterwarnings("ignore", category=RuntimeWarning)
CMP = os.environ.get("CMP_DIR", "cmp")
cf = json.load(open(f"{CMP}/rugarch_results.json"))["E_arma11_X_t"]["coef"]
y = np.loadtxt(f"{CMP}/E_arma11_X_t_y.csv"); X = np.loadtxt(f"{CMP}/E_arma11_X_t_X.csv", delimiter=",")
phi = cf["ar1"]; g = np.array([cf["mxreg1"], cf["mxreg2"]])
Xaug = np.column_stack([X, np.vstack([np.zeros((1, 2)), X[:-1]])])
th = np.r_[cf["mu"] * (1 - phi), phi, cf["ma1"], g, -phi * g, cf["omega"], cf["alpha1"], cf["beta1"], cf["shape"]]
eps, s2, _ = J._filter(th, y, Xaug, 1, 1, "t", *MODE); ll = J._loglik_obs(th, y, Xaug, 1, 1, "t", *MODE)
r = np.genfromtxt(f"{CMP}/E_rug_filter_at_rugMLE.csv", delimiter=",", skip_header=1)[1:]
d = np.abs(np.column_stack([eps, np.sqrt(s2), ll]) - r)[300:]
print(f"E reparametrised, t>=300: max|eps|={d[:,0].max():.1e} max|sigma|={d[:,1].max():.1e} max|loglik_t|={d[:,2].max():.1e}")
