import sys, warnings, numpy as np
import os; sys.path.insert(0, os.environ.get("JAG_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))
import joint_armax_garch as J
warnings.filterwarnings("ignore", category=RuntimeWarning)
np.set_printoptions(precision=5, suppress=True, linewidth=150)
# realistic daily-returns-like process: high persistence
y, _, _ = J.simulate(2500, 0.05, [0.3], [0.2], 0.02, 0.06, 0.93, dist="t", nu=6, seed=3)
base = J.fit_joint(y, p=1, q=1, dist="t")
print("scale=1   std=%.3f  se:" % y.std(), base.se)
for s in (0.1, 0.05, 0.01):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", UserWarning)
        r = J.fit_joint(s * y, p=1, q=1, dist="t")
    unscale = np.r_[1/s, 1, 1, 1/s**2, 1, 1, 1]
    print(f"scale={s:<5} std={s*y.std():.4f} warned={any(issubclass(x.category, UserWarning) for x in w)}")
    print("   params rescaled back:", r.params * unscale, " (base:", base.params, ")")
    print("   robust SE rescaled  :", r.se * unscale)
    print("   hessian SE rescaled :", r.se_hessian * unscale, " base:", base.se_hessian)
    print("   omega=%.2e  Hessian step on omega = 1e-4 -> omega-h %s 0" % (r['omega'], "<" if r['omega'] < 1e-4 else ">="))
