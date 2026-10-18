"""Monte Carlo: bias and 95% CI coverage of the script's robust and Hessian SEs."""
import sys, warnings, time, numpy as np
import os; sys.path.insert(0, os.environ.get("JAG_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))
import joint_armax_garch as J
warnings.filterwarnings("ignore")
R, T = 400, 2000
truth = np.r_[0.2, 0.5, 0.3, 0.05, 0.08, 0.88, 6.0]
est, se, seh, conv = [], [], [], []
t0 = time.time()
for r in range(R):
    y, _, _ = J.simulate(T, 0.2, [0.5], [0.3], 0.05, 0.08, 0.88, dist="t", nu=6.0, seed=1000 + r)
    f = J.fit_joint(y, p=1, q=1, dist="t")
    est.append(f.params); se.append(f.se); seh.append(f.se_hessian); conv.append(f.converged)
est, se, seh = map(np.array, (est, se, seh))
names = ["const", "phi1", "theta1", "omega", "alpha", "beta", "nu"]
sd = est.std(0, ddof=1)
print(f"R={R} reps, T={T}, ARMA(1,1)-GARCH(1,1)-t(6); {time.time()-t0:.0f}s; converged={np.mean(conv):.3f}")
print(f"{'param':>7} {'truth':>7} {'mean est':>9} {'bias/SD':>8} {'MC SD':>8} {'mean robSE':>10} {'robSE/SD':>9} {'cover95 rob':>11} {'mean hesSE':>10} {'cover95 hes':>11}")
for i, n in enumerate(names):
    cr = np.mean(np.abs(est[:, i] - truth[i]) < 1.96 * se[:, i])
    ch = np.mean(np.abs(est[:, i] - truth[i]) < 1.96 * seh[:, i])
    print(f"{n:>7} {truth[i]:7.3f} {est[:, i].mean():9.4f} {(est[:, i].mean()-truth[i])/sd[i]:8.3f} {sd[i]:8.4f} "
          f"{np.median(se[:, i]):10.4f} {np.median(se[:, i])/sd[i]:9.3f} {cr:11.3f} {np.median(seh[:, i]):10.4f} {ch:11.3f}")
print(f"(binomial MC error on coverage at R={R}: +/-{1.96*np.sqrt(.95*.05/R):.3f})")
