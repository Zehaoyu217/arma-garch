"""E5 -- does converged=False mean a bad fit?

fit_joint sets converged = L-BFGS-B's success flag, although the answer it returns may come from the
Nelder-Mead polish or the SLSQP step that follow.  For fits flagged False (and a sample flagged True),
re-fit from 8 perturbed starts and from the returned optimum, and compare the best log-likelihood found.
Design: E2's D2 (AR(1)-GARCH(1,1)-t5, T = 250), where the flag is False most often.
"""
import json
import warnings
from multiprocessing import Pool

import numpy as np

import common
from common import _core

TRUE = dict(c=0.05, phi=0.5, omega=0.05, alpha=0.08, beta=0.90)


def one(seed):
    rng = np.random.default_rng(seed)
    z = common.draw_z(rng, 250 + 500, "t", nu=5.0)
    y, *_ = common.simulate(250, **TRUE, z=z)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = _core.fit_joint(y, p=1, q=0, dist="t", compute_cov=False)
        best = r.loglik
        for j in range(8):
            st = r.params * np.exp(rng.normal(0, 0.15, r.params.size))
            st[3:5] = np.clip(st[3:5], 0.001, 0.98)
            if st[3] + st[4] >= 0.999:
                st[3:5] *= 0.99 / (st[3] + st[4])
            st[-1] = np.clip(st[-1], 2.5, 150)
            rr = _core.fit_joint(y, p=1, q=0, dist="t", start=st, compute_cov=False)
            best = max(best, rr.loglik)
    return dict(converged=bool(r.converged), gap=float(best - r.loglik))


def main():
    with Pool(12) as pool:
        rows = pool.map(one, range(70_000, 70_400))
    conv = np.array([r["converged"] for r in rows])
    gap = np.array([r["gap"] for r in rows])
    out = dict(n=len(rows), flagged_not_converged=int((~conv).sum()),
               gap_flagged=dict(max=float(gap[~conv].max()) if (~conv).any() else None,
                                share_below_1e_4=float(np.mean(gap[~conv] < 1e-4)) if (~conv).any() else None),
               gap_converged=dict(max=float(gap[conv].max()), share_below_1e_4=float(np.mean(gap[conv] < 1e-4))),
               all_fits_gap_over=dict((str(th), int(np.sum(gap > th))) for th in (1e-4, 1e-2, 0.1, 0.5, 1.0)))
    print(json.dumps(out, indent=1))
    with open(f"{common.OUT}/e5_converged.json", "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
