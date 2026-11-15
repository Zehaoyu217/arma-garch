"""Settle the restricted bootstrap's size at T = 120 on FRESH seeds (E11's 300 series were reused by every variant).
Part 1: LR null distribution, true DGP vs the bootstrap DGP with the true parameters, 6000 draws each.
Part 2: rejection rate of the shipped garch_test (B = 99, residual resampling, estimated null), 1000 new series.
usage: python e11_size_check.py"""
import json
import warnings
from multiprocessing import Pool

import numpy as np
from scipy import stats

import common
from common import _core
from sarimax_garch import garch_test
from sarimax_garch.compare import _simulate_null

import e11_valid_inference as e
from e11_lr_paths import fit_lr


def draw_pair(seed):
    rng = np.random.default_rng(seed)
    y, X, _ = e.series("AR1", "none", 120, rng)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        np.seterr(all="ignore")
        a = fit_lr(y, X)
        rt = _core.fit_joint(y, X=X, p=1, q=0, d=0, dist="normal", compute_cov=False,
                             fixed=dict(const=0.1, phi1=0.5, gamma1=0.5, omega=1.0, alpha=0.0, beta=0.0))
        yc = _simulate_null(rt, np.random.default_rng(seed + 13).standard_normal((1, 320)))[0]
        c = fit_lr(yc, X)
    return a, c


def size_one(seed):
    rng = np.random.default_rng(seed)
    y, X, order = e.series("AR1", "none", 120, rng)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        np.seterr(all="ignore")
        r = garch_test(y, exog=X, order=order, trend="c", B=99, seed=seed)
    return r["p_bootstrap"] <= 0.05, r["p_chi2"] < 0.05


if __name__ == "__main__":
    with Pool(12) as pool:
        ac = np.array(pool.map(draw_pair, range(7_000_000, 7_006_000)))
        a, c = ac[:, 0], ac[:, 1]
        ks = stats.ks_2samp(a, c)
        print("part 1 (6000 each):  " + "  ".join(f"q{int(q * 100)} {np.quantile(a, q):.2f}/{np.quantile(c, q):.2f}"
                                               for q in (0.5, 0.9, 0.95, 0.99))
              + f"  mean {a.mean():.3f}/{c.mean():.3f}  KS p {ks.pvalue:.3f}   (true DGP / bootstrap DGP)")
        rows = np.array(pool.map(size_one, range(8_000_000, 8_001_000)))
        boot, chi2 = rows[:, 0].mean(), rows[:, 1].mean()
        se = np.sqrt(0.05 * 0.95 / len(rows))
        print(f"part 2 (1000 fresh series): restricted bootstrap size {boot:.3f} (MC SE {se:.3f}), chi2(2) size {chi2:.3f}")
    with open(f"{common.OUT}/e11_size_check.json", "w") as fh:
        json.dump(dict(ks_p=float(ks.pvalue), q95=[float(np.quantile(a, .95)), float(np.quantile(c, .95))],
                       boot_size=float(boot), chi2_size=float(chi2), n=int(len(rows))), fh, indent=1)
