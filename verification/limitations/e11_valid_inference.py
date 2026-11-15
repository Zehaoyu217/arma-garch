"""E11 -- when the textbook p-values are not valid: the restricted bootstrap for the GARCH test, and
post-selection inference for the mean.

(a) Testing H0: no GARCH.  alpha = 0 is on the boundary and beta is unidentified, so the LR has no chi2 law.
    Compared at 5%: the chi2(2) cut-off (5.99), ARCH-LM(4) on SARIMAX's residuals, and garch_test(): the
    RESTRICTED bootstrap (Cavaliere, Nielsen & Rahbek 2017 -- resample from the model fitted under H0,
    B = 99 residual-resampling draws).  DGP: y = 0.5 x + u, u AR(1) phi .5 (or MA(1) theta .9: the start-up
    case of E6), GARCH none / typical (alpha .10, beta .75) / persistent (alpha .08, beta .90).
(b) Post-selection inference.  The workflow picks GARCH when the LR test rejects, then reports the chosen
    model's p-values.  Does that selection step distort the 95% intervals for the slope g and for phi?
    Unconditional coverage, and coverage conditional on each selection outcome (selective inference asks for
    the latter: valid "given that the question was asked", Fithian, Sun & Taylor), for: always SARIMAX opg,
    always SARIMAX robust, always GARCH (robust), and select-then-report (GARCH robust if selected, else
    SARIMAX opg = the defaults; or else SARIMAX robust).
"""
import json
import sys
import warnings
from multiprocessing import Pool

import numpy as np
from scipy import stats

import common
from sarimax_garch import SARIMAXGARCH, arch_lm_test, garch_test

GARCHS = {"none": (1.0, 0.0, 0.0), "typical": (0.15, 0.10, 0.75), "persistent": (0.02, 0.08, 0.90)}


def series(mean, gk, T, rng):
    x = common.ar1_exog(T, 0.7, rng)
    om, al, be = GARCHS[gk]
    phi, th = (0.5, 0.0) if mean == "AR1" else (0.0, 0.9)
    y, *_ = common.simulate(T, 0.1, phi, om, al, be, rng.standard_normal(T + 500), X=x[:, None], g=[0.5], theta=th)
    return y, x[:, None], ((1, 0, 0) if mean == "AR1" else (0, 0, 1))


def part_a(args):
    mean, gk, T, seed = args
    rng = np.random.default_rng(seed)
    y, X, order = series(mean, gk, T, rng)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        np.seterr(all="ignore")
        sm = SARIMAXGARCH(y, exog=X, order=order, trend="c").fit()
        lm_p = arch_lm_test(np.asarray(sm.resid), 4)[1]
        r = garch_test(y, exog=X, order=order, trend="c", B=99, seed=seed)
    # Monte Carlo test rule: with B = 99, reject when p* <= 0.05 (exact level 5% since 0.05 * (B + 1) is an integer)
    return dict(chi2=r["p_chi2"] < 0.05, boot=r["p_bootstrap"] <= 0.05, lm=lm_p < 0.05)


def part_b(args):
    gk, T, seed = args
    rng = np.random.default_rng(seed)
    y, X, order = series("AR1", gk, T, rng)
    tv = {"x1": 0.5, "ar.L1": 0.5}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        np.seterr(all="ignore")
        sm = SARIMAXGARCH(y, exog=X, order=order, trend="c").fit()
        mod = SARIMAXGARCH(y, exog=X, order=order, trend="c", garch=(1, 1))
        g0 = mod.fit(fixed_params={"alpha.L1": 0.0, "beta.L1": 0.0})
        g = mod.fit()
    names = list(tv)
    idx = [list(sm.model.param_names).index(n) for n in names]
    p_sm = np.asarray(sm.params)[idx]
    se_opg = np.sqrt(np.diag(np.asarray(sm.cov_params_opg)))[idx]
    se_rob = np.sqrt(np.diag(np.asarray(sm.cov_params_robust)))[idx]
    p_g, se_g = g.params[names].values, g.bse[names].values
    t = np.array([tv[n] for n in names])
    cov = lambda p_, se_: (np.abs(p_ - t) <= 1.96 * se_).tolist()
    sel = bool(2 * (g.llf - g0.llf) > 5.99)                      # compare()'s pre-test: GARCH if the LR rejects
    return dict(sel=sel, sm_opg=cov(p_sm, se_opg), sm_rob=cov(p_sm, se_rob), garch=cov(p_g, se_g),
                post_default=cov(p_g, se_g) if sel else cov(p_sm, se_opg),
                post_robust=cov(p_g, se_g) if sel else cov(p_sm, se_rob))


def main():
    R_a = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    R_b = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    res = dict(a={}, b={})
    designs_a = ([("AR1", gk, T) for gk in GARCHS for T in (120, 250)] + [("MA1 theta=.9", "none", T) for T in (250, 1000)])
    with Pool(12) as pool:
        for i, (mean, gk, T) in enumerate(designs_a):
            R = R_a if T < 1000 else max(100, R_a // 2)
            rows = pool.map(part_a, [(mean, gk, T, 3_000_000 + 10_000 * i + r) for r in range(R)])
            key = f"{mean} | GARCH {gk} | T={T}"
            res["a"][key] = {k: round(float(np.mean([r[k] for r in rows])), 3) for k in ("chi2", "boot", "lm")}
            res["a"][key]["R"] = R
            print(f"(a) {key:34} reject at 5%: chi2(2) {res['a'][key]['chi2']:.3f}  restricted bootstrap "
                  f"{res['a'][key]['boot']:.3f}  ARCH-LM(4) {res['a'][key]['lm']:.3f}  (R={R})", flush=True)
        for j, (gk, T) in enumerate([(gk, T) for gk in GARCHS for T in (120, 250)]):
            rows = pool.map(part_b, [(gk, T, 4_000_000 + 10_000 * j + r) for r in range(R_b)])
            sel = np.array([r["sel"] for r in rows])
            key = f"AR1 | GARCH {gk} | T={T}"
            out = dict(R=R_b, share_garch_selected=round(float(sel.mean()), 3), coverage={}, conditional={})
            for m in ("sm_opg", "sm_rob", "garch", "post_default", "post_robust"):
                c = np.array([r[m] for r in rows], float)          # (R, 2): x1, ar.L1
                out["coverage"][m] = dict(zip(["x1", "ar.L1"], c.mean(0).round(3).tolist()))
                out["conditional"][m] = {
                    "given GARCH selected": dict(zip(["x1", "ar.L1"], (c[sel].mean(0).round(3).tolist() if sel.any() else [None, None]))),
                    "given SARIMAX selected": dict(zip(["x1", "ar.L1"], (c[~sel].mean(0).round(3).tolist() if (~sel).any() else [None, None])))}
            res["b"][key] = out
            print(f"(b) {key:28} GARCH selected {out['share_garch_selected']:.2f} | coverage x1 / ar.L1: "
                  + " | ".join(f"{m} {out['coverage'][m]['x1']:.3f}/{out['coverage'][m]['ar.L1']:.3f}" for m in out["coverage"])
                  + f" || post_default given GARCH {out['conditional']['post_default']['given GARCH selected']} "
                  f"given SARIMAX {out['conditional']['post_default']['given SARIMAX selected']}", flush=True)
    with open(f"{common.OUT}/e11_valid_inference.json", "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
