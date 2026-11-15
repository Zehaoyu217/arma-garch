"""E13 -- can the mean model, or the shock distribution, manufacture GARCH?  (identification and the residuals)

E12 showed the low power of GARCH tests is GARCH's own: beta is not identified when alpha = 0, and only weakly
when alpha is small, even with the true shocks.  This study asks whether identification problems of the MEAN
model reach the residuals that GARCH is tested on, and whether fat tails alone fake GARCH.

(a) ARMA identification.  One shock series per replication drives every design (paired):
  oracle               the true shocks (no mean model)
  white noise          y = 0.1 + eps, fitted with a constant only (correct)
  common factor        the same y fitted as ARMA(1,1): phi and theta are not identified (the ridge theta = -phi);
                       only the net filter (1 - phi L) / (1 + theta L) = 1 is
  near-cancelling      (1 - .5L)(y - .1) = (1 - .4L) eps, fitted as ARMA(1,1): weakly identified
  MA .9                y = 0.1 + eps_t + .9 eps_{t-1}, fitted as MA(1) (E12's design)
  MA .9, eps_0 = 0     the same with the pre-sample shock set to zero, so the conditional likelihood's start is exact
  over-differenced     y = 0.1 + eps fitted as ARIMA(0,1,1) without drift: the true MA root is exactly 1
  GARCH none | typical; T = 120, 240, 480.  Tests at 5%: LR one-stage (engine), LR two-stage (exact SARIMAX
  residuals), ARCH-LM(4).  Also: the spread of the AR/MA estimates and how far the residuals are from the shocks.
(b) Non-fundamental shocks (the residuals themselves not identified): y = 0.1 + eps_t + 2 eps_{t-1} has its MA root
  inside the unit circle; its invertible twin y = 0.1 + eps_t + .5 eps_{t-1} has the same autocorrelations.  Every
  fit returns the invertible MA(1), theta-hat ~ .5; for theta = 2 its residuals are an all-pass filter of the shocks:
  uncorrelated, but not independent unless the shocks are normal (Breidt, Davis & Trindade 2001).  No GARCH.
(c) Fat tails: AR(1) with shocks normal | t(8) | t(5) | contaminated normal | Hansen skew-t (nu 8, lam -.4), GARCH
  none (size) | typical | persistent (power).  Under iid shocks the Gaussian quasi-LR of a variance parameter is
  (kurtosis - 1) / 2 times its normal-theory value, so compare()'s Gaussian LR is checked against: the LR divided by
  (kurt-hat - 1) / 2 of the null residuals, the LR divided by max(1, (kurt-hat - 1) / 2) of the GARCH fit's
  standardised residuals (GARCH itself fattens the unconditional tails, which the null residuals would count),
  the t LR, ARCH-LM(4), and the restricted bootstrap (garch_test, B = 99) in two cells.
(d) Block diagonality: long AR(1)-GARCH and MA(1)-GARCH series (T = 50,000), shocks normal | t(8) | Hansen skew-t
  (nu 8, lam -.4); the robust-sandwich correlations between the mean and the GARCH estimates.
(e) compare() end to end: how often it recommends GARCH under the old rule (Gaussian LR), the new default (t LR) and
  the normal-only rule (kurtosis-scaled Gaussian LR); AR(1), shocks normal | t(8) | t(5) | contaminated normal,
  GARCH none | typical, T = 120, 240, 480.
(f) Non-fundamental MA: holdout log score of MA(1)-GARCH-t against MA(1)-t (does the GARCH layer still help?).
Usage: python e13_identification.py [parts=abcdef] [R=400] [workers=12] [R_boot=200]
"""
import json
import sys
import warnings
from multiprocessing import Pool

import numpy as np

import common
from common import _core
from sarimax_garch import SARIMAXGARCH, arch_lm_test, compare, dm_test, garch_test, holdout_logscore

GARCHS = {"none": (1.0, 0.0, 0.0), "typical": (0.15, 0.10, 0.75), "persistent": (0.02, 0.08, 0.90)}
A_GARCHS = ("none", "typical")
TS = (120, 240, 480)
BURN = 600
CRIT = 5.99          # chi2(2) 5% cut-off, as in E6 / E10 / E12

A_DESIGNS = {   # name: (series, engine (p, q, d, fixed), SARIMAX (order, trend))
    "white noise, fitted as constant": ("wn", (0, 0, 0, None), ((0, 0, 0), "c")),
    "white noise, fitted as ARMA(1,1) (common factor)": ("wn", (1, 1, 0, None), ((1, 0, 1), "c")),
    "ARMA(1,1) .5 / -.4 (near-cancelling)": ("nc", (1, 1, 0, None), ((1, 0, 1), "c")),
    "MA(1) .9": ("ma9", (0, 1, 0, None), ((0, 0, 1), "c")),
    "MA(1) .9, pre-sample shock 0": ("ma9z", (0, 1, 0, None), ((0, 0, 1), "c")),
    "white noise, fitted as ARIMA(0,1,1) (over-differenced)": ("wn", (0, 1, 1, {"const": 0.0}), ((0, 1, 1), None)),
}
B_DESIGNS = {"MA(1) theta = 2, normal": (2.0, "normal", None), "MA(1) theta = 2, t(8)": (2.0, "t", 8.0),
             "MA(1) theta = 2, t(5)": (2.0, "t", 5.0), "MA(1) theta = .5, t(5) (invertible twin)": (0.5, "t", 5.0)}
C_SHOCKS = {"normal": ("normal", None), "t(8)": ("t", 8.0), "t(5)": ("t", 5.0),
            "contaminated normal": ("cn", None), "skew-t (nu 8, lam -.4)": ("skewt", 8.0)}
C_GARCHS = ("none", "typical", "persistent")
BOOT_CELLS = (("t(5)", "none", 240), ("t(5)", "typical", 240), ("contaminated normal", "none", 240),
              ("skew-t (nu 8, lam -.4)", "none", 240))


def garch_from_z(z, gk):
    om, al, be = GARCHS[gk]
    eps = np.empty(len(z))
    s2 = om / (1 - al - be)
    for t in range(len(z)):
        if t:
            s2 = om + al * eps[t - 1] ** 2 + be * s2
        eps[t] = np.sqrt(s2) * z[t]
    return eps


def draw(rng, n, zdist="normal", nu=None):
    """Standardised iid shocks; "cn" = contaminated normal, N(0, 1) w.p. .9 and N(0, 9) w.p. .1 (kurtosis 8.3)."""
    if zdist == "cn":
        return rng.standard_normal(n) * np.where(rng.random(n) < 0.1, 3.0, 1.0) / np.sqrt(1.8)
    if zdist == "skewt":
        return common.draw_z(rng, n, "skewt", nu, -0.4)
    return common.draw_z(rng, n, zdist, nu if nu else 6.0)


def shocks(rng, n, gk, zdist="normal", nu=None):
    return garch_from_z(draw(rng, n, zdist, nu), gk)


def fit_pair(y, p, q, d=0, dist="normal", fixed=None):
    """One-stage: the engine's joint GARCH fit and its constant-variance null; LR = 2 (llf_1 - llf_0)."""
    fx = dict(fixed or {})
    g = _core.fit_joint(y, p=p, q=q, d=d, dist=dist, fixed=fx or None, compute_cov=False)
    g0 = _core.fit_joint(y, p=p, q=q, d=d, dist=dist, fixed={**fx, "alpha": 0.0, "beta": 0.0}, compute_cov=False)
    return max(0.0, 2 * (g.loglik - g0.loglik)), g, g0


def sarimax_resid(y, order, trend):
    """SARIMAX (exact Kalman likelihood) one-step errors after the burn, on the scale of sigma (as in E12)."""
    sm = SARIMAXGARCH(y, order=order, trend=trend).fit()
    burn = int(sm.loglikelihood_burn)
    r = np.asarray(sm.standardized_forecasts_error)[0, burn:] * np.sqrt(sm.params["sigma2"])
    return sm, r[np.isfinite(r)]


def two_stage(r):
    return fit_pair(r, 0, 0, fixed={"const": 0.0})[0]


def kurt(g0):
    z = g0.eps / np.sqrt(g0.sig2)
    return float(np.mean(z ** 4) / np.mean(z ** 2) ** 2)


# ---------------------------------------------------------------------------------------------------------------- (a)
def build_a(e, n, T):
    nc = np.zeros(n)
    for t in range(1, n):
        nc[t] = 0.5 * nc[t - 1] + e[t] - 0.4 * e[t - 1]
    ma9 = 0.1 + e.copy()
    ma9[1:] += 0.9 * e[:-1]
    ma9z = ma9.copy()
    ma9z[n - T] -= 0.9 * e[n - T - 1]            # the first in-sample value without the pre-sample shock
    return {k: v[-T:] for k, v in {"wn": 0.1 + e, "nc": 0.1 + nc, "ma9": ma9, "ma9z": ma9z}.items()}


def one_a(args):
    gk, T, seed = args
    rng = np.random.default_rng(seed)
    n = T + BURN
    e = shocks(rng, n, gk)
    ys, et = build_a(e, n, T), e[-T:]
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        np.seterr(all="ignore")
        lr_o = two_stage(et)
        out["oracle (true shocks)"] = dict(lr1=lr_o, lr2=lr_o, lm=arch_lm_test(et, 4)[1])
        for name, (key, (p, q, d, fx), (order, trend)) in A_DESIGNS.items():
            y = ys[key]
            lr1, g, g0 = fit_pair(y, p, q, d, fixed=fx)
            sm, r = sarimax_resid(y, order, trend)
            truth = et[d + p:]
            o = dict(lr1=lr1, lr2=two_stage(r), lm=arch_lm_test(r, 4)[1],
                     err=float(np.sqrt(np.mean((g0.eps - truth) ** 2)) / np.std(truth)),
                     err10=float(np.sqrt(np.mean((g0.eps[10:] - truth[10:]) ** 2)) / np.std(truth)))
            if p:
                o["phi"] = float(g0["phi1"])
            if q:
                o["theta"] = float(g0["theta1"])
            out[name] = o
    return out


def summarise_a(rows):
    s = {}
    for m in rows[0]:
        v = [r[m] for r in rows]
        d = {"LR one-stage": np.mean([x["lr1"] > CRIT for x in v]), "LR two-stage": np.mean([x["lr2"] > CRIT for x in v]),
             "ARCH-LM(4)": np.mean([x["lm"] < 0.05 for x in v])}
        for k, lab in (("err", "resid error / sd"), ("err10", "resid error / sd, after 10 obs")):
            if k in v[0]:
                d[lab] = np.median([x[k] for x in v])
        for k in ("phi", "theta"):
            if k in v[0]:
                d[f"{k}-hat 5/50/95%"] = np.percentile([x[k] for x in v], [5, 50, 95]).round(3).tolist()
        s[m] = {k: (round(float(x), 3) if np.isscalar(x) else x) for k, x in d.items()}
    return s


# ---------------------------------------------------------------------------------------------------------------- (b)
def one_b(args):
    name, T, seed = args
    th, zd, nu = B_DESIGNS[name]
    rng = np.random.default_rng(seed)
    n = T + BURN
    e = common.draw_z(rng, n, zd, nu if nu else 6.0)
    y = 0.1 + e.copy()
    y[1:] += th * e[:-1]
    y, et = y[-T:], e[-T:]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        np.seterr(all="ignore")
        lr_n, g, g0 = fit_pair(y, 0, 1)
        lr_t = fit_pair(y, 0, 1, dist="t")[0]
        sm, r = sarimax_resid(y, (0, 0, 1), "c")
        r2 = r ** 2 - np.mean(r ** 2)
        return dict(lm_oracle=arch_lm_test(et, 4)[1], lm=arch_lm_test(r, 4)[1], lr_normal=lr_n,
                    lr_scaled=lr_n / ((kurt(g0) - 1) / 2), lr_t=lr_t, theta_hat=float(sm.params["ma.L1"]),
                    acf_r2=float(np.sum(r2[1:] * r2[:-1]) / np.sum(r2 ** 2)))


def summarise_b(rows):
    return {"ARCH-LM(4), true shocks": round(float(np.mean([r["lm_oracle"] < 0.05 for r in rows])), 3),
            "ARCH-LM(4), residuals": round(float(np.mean([r["lm"] < 0.05 for r in rows])), 3),
            "LR one-stage, Gaussian": round(float(np.mean([r["lr_normal"] > CRIT for r in rows])), 3),
            "LR one-stage, Gaussian / ((kurt - 1)/2)": round(float(np.mean([r["lr_scaled"] > CRIT for r in rows])), 3),
            "LR one-stage, t": round(float(np.mean([r["lr_t"] > CRIT for r in rows])), 3),
            "theta-hat median": round(float(np.median([r["theta_hat"] for r in rows])), 3),
            "lag-1 autocorrelation of squared residuals, median": round(float(np.median([r["acf_r2"] for r in rows])), 3)}


# ---------------------------------------------------------------------------------------------------------------- (c)
def one_c(args):
    sk, gk, T, seed, boot = args
    zd, nu = C_SHOCKS[sk]
    rng = np.random.default_rng(seed)
    n = T + BURN
    e = shocks(rng, n, gk, zd, nu)
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = 0.1 + 0.5 * y[t - 1] + e[t]
    y = y[-T:]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        np.seterr(all="ignore")
        lr_n, g, g0 = fit_pair(y, 1, 0)
        lr_t, _, gt0 = fit_pair(y, 1, 0, dist="t")
        sm, r = sarimax_resid(y, (1, 0, 0), "c")
        k, k1 = kurt(g0), kurt(g)
        o = dict(lr_normal=lr_n, kurt=k, kurt1=k1, lr_scaled=lr_n / ((k - 1) / 2),
                 lr_scaled1=lr_n / max(1.0, (k1 - 1) / 2), lr_t=lr_t, lm=arch_lm_test(r, 4)[1], nu0=float(gt0["nu"]))
        if boot:
            o["p_boot"] = garch_test(y, order=(1, 0, 0), trend="c", dist="normal", B=99, seed=seed)["p_bootstrap"]
    return o


def summarise_c(rows):
    s = {"LR Gaussian (compare() default)": np.mean([r["lr_normal"] > CRIT for r in rows]),
         "LR Gaussian / ((kurt - 1)/2), null residuals": np.mean([r["lr_scaled"] > CRIT for r in rows]),
         "LR Gaussian / max(1, (kurt - 1)/2), GARCH std. residuals": np.mean([r["lr_scaled1"] > CRIT for r in rows]),
         "LR t": np.mean([r["lr_t"] > CRIT for r in rows]),
         "ARCH-LM(4)": np.mean([r["lm"] < 0.05 for r in rows]),
         "kurtosis of null residuals, median": np.median([r["kurt"] for r in rows]),
         "kurtosis of GARCH std. residuals, median": np.median([r["kurt1"] for r in rows]),
         "nu-hat of the constant-variance t fit, median": np.median([r["nu0"] for r in rows])}
    if "p_boot" in rows[0]:
        s["restricted bootstrap LR (B = 99, p <= .05)"] = np.mean([r["p_boot"] <= 0.05 for r in rows])
    return {k: round(float(v), 3) for k, v in s.items()}


# ---------------------------------------------------------------------------------------------------------------- (d)
def one_d(args):
    model, sk, seed = args
    zd, nu, lam = {"normal": ("normal", 6.0, 0.0), "t(8)": ("t", 8.0, 0.0),
                   "skew-t (nu 8, lam -.4)": ("skewt", 8.0, -0.4)}[sk]
    rng = np.random.default_rng(9_000_000 + seed)
    T = 50_000
    n = T + BURN
    e = garch_from_z(common.draw_z(rng, n, zd, nu, lam), "typical")
    if model == "AR(1)":
        y = np.zeros(n)
        for t in range(1, n):
            y[t] = 0.1 + 0.5 * y[t - 1] + e[t]
        p, q, m = 1, 0, "phi1"
    else:
        y = 0.1 + e.copy()
        y[1:] += 0.5 * e[:-1]
        p, q, m = 0, 1, "theta1"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        np.seterr(all="ignore")
        g = _core.fit_joint(y[-T:], p=p, q=q, dist="normal", compute_cov=True)
    s = np.sqrt(np.diag(g.cov))
    R = g.cov / np.outer(s, s)
    ix = g.names.index
    return {f"corr({a}, {b})": round(float(R[ix(a), ix(b)]), 3)
            for a in ("const", m) for b in ("alpha", "beta")}


# ---------------------------------------------------------------------------------------------------------------- (e)
E_SHOCKS = ("normal", "t(8)", "t(5)", "contaminated normal")
E_TS = (120, 240, 480)


def one_e(args):
    """compare() end to end on an AR(1): does it recommend GARCH?  Three decision rules from one call:
    old = Gaussian LR (package <= 0.3.0), new = t LR (the default dists), normal-only = kurtosis-scaled Gaussian LR
    (what compare(dists=("normal",)) decides by); all three then demote a boundary GARCH fit to SARIMAX (L5)."""
    sk, gk, T, seed = args
    zd, nu = C_SHOCKS[sk]
    rng = np.random.default_rng(seed)
    n = T + BURN
    e = shocks(rng, n, gk, zd, nu)
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = 0.1 + 0.5 * y[t - 1] + e[t]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        np.seterr(all="ignore")
        res = compare(y[-T:], order=(1, 0, 0), trend="c")
    p = res.tests["p_value"]
    boundary = any(f["code"] == "L5" and "boundary" in f["message"] for f in res.flags)
    return dict(old=bool(p["LR GARCH vs constant variance (normal, within engine)"] < 0.05 and not boundary),
                new=res.recommendation["model"] != "SARIMAX",
                normal_only=bool(p["LR GARCH vs constant variance (normal, kurtosis-scaled, within engine)"] < 0.05
                                 and not boundary),
                raw_gaussian=bool(p["LR GARCH vs constant variance (normal, within engine)"] < 0.05))


# ---------------------------------------------------------------------------------------------------------------- (f)
F_DESIGNS = ("MA(1) theta = 2, normal", "MA(1) theta = 2, t(5)", "MA(1) theta = .5, t(5) (invertible twin)")


def one_f(args):
    """Non-fundamental MA: does the GARCH layer still help the one-step density forecast?  Holdout log score (last
    25%, parameters fitted on the rest) of MA(1)-GARCH(1,1)-t minus MA(1)-t with constant variance."""
    name, T, seed = args
    th, zd, nu = B_DESIGNS[name]
    rng = np.random.default_rng(seed)
    e = common.draw_z(rng, T + BURN, zd, nu if nu else 6.0)
    y = 0.1 + e.copy()
    y[1:] += th * e[:-1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        np.seterr(all="ignore")
        kw = dict(order=(0, 0, 1), trend="c", dist="t", holdout=0.25)
        d = holdout_logscore(y[-T:], garch=(1, 1), **kw) - holdout_logscore(y[-T:], garch=None, **kw)
    return dict(gain=float(np.mean(d)), dm_p=float(dm_test(d)[1]))


def main():
    parts = sys.argv[1] if len(sys.argv) > 1 else "abcd"
    R = int(sys.argv[2]) if len(sys.argv) > 2 else 400
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 12
    R_boot = int(sys.argv[4]) if len(sys.argv) > 4 else 200
    with Pool(workers) as pool:
        if "a" in parts:
            res = {}
            for i, gk in enumerate(A_GARCHS):
                for j, T in enumerate(TS):
                    rows = pool.map(one_a, [(gk, T, 7_000_000 + 100_000 * i + 10_000 * j + r) for r in range(R)])
                    key = f"GARCH {gk} | T={T}"
                    res[key] = summarise_a(rows)
                    print(f"(a) {key}: " + " | ".join(f"{m.split(' (')[0]} {v['LR one-stage']:.3f}/{v['LR two-stage']:.3f}"
                                                      f"/{v['ARCH-LM(4)']:.3f}" for m, v in res[key].items()), flush=True)
            json.dump(dict(R=R, cells=res), open(f"{common.OUT}/e13_a.json", "w"), indent=1)
        if "b" in parts:
            res = {}
            for i, name in enumerate(B_DESIGNS):
                for j, T in enumerate(TS):
                    rows = pool.map(one_b, [(name, T, 7_500_000 + 100_000 * i + 10_000 * j + r) for r in range(R)])
                    res[f"{name} | T={T}"] = summarise_b(rows)
                    print(f"(b) {name} | T={T}: {res[f'{name} | T={T}']}", flush=True)
            json.dump(dict(R=R, cells=res), open(f"{common.OUT}/e13_b.json", "w"), indent=1)
        if "c" in parts:
            res = {}
            for i, sk in enumerate(C_SHOCKS):
                for j, gk in enumerate(C_GARCHS):
                    for k, T in enumerate(TS):
                        rows = pool.map(one_c, [(sk, gk, T, 8_000_000 + 100_000 * i + 30_000 * j + 10_000 * k + r, False)
                                                for r in range(R)])
                        res[f"{sk} | GARCH {gk} | T={T}"] = summarise_c(rows)
                        print(f"(c) {sk} | GARCH {gk} | T={T}: {res[f'{sk} | GARCH {gk} | T={T}']}", flush=True)
            boot = {}
            for (sk, gk, T) in BOOT_CELLS:
                i, j, k = list(C_SHOCKS).index(sk), C_GARCHS.index(gk), TS.index(T)
                rows = pool.map(one_c, [(sk, gk, T, 8_000_000 + 100_000 * i + 30_000 * j + 10_000 * k + r, True)
                                        for r in range(R_boot)])
                boot[f"{sk} | GARCH {gk} | T={T}"] = summarise_c(rows)
                print(f"(c, bootstrap) {sk} | GARCH {gk} | T={T}: {boot[f'{sk} | GARCH {gk} | T={T}']}", flush=True)
            json.dump(dict(R=R, R_boot=R_boot, cells=res, boot=boot), open(f"{common.OUT}/e13_c.json", "w"), indent=1)
        if "d" in parts:
            jobs = [(m, sk, s) for m in ("AR(1)", "MA(1)") for sk in ("normal", "t(8)", "skew-t (nu 8, lam -.4)")
                    for s in (0, 1)]
            rows = pool.map(one_d, jobs)
            res = {}
            for (m, sk, s), r in zip(jobs, rows):
                res.setdefault(f"{m} | {sk}", []).append(r)
            for k, v in res.items():
                print(f"(d) {k}: {v}", flush=True)
            json.dump(res, open(f"{common.OUT}/e13_d.json", "w"), indent=1)
        if "e" in parts:
            res = {}
            for i, sk in enumerate(E_SHOCKS):
                for j, gk in enumerate(A_GARCHS):
                    for k, T in enumerate(E_TS):
                        rows = pool.map(one_e, [(sk, gk, T, 8_500_000 + 100_000 * i + 30_000 * j + 10_000 * k + r)
                                                for r in range(R)])
                        key = f"{sk} | GARCH {gk} | T={T}"
                        res[key] = {lab: round(float(np.mean([r[c] for r in rows])), 3) for c, lab in
                                    (("raw_gaussian", "Gaussian LR rejects"), ("old", "old compare() picks GARCH"),
                                     ("new", "new compare() picks GARCH (t LR)"),
                                     ("normal_only", "new compare(dists=('normal',)) picks GARCH"))}
                        print(f"(e) {key}: {res[key]}", flush=True)
            json.dump(dict(R=R, cells=res), open(f"{common.OUT}/e13_e.json", "w"), indent=1)
        if "f" in parts:
            res = {}
            for i, name in enumerate(F_DESIGNS):
                for j, T in enumerate((240, 480)):
                    rows = pool.map(one_f, [(name, T, 8_800_000 + 100_000 * i + 10_000 * j + r) for r in range(R)])
                    g = np.array([r["gain"] for r in rows])
                    gains = [r for r in rows]
                    res[f"{name} | T={T}"] = {
                        "mean holdout log-score gain of GARCH (nats/obs)": round(float(g.mean()), 4),
                        "its MC standard error": round(float(g.std(ddof=1) / np.sqrt(len(g))), 4),
                        "share of series where GARCH scores better": round(float(np.mean(g > 0)), 3),
                        "share where DM favours GARCH at 5%": round(float(np.mean([r["dm_p"] < 0.05 and r["gain"] > 0
                                                                                   for r in gains])), 3),
                        "share where DM favours constant variance at 5%": round(float(np.mean(
                            [r["dm_p"] < 0.05 and r["gain"] < 0 for r in gains])), 3)}
                    print(f"(f) {name} | T={T}: {res[f'{name} | T={T}']}", flush=True)
            json.dump(dict(R=R, cells=res), open(f"{common.OUT}/e13_f.json", "w"), indent=1)


if __name__ == "__main__":
    main()
