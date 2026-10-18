"""Grid comparison script vs rugarch: filter and forecast at identical parameters, and free fits."""
import json
import os
import warnings

import numpy as np

from common import J, OUT

warnings.filterwarnings("ignore", category=RuntimeWarning)
D = os.path.join(OUT, "rugarch")
S = json.load(open(f"{D}/script.json"))
R = json.load(open(f"{D}/rugarch.json"))
BURN = 300


def rug_to_script(m, cf):
    p, q, k = m["p"], m["q"], m["k"]
    ar = [cf[f"ar{i + 1}"] for i in range(p)]
    return np.r_[cf["mu"] * (1 - sum(ar)), ar, [cf[f"ma{j + 1}"] for j in range(q)],
                 [cf[f"mxreg{i + 1}"] for i in range(k)], cf["omega"], cf["alpha1"], cf["beta1"],
                 ([cf["shape"]] if m["dist"] == "t" else [])]


def rug_hess_se(m, cf, V):
    """rugarch Hessian SEs mapped to the script's parametrisation (delta method for const)."""
    names = list(cf)
    V = np.array(V)
    ix = {n: i for i, n in enumerate(names)}
    ar = [f"ar{i + 1}" for i in range(m["p"])]
    g = np.zeros(len(names)); g[ix["mu"]] = 1 - sum(cf[a] for a in ar)
    for a in ar:
        g[ix[a]] = -cf["mu"]
    order = ["mu"] + ar + [f"ma{j + 1}" for j in range(m["q"])] + [f"mxreg{i + 1}" for i in range(m["k"])] \
        + ["omega", "alpha1", "beta1"] + (["shape"] if m["dist"] == "t" else [])
    se = np.sqrt(np.diag(V))
    out = np.array([se[ix[n]] for n in order])
    out[0] = np.sqrt(g @ V @ g)
    return out


rows = []
print(f"{'case':42s} {'filter':>8s} {'fcast':>8s} | {'max|d|/SE':>9s} {'hesSE ratio':>13s} | "
      f"{'LLs(own-rug)':>12s} {'LLr(own-scr)':>12s} | {'max|est-true|/SE':>16s}")
for nm, m in S.items():
    s = np.loadtxt(f"{D}/{nm}_script_filter.csv", delimiter=",")
    r = np.genfromtxt(f"{D}/{nm}_rug_filter.csv", delimiter=",", skip_header=1)[m["p"]:]
    filt = np.abs(s - r)[BURN:].max()
    filt_late = np.abs(s - r)[1000:].max()
    fc = max(np.abs(np.array(m["fc_mean"]) - R[nm]["fc_mean"]).max(),
             np.abs(np.array(m["fc_sigma"]) - R[nm]["fc_sigma"]).max())
    cf = R[nm]["coef"]
    rs = rug_to_script(m, cf)
    sp, se = np.array(m["params"]), np.array(m["se"])
    dse = np.abs(sp - rs) / se
    hr = np.array(m["se_hess"]) / rug_hess_se(m, cf, R[nm]["vcov_hess"])
    y = np.loadtxt(f"{D}/{nm}_y.csv")
    X = np.loadtxt(f"{D}/{nm}_X.csv", delimiter=",") if m["k"] else None
    ll_s_at_r = -J._neg_loglik(rs, y, X, m["p"], m["q"], m["dist"])
    gap_s = m["loglik"] - ll_s_at_r                       # >= 0: script's optimum is at least as good
    gap_r = R[nm]["LL"] - R[nm]["LL_at_script"]           # >= 0: rugarch's optimum is at least as good
    ztrue = np.abs(sp - np.array(m["truth"])) / se
    rows.append(dict(name=nm, filt=filt, filt_late=filt_late, fc=fc, dse=dse.max(), hr_lo=hr.min(), hr_hi=hr.max(),
                     gap_s=gap_s, gap_r=gap_r, ztrue=ztrue.max(), conv=R[nm]["conv"]))
    print(f"{nm:42s} {filt:8.1e} {fc:8.1e} | {dse.max():9.3f} {hr.min():6.3f}-{hr.max():5.3f} | "
          f"{gap_s:12.4f} {gap_r:12.4f} | {ztrue.max():16.2f}")
print("\nSUMMARY over", len(rows), "cases")
print(f"  filter at identical params (t>={BURN}): max |diff| = {max(r['filt'] for r in rows):.1e}   "
      f"(t>=1000: {max(r['filt_late'] for r in rows):.1e}; slow start-up decay when beta ~ 0.95)")
print(f"  forecasts at identical params        : max |diff| = {max(r['fc'] for r in rows):.1e}")
bad = [r for r in rows if r["gap_r"] < -0.01]          # rugarch's own optimum beaten under its own likelihood
good = [r for r in rows if r["gap_r"] >= -0.01]
failures = ", ".join(f"{r['name']} (+{-r['gap_r']:.1f})" for r in bad) or "none"
print(f"  rugarch optimiser failures (its LL at the script's estimate is higher than at its own): {failures}")
print(f"  free fits, excluding those: max |script - rugarch| / SE = {max(r['dse'] for r in good):.3f}")
print(f"  Hessian SE ratio script/rugarch       : {min(r['hr_lo'] for r in good):.3f} - {max(r['hr_hi'] for r in good):.3f}   "
      f"(>1.06 only in high-persistence cases: rugarch's default hess.d=0.1 step, see rugarch_hessian_step.R)")
print(f"  script LL(own) - LL(at rugarch est)   : min {min(r['gap_s'] for r in rows):.4f}  (>= 0: script optimiser not beaten)")
print(f"  rugarch LL(own) - LL(at script est)   : min {min(r['gap_r'] for r in rows):.4f}")
print(f"  rugarch converged in {sum(r['conv'] == 0 for r in rows)}/{len(rows)}; "
      f"max |script est - truth| / SE = {max(r['ztrue'] for r in rows):.2f}")
json.dump(rows, open(f"{D}/summary.json", "w"), indent=1)
