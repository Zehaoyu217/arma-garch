import sys, json, warnings, numpy as np
import os; sys.path.insert(0, os.environ.get("JAG_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))
CMP = os.environ.get("CMP_DIR", "cmp")
import joint_armax_garch as J
warnings.filterwarnings("ignore", category=RuntimeWarning)
S = json.load(open(f"{CMP}/script_results.json")); R = json.load(open(f"{CMP}/rugarch_results.json"))
BURN = 300

def rug_to_script(m, cf):
    """rugarch coef dict -> script parameter vector (const = mu*(1-sum ar))."""
    p, q, k = m["p"], m["q"], m["k"]
    ar = [cf[f"ar{i+1}"] for i in range(p)]
    v = [cf["mu"] * (1 - sum(ar))] + ar + [cf[f"ma{j+1}"] for j in range(q)] \
        + [cf[f"mxreg{i+1}"] for i in range(k)] + [cf["omega"], cf["alpha1"], cf["beta1"]] \
        + ([cf["shape"]] if m["dist"] == "t" else [])
    return np.array(v)

def rug_se_script(m, cf, V, names_r):
    """delta-method SEs in script parametrisation (only const needs it)."""
    V = np.array(V); idx = {n: i for i, n in enumerate(names_r)}
    p = m["p"]; ar = [f"ar{i+1}" for i in range(p)]
    se = np.sqrt(np.diag(V))
    g = np.zeros(len(names_r)); g[idx["mu"]] = 1 - sum(cf[a] for a in ar)
    for a in ar: g[idx[a]] = -cf["mu"]
    se_c = np.sqrt(g @ V @ g)
    order = ["mu"] + ar + [f"ma{j+1}" for j in range(m["q"])] + [f"mxreg{i+1}" for i in range(m["k"])] \
            + ["omega", "alpha1", "beta1"] + (["shape"] if m["dist"] == "t" else [])
    out = np.array([se[idx[n]] for n in order]); out[0] = se_c
    return out

print("=" * 100)
print("(1) FILTER 1:1 at identical parameters: script _filter vs rugarch ugarchfilter (t >= p, after BURN=%d obs)" % BURN)
print(f"{'case':22s} {'max|eps diff|':>14s} {'max|sigma diff|':>16s} {'max|loglik_t diff|':>19s}   first-obs |sigma diff|   decay")
for nm, m in S.items():
    s = np.loadtxt(f"{CMP}/{nm}_script_filter.csv", delimiter=",")
    r = np.genfromtxt(f"{CMP}/{nm}_rugarch_filter.csv", delimiter=",", skip_header=1)[m["p"]:]
    d = np.abs(s - r)
    tail = d[BURN:]
    print(f"{nm:22s} {tail[:,0].max():14.2e} {tail[:,1].max():16.2e} {tail[:,2].max():19.2e}   {d[0,1]:.2e}"
          f"   t=50:{d[50,1]:.1e} t=150:{d[150,1]:.1e}")

print("\n" + "=" * 100)
print("(2) FREE FITS: script fit_joint vs rugarch ugarchfit(solver='hybrid')")
for nm, m in S.items():
    cf = R[nm]["coef"]; names_r = list(cf.keys())
    rs = rug_to_script(m, cf); sp = np.array(m["params"])
    se_s, seh_s = np.array(m["se"]), np.array(m["se_hess"])
    se_r = rug_se_script(m, cf, R[nm]["vcov_rob"], names_r); seh_r = rug_se_script(m, cf, R[nm]["vcov_hess"], names_r)
    y = np.loadtxt(f"{CMP}/{nm}_y.csv"); X = np.loadtxt(f"{CMP}/{nm}_X.csv", delimiter=",") if m["k"] else None
    ll_script_at_rug = -J._neg_loglik(rs, y, X, m["p"], m["q"], m["dist"])
    print(f"\n--- {nm}   (nobs script={m['nobs']}, rugarch={len(y)})")
    print(f"  loglik  script@scriptMLE={m['loglik']:.4f}  script@rugarchMLE={ll_script_at_rug:.4f}   "
          f"rugarch@rugarchMLE={R[nm]['loglik_fit']:.4f}  rugarch@scriptMLE={R[nm]['loglik_at_script']:.4f}")
    print(f"  {'param':>7} {'script':>10} {'rugarch':>10} {'diff/SE':>8} | {'robSE s':>8} {'robSE r':>8} {'ratio':>6} | {'hesSE s':>8} {'hesSE r':>8} {'ratio':>6}")
    for i, n in enumerate(m["names"]):
        print(f"  {n:>7} {sp[i]:10.5f} {rs[i]:10.5f} {(sp[i]-rs[i])/se_s[i]:8.3f} | {se_s[i]:8.5f} {se_r[i]:8.5f} {se_s[i]/se_r[i]:6.3f}"
              f" | {seh_s[i]:8.5f} {seh_r[i]:8.5f} {seh_s[i]/seh_r[i]:6.3f}")

print("\n" + "=" * 100)
print("(3) FORECASTS at identical parameters (script MLE): script forecast() vs rugarch ugarchforecast, h=1..10")
for nm, m in S.items():
    dm = np.abs(np.array(m["fc_mean_diff"]) - np.array(R[nm]["fc_mean"])).max()
    ds = np.abs(np.array(m["fc_sigma"]) - np.array(R[nm]["fc_sigma"])).max()
    print(f"  {nm:22s} max|mean diff|={dm:.2e}   max|sigma_T+h diff|={ds:.2e}")
