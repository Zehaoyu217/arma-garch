import os
O = os.path.join(os.environ.get("CMP_DIR", "cmp"), "new_modes")
import json, numpy as np
S = json.load(open(f"{O}/script.json")); R = json.load(open(f"{O}/rugarch.json"))
for nm, m in S.items():
    s = np.loadtxt(f"{O}/{nm}_script_filter.csv", delimiter=","); r = np.genfromtxt(f"{O}/{nm}_rug_filter.csv", delimiter=",", skip_header=1)[1:]
    d = np.abs(s - r)[300:].max(0)
    cf = R[nm]["coef"]; th = dict(zip(m["names"], m["params"])); se = dict(zip(m["names"], m["se"]))
    rug = {"const": cf.get("mu", 0.0) * (1 - cf["ar1"]), "phi1": cf["ar1"], "theta1": cf["ma1"], "omega": cf["omega"],
           "alpha": cf["alpha1"], "beta": cf["beta1"], "nu": cf["shape"]}
    for i in range(m["k"]): rug[f"gamma{i+1}"] = cf[f"mxreg{i+1}"]
    print(f"\n{nm}: filter 1:1 (t>=300) max|eps|={d[0]:.1e} max|sigma|={d[1]:.1e} max|ll_t|={d[2]:.1e}")
    print(f"  forecast 1:1  max|mean|={np.abs(np.array(m['fc_mean'])-R[nm]['fc_mean']).max():.1e} "
          f"max|sigma|={np.abs(np.array(m['fc_sigma'])-R[nm]['fc_sigma']).max():.1e}")
    print(f"  loglik: rugarch@own={R[nm]['LL']:.4f}  rugarch@script={R[nm]['LL_at_script']:.4f}")
    print("  " + "  ".join(f"{n}: {th[n]:.4f} vs {rug[n]:.4f} ({(th[n]-rug[n])/se[n]:+.2f}SE)" if se[n] > 0
                           else f"{n}: fixed 0 vs rugarch excluded" for n in m["names"]))
