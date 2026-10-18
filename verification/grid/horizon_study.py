"""Multi-step, macro-conditioned forecasts: levels vs differences, with and without GARCH.

The macro path X (two random walks) is GIVEN over the horizon: X_future = the realised path, H = 36 months.
  DGP L (levels, cointegrated):  y_t  = c + b'X_t  + u_t,  u AR(1) phi = .8        -- the level of y is tied to X
  DGP D (differences):           dy_t = c + b'dX_t + u_t,  u AR(1) phi = .3, y = 100 + cumsum  -- it is not
Fits on each sample: levels model ARMA(1,0) d=0, differences model ARMA(1,1) d=1; each with GARCH (normal
errors) and with GARCH off (constant variance).  Oracle = the true form with the true parameters.
Scores at h = 1, 3, 6, 12, 24, 36:  RMSE of the point forecast (ratio to the oracle's), coverage of the 90%
interval (simulated quantiles), CRPS of the predictive distribution (ratio to the oracle's).
"Better" = lower RMSE, coverage nearer 0.90, lower CRPS.
Usage: python horizon_study.py [--quick]"""
import os

os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402

import numpy as np  # noqa: E402

from common import J, OUT  # noqa: E402

QUICK = "--quick" in sys.argv
R = 100 if QUICK else 300
H, HS, NS = 36, [1, 3, 6, 12, 24, 36], 2000
GARCH = {"typical (pers .85)": (0.15, 0.10, 0.75), "high-pers (pers .98)": (0.02, 0.08, 0.90)}
B = [1.0, -0.5]
VARIANTS = ["levels-garch", "levels-const", "diffs-garch", "diffs-const"]


def crps(s, x):
    """CRPS of a sample s against the realisation x (sorted O(n log n) formula)."""
    s = np.sort(s)
    n = len(s)
    return float(np.mean(np.abs(s - x)) - np.sum((2 * np.arange(1, n + 1) - n - 1) * s) / n ** 2)


def gen(dgp, T, garch, seed):
    rng = np.random.default_rng(seed)
    Xd = rng.normal(size=(T + H, 2))
    X = np.cumsum(Xd, axis=0)
    om, al, be = GARCH[garch]
    if dgp == "L":
        y, _, _ = J.simulate(T + H, 0.1, [0.8], [], om, al, be, dist="normal", X=X, gamma=B, seed=seed + 1)
        truth, d = dict(const=0.1, phi1=0.8, gamma1=1.0, gamma2=-0.5, omega=om, alpha=al, beta=be), 0
    else:
        w, _, _ = J.simulate(T + H, 0.1, [0.3], [], om, al, be, dist="normal", X=Xd, gamma=B, seed=seed + 1)
        y = 100.0 + np.cumsum(w)
        truth, d = dict(const=0.1, phi1=0.3, gamma1=1.0, gamma2=-0.5, omega=om, alpha=al, beta=be), 1
    return y, X, truth, d


def score(fc, y, T):
    out = {}
    for h in HS:
        s, x = fc["sims"][:, h - 1], y[T + h - 1]
        q = np.quantile(s, [0.05, 0.95])
        out[h] = dict(err=float(fc["mean"][h - 1] - x), cov=bool(q[0] <= x <= q[1]), crps=crps(s, x))
    return out


def one(args):
    dgp, garch, T, seeds = args
    warnings.filterwarnings("ignore")
    out = []
    for s in seeds:
        y, X, truth, d = gen(dgp, T, garch, s)
        ys, Xs, Xf = y[:T], X[:T], X[T:]
        res = {}
        for form, (p, q, dd) in (("levels", (1, 0, 0)), ("diffs", (1, 1, 1))):
            for g, fx in (("garch", None), ("const", {"alpha": 0.0, "beta": 0.0})):
                try:
                    r = J.fit_joint(ys, X=Xs, p=p, q=q, d=dd, dist="normal", fixed=fx, compute_cov=False)
                    res[f"{form}-{g}"] = score(J.forecast(r, H, X_future=Xf, n_sims=NS, seed=s), y, T)
                except Exception as e:  # noqa: BLE001
                    res[f"{form}-{g}"] = {"error": f"{type(e).__name__}: {e}"}
        ro = J.fit_joint(ys, X=Xs, p=1, q=0, d=d, dist="normal", fixed=truth, compute_cov=False)
        res["oracle"] = score(J.forecast(ro, H, X_future=Xf, n_sims=NS, seed=s), y, T)
        out.append(res)
    return dgp, garch, T, out


def main():
    cells = [(dgp, garch, T) for dgp in ("L", "D") for garch in GARCH for T in (120, 180, 360)]
    tasks = []
    for i, (dgp, garch, T) in enumerate(cells):
        seeds = [200_000 + 1000 * i + j for j in range(R)]
        tasks += [(dgp, garch, T, seeds[k:k + 10]) for k in range(0, R, 10)]
    res, t0 = {}, time.time()
    with ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) - 1)) as ex:
        for dgp, garch, T, out in ex.map(one, tasks):
            res.setdefault((dgp, garch, T), []).extend(out)
    print(f"{len(cells) * R} samples x 5 fits in {time.time() - t0:.0f}s\n")

    lines, table = [], []
    for (dgp, garch, T), rr in res.items():
        head = f"DGP {dgp} ({'levels, cointegrated' if dgp == 'L' else 'differences'}) · GARCH {garch} · T={T} ({T // 12}y)"
        lines.append(head + "\n" + f"{'':14s}{'':13s}" + "".join(f"{'h=' + str(h):>8s}" for h in HS))
        orc = [r["oracle"] for r in rr]
        o_rmse = {h: np.sqrt(np.mean([o[h]["err"] ** 2 for o in orc])) for h in HS}
        o_crps = {h: np.mean([o[h]["crps"] for o in orc]) for h in HS}
        o_cov = {h: np.mean([o[h]["cov"] for o in orc]) for h in HS}
        lines.append(f"  {'oracle':14s}{'coverage90':13s}" + "".join(f"{o_cov[h]:8.2f}" for h in HS))
        for v in VARIANTS:
            ok = [r[v] for r in rr if "error" not in r[v]]
            fails = len(rr) - len(ok)
            rm = {h: np.sqrt(np.mean([o[h]["err"] ** 2 for o in ok])) / o_rmse[h] for h in HS}
            cv = {h: np.mean([o[h]["cov"] for o in ok]) for h in HS}
            cr = {h: np.mean([o[h]["crps"] for o in ok]) / o_crps[h] for h in HS}
            lines.append(f"  {v:14s}{'RMSE/oracle':13s}" + "".join(f"{rm[h]:8.2f}" for h in HS) + (f"   ({fails} fit failures)" if fails else ""))
            lines.append(f"  {'':14s}{'coverage90':13s}" + "".join(f"{cv[h]:8.2f}" for h in HS))
            lines.append(f"  {'':14s}{'CRPS/oracle':13s}" + "".join(f"{cr[h]:8.2f}" for h in HS))
            for h in HS:
                table.append(dict(dgp=dgp, garch=garch, T=T, variant=v, h=h, rmse_ratio=rm[h], cov=cv[h], crps_ratio=cr[h], fails=fails))
        lines.append("")
    txt = "\n".join(lines)
    print(txt)
    json.dump(table, open(os.path.join(OUT, "horizon_study.json"), "w"), indent=1)
    open(os.path.join(OUT, "horizon_study.txt"), "w").write(txt + "\n")


if __name__ == "__main__":
    main()
