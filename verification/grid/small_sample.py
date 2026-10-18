"""Small samples: monthly data with 5-30 years (T = 60, 120, 180, 360).

Monthly-realistic designs: AR(1) or ARMA(1,1) mean with a seasonal pair x = [sin, cos](2 pi t / 12),
GARCH persistence 0.85 (monthly volatility is less persistent than daily), unconditional variance 1.
  DGPs:  AR1-N / AR1-t6 / ARMA11-t6 with "monthly" GARCH;  AR1-N with NO GARCH (homoskedastic);
         AR1-N with daily-like high persistence (alpha .08, beta .90).
  Fits:  full      -- the DGP's ARMA order, dist = DGP's (t fits nu)
         normal    -- same, dist="normal"            (QMLE: fewer parameters)
         no-garch  -- fixed alpha = beta = 0, normal  (regression with ARMA errors, constant variance)
Metrics per cell: estimation (bias, RMSE, coverage of robust and Hessian SEs, boundary/cap/nu-at-bound
rates) and forecasting (1-step variance forecast QLIKE loss vs the TRUE sigma^2_{T+1}, and point-forecast
RMSE at h = 1 and 12 relative to an oracle that knows the true parameters).
Usage: python small_sample.py [--quick]"""
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
H = 12
GAMMA = [1.0, -0.5]
DGPS = {
    "AR1-N monthly":        dict(phi=[0.5], theta=[], garch=(0.15, 0.10, 0.75), dist="normal", nu=0.0),
    "AR1-t6 monthly":       dict(phi=[0.5], theta=[], garch=(0.15, 0.10, 0.75), dist="t", nu=6.0),
    "ARMA11-t6 monthly":    dict(phi=[0.5], theta=[0.3], garch=(0.15, 0.10, 0.75), dist="t", nu=6.0),
    "AR1-N no-GARCH":       dict(phi=[0.5], theta=[], garch=(1.0, 0.0, 0.0), dist="normal", nu=0.0),
    "AR1-N high-pers":      dict(phi=[0.5], theta=[], garch=(0.02, 0.08, 0.90), dist="normal", nu=0.0),
}
FITS = {"full": None, "normal": "normal", "no-garch": "no-garch"}
TS = (60, 120, 180, 360)


def seasonal_X(n):
    t = np.arange(n)
    return np.column_stack([np.sin(2 * np.pi * t / 12), np.cos(2 * np.pi * t / 12)])


def truth_vec(g, fit):
    base = [0.1, *g["phi"], *g["theta"], *GAMMA, *g["garch"]]
    if fit == "full" and g["dist"] == "t":
        base.append(g["nu"])
    return np.array(base)


def one(args):
    name, T, fit, seeds = args
    warnings.filterwarnings("ignore")
    g = DGPS[name]
    p, q = len(g["phi"]), len(g["theta"])
    X = seasonal_X(T + H)
    out = []
    for s in seeds:
        y, eps, sig2 = J.simulate(T + H, 0.1, g["phi"], g["theta"], *g["garch"], dist=g["dist"], nu=g["nu"] or 6.0,
                                  X=X, gamma=GAMMA, seed=s)
        ys, Xs, Xf, yf = y[:T], X[:T], X[T:], y[T:]
        v_true = sig2[T]                       # true sigma^2_{T+1} given the sample
        kw = dict(p=p, q=q)
        if fit == "full":
            kw["dist"] = g["dist"]
        elif fit == "normal":
            kw["dist"] = "normal"
        else:
            kw.update(dist="normal", fixed={"alpha": 0.0, "beta": 0.0})
        try:
            r = J.fit_joint(ys, X=Xs, **kw)
            fc = J.forecast(r, H, X_future=Xf, n_sims=0)
            # oracle: true parameters, same filter/forecast machinery
            tr = truth_vec(g, "full")
            ro = J.fit_joint(ys, X=Xs, p=p, q=q, dist=g["dist"], fixed=dict(zip(J._names(p, q, 2, g["dist"]), tr)),
                             compute_cov=False)
            fo = J.forecast(ro, H, X_future=Xf, n_sims=0)
            v_hat = fc["variance_innovation"][0]
            out.append(dict(ok=True, params=r.params.tolist(), se=r.se.tolist(), se_h=r.se_hessian.tolist(),
                            names=r.names, conv=r.converged, cap=r.at_persistence_cap, bnd=r.at_boundary,
                            nub=r.nu_at_bound, qlike=float(v_true / v_hat - np.log(v_true / v_hat) - 1),
                            qlike_oracle=float(v_true / fo["variance_innovation"][0] - np.log(v_true / fo["variance_innovation"][0]) - 1),
                            e1=float(fc["mean"][0] - yf[0]), e12=float(fc["mean"][-1] - yf[-1]),
                            o1=float(fo["mean"][0] - yf[0]), o12=float(fo["mean"][-1] - yf[-1])))
        except Exception as e:  # noqa: BLE001
            out.append(dict(ok=False, err=f"{type(e).__name__}: {e}"))
    return name, T, fit, out


def main():
    R = 150 if QUICK else 600
    tasks, cid = [], 0
    for name in DGPS:
        for T in TS:
            for fit in FITS:
                seeds = [10_000 * (cid + 1) + i for i in range(R)]
                for i in range(0, R, 25):
                    tasks.append((name, T, fit, seeds[i:i + 25]))
                cid += 1
    res = {}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) - 1)) as ex:
        for name, T, fit, out in ex.map(one, tasks):
            res.setdefault((name, T, fit), []).extend(out)
    print(f"{sum(len(v) for v in res.values())} fits in {time.time() - t0:.0f}s\n")
    json.dump([dict(dgp=k[0], T=k[1], fit=k[2], results=v) for k, v in res.items()],
              open(os.path.join(OUT, "small_sample_raw.json"), "w"))
    report(res)


def report(res):
    lines = []

    def emit(s=""):
        print(s)
        lines.append(s)

    emit("ESTIMATION  (full fit = the DGP's model).  cover = 95% CI coverage, robust / Hessian SEs.")
    emit(f"{'DGP':20s} {'T':>4s} {'fails':>5s} {'bnd':>5s} {'cap':>5s} {'nu@200':>6s} | "
         f"{'phi1 bias':>9s} {'RMSE':>6s} {'cov r/h':>11s} | {'alpha RMSE':>10s} {'cov r/h':>11s} | "
         f"{'beta RMSE':>9s} {'cov r/h':>11s} | {'gamma1 RMSE':>11s} {'cov r/h':>11s}")
    for name, g in DGPS.items():
        for T in TS:
            rr = [r for r in res[(name, T, "full")] if r["ok"]]
            tr = truth_vec(g, "full")
            P = np.array([r["params"] for r in rr]); SE = np.array([r["se"] for r in rr]); SH = np.array([r["se_h"] for r in rr])
            names = rr[0]["names"]
            cov_r = np.mean(np.abs(P - tr) < 1.96 * SE, 0); cov_h = np.mean(np.abs(P - tr) < 1.96 * SH, 0)
            rmse = np.sqrt(np.mean((P - tr) ** 2, 0)); bias = P.mean(0) - tr

            def cell(n):
                i = names.index(n)
                return f"{rmse[i]:6.3f} {cov_r[i]:5.2f}/{cov_h[i]:4.2f}"
            fails = len(res[(name, T, "full")]) - len(rr)
            emit(f"{name:20s} {T:4d} {fails:5d} {np.mean([r['bnd'] for r in rr]):5.2f} {np.mean([r['cap'] for r in rr]):5.2f} "
                 f"{np.mean([r['nub'] for r in rr]):6.2f} | {bias[names.index('phi1')]:+9.3f} {cell('phi1')} | "
                 f"    {cell('alpha')} |    {cell('beta')} |      {cell('gamma1')}")
    emit("\nFORECASTING: 1-step variance forecast QLIKE loss vs the true sigma^2_{T+1} (0 = perfect, oracle = true params),")
    emit("and point-forecast RMSE at h=1 and h=12 relative to the oracle (1.00 = as good as knowing the truth).")
    emit(f"{'DGP':20s} {'T':>4s} | {'QLIKE full':>10s} {'normal':>7s} {'no-garch':>8s} {'oracle':>7s} | "
         f"{'RMSE1 full':>10s} {'normal':>7s} {'no-garch':>8s} | {'RMSE12 full':>11s} {'normal':>7s} {'no-garch':>8s}")
    table = []
    for name in DGPS:
        for T in TS:
            row = {"dgp": name, "T": T}
            for fit in FITS:
                rr = [r for r in res[(name, T, fit)] if r["ok"]]
                row[f"qlike_{fit}"] = float(np.mean([r["qlike"] for r in rr]))
                row[f"rmse1_{fit}"] = float(np.sqrt(np.mean([r["e1"] ** 2 for r in rr])) / np.sqrt(np.mean([r["o1"] ** 2 for r in rr])))
                row[f"rmse12_{fit}"] = float(np.sqrt(np.mean([r["e12"] ** 2 for r in rr])) / np.sqrt(np.mean([r["o12"] ** 2 for r in rr])))
                row[f"fails_{fit}"] = len(res[(name, T, fit)]) - len(rr)
            row["qlike_oracle"] = float(np.mean([r["qlike_oracle"] for r in res[(name, T, "full")] if r["ok"]]))
            table.append(row)
            emit(f"{name:20s} {T:4d} | {row['qlike_full']:10.3f} {row['qlike_normal']:7.3f} {row['qlike_no-garch']:8.3f} "
                 f"{row['qlike_oracle']:7.3f} | {row['rmse1_full']:10.3f} {row['rmse1_normal']:7.3f} {row['rmse1_no-garch']:8.3f} | "
                 f"{row['rmse12_full']:11.3f} {row['rmse12_normal']:7.3f} {row['rmse12_no-garch']:8.3f}")
    json.dump(table, open(os.path.join(OUT, "small_sample_table.json"), "w"), indent=1)
    open(os.path.join(OUT, "small_sample_summary.txt"), "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
