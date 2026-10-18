"""Small-sample follow-ups (monthly, T = 60-180):
  (i)   bootstrap percentile CIs vs Wald CIs (robust / Hessian): coverage at T = 120, 180   (slow: ~15 min)
  (ii)  the same 150 series fitted by rugarch: RMSE per parameter and boundary rates (data or code?)
  (iii) ARMA part only (GARCH off): script's conditional likelihood vs SARIMAX's exact one, T = 60/120/180
Usage: python small_sample_extra.py [--quick] [--skip-boot]"""
import os

os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402

import numpy as np  # noqa: E402

from common import HERE, J, OUT  # noqa: E402
from small_sample import DGPS, GAMMA, seasonal_X, truth_vec  # noqa: E402

QUICK = "--quick" in sys.argv
SKIP_BOOT = "--skip-boot" in sys.argv
R = 60 if QUICK else 150
NB = 99 if QUICK else 199
D = os.path.join(OUT, "small")
os.makedirs(D, exist_ok=True)


def gen(name, T, seed):
    g = DGPS[name]
    X = seasonal_X(T)
    y, _, _ = J.simulate(T, 0.1, g["phi"], g["theta"], *g["garch"], dist=g["dist"], nu=g["nu"] or 6.0,
                         X=X, gamma=GAMMA, seed=seed)
    return y, X


def boot_one(args):
    name, T, seeds = args
    warnings.filterwarnings("ignore")
    g = DGPS[name]
    p, q = len(g["phi"]), len(g["theta"])
    out = []
    for s in seeds:
        y, X = gen(name, T, s)
        r = J.fit_joint(y, X=X, p=p, q=q, dist=g["dist"])
        bs = J.bootstrap(r, n_boot=NB, seed=s)
        out.append(dict(params=r.params.tolist(), se=r.se.tolist(), se_h=r.se_hessian.tolist(),
                        ci=bs["ci"].tolist(), n_ok=bs["n_ok"], names=r.names))
    return name, T, out


def sarimax_one(args):
    T, seeds = args
    warnings.filterwarnings("ignore")
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    out = []
    for s in seeds:
        X = seasonal_X(T)
        y, _, _ = J.simulate(T, 0.1, [0.5], [0.3], 1.0, 0.0, 0.0, dist="normal", X=X, gamma=GAMMA, seed=s)
        r = J.fit_joint(y, X=X, p=1, q=1, dist="normal", fixed={"alpha": 0.0, "beta": 0.0})
        f = SARIMAX(y, exog=X, order=(1, 0, 1), trend="c").fit(disp=False)
        P = dict(zip(f.model.param_names, f.params))
        out.append(dict(script=[r["const"], r["phi1"], r["theta1"], r["gamma1"], r["gamma2"]],
                        sarimax=[P["intercept"], P["ar.L1"], P["ma.L1"], P["x1"], P["x2"]]))
    return T, out


def main():
    t0 = time.time()
    workers = max(1, (os.cpu_count() or 2) - 1)
    lines = []

    def emit(s=""):
        print(s)
        lines.append(s)

    # ---- (ii) export series for rugarch, run R in the background ----
    export = {}
    for T in (120, 180):
        seeds = [70_000 + T * 10 + i for i in range(R)]
        Y = np.column_stack([gen("AR1-t6 monthly", T, s)[0] for s in seeds])
        np.savetxt(f"{D}/T{T}_y.csv", Y, delimiter=",")
        np.savetxt(f"{D}/T{T}_X.csv", seasonal_X(T), delimiter=",")
        export[T] = seeds
    rproc = subprocess.Popen(["Rscript", os.path.join(HERE, "small_rugarch.R"), D],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # ---- (i) bootstrap coverage ----
    if SKIP_BOOT:
        emit("(i) bootstrap coverage skipped (--skip-boot)")
    else:
        tasks = []
        for name in ("AR1-N monthly", "AR1-t6 monthly"):
            for T in (120, 180):
                seeds = [50_000 + T * 10 + i for i in range(R)]
                tasks += [(name, T, seeds[i:i + 5]) for i in range(0, R, 5)]
        res = {}
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for name, T, out in ex.map(boot_one, tasks):
                res.setdefault((name, T), []).extend(out)
        emit(f"(i) 95% CI coverage: Wald robust / Wald Hessian / bootstrap percentile ({NB} residual-bootstrap refits), {R} reps")
        emit(f"{'DGP':18s} {'T':>4s} | {'phi1':>16s} | {'gamma1':>16s} | {'omega':>16s} | {'alpha':>16s} | {'beta':>16s} | boot fails")
        for (name, T), rr in res.items():
            tr = truth_vec(DGPS[name], "full")
            P = np.array([r["params"] for r in rr]); SE = np.array([r["se"] for r in rr]); SH = np.array([r["se_h"] for r in rr])
            CI = np.array([r["ci"] for r in rr])
            names = rr[0]["names"]
            cr = np.mean(np.abs(P - tr) < 1.96 * SE, 0); ch = np.mean(np.abs(P - tr) < 1.96 * SH, 0)
            cb = np.mean((CI[:, 0, :] <= tr) & (tr <= CI[:, 1, :]), 0)
            cell = lambda n: f"{cr[names.index(n)]:4.2f}/{ch[names.index(n)]:4.2f}/{cb[names.index(n)]:4.2f}"
            emit(f"{name:18s} {T:4d} | {cell('phi1'):>16s} | {cell('gamma1'):>16s} | {cell('omega'):>16s} | "
                 f"{cell('alpha'):>16s} | {cell('beta'):>16s} | {np.mean([NB - r['n_ok'] for r in rr]):.1f}/{NB}")

    # ---- (iii) conditional vs exact likelihood for the ARMA part ----
    tasks = [(T, [60_000 + T * 10 + i for i in range(R * 2)][j:j + 10]) for T in (60, 120, 180) for j in range(0, R * 2, 10)]
    res3 = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for T, out in ex.map(sarimax_one, tasks):
            res3.setdefault(T, []).extend(out)
    emit(f"\n(iii) ARMA(1,1)+seasonal X, no GARCH: RMSE of the script (conditional likelihood) vs SARIMAX (exact), {2 * R} reps")
    emit(f"{'T':>4s} | {'const':>13s} | {'phi1':>13s} | {'theta1':>13s} | {'gamma1':>13s} | {'gamma2':>13s} | |theta|>0.95 script/sarimax")
    tr3 = np.array([0.1, 0.5, 0.3, 1.0, -0.5])
    for T, rr in res3.items():
        S = np.array([r["script"] for r in rr]); M = np.array([r["sarimax"] for r in rr])
        rs, rm = np.sqrt(np.mean((S - tr3) ** 2, 0)), np.sqrt(np.mean((M - tr3) ** 2, 0))
        emit(f"{T:4d} | " + " | ".join(f"{a:5.3f} vs {b:5.3f}" for a, b in zip(rs, rm)) +
             f" | {np.mean(np.abs(S[:, 2]) > 0.95):.2f} / {np.mean(np.abs(M[:, 2]) > 0.95):.2f}")

    # ---- (ii) collect rugarch ----
    rproc.wait()
    emit(f"\n(ii) AR(1)-GARCH-t + seasonal X, the same {R} series per T: script vs rugarch (solver='hybrid')")
    emit(f"{'T':>4s} | {'RMSE phi1':>15s} | {'RMSE alpha':>15s} | {'RMSE beta':>15s} | {'RMSE nu':>15s} | "
         f"nu at bound | boundary rate | rugarch errors")
    tr = truth_vec(DGPS["AR1-t6 monthly"], "full")
    for T, seeds in export.items():
        rug = np.genfromtxt(f"{D}/T{T}_rugarch.csv", delimiter=",", skip_header=1)   # mu ar1 mx1 mx2 omega alpha beta shape conv
        Y = np.loadtxt(f"{D}/T{T}_y.csv", delimiter=","); X = seasonal_X(T)
        S = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for i in range(Y.shape[1]):
                r = J.fit_joint(Y[:, i], X=X, p=1, q=0, dist="t", compute_cov=False)
                S.append(r.params)
        S = np.array(S)
        ok = np.isfinite(rug).all(1) & (rug[:, -1] == 0)
        Rm = np.column_stack([rug[:, 0] * (1 - rug[:, 1]), rug[:, 1], rug[:, 2], rug[:, 3], rug[:, 4], rug[:, 5], rug[:, 6], rug[:, 7]])
        names = ["const", "phi1", "gamma1", "gamma2", "omega", "alpha", "beta", "nu"]
        rs = np.sqrt(np.mean((S[ok] - tr) ** 2, 0)); rr_ = np.sqrt(np.mean((Rm[ok] - tr) ** 2, 0))
        cell = lambda n: f"{rs[names.index(n)]:6.3f} vs {rr_[names.index(n)]:6.3f}"
        bnd_s = np.mean((S[ok, 5] < 1e-4) | (S[ok, 6] < 1e-4)); bnd_r = np.mean((Rm[ok, 5] < 1e-4) | (Rm[ok, 6] < 1e-4))
        nub_s = np.mean(S[ok, 7] > 199); nub_r = np.mean(Rm[ok, 7] > 99)         # rugarch's shape bound is 100
        emit(f"{T:4d} | {cell('phi1'):>15s} | {cell('alpha'):>15s} | {cell('beta'):>15s} | {cell('nu'):>15s} | "
             f"{nub_s:.2f} vs {nub_r:.2f} | {bnd_s:.2f} vs {bnd_r:.2f}   | {int((~ok).sum())}/{len(ok)}")
    emit(f"\n{time.time() - t0:.0f}s")
    open(os.path.join(OUT, "small_sample_extra.txt"), "a" if SKIP_BOOT else "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
