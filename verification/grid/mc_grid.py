"""Monte Carlo validation of the estimator: bias, SE calibration, 95% CI coverage, asymptotics.

Blocks (regression with ARMA-GARCH errors, X = [sin, N(0,1)], gamma = [1, -0.5], intercept 0.1):
  M    7 ARMA orders x d in {0,1,2}, typical GARCH, t(6)
  G    ARMA(1,1), 5 GARCH sets x {normal, t(5), t(10)}
  A22  ARMA(2,2) x {high-alpha, low-pers}, t(6)    (the two grid cases whose alpha landed > 3 SE out)
  Q    QMLE: t(5) data fitted with dist="normal"  -> robust SEs must cover, Hessian SEs need not
  C    asymptotics: ARMA(1,1) typical t(6), T = 500 .. 8000; high-persistence normal, T = 1000 .. 16000
Usage: python mc_grid.py [--quick]      (quick = a quarter of the replications)"""
import os

os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402

import numpy as np  # noqa: E402

from common import GARCH, J, ORDERS, OUT, make_data, truth  # noqa: E402

QUICK = "--quick" in sys.argv


def configs():
    c = []
    for order in ORDERS:
        for d in (0, 1, 2):
            c.append(dict(block="M", order=order, d=d, garch="typical", dist="t", nu=6.0, fit="t", T=2000,
                          R=300 if d == 0 else 150))
    for garch in GARCH:
        for dist, nu in (("normal", 0.0), ("t", 5.0), ("t", 10.0)):
            c.append(dict(block="G", order=(1, 1), d=0, garch=garch, dist=dist, nu=nu, fit=dist, T=2000, R=300))
    for garch in ("high-alpha", "low-pers"):
        c.append(dict(block="A22", order=(2, 2), d=0, garch=garch, dist="t", nu=6.0, fit="t", T=2000, R=300))
    for garch in ("typical", "high-alpha"):
        c.append(dict(block="Q", order=(1, 1), d=0, garch=garch, dist="t", nu=5.0, fit="normal", T=2000, R=400))
    for T in (500, 1000, 2000, 4000, 8000):
        c.append(dict(block="C", order=(1, 1), d=0, garch="typical", dist="t", nu=6.0, fit="t", T=T, R=400))
    for T in (1000, 4000, 16000):
        c.append(dict(block="C", order=(1, 1), d=0, garch="high-pers", dist="normal", nu=0.0, fit="normal", T=T, R=300))
    for i, cfg in enumerate(c):
        cfg["id"] = i
        if QUICK:
            cfg["R"] = max(40, cfg["R"] // 4)
    return c


def run_chunk(args):
    cfg, seeds = args
    warnings.filterwarnings("ignore")
    out = []
    for s in seeds:
        y, X, _ = make_data(cfg["T"], tuple(cfg["order"]), cfg["d"], cfg["garch"], cfg["dist"], cfg["nu"] or 6.0, True, seed=s)
        try:
            r = J.fit_joint(y, X=X, p=cfg["order"][0], q=cfg["order"][1], d=cfg["d"], dist=cfg["fit"])
            out.append(dict(ok=True, params=r.params.tolist(), se=r.se.tolist(), se_h=r.se_hessian.tolist(),
                            conv=r.converged, cap=r.at_persistence_cap, boundary=r.at_boundary,
                            m4=r.fourth_moment_exists))
        except Exception as e:  # noqa: BLE001
            out.append(dict(ok=False, err=f"{type(e).__name__}: {e}"))
    return cfg["id"], out


def main():
    cfgs = configs()
    tasks = []
    for cfg in cfgs:
        seeds = [100_000 * (cfg["id"] + 1) + r for r in range(cfg["R"])]
        for i in range(0, len(seeds), 20):
            tasks.append((cfg, seeds[i:i + 20]))
    results = {cfg["id"]: [] for cfg in cfgs}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) - 1)) as ex:
        for cid, out in ex.map(run_chunk, tasks):
            results[cid].extend(out)
    print(f"{sum(c['R'] for c in cfgs)} fits in {time.time() - t0:.0f}s")
    json.dump(dict(configs=cfgs, results={str(k): v for k, v in results.items()}),
              open(os.path.join(OUT, "mc_raw.json"), "w"))
    summarize(cfgs, results)


def summarize(cfgs, results):
    lines = []

    def emit(s=""):
        print(s)
        lines.append(s)
    emit("Per configuration (worst parameter shown): bias/SD, robust SE/MC SD, 95% coverage robust & Hessian")
    emit(f"{'id':>2} {'block':5} {'model':30} {'T':>5} {'ok':>4} {'cap':>4} | {'max|bias|/SD':>12} "
         f"{'robSE/SD':>13} {'cover rob':>13} {'cover hes':>13}")
    allcov, table = [], []
    for cfg in cfgs:
        res = [r for r in results[cfg["id"]] if r["ok"]]
        P = np.array([r["params"] for r in res]); SE = np.array([r["se"] for r in res]); SH = np.array([r["se_h"] for r in res])
        tr = truth(tuple(cfg["order"]), cfg["garch"], cfg["fit"], cfg["nu"], True)
        if cfg["fit"] == "t" and cfg["dist"] == "normal":
            tr = tr[:-1]
        names = J._names(cfg["order"][0], cfg["order"][1], 2, cfg["fit"])
        sd = P.std(0, ddof=1)
        bias = (P.mean(0) - tr) / sd
        ratio = np.nanmedian(SE, 0) / sd
        cov_r = np.mean(np.abs(P - tr) < 1.96 * SE, 0)
        cov_h = np.mean(np.abs(P - tr) < 1.96 * SH, 0)
        z = (P - tr) / SE
        model = (f"ARIMA({cfg['order'][0]},{cfg['d']},{cfg['order'][1]}) {cfg['garch']} "
                 f"{cfg['dist']}{int(cfg['nu']) if cfg['dist'] == 't' else ''}" + (" fit=N" if cfg["block"] == "Q" else ""))
        ncap = sum(r["cap"] for r in res)
        wb, wr, wc, wh = (np.argmax(np.abs(bias)), np.argmax(np.abs(np.log(ratio))),
                          np.argmax(np.abs(cov_r - 0.95)), np.argmax(np.abs(cov_h - 0.95)))
        emit(f"{cfg['id']:>2} {cfg['block']:5} {model:30} {cfg['T']:>5} {len(res):>4} {ncap:>4} | "
             f"{abs(bias[wb]):6.2f} {names[wb]:>5} {ratio[wr]:6.2f} {names[wr]:>6} {cov_r[wc]:6.3f} {names[wc]:>6} "
             f"{cov_h[wh]:6.3f} {names[wh]:>6}")
        for i, n in enumerate(names):
            table.append(dict(id=cfg["id"], block=cfg["block"], model=model, T=cfg["T"], param=n, truth=float(tr[i]),
                              bias_sd=float(bias[i]), sd=float(sd[i]), se_ratio=float(ratio[i]), cov_rob=float(cov_r[i]),
                              cov_hes=float(cov_h[i]), z_mean=float(np.nanmean(z[:, i])), z_sd=float(np.nanstd(z[:, i])),
                              n=len(res), cap=ncap, fails=len(results[cfg["id"]]) - len(res)))
            if cfg["block"] != "Q":
                allcov.append((cov_r[i], len(res)))
    cov = np.array([c for c, _ in allcov]); n = np.array([m for _, m in allcov])
    band = 1.96 * np.sqrt(0.95 * 0.05 / n)
    emit(f"\nRobust-SE 95% coverage over {len(cov)} (config, parameter) pairs, blocks M/G/A22/C:")
    emit(f"  inside the binomial MC band: {np.mean(np.abs(cov - 0.95) <= band):.1%}   "
         f"median {np.median(cov):.3f}   range {cov.min():.3f} - {cov.max():.3f}")
    emit(f"  (by chance alone ~95% fall inside the band)")
    fails = sum(1 for cfg in cfgs for r in results[cfg["id"]] if not r["ok"])
    emit(f"  fit exceptions: {fails}")
    json.dump(table, open(os.path.join(OUT, "mc_table.json"), "w"), indent=1)
    open(os.path.join(OUT, "mc_summary.txt"), "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
