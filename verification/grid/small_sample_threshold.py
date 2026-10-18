"""When does a GARCH variance forecast beat the sample variance on monthly data?

For AR(1) + seasonal X with normal errors, by ARCH strength / persistence and sample size:
  QLIKE loss of the 1-step variance forecast (GARCH fit with dist="normal" vs the constant-variance fit,
  both against the TRUE sigma^2_{T+1}), the win rate (share of samples where GARCH's loss is lower), and
  the power of the ARCH-LM pre-test (12 lags, 5%) on the constant-variance residuals.
Usage: python small_sample_threshold.py [--quick]"""
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
from small_sample import GAMMA, seasonal_X  # noqa: E402

QUICK = "--quick" in sys.argv
R = 150 if QUICK else 600
DGPS = {                                   # omega chosen so the unconditional variance is 1
    "weak     a=.05 b=.85 (pers .90)": (0.10, 0.05, 0.85),
    "typical  a=.10 b=.75 (pers .85)": (0.15, 0.10, 0.75),
    "strong   a=.15 b=.75 (pers .90)": (0.10, 0.15, 0.75),
    "daily-like a=.08 b=.90 (pers .98)": (0.02, 0.08, 0.90),
}
TS = (120, 180, 240, 300, 360)


def one(args):
    name, T, seeds = args
    warnings.filterwarnings("ignore")
    om, al, be = DGPS[name]
    X = seasonal_X(T + 1)
    out = []
    for s in seeds:
        y, _, sig2 = J.simulate(T + 1, 0.1, [0.5], [], om, al, be, dist="normal", X=X, gamma=GAMMA, seed=s)
        v_true = sig2[T]
        base = J.fit_joint(y[:T], X=X[:T], p=1, q=0, dist="normal", fixed={"alpha": 0.0, "beta": 0.0}, compute_cov=False)
        garch = J.fit_joint(y[:T], X=X[:T], p=1, q=0, dist="normal", compute_cov=False)
        ql = lambda v: float(v_true / v - np.log(v_true / v) - 1)
        out.append(dict(q_const=ql(base["omega"]), q_garch=ql(J.forecast(garch, 1, X_future=X[T:], n_sims=0)["variance_innovation"][0]),
                        lm_p=J.arch_lm_test(base.eps, 12)[1]))
    return name, T, out


def main():
    tasks = [(name, T, [90_000 + 1000 * i + T + j for j in range(R)][k:k + 25])
             for i, name in enumerate(DGPS) for T in TS for k in range(0, R, 25)]
    res, t0 = {}, time.time()
    with ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) - 1)) as ex:
        for name, T, out in ex.map(one, tasks):
            res.setdefault((name, T), []).extend(out)
    print(f"{2 * sum(len(v) for v in res.values())} fits in {time.time() - t0:.0f}s\n")
    lines = [f"1-step variance forecast on monthly data, AR(1)+seasonal X, normal errors, {R} reps per cell",
             f"{'ARCH strength':34s} {'T':>4s} {'yrs':>4s} | {'QLIKE GARCH':>11s} {'constant':>9s} {'ratio':>6s} {'win rate':>9s} | ARCH-LM power"]
    table = []
    for name in DGPS:
        for T in TS:
            rr = res[(name, T)]
            qg, qc = np.mean([r["q_garch"] for r in rr]), np.mean([r["q_const"] for r in rr])
            win = np.mean([r["q_garch"] < r["q_const"] for r in rr])
            pw = np.mean([r["lm_p"] < 0.05 for r in rr])
            table.append(dict(dgp=name, T=T, q_garch=qg, q_const=qc, ratio=qg / qc, win=win, power=pw))
            lines.append(f"{name:34s} {T:4d} {T // 12:4d} | {qg:11.3f} {qc:9.3f} {qg / qc:6.2f} {win:9.2f} | {pw:.2f}")
        lines.append("")
    print("\n".join(lines))
    json.dump(table, open(os.path.join(OUT, "small_sample_threshold.json"), "w"), indent=1)
    open(os.path.join(OUT, "small_sample_threshold.txt"), "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
