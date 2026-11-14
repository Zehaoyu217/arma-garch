"""E9 -- is the Diebold-Mariano statistic N(0, 1) under its null in our setting?

Part A, the null exactly true (equal expected log score by construction).  Data: u_t = 0.1 + 0.5 u_{t-1} +
eps_t with persistent GARCH (alpha .08, beta .90), normal or t(5) shocks (t(5) leaves eps with no fourth
moment).  Two forecasters issue Gaussian densities centred at the TRUE conditional mean +delta*s and
-delta*s (delta = 0.25): by symmetry of the true predictive distribution E[d_t] = 0 exactly, where
d_t = log f_A(y) - log f_B(y) = 2 delta (y - m) / s.
  A1  h = 1, s = the unconditional sd (a constant-variance forecaster, like SARIMAX): d_t is a martingale
      difference but heteroskedastic (GARCH) and, with t(5), heavy-tailed
  A2  h = 6, s = the true 6-step sd: d_t is MA(5) (overlapping forecasts), so a HAC variance is needed
Tests at 5% two-sided, holdout n = 60, 120, 250, 1000, 4000 replications:
  naive   paired t-test, variance gamma_0 only, N(0, 1)
  NW      Newey-West, lag floor(4 (n/100)^(2/9)), N(0, 1)          (dm_test before the E9 fix)
  NW-h    Newey-West with lag max(h - 1, floor(4 (n/100)^(2/9))), N(0, 1)
  HLN     NW-h times the Harvey-Leybourne-Newbold factor sqrt((n + 1 - 2h + h(h-1)/n) / n), against t(n-1)
          = dm_test(d, h=h) as shipped (asserted)
Part B, the nested population null (E8, GARCH absent: SARIMAX is the true model and the GARCH fit nests it):
the distribution of the shipped DM statistic for GARCH - SARIMAX log score and CRPS, from e8_dm_stats.json.
"""
import json
import sys

import numpy as np
from scipy import stats

import common
from sarimax_garch import dm_test

C, PHI, OM, AL, BE, DELTA = 0.1, 0.5, 0.02, 0.08, 0.90, 0.25
NS, R, BURN = (60, 120, 250, 1000), 4000, 500


def lrv(d, lag):
    dc = d - d.mean(1, keepdims=True)
    n = d.shape[1]
    v = np.einsum("ij,ij->i", dc, dc) / n
    for k in range(1, lag + 1):
        v = v + 2 * (1 - k / (lag + 1)) * np.einsum("ij,ij->i", dc[:, k:], dc[:, :-k]) / n
    return v


def simulate_paths(n_total, zdist, rng):
    """R paths of u and the true one-step quantities (vectorised over replications)."""
    z = common.draw_z(rng, (R, n_total + BURN), **zdist)
    ubar = OM / (1 - AL - BE)
    s2 = np.full(R, ubar); e = np.zeros(R); u = np.full(R, C / (1 - PHI))
    U, S2N = np.empty((R, n_total)), np.empty((R, n_total))         # u_t and sigma^2_{t+1} known at t
    for t in range(n_total + BURN):
        s2 = OM + AL * e * e + BE * s2
        e = np.sqrt(s2) * z[:, t]
        u = C + PHI * u + e
        if t >= BURN:
            U[:, t - BURN] = u
            S2N[:, t - BURN] = OM + AL * e * e + BE * s2
    return U, S2N


def h_step_moments(U, S2N, h):
    """True conditional mean and variance of u_{t+h} given information at t (AR(1)-GARCH(1,1))."""
    ubar, pers = OM / (1 - AL - BE), AL + BE
    m = C * sum(PHI ** j for j in range(h)) + PHI ** h * U
    v = np.zeros_like(U)
    for j in range(h):                                              # psi_j = PHI^j, sigma^2 at step h - j
        k = h - j                                                   # E_t sigma^2_{t+k} = ubar + pers^(k-1)(S2N - ubar)
        v = v + PHI ** (2 * j) * (ubar + pers ** (k - 1) * (S2N - ubar))
    return m, v


def part_a():
    rng = np.random.default_rng(2026)
    out = {}
    for zname, zdist in (("normal", dict(dist="normal")), ("t5", dict(dist="t", nu=5.0))):
        for h, scale in ((1, "constant"), (6, "true")):
            for n in NS:
                U, S2N = simulate_paths(n + h, zdist, rng)
                m, v = h_step_moments(U[:, :n], S2N[:, :n], h)
                y_future = U[:, h:n + h]                            # u_{t+h} for origins t = 0..n-1
                s = np.sqrt(OM / (1 - AL - BE)) * np.sqrt(sum(PHI ** (2 * j) for j in range(h))) \
                    if scale == "constant" else np.sqrt(v)
                d = 2 * DELTA * (y_future - m) / s                  # log N(y; m + ds, s^2) - log N(y; m - ds, s^2)
                dbar = d.mean(1)
                lag0 = int(np.floor(4 * (n / 100) ** (2 / 9)))
                lag_h = max(h - 1, lag0)
                naive = dbar / np.sqrt(lrv(d, 0) / n)
                nw = dbar / np.sqrt(np.maximum(lrv(d, lag0), 1e-300) / n)
                nw_h = dbar / np.sqrt(np.maximum(lrv(d, lag_h), 1e-300) / n)
                hln = nw_h * np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
                shipped = np.array([dm_test(di, h=h)[0] for di in d[:50]])
                assert np.allclose(shipped, hln[:50]), "dm_test(h=h) must equal the HLN variant"
                tcrit = stats.t.ppf(0.975, n - 1)
                key = f"{zname} shocks, h={h} ({scale} scale), n={n}"
                out[key] = dict(
                    naive=float(np.mean(np.abs(naive) > 1.96)), nw=float(np.mean(np.abs(nw) > 1.96)),
                    nw_h=float(np.mean(np.abs(nw_h) > 1.96)), hln=float(np.mean(np.abs(hln) > tcrit)),
                    nw_mean=float(nw.mean()), nw_sd=float(nw.std()), hln_sd=float(hln.std()),
                    hln_q=[float(v_) for v_ in np.quantile(hln, [0.025, 0.975])], tcrit=float(tcrit),
                    lag=lag0, lag_h=lag_h)
                o = out[key]
                print(f"A {key:42} reject at 5%: naive {o['naive']:.3f} NW {o['nw']:.3f} NW-h {o['nw_h']:.3f} "
                      f"HLN {o['hln']:.3f} | HLN stat sd {o['hln_sd']:.3f}, 2.5/97.5% {o['hln_q'][0]:+.2f}/"
                      f"{o['hln_q'][1]:+.2f} vs t(n-1) +-{tcrit:.2f} (lags {lag0}/{lag_h})", flush=True)
    return out


def part_b():
    with open(f"{common.OUT}/e8_dm_stats.json") as fh:
        st = json.load(fh)
    out = {}
    for key, v in st.items():
        if not key.startswith("none"):
            continue
        n_ho = int(key.split("holdout ")[1])
        crit = stats.t.ppf(0.975, n_ho - 1)                         # the shipped dm_test's 5% rule
        for score in ("ls", "crps"):
            a = np.array(v[score])
            out[f"{key} | {score}"] = dict(mean=float(a.mean()), sd=float(a.std()), crit=float(crit),
                                          p_garch=float(np.mean(a > crit)), p_sarimax=float(np.mean(a < -crit)))
            o = out[f"{key} | {score}"]
            print(f"B {key} {score:5}: DM mean {o['mean']:+.3f} sd {o['sd']:.3f} | P(DM > {crit:.2f}, GARCH better) "
                  f"{o['p_garch']:.3f}  P(DM < -{crit:.2f}, SARIMAX better) {o['p_sarimax']:.3f}")
    return out


if __name__ == "__main__":
    res = dict(part_a=part_a(), part_b=part_b() if "--no-b" not in sys.argv else {})
    with open(f"{common.OUT}/e9_dm_null.json", "w") as fh:
        json.dump(res, fh, indent=1)
