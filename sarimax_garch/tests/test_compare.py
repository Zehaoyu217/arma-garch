"""compare(): the GUIDE.md workflow as code -- end-to-end cases; holdout log score and predictive densities."""
import numpy as np
from scipy import stats

from sarimax_garch import SARIMAXGARCH, _core, compare, dm_test, garch_test, holdout_logscore


def _x(T, seed):
    return np.cumsum(np.random.default_rng(seed).normal(size=T)) * 0.1


def test_homoskedastic_keeps_sarimax_and_never_compares_across_backends():
    T = 500
    x = _x(T, 1)
    y, _, _ = _core.simulate(T, 0.1, [0.5], [], 1.0, 0.0, 0.0, dist="normal", X=x[:, None], gamma=[0.5], seed=2)
    res = compare(y, exog=x, order=(1, 0, 0), trend="c", dists=("normal",))
    assert res.recommendation["model"] == "SARIMAX"
    assert any(f["code"] == "L4" for f in res.flags)                       # backend warning always present
    assert set(res.table["compare_group"]) == {"SARIMAX alone", "engine"}
    assert res.table.loc["SARIMAX", "backend"] == "statsmodels (exact)"
    assert "SARIMAX" in res.summary()


def test_ma_garch_is_detected_and_recommended():
    T = 1500
    x = _x(T, 3)
    y, _, _ = _core.simulate(T, 0.1, [], [0.5], 0.02, 0.08, 0.90, dist="normal", X=x[:, None], gamma=[0.5], seed=4)
    res = compare(y, exog=x, order=(0, 0, 1), trend="c")
    assert res.recommendation["model"].startswith("SARIMAX + GARCH(1,1)")
    assert res.tests.loc["LR GARCH vs constant variance (normal, within engine)", "p_value"] < 1e-3
    assert res.tests.loc["ARCH-LM(4) on SARIMAX residuals", "p_value"] < 1e-3
    g = res.models["engine GARCH(1,1) (normal)"]
    assert abs(g.params["ma.L1"] - 0.5) < 0.1 and 0.9 < g.persistence < 1.0


def test_holdout_logscore_is_comparable_across_backends():
    T = 1200
    x = np.random.default_rng(1).normal(size=T)
    y0, _, _ = _core.simulate(T, 0.1, [0.5], [], 1.0, 0.0, 0.0, dist="normal", X=x[:, None], gamma=[0.5], seed=2)
    sm = holdout_logscore(y0, exog=x, order=(1, 0, 0), holdout=300)              # statsmodels backend
    gn = holdout_logscore(y0, exog=x, order=(1, 0, 0), garch=(1, 1), holdout=300)  # engine
    assert len(sm) == len(gn) == 300 and abs(np.mean(gn - sm)) < 0.01          # no ARCH: the backends agree
    y1, _, _ = _core.simulate(T, 0.1, [0.5], [], 0.02, 0.08, 0.90, dist="normal", X=x[:, None], gamma=[0.5], seed=2)
    d = holdout_logscore(y1, exog=x, order=(1, 0, 0), garch=(1, 1), holdout=300) - \
        holdout_logscore(y1, exog=x, order=(1, 0, 0), holdout=300)
    assert d.mean() > 0.02 and dm_test(d)[1] < 0.05                            # GARCH wins out of sample
    res = compare(y1, exog=x, order=(1, 0, 0), trend="c", dists=("normal",), holdout=300)
    assert np.isfinite(res.table.loc["SARIMAX", "holdout_logscore"])


def test_dm_test_statistic_lag_and_small_sample_rule():
    d = np.random.default_rng(5).normal(0.1, 1.0, size=80)
    n = len(d)
    stat_n, p_n = dm_test(d, small_sample=False)
    L = int(np.floor(4 * (n / 100) ** (2 / 9)))                                   # = 3 at n = 80
    dc = d - d.mean()
    lrv = dc @ dc / n + 2 * sum((1 - k / (L + 1)) * (dc[k:] @ dc[:-k]) / n for k in range(1, L + 1))
    assert abs(stat_n - d.mean() / np.sqrt(lrv / n)) < 1e-12 and abs(p_n - 2 * stats.norm.sf(abs(stat_n))) < 1e-12
    stat, p = dm_test(d)                                                          # HLN factor, t(n - 1)
    assert abs(stat - stat_n * np.sqrt((n - 1) / n)) < 1e-12 and abs(p - 2 * stats.t.sf(abs(stat), n - 1)) < 1e-12
    s6, _ = dm_test(d, h=6, small_sample=False)                                   # lag = max(h - 1, L) = 5
    lrv6 = dc @ dc / n + 2 * sum((1 - k / 6) * (dc[k:] @ dc[:-k]) / n for k in range(1, 6))
    assert abs(s6 - d.mean() / np.sqrt(lrv6 / n)) < 1e-12


def test_predictive_logpdf_exact_at_h1_and_a_density_beyond():
    w, _, _ = _core.simulate(1500, 0.05, [0.5], [0.3], 0.05, 0.08, 0.90, dist="t", nu=6.0, seed=3)
    r = _core.fit_joint(np.cumsum(w), p=1, q=1, d=1, dist="t")
    fc = _core.forecast(r, 4, n_sims=5000, seed=1)
    nu, s1 = r["nu"], np.sqrt(fc["variance_innovation"][0])
    k = np.sqrt(nu / (nu - 2))
    x = fc["mean"] + np.r_[1.3 * s1, 0, 0, 0]
    exact = stats.t.logpdf(1.3 * k, nu) + np.log(k) - np.log(s1)
    assert abs(_core.predictive_logpdf(fc, x)[0] - exact) < 1e-10
    sd = np.sqrt(fc["variance"][3])
    grid = np.linspace(fc["mean"][3] - 12 * sd, fc["mean"][3] + 12 * sd, 2001)
    dens = [np.exp(_core.predictive_logpdf(fc, np.r_[fc["mean"][:3], g])[3]) for g in grid]
    assert abs(np.sum(dens) * (grid[1] - grid[0]) - 1) < 1e-3                   # integrates to 1 at h = 4 (d = 1)


def test_backtest_reports_logscore_on_both_backends():
    T = 300
    x = np.random.default_rng(4).normal(size=T)
    y, _, _ = _core.simulate(T, 0.1, [0.5], [], 0.1, 0.1, 0.8, dist="normal", X=x[:, None], gamma=[0.5], seed=4)
    for garch in (None, (1, 1)):
        bt = SARIMAXGARCH(y, exog=x, order=(1, 0, 0), trend="c", garch=garch).backtest(horizon=3, start=250, step=10,
                                                                                         n_sims=500)
        assert "logscore" in bt.columns and np.all(np.isfinite(bt["logscore"])) and bt.columns[-1] == "coverage90"


def test_garch_test_restricted_bootstrap():
    T = 300
    x = np.random.default_rng(6).normal(size=T)
    y1, _, _ = _core.simulate(T, 0.1, [0.5], [], 0.02, 0.08, 0.90, dist="normal", X=x[:, None], gamma=[0.5], seed=6)
    r1 = garch_test(y1, exog=x, order=(1, 0, 0), B=19, seed=1)
    assert r1["B"] == 19 and r1["p_bootstrap"] <= 0.05 and r1["lr"] > np.quantile(r1["lr_boot"], 0.95)
    y0, _, _ = _core.simulate(T, 0.1, [0.5], [], 1.0, 0.0, 0.0, dist="normal", X=x[:, None], gamma=[0.5], seed=7)
    r0 = garch_test(y0, exog=x, order=(1, 0, 0), B=19, seed=1)
    assert r0["p_bootstrap"] > 0.05 and np.quantile(r0["lr_boot"], 0.95) < 5.99   # chi2(2) is conservative here
    res = compare(y1, exog=x, order=(1, 0, 0), dists=("normal",), bootstrap_lr=19)
    assert "restricted-bootstrap" in res.recommendation["reason"] and res.recommendation["model"] != "SARIMAX"


def test_short_sample_stops_garch():
    T = 90
    y, _, _ = _core.simulate(T, 0.1, [], [0.4], 0.1, 0.1, 0.8, dist="normal", seed=5)
    res = compare(y, order=(0, 0, 1), trend="c", dists=("normal",))
    assert res.recommendation["model"] == "SARIMAX"
    assert any(f["code"] == "L3" and f["level"] == "stop" for f in res.flags)
