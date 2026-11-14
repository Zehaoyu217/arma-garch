"""compare(): the GUIDE.md workflow as code -- three end-to-end cases."""
import numpy as np

from sarimax_garch import _core, compare


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


def test_short_sample_stops_garch():
    T = 90
    y, _, _ = _core.simulate(T, 0.1, [], [0.4], 0.1, 0.1, 0.8, dist="normal", seed=5)
    res = compare(y, order=(0, 0, 1), trend="c", dists=("normal",))
    assert res.recommendation["model"] == "SARIMAX"
    assert any(f["code"] == "L3" and f["level"] == "stop" for f in res.flags)
