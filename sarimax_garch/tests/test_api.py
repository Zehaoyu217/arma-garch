import os

import numpy as np
import pandas as pd
import pytest

from sarimax_garch import SARIMAXGARCH, _core

T = 1500


@pytest.fixture(scope="module")
def data():
    idx = pd.date_range("1900-01-01", periods=T, freq="MS")
    t = np.arange(T)
    X = pd.DataFrame({"sin12": np.sin(2 * np.pi * t / 12), "shock": np.random.default_rng(1).normal(size=T)}, index=idx)
    y, _, _ = _core.simulate(T, 0.2, [0.6], [0.3], 0.05, 0.08, 0.88, dist="t", nu=6.0,
                             X=X.values, gamma=[1.0, -0.5], seed=7)
    return pd.Series(y, index=idx, name="y"), X


def future_exog(X, steps):
    idx = pd.date_range(X.index[-1] + pd.offsets.MonthBegin(1), periods=steps, freq="MS")
    t = np.arange(len(X), len(X) + steps)
    return pd.DataFrame({"sin12": np.sin(2 * np.pi * t / 12), "shock": 0.0}, index=idx)


def test_garch_off_is_statsmodels(data):
    from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResultsWrapper
    y, X = data
    res = SARIMAXGARCH(y, exog=X, order=(1, 0, 1), trend="c").fit(disp=False)
    ref = SARIMAX(y, exog=X, order=(1, 0, 1), trend="c").fit(disp=False)
    assert isinstance(res, SARIMAXResultsWrapper)
    assert list(res.params.index) == list(ref.params.index)
    assert np.allclose(res.params.values, ref.params.values)
    assert abs(res.llf - ref.llf) < 1e-8


def test_garch_on_names_shapes_and_estimates(data):
    y, X = data
    res = SARIMAXGARCH(y, exog=X, order=(1, 0, 1), trend="c", garch=(1, 1), dist="t").fit()
    names = ["intercept", "sin12", "shock", "ar.L1", "ma.L1", "omega", "alpha.L1", "beta.L1", "nu"]
    assert list(res.params.index) == names
    for attr in ("bse", "tvalues", "pvalues"):
        assert list(getattr(res, attr).index) == names
    assert res.conf_int().shape == (9, 2) and res.cov_params().shape == (9, 9)
    assert np.isfinite(res.llf) and res.nobs == T - 1 and res.df_model == 9
    assert res.resid.index.equals(y.index) and res.resid.isna().sum() == 1
    assert np.allclose((res.fittedvalues + res.resid).dropna(), y.iloc[1:])
    truth = pd.Series(dict(zip(names, [0.2, 1.0, -0.5, 0.6, 0.3, 0.05, 0.08, 0.88, 6.0])))
    assert (((res.params - truth) / res.bse).abs() < 4).all()
    txt = res.summary().as_text()
    for s in ("ar.L1", "alpha.L1", "Log Likelihood", "Persistence", "P>|z|"):
        assert s in txt


def test_forecast_api(data):
    y, X = data
    Xf = future_exog(X, 12)
    res = SARIMAXGARCH(y, exog=X, order=(1, 0, 1), trend="c", garch=(1, 1)).fit()
    fc = res.get_forecast(12, exog=Xf, seed=0)
    assert fc.predicted_mean.index.equals(Xf.index)
    sf = fc.summary_frame()
    assert list(sf.columns) == ["mean", "mean_se", "mean_ci_lower", "mean_ci_upper"]
    assert (sf["mean_ci_lower"] < sf["mean"]).all() and (sf["mean"] < sf["mean_ci_upper"]).all()
    ci = fc.conf_int(method="simulated")
    assert ci.shape == (12, 2) and list(ci.columns) == ["lower y", "upper y"]
    assert np.allclose(res.forecast(12, exog=Xf).values, fc.predicted_mean.values)
    # volatility path: sigma^2_{T+h}; equals Var(y_{T+h}) at h=1, smaller afterwards (ARMA accumulation)
    cv = fc.conditional_variance
    assert cv.index.equals(Xf.index) and (cv > 0).all()
    assert np.isclose(cv.iloc[0], fc.var_pred_mean.iloc[0]) and (fc.var_pred_mean.iloc[1:] > cv.iloc[1:]).all()
    assert np.allclose(fc.conditional_volatility ** 2, cv)
    s2_T1 = res.params["omega"] + res.params["alpha.L1"] * res.resid.iloc[-1] ** 2 \
        + res.params["beta.L1"] * res.conditional_variance.iloc[-1]
    assert np.isclose(cv.iloc[0], s2_T1)                      # the GARCH recursion one step past the sample
    assert res.simulate(6, repetitions=50, exog=Xf.iloc[:6], seed=1).shape == (6, 50)
    with pytest.raises(ValueError):
        res.get_forecast(12)
    with pytest.raises(ValueError):
        res.get_forecast(12, exog=Xf.iloc[:5])


def test_garch_fixed_off_matches_statsmodels(data):
    y, X = data
    ours = SARIMAXGARCH(y, exog=X, order=(1, 0, 1), trend="c", garch=(1, 1)).fit(
        fixed_params={"alpha.L1": 0.0, "beta.L1": 0.0})
    sm = SARIMAXGARCH(y, exog=X, order=(1, 0, 1), trend="c").fit(disp=False)
    common = ["intercept", "sin12", "shock", "ar.L1", "ma.L1"]
    assert np.allclose(ours.params[common].values, sm.params[common].values, atol=0.02)
    assert abs(ours.params["omega"] - sm.params["sigma2"]) < 0.05
    assert np.isnan(ours.bse["alpha.L1"]) and ours.fixed_params == {"alpha.L1": 0.0, "beta.L1": 0.0}
    assert np.allclose(np.sort(np.abs(ours.arroots)), np.sort(np.abs(sm.arroots)), rtol=0.05)   # statsmodels names
    assert np.allclose(np.sort(np.abs(ours.maroots)), np.sort(np.abs(sm.maroots)), rtol=0.05)


def test_t_without_garch_uses_joint_backend(data):
    y, X = data
    mod = SARIMAXGARCH(y, exog=X, order=(1, 0, 1), trend="c", dist="t")
    assert mod.backend == "garch"
    res = mod.fit()
    assert list(res.params.index) == ["intercept", "sin12", "shock", "ar.L1", "ma.L1", "sigma2", "nu"]
    assert 3 < res.params["nu"] < 12


def test_no_trend_default_matches_statsmodels_convention(data):
    y, X = data
    res = SARIMAXGARCH(y, exog=X, order=(1, 0, 0), garch=(1, 1)).fit()
    assert "intercept" not in res.params.index and "const" not in res.params.index


def test_fix_params_context_and_start_params(data):
    y, X = data
    mod = SARIMAXGARCH(y, exog=X, order=(1, 0, 1), trend="c", garch=(1, 1))
    with mod.fix_params({"ar.L1": 0.5}):
        res = mod.fit()
    assert res.params["ar.L1"] == 0.5 and np.isnan(res.bse["ar.L1"]) and res.fixed_params == {"ar.L1": 0.5}
    assert "ar.L1" not in mod._fixed                      # the context manager restored the model
    res2 = mod.fit(start_params=res.params)               # warm start from a Series in display names
    assert res2.fixed_params == {} and abs(res2.params["ar.L1"] - 0.6) < 0.1


def test_d1_levels_trend_c_is_sarimax_intercept(data):
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    y, X = data
    lvl, Xl = 100 + y.cumsum(), X.cumsum()
    res = SARIMAXGARCH(lvl, exog=Xl, order=(1, 1, 0), trend="c", garch=(1, 1)).fit()
    assert res.params.index[0] == "intercept" and res.resid.isna().sum() == 2
    Xf = Xl.iloc[-1].values + np.cumsum(np.full((6, 2), 0.1), axis=0)
    fc = res.get_forecast(6, exog=Xf, n_sims=0)
    assert len(fc.predicted_mean) == 6 and fc.var_pred_mean.is_monotonic_increasing
    # same meaning as SARIMAX trend="c" with d=1: the constant of the differenced equation (the drift)
    ours = SARIMAXGARCH(lvl, exog=Xl, order=(1, 1, 0), trend="c", garch=(1, 1)).fit(
        fixed_params={"alpha.L1": 0.0, "beta.L1": 0.0})
    sm = SARIMAX(lvl, exog=Xl, order=(1, 1, 0), trend="c").fit(disp=False)
    for name in ("intercept", "sin12", "shock", "ar.L1"):
        assert abs(ours.params[name] - sm.params[name]) < 0.03, name
    with pytest.raises(NotImplementedError):                 # SARIMAX's "t" is a time trend inside dy
        SARIMAXGARCH(lvl, exog=Xl, order=(1, 1, 0), trend="t", garch=(1, 1))


def test_all_params_fixed_is_a_filter(data):
    y, X = data
    mod = SARIMAXGARCH(y, exog=X, order=(1, 0, 1), trend="c", garch=(1, 1))
    res = mod.fit()
    new = SARIMAXGARCH(y.iloc[:1000], exog=X.iloc[:1000], order=(1, 0, 1), trend="c", garch=(1, 1)).fit(
        fixed_params=dict(res.params))
    assert np.allclose(new.params.values, res.params.values) and new.bse.isna().all()
    assert np.allclose(new.conditional_variance.dropna().values, res.conditional_variance.iloc[1:1000].values)


def test_unsupported_options():
    y = np.random.default_rng(0).normal(size=300)
    with pytest.raises(NotImplementedError):
        SARIMAXGARCH(y, order=(1, 0, 0), seasonal_order=(1, 0, 0, 12), garch=(1, 1))
    with pytest.raises(NotImplementedError):
        SARIMAXGARCH(y, order=(1, 0, 0), trend="ct", garch=(1, 1))
    with pytest.raises(NotImplementedError):
        SARIMAXGARCH(y, order=(1, 0, 0), trend="t", garch=(1, 1))
    with pytest.raises(NotImplementedError):
        SARIMAXGARCH(y, order=(1, 0, 0), garch=(2, 1))
    with pytest.raises(NotImplementedError):
        SARIMAXGARCH(y, order=(1, 0, 0), garch=(1, 1), simple_differencing=True)
    with pytest.raises(ValueError):
        SARIMAXGARCH(y, order=(1, 0, 0), garch=(1, 1)).fit(fixed_params={"sigma2": 1.0})


def test_diagnostics_and_bootstrap(data):
    y, X = data
    res = SARIMAXGARCH(y, exog=X, order=(1, 0, 1), trend="c", garch=(1, 1)).fit()
    lb = res.test_serial_correlation(lags=12)
    assert "lb_pvalue" in lb.columns and len(lb) == 12
    assert 0 <= res.test_arch(12)["lm_pvalue"] <= 1 and 0 <= res.test_normality()["jb_pvalue"] <= 1
    bs = res.bootstrap(n_boot=19, seed=0)
    assert list(bs.index) == list(res.params.index) and (bs["ci_lower"] <= bs["ci_upper"]).all()
    pr = res.get_prediction()
    assert pr.predicted_mean.index.equals(y.index) and np.allclose(pr.predicted_mean.dropna(), res.fittedvalues.dropna())


def test_backtest_both_backends(data):
    y, X = data
    y2, X2 = y.iloc[:400], X.iloc[:400]
    for garch in ((1, 1), None):
        bt = SARIMAXGARCH(y2, exog=X2, order=(1, 0, 1), trend="c", garch=garch).backtest(horizon=6, start=300, step=25)
        assert list(bt.columns) == ["rmse", "mae", "crps", "coverage90"] and len(bt) == 6
        assert np.isfinite(bt.values).all() and (bt["crps"] <= bt["rmse"]).all()


def _trending_exog_case():
    """Sales cointegrated with two I(1) macro drivers (levels ~120, ~6, ~2): statsmodels' default
    SARIMAX fit (maxiter=50) stops early on this data."""
    T = 168
    idx = pd.date_range("2011-01-01", periods=T, freq="MS")
    rng = np.random.default_rng(101)
    X = pd.DataFrame({"unemployment": 6.0 + np.cumsum(rng.normal(0, 0.15, T + 24))[:T],
                      "gdp_growth": 2.0 + np.cumsum(rng.normal(0, 0.20, T + 24))[:T]}, index=idx)
    full = np.column_stack([6.0 + np.cumsum(np.random.default_rng(101).normal(0, 0.15, T + 24)),
                            2.0 + np.cumsum(np.random.default_rng(101).normal(0, 0.20, T + 24))])
    y, _, _ = _core.simulate(T + 24, 0.0, [0.8], [], 1.0, 0.0, 0.0, dist="normal", X=full, gamma=[-3.0, 2.0], seed=102)
    return pd.Series(120.0 + y[:T], index=idx, name="sales"), X


def test_statsmodels_backend_reaches_the_mle_on_trending_exog():
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    y, X = _trending_exog_case()
    res = SARIMAXGARCH(y, exog=X, order=(1, 0, 0), trend="c").fit(disp=False)
    nm = SARIMAX(y, exog=X, order=(1, 0, 0), trend="c").fit(disp=False, method="nm", maxiter=50_000)
    plain = SARIMAX(y, exog=X, order=(1, 0, 0), trend="c").fit(disp=False)        # statsmodels' defaults
    assert res.mle_retvals["converged"]
    assert res.llf >= nm.llf - 0.05 and res.llf >= plain.llf - 1e-9    # flat ridge: within 0.05 loglik units
    assert abs(res.params["ar.L1"] - nm.params["ar.L1"]) < 0.02
    if not plain.mle_retvals["converged"]:                     # the failure this guards against
        assert res.llf > plain.llf + 1.0


def test_backtest_reports_convergence():
    y, X = _trending_exog_case()
    bt = SARIMAXGARCH(y, exog=X, order=(1, 0, 0), trend="c").backtest(horizon=12, start=96, step=12)
    assert bt.attrs["n_origins"] == 6 and bt.attrs["n_not_converged"] == 0


def test_core_in_sync_with_standalone_script():
    here = os.path.dirname(os.path.abspath(__file__))
    standalone = os.path.join(here, "..", "..", "fixed", "joint_armax_garch.py")
    if not os.path.exists(standalone):
        pytest.skip("standalone script not present")
    core = os.path.join(here, "..", "src", "sarimax_garch", "_core.py")
    assert open(core).read() == open(standalone).read(), "_core.py differs from fixed/joint_armax_garch.py"
