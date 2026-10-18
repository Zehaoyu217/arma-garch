"""sarimax_garch -- statsmodels-SARIMAX-style regression with ARIMA errors, GARCH opt-in.

    from sarimax_garch import SARIMAXGARCH

    res = SARIMAXGARCH(y, exog=X, order=(1, 0, 1), trend="c").fit()                 # == statsmodels SARIMAX
    res = SARIMAXGARCH(y, exog=X, order=(1, 0, 1), trend="c", garch=(1, 1)).fit()   # joint ARMA-GARCH MLE
    res = SARIMAXGARCH(y, exog=X, order=(1, 1, 0), trend="t", garch=(1, 1), dist="t").fit()

    res.params, res.bse, res.pvalues, res.llf, res.aic, res.summary()
    fc = res.get_forecast(12, exog=X_future); fc.predicted_mean, fc.se_mean, fc.summary_frame()
"""
from ._core import arch_lm_test, ljung_box
from .model import GARCHPredictionResults, SARIMAXGARCH, SARIMAXGARCHResults

__version__ = "0.1.0"
__all__ = ["SARIMAXGARCH", "SARIMAXGARCHResults", "GARCHPredictionResults", "arch_lm_test", "ljung_box"]
