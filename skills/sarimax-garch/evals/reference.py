"""Reference answers for grading, computed with the package itself."""
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

from sarimax_garch import SARIMAXGARCH, arch_lm_test

warnings.filterwarnings("ignore")
WS = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else sys.exit("usage: python reference.py <workspace-dir>")
I = os.path.join(WS, "inputs")
ref = {}

# eval 1
s = pd.read_csv(f"{I}/eval-1/sales.csv", parse_dates=["date"], index_col="date").asfreq("MS")["sales"]
m = pd.read_csv(f"{I}/eval-1/macro.csv", parse_dates=["date"], index_col="date").asfreq("MS")
res0 = SARIMAXGARCH(s, exog=m, order=(1, 0, 0), trend="c").fit(disp=False)
stat, p = arch_lm_test(res0.resid.iloc[res0.loglikelihood_burn:], 12)
bt_l = SARIMAXGARCH(s, exog=m, order=(1, 0, 0), trend="c").backtest(horizon=24, start=96, step=6)
bt_d = SARIMAXGARCH(s, exog=m, order=(1, 1, 1), trend="c").backtest(horizon=24, start=96, step=6)
ref["eval-1"] = dict(arch_lm_p=p, min_ar_root=float(np.abs(res0.arroots).min()),
                     crps_levels=bt_l["crps"].round(3).tolist(), crps_diffs=bt_d["crps"].round(3).tolist(),
                     cov_levels=bt_l["coverage90"].round(2).tolist(), cov_diffs=bt_d["coverage90"].round(2).tolist())

# eval 2
r = pd.read_csv(f"{I}/eval-2/returns.csv", parse_dates=["date"], index_col="date")["ret"]
res2 = SARIMAXGARCH(r, order=(1, 0, 0), trend="c", garch=(1, 1), dist="t").fit()
fc = res2.get_forecast(10, n_sims=0)
ref["eval-2"] = dict(params=res2.params.round(4).to_dict(), persistence=res2.persistence, half_life=res2.half_life,
                     cond_vol=fc.conditional_volatility.round(5).tolist(), se_mean=fc.se_mean.round(5).tolist(),
                     ratio_h10=float(fc.se_mean.iloc[-1] / fc.conditional_volatility.iloc[-1]))
res2n = SARIMAXGARCH(r, order=(1, 0, 0), garch=(1, 1), dist="t").fit()               # trend=None: no constant
ref["eval-2"]["cond_vol_no_constant_h10"] = float(res2n.get_forecast(10, n_sims=0).conditional_volatility.iloc[-1])

# eval 3: the team's original code runs and what a ported GARCH model looks like
s3 = pd.read_csv(f"{I}/eval-3/sales_seasonal.csv", parse_dates=["date"], index_col="date").asfreq("MS")["sales"]
m3 = pd.read_csv(f"{I}/eval-3/macro.csv", parse_dates=["date"], index_col="date").asfreq("MS")
orig = SARIMAXGARCH(s3, exog=m3, order=(1, 1, 1), seasonal_order=(1, 0, 0, 12), trend="c").fit(disp=False)
try:
    SARIMAXGARCH(s3, exog=m3, order=(1, 1, 1), seasonal_order=(1, 0, 0, 12), trend="c", garch=(1, 1))
    raised = False
except NotImplementedError:
    raised = True
ref["eval-3"] = dict(original_llf=float(orig.llf), seasonal_plus_garch_raises=raised)
json.dump(ref, open(os.path.join(WS, "reference.json"), "w"), indent=1, default=float)
print(json.dumps(ref, indent=1, default=float)[:2500])
