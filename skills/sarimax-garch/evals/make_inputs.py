"""Synthetic inputs for the three skill evals, with known 'right answers' recorded in truth.json."""
import json
import os
import sys

import numpy as np
import pandas as pd

if len(sys.argv) < 2:
    sys.exit("usage: python make_inputs.py <workspace-dir>")
from sarimax_garch import _core as J  # noqa: E402

WS = os.path.abspath(sys.argv[1])
out = os.path.join(WS, "inputs")
truth = {}

# ---- eval 1: monthly sales cointegrated with two I(1) macro drivers, NO ARCH; 14y history + 24m scenario
T, H = 168, 24
idx = pd.date_range("2011-01-01", periods=T + H, freq="MS")
rng = np.random.default_rng(101)
unemp = 6.0 + np.cumsum(rng.normal(0, 0.15, T + H))
gdp = 2.0 + np.cumsum(rng.normal(0, 0.20, T + H))
X = np.column_stack([unemp, gdp])
y, _, _ = J.simulate(T + H, 0.0, [0.8], [], 1.0, 0.0, 0.0, dist="normal", X=X, gamma=[-3.0, 2.0], seed=102)
sales = 120.0 + y
d = os.path.join(out, "eval-1"); os.makedirs(d, exist_ok=True)
pd.DataFrame({"date": idx[:T].strftime("%Y-%m-%d"), "sales": np.round(sales[:T], 3)}).to_csv(f"{d}/sales.csv", index=False)
pd.DataFrame({"date": idx[:T].strftime("%Y-%m-%d"), "unemployment": np.round(unemp[:T], 3),
              "gdp_growth": np.round(gdp[:T], 3)}).to_csv(f"{d}/macro.csv", index=False)
pd.DataFrame({"date": idx[T:].strftime("%Y-%m-%d"), "unemployment": np.round(unemp[T:], 3),
              "gdp_growth": np.round(gdp[T:], 3)}).to_csv(f"{d}/macro_scenario.csv", index=False)
truth["eval-1"] = dict(dgp="levels, cointegrated (u AR(1) phi=.8), homoskedastic", right_d=0, right_garch=False,
                       future_dates=list(idx[T:].strftime("%Y-%m-%d")),
                       realised_sales_future=list(np.round(sales[T:], 3)))

# ---- eval 2: daily returns in percent, AR(1) phi=.3 + GARCH(.08,.90) t(5); 2000 business days
T2 = 2000
bidx = pd.bdate_range("2017-01-02", periods=T2)
r, _, _ = J.simulate(T2, 0.03, [0.3], [], 0.02, 0.08, 0.90, dist="t", nu=5.0, seed=202)
d = os.path.join(out, "eval-2"); os.makedirs(d, exist_ok=True)
pd.DataFrame({"date": bidx.strftime("%Y-%m-%d"), "ret": np.round(r, 5)}).to_csv(f"{d}/returns.csv", index=False)
truth["eval-2"] = dict(dgp="AR(1) phi=.3, c=.03, GARCH omega=.02 alpha=.08 beta=.90, t(5)", horizon=10)

# ---- eval 3: monthly level series with seasonality, d=1, macro regressor, mild GARCH; 15 years
T3 = 180
midx = pd.date_range("2010-01-01", periods=T3, freq="MS")
rng = np.random.default_rng(303)
macro = np.cumsum(rng.normal(0, 0.3, T3))
season = 2.0 * np.sin(2 * np.pi * np.arange(T3) / 12) + 1.0 * np.cos(2 * np.pi * np.arange(T3) / 12)
w, _, _ = J.simulate(T3, 0.15, [0.4], [0.2], 0.10, 0.10, 0.80, dist="normal",
                     X=np.diff(np.r_[0.0, macro])[:, None], gamma=[1.5], seed=304)
level = 200.0 + np.cumsum(w) + season
d = os.path.join(out, "eval-3"); os.makedirs(d, exist_ok=True)
pd.DataFrame({"date": midx.strftime("%Y-%m-%d"), "sales": np.round(level, 3)}).to_csv(f"{d}/sales_seasonal.csv", index=False)
pd.DataFrame({"date": midx.strftime("%Y-%m-%d"), "macro": np.round(macro, 4)}).to_csv(f"{d}/macro.csv", index=False)
with open(f"{d}/team_sarimax.py", "w") as fh:
    fh.write('''import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

sales = pd.read_csv("sales_seasonal.csv", parse_dates=["date"], index_col="date").asfreq("MS")["sales"]
macro = pd.read_csv("macro.csv", parse_dates=["date"], index_col="date").asfreq("MS")
res = SARIMAX(sales, exog=macro, order=(1, 1, 1), seasonal_order=(1, 0, 0, 12), trend="c").fit(disp=False)
print(res.summary())
''')
truth["eval-3"] = dict(dgp="d=1 ARMA(1,1) + macro (levels) + deterministic seasonal sin/cos in the level, GARCH(.10,.80)")
json.dump(truth, open(os.path.join(WS, "truth.json"), "w"), indent=1)
for k in ("eval-1", "eval-2", "eval-3"):
    print(k, sorted(os.listdir(os.path.join(out, k))))
