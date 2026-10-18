# arma-garch

Regression with ARIMA(p,d,q)-GARCH(1,1) errors — normal or Student-t — estimated by joint conditional
maximum likelihood, with a statsmodels-SARIMAX-style package on top and a verification suite against R
rugarch, statsmodels SARIMAX and Python `arch`.

| Path | What it is |
|---|---|
| `sarimax_garch/` | **The package.** `SARIMAXGARCH(endog, exog, order, trend, garch=None/(1,1), dist="normal"/"t")`: with GARCH off it *is* statsmodels SARIMAX (its own results object); with GARCH or t errors it switches to the joint estimator behind the same attribute names (`params`, `bse`, `summary()`, `get_forecast()`, …). GARCH or seasonal: pick one. See `sarimax_garch/README.md`. |
| `skills/sarimax-garch/` | **A Claude skill** for using the package. It covers setup, the two backends, and the modelling workflow: levels vs differences, GARCH or not, small samples, forecasting with a given exog path, backtesting and the gotchas. It also has an API reference, the evidence behind its advice, and evals (`evals/`). Install for Claude Code: `ln -s "$PWD/skills/sarimax-garch" ~/.claude/skills/sarimax-garch`. The package's test suite runs every code example in the skill. |
| `fixed/joint_armax_garch.py` | The estimation engine, v3.2 (single file, numpy/scipy only). The package ships a byte-identical copy as `_core.py`; a test keeps them in sync. `v2.1_to_v3.2.diff` is the full change log against the original. |
| `joint_armax_garch.py` | The original v2.1 script, kept unchanged for reference and regression testing. |
| `verification/` | Everything used to validate the engine. Results and recommendations are in `verification/README.md`. |

## Quick start

```bash
cd sarimax_garch && pip install -e . && pytest          # 16 tests
```
```python
from sarimax_garch import SARIMAXGARCH
res = SARIMAXGARCH(y, exog=X, order=(1, 0, 1), trend="c", garch=(1, 1)).fit()
print(res.summary())
fc = res.get_forecast(12, exog=X_future).summary_frame()
```
The engine also runs standalone: `python fixed/joint_armax_garch.py` runs its 53-check self-test.

## What has been verified (details in `verification/README.md`)

- **Same model, same numbers:** residuals, σ_t, per-observation log-likelihood and forecasts match rugarch
  (28 cases), statsmodels SARIMAX with GARCH off (42 cases, including d = 1, 2 level forecasts) and `arch`
  (26 cases) to about 1e-14; it reproduces the published Fiorentini–Calzolari–Panattoni (1996) GARCH benchmark.
- **Statistical behaviour:** a 13,000-fit Monte Carlo (ARMA orders up to (2,2), d = 0/1/2, five GARCH
  settings, normal/t errors, T = 500–16,000) shows consistent, asymptotically normal estimates with correctly
  sized robust standard errors.
- **Small samples and forecasting:** monthly data with 5–30 years (36,000 fits), GARCH vs constant variance
  by ARCH strength, and multi-step macro-conditioned forecasts in levels vs differences.

## Re-running the verification

```bash
cd verification
Rscript install_r_packages.R      # once: rugarch, fGarch, numDeriv, jsonlite (needs R; macOS: brew install r nlopt pkgconf)
./run_all.sh                      # v3.2: self-test, rugarch/statsmodels 1:1 checks, Monte Carlo, package tests
./run_all.sh ..                   # the same against the original v2.1 script
GRID=1 ./run_all.sh               # plus the full grids and small-sample studies (MC_QUICK=1 for a quick pass)
```
Set `PY=/path/to/python` to choose the interpreter (one with numpy, scipy, pandas, statsmodels; `arch` optional).
