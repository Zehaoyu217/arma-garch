#!/usr/bin/env bash
# Re-run the full verification of joint_armax_garch.py against R rugarch + the FCP (1996) benchmark.
#   ./run_all.sh            -> verifies ../fixed/joint_armax_garch.py (v2.2)
#   ./run_all.sh ..         -> verifies ../joint_armax_garch.py      (original v2.1)
# Needs: python3 with numpy/scipy (arch optional, for self-test 2); R with rugarch, fGarch, jsonlite
#        (one-time: Rscript install_r_packages.R  -- R itself: brew install r nlopt pkgconf)
set -euo pipefail
cd "$(dirname "$0")"
export JAG_DIR="$(cd "${1:-../fixed}" && pwd)"
export CMP_DIR="cmp_$(basename "$JAG_DIR")"
PY="${PY:-python3}"
mkdir -p "$CMP_DIR"
echo "== verifying $JAG_DIR/joint_armax_garch.py  (outputs in verification/$CMP_DIR)"
echo "== 1. built-in self-test";               "$PY" -W ignore::RuntimeWarning "$JAG_DIR/joint_armax_garch.py" | grep -E "\[(PASS|FAIL)\]|^[0-9]\)|OVERALL"
echo "== 2. vectorised vs naive-loop loglik";  "$PY" -W ignore ref_loop.py
echo "== 3. scale invariance of SEs";          "$PY" scale_probe.py 2>&1 | grep -E "scale=|robust SE|Error" || true
echo "== 4. fit 7 cases with the script";      "$PY" make_cases.py | cut -c1-70
echo "== 5. same cases in rugarch";            Rscript rugarch_side.R "$CMP_DIR" 2>&1 | grep -v "^$" | tail -7
echo "== 6. filter / fit / forecast 1:1";      "$PY" compare.py | tee "$CMP_DIR/compare_report.txt"
echo "== 7. rugarch robust-SE internals";      Rscript rugarch_robust_se_check.R "$CMP_DIR" 2>&1 | tail -8
echo "== 8. xreg model-definition check";      Rscript rugarch_xreg_filter.R "$CMP_DIR" && "$PY" xreg_reparam_check.py
echo "== 9. Monte Carlo SE coverage";          "$PY" mc.py
if grep -q 'xreg="armax"' "$JAG_DIR/joint_armax_garch.py"; then      # v2.3+ options
echo "== 10. xreg='regression' + fixed const vs rugarch"
"$PY" new_modes_make.py && Rscript new_modes_rugarch.R "$CMP_DIR" 2>&1 | grep -v "^$" | tail -2 && "$PY" new_modes_compare.py
echo "== 11. xreg='regression' vs statsmodels ARIMA(exog)"; "$PY" statsmodels_check.py
echo "== 12. xreg='regression' vs statsmodels SARIMAX(exog)"; "$PY" sarimax_check.py
fi
if [ "${GRID:-0}" = "1" ] && grep -q '__version__ = "3' "$JAG_DIR/joint_armax_garch.py"; then   # GRID=1 ./run_all.sh
  cd grid
  echo "== G1. 28-case grid vs rugarch";       "$PY" rugarch_make.py > /dev/null && Rscript rugarch_grid.R out > /dev/null 2>&1 \
                                               && "$PY" rugarch_compare.py | sed -n '/^SUMMARY/,$p'
  echo "== G2. 42-case ARIMA grid vs SARIMAX"; "$PY" statsmodels_grid.py | sed -n '/^SUMMARY/,$p'
  echo "== G3. 26-case GARCH grid vs arch";    "$PY" arch_grid.py | sed -n '/^SUMMARY/,$p'
  echo "== G4. Monte Carlo grid";              "$PY" mc_grid.py ${MC_QUICK:+--quick} | tail -6
  echo "== G5. small samples (monthly, 5-30y)"; "$PY" small_sample.py ${MC_QUICK:+--quick} | tail -24
  echo "== G6. small-sample follow-ups";        "$PY" small_sample_extra.py ${MC_QUICK:+--quick}
  echo "== G7. GARCH vs constant variance by ARCH strength"; "$PY" small_sample_threshold.py ${MC_QUICK:+--quick} | tail -24
  echo "== G8. simulator vs theoretical moments"; "$PY" simulator_check.py | grep -E "FAIL|SIMULATOR"
  echo "== G9. multi-horizon, levels vs differences, macro given"; "$PY" horizon_study.py ${MC_QUICK:+--quick} | head -1
  cd ..
fi
if [ -d ../sarimax_garch ] && "$PY" -c "import sarimax_garch" 2>/dev/null; then
  echo "== P1. package tests (sarimax_garch, statsmodels parity + engine sync)"
  (cd ../sarimax_garch && "$PY" -m pytest -q 2>&1 | tail -1)
fi
