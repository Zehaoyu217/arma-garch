# Evidence behind the package and the skill's recommendations

Full write-up, scripts and raw numbers: `verification/README.md` in the arma-garch repo
(re-run with `verification/run_all.sh`; `GRID=1` adds the grids). Summary:

## Correctness (same model, same numbers)
- At identical parameters, residuals, σₜ and per-observation log-likelihood match **R rugarch** (28 cases: ARMA
  orders up to (2,2), normal and t, d = 0/1/2, five GARCH settings) to ~1e-14 once the start-up has decayed;
  forecasts match to ~5e-15.
- With GARCH off, residuals match **statsmodels SARIMAX** to 1e-15 in 42 cases, and level forecasts for
  d = 0/1/2 match in mean (~1e-14 relative) and variance (2e-15 relative).
- **Python arch** (26 AR-GARCH cases): σₜ match ~1e-15; estimates within 0.015 SE; robust and Hessian SEs
  within 0.8%.
- Reproduces the published Fiorentini–Calzolari–Panattoni (1996) GARCH(1,1) benchmark (coefficients to 1e-7,
  log-likelihood exactly, SEs to 3e-5).
- Known rugarch quirks (not package bugs): its reported robust SEs differ from its own H⁻¹S′SH⁻¹ by 5–10%; its
  default Hessian step (`hess.d = 0.1`) halves the GARCH SEs when α+β ≈ 1; `ugarchfit` can stop at α = 0 while
  reporting convergence.

## Statistical behaviour (Monte Carlo, 13,000 fits, T = 500–16,000)
Estimates are consistent and asymptotically normal: bias/SD shrinks with T, SD·√T is flat, robust SE / true SD
goes from 0.3–1.05 at T = 500 to 0.97–1.05 at T = 8,000, and 95% coverage reaches nominal (α: 0.87 at
T = 500 → 0.965 at 8,000). With t(5) data fitted as normal, robust-SE coverage is 0.90 while Hessian-SE
coverage is 0.74 — use the default robust SEs.

## Small samples (monthly, AR(1) + seasonal exog, GARCH persistence .85, 600 reps per cell)

| Years (T) | φ₁ / γ coverage | fits with α or β on a boundary | α / β Wald coverage | ν at its bound (t) |
|---|---|---|---|---|
| 5 (60) | 0.87 / 0.89 | 71–77% | 0.55–0.61 / 0.50–0.56 | 29–34% |
| 10 (120) | 0.92–0.94 / 0.91–0.93 | 46–51% | 0.59–0.67 / 0.56–0.58 | 14–16% |
| 15 (180) | 0.93–0.94 / 0.93–0.94 | 29–32% | 0.75–0.77 / 0.67–0.68 | 5–6% |
| 30 (360) | 0.94–0.95 / 0.94–0.95 | 9–13% | 0.80–0.85 / 0.74–0.77 | 0% |

Bootstrap percentile CIs at T = 120 / 180 cover ω, α, β at 0.85 / 0.73 / 0.93 and 0.94 / 0.77 / 0.91 (vs Wald
0.53 / 0.57 / 0.49 and 0.65 / 0.73 / 0.66), but under-cover the mean parameters (0.81–0.84 at 120) — use Wald
for those. rugarch on the same series behaves the same (more boundary fits). For the ARMA part alone, the
conditional likelihood and SARIMAX's exact likelihood give the same RMSE from T = 60 on.

## When does GARCH beat a constant variance? (1-step variance forecast, QLIKE loss ratio GARCH / constant, share of samples GARCH wins, ARCH-LM power)

| ARCH strength (α, β) | 10 years | 15 years | 20 years | 30 years |
|---|---|---|---|---|
| weak (.05, .85) | 1.58 · 45% · 7% | 1.25 · 46% · 11% | 0.89 · 54% · 15% | 0.71 · 60% · 22% |
| typical (.10, .75) | 1.09 · 49% · 10% | 0.59 · 59% · 20% | 0.48 · 64% · 28% | 0.33 · 73% · 43% |
| strong (.15, .75) | 0.38 · 66% · 22% | 0.24 · 74% · 46% | 0.20 · 71% · 63% | 0.13 · 82% · 82% |
| persistent (.08, .90) | 0.51 · 62% · 16% | 0.34 · 71% · 30% | 0.22 · 77% · 47% | 0.12 · 84% · 69% |

## Multi-step, macro path given (36 months; CRPS / oracle, 90% coverage in brackets; at 1 · 12 · 36 months)

| Truth | Years | Levels model (d = 0) | Differences model (d = 1) |
|---|---|---|---|
| y cointegrated with X | 10 | 1.02 (.86) · 1.12 (.83) · 1.18 (.79) | 1.03 (.86) · 1.44 (.84) · 1.76 (.86) |
| y cointegrated with X | 30 | 1.01 (.90) · 1.04 (.85) · 1.08 (.86) | 1.02 (.88) · 1.16 (.84) · 1.38 (.89) |
| Δy driven by ΔX only | 10 | 1.09 (.89) · 1.20 (.68) · 1.49 (.55) | 1.03 (.87) · 1.10 (.83) · 1.22 (.83) |
| Δy driven by ΔX only | 30 | 1.03 (.89) · 1.11 (.71) · 1.22 (.70) | 0.99 (.87) · 1.01 (.92) · 1.07 (.90) |

GARCH vs constant variance changed CRPS, RMSE and coverage by at most 0.02–0.03 at every horizon here (a 2–4%
CRPS edge through ~12 months only when persistence is .98). The levels-vs-differences choice is what matters.
