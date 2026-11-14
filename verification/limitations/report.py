"""Print the markdown tables of README.md from out/*.json (run after the e*.py scripts)."""
import json
import os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")


def load(name):
    with open(os.path.join(OUT, f"{name}.json")) as f:
        return json.load(f)


def e1():
    d = load("e1_backends")
    print("| design | nobs statsmodels / engine | llf gap engine − statsmodels | LR > 5.99: within / across | AIC picks GARCH: within / across | median \\|Δ\\| intercept · x1 · ar.L1 (SE) |")
    print("|---|---|---|---|---|---|")
    for k, v in d.items():
        m = v["median_dparam_in_sm_se"]
        print(f"| {k} | {v['nobs_sm']} / {v['nobs_engine']} | {v['gap_mean']:+.2f} (sd {v['gap_sd']:.2f}) | "
              f"{v['lr_within_gt_5_99']:.1%} / **{v['lr_cross_gt_5_99']:.1%}** | {v['aic_picks_garch_within']:.1%} / "
              f"**{v['aic_picks_garch_cross']:.1%}** | {m['intercept']:.2f} · {m['x1']:.2f} · {m['ar.L1']:.2f} |")


def e2():
    d = load("e2_bands")
    for name, s in d.items():
        print(f"\n**{name}** (R = {s['R']}; median ν̂ {s['nu_hat_median']:.1f}; true predictive excess kurtosis "
              f"h1 {s['true_excess_kurtosis_h']['h1']:.1f}, h20 {s['true_excess_kurtosis_h']['h20']:.1f}; point-forecast RMSE "
              f"vs the true conditional mean = {s['point_rmse_over_sd']['h1']:.2f} / {s['point_rmse_over_sd']['h20']:.2f} "
              f"predictive SD at h1 / h20)\n")
        print("| band | 80% (h1 · h20) | 90% | 95% | 99% | 99% band: below / above at h1 (nominal 0.5% each) |")
        print("|---|---|---|---|---|---|")
        for band, lv in s["coverage"].items():
            cells = [f"{lv[L]['h1'][0]:.3f} · {lv[L]['h20'][0]:.3f}" for L in ("80", "90", "95", "99")]
            b99 = lv["99"]["h1"]
            print(f"| {band} | " + " | ".join(cells) + f" | {b99[1]:.2%} / {b99[2]:.2%} |")


def e3():
    d = load("e3_skew")
    print("| design | z skew | t-MLE centre of z (theory) | estimator | c bias (bias/SD) | μ bias | φ bias/SD | g bias/SD | SD(g) | 95% CI cover c |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for k, v in d.items():
        for nm in ("SM", "GN", "GT", "GS"):
            p = v["params"][nm]
            cov = v["coverage95"].get(nm, {}).get("c")
            first = nm == "SM"
            print(f"| {k if first else ''} | {v['skew']:+.2f} | {v['pseudo_true_t']['loc']:+.3f} SD |"[0 if first else 0:] if first
                  else f"| | | |", end="")
            print(f" {nm} | {p['c']['bias']:+.4f} ({p['c']['bias_over_sd']:+.1f}) | {p['mu']['bias']:+.4f} | "
                  f"{p['phi']['bias_over_sd']:+.1f} | {p['g']['bias_over_sd']:+.1f} | {p['g']['sd']:.4f} | "
                  f"{'' if cov is None else f'{cov:.2f}'} |")
    print("\n1-step forecasts: point error vs the true conditional mean (in true σ_{T+1}) and the TRUE probability of"
          " falling outside each model's quantiles (nominal 1% / 1% / 5% / 5%)\n")
    print("| design | model | location error / σ | P(y < q01) | P(y > q99) | P(y < q05) | P(y > q95) |")
    print("|---|---|---|---|---|---|---|")
    for k, v in d.items():
        for i, (nm, le) in enumerate(v["loc_err"].items()):
            t = v["tails"][nm]
            print(f"| {k if i == 0 else ''} | {nm} | {le['mean']:+.3f} | {t[0]:.2%} | {t[1]:.2%} | {t[2]:.1%} | {t[3]:.1%} |")
    print("\ncheck available today, (c_t − c_normal) / SE(c_normal): mean · share > 2 · share |.| > 2 · mean skew of"
          " the normal fit's std_resid —", "; ".join(
              f"{k}: {v['check_t_vs_normal_intercept']['mean_dc_over_se']:+.2f} · "
              f"{v['check_t_vs_normal_intercept']['share_dc_over_2se']:.0%} · "
              f"{v['check_t_vs_normal_intercept']['share_abs_dc_over_2se']:.0%} · "
              f"{v['check_t_vs_normal_intercept']['mean_skew_std_resid']:+.2f}" for k, v in d.items()))
    print("\nskew-t prototype: mean λ̂ and the LR test of λ = 0 (5%) —",
          "; ".join(f"{k}: λ̂ {v['lam_s_mean']:+.3f}, rejects {v['lr_skew_reject_5pct']:.0%}" for k, v in d.items()))


def e4():
    d = load("e4_leverage")
    print("| design | estimator | c bias (bias/SD) | μ bias | φ bias/SD | g bias/SD |")
    print("|---|---|---|---|---|---|")
    for k, v in d.items():
        for i, nm in enumerate(("SM", "GN", "GT", "GJR")):
            p = v["params"][nm]
            print(f"| {k if i == 0 else ''} | {nm} | {p['c']['bias']:+.4f} ({p['c']['bias_over_sd']:+.2f}) | "
                  f"{p['mu']['bias']:+.4f} | {p['phi']['bias_over_sd']:+.2f} | {p['g']['bias_over_sd']:+.2f} |")
    print("\nGN 95% robust-Wald coverage of c, and the GJR prototype's LR test of lev = 0 —",
          "; ".join(f"{k}: cover c {v['gn_coverage95']['c']:.2f}, lev̂ {v['lev_hat_mean']:.3f}, LR rejects "
                    f"{v['lr_lev_reject_5pct']:.0%}" for k, v in d.items()))


def e5():
    d = load("e5_converged")
    print(f"{d['n']} fits; flagged converged=False: {d['flagged_not_converged']} "
          f"(of these {d['gap_flagged']['share_below_1e_4']:.0%} are at the best optimum found); "
          f"fits more than x log-lik units below the best of 9 starts: {d['all_fits_gap_over']}; "
          f"worst 'converged=True' fit: {d['gap_converged']['max']:.2f} units below")


if __name__ == "__main__":
    for name, fn in (("E1", e1), ("E2", e2), ("E3", e3), ("E4", e4), ("E5", e5)):
        print(f"\n### {name}\n")
        fn()
