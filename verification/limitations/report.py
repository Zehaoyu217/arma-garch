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


def e6():
    d = load("e6_sarimax_vs_garch")
    print("Mean parameters: RMSE GN / SM (< 1 = GARCH more accurate) and 95% CI coverage (SM opg · SM robust · GN robust)\n")
    print("| design | ARMA coef RMSE SM → GN | slope RMSE SM → GN | ARMA coef cover | slope cover | intercept cover |")
    print("|---|---|---|---|---|---|")
    for k, v in d.items():
        arma = "phi" if "AR1" in k else "theta"
        P, C = v["params"], v["coverage"]
        cov = lambda p: " · ".join(f"{C[n][p]:.2f}" for n in ("SM opg", "SM robust", "GN robust"))
        print(f"| {k.replace(' | ', ', ')} | {P['SM'][arma]['rmse']:.3f} → {P['GN'][arma]['rmse']:.3f} | {P['SM']['x1']['rmse']:.3f} → "
              f"{P['GN']['x1']['rmse']:.3f} | {cov(arma.replace('phi', 'phi').replace('theta', 'theta'))} | {cov('x1')} | "
              f"{cov('intercept')} |")
    print("\nForecasts: CRPS GN / SM at h1 · h6 · h12; 90% coverage at h1 (SM · GN) overall, calm, turbulent; "
          "point RMSE vs the true conditional mean (in predictive SD) at h1, SM · GN\n")
    print("| design | CRPS GN/SM h1 · h6 · h12 | cover h1 all | calm | turbulent | CRPS GN/SM calm · turbulent | point RMSE h1 |")
    print("|---|---|---|---|---|---|---|")
    for k, v in d.items():
        F = v["forecast"]
        f1 = F["h1"]
        print(f"| {k.replace(' | ', ', ')} | {f1['crps_ratio']:.3f} · {F['h6']['crps_ratio']:.3f} · {F['h12']['crps_ratio']:.3f} | "
              f"{f1['cov_sm']:.2f} · {f1['cov_gn']:.2f} | {f1['cov_sm_calm']:.2f} · {f1['cov_gn_calm']:.2f} | "
              f"{f1['cov_sm_turb']:.2f} · {f1['cov_gn_turb']:.2f} | {f1['crps_ratio_calm']:.3f} · "
              f"{f1['crps_ratio_turb']:.3f} | {f1['rmse_sm']:.3f} · {f1['rmse_gn']:.3f} |")
    print("\nDetection: share of series where each rule chooses GARCH (truth 'none' = false positives)\n")
    rules = list(next(iter(d.values()))["detection"])
    print("| design | " + " | ".join(rules) + " | GN on boundary |")
    print("|---|" + "---|" * (len(rules) + 1))
    for k, v in d.items():
        print(f"| {k.replace(' | ', ', ')} | " + " | ".join(f"{v['detection'][r]:.2f}" for r in rules) + f" | {v['flags']['boundary']:.2f} |")


def e8():
    d = load("e8_holdout_logscore")
    print("Share of series where a Diebold-Mariano test (5%) on the holdout scores says the first model is "
          "significantly better (in brackets: the second), and the in-sample LR test on the training data\n")
    print("| design | LR in-sample | GN vs SM: log score | GN vs SM: CRPS | Δ log score (nats/obs) · CRPS ratio | "
          "GT vs GN: log score | GT vs GN: CRPS | Δ log score · CRPS ratio |")
    print("|---|---|---|---|---|---|---|---|")
    for k, v in d.items():
        print(f"| {k.replace(' | ', ', ')} | {v['lr_in_sample']:.2f} | {v['ls_win']:.2f} ({v['ls_loss']:.2f}) | "
              f"{v['crps_win']:.2f} ({v['crps_loss']:.2f}) | {v['d_logscore']:+.4f} · {v['crps_ratio']:.4f} | "
              f"{v['ls_t_win']:.2f} ({v['ls_t_loss']:.2f}) | {v['crps_t_win']:.2f} ({v['crps_t_loss']:.2f}) | "
              f"{v['d_logscore_t']:+.4f} · {v['crps_ratio_t']:.4f} |")


def e10():
    d = load("e10_sample_size")
    for fam in ("ARIMA(1,1,1)", "reg + ARIMA(1,1,1)", "airline"):
        names = list(d[f"{fam} | GARCH none | T=60"]["wald_size"]["SM opg"])
        print(f"\n**{fam}** — detection = share choosing GARCH (5%); boundary = GARCH fit with alpha or beta ~ 0; QLIKE = "
              f"one-step variance-forecast loss GARCH / constant; Wald size at the true {', '.join(names)}: "
              "SM opg · SM robust · OS robust\n")
        print("| GARCH | T | ARCH-LM(12) | LR one-stage | LR two-stage | boundary OS · TS | QLIKE OS · TS | "
              "persistence RMSE OS · TS | Wald size |")
        print("|---|---|---|---|---|---|---|---|---|")
        for g in ("none", "typical", "persistent"):
            for T in (60, 120, 240, 360, 600):
                v = d[f"{fam} | GARCH {g} | T={T}"]
                f2 = lambda dd, k: f"{dd[k]:.2f}" if k in dd else "–"
                ws = " · ".join("/".join(f"{x:.2f}" for x in v["wald_size"][k].values())
                                for k in ("SM opg", "SM robust", "OS robust") if k in v["wald_size"])
                pr = v["persistence_rmse"]
                print(f"| {g} | {T} | {f2(v['detection'], 'ARCH-LM(12)')} | {f2(v['detection'], 'LR one-stage')} | "
                      f"{f2(v['detection'], 'LR two-stage')} | {f2(v['boundary'], 'OS')} · {f2(v['boundary'], 'TS')} | "
                      f"{f2(v['qlike_ratio'], 'OS')} · {f2(v['qlike_ratio'], 'TS')} | "
                      f"{(f2(pr, 'OS') + ' · ' + f2(pr, 'TS')) if pr else '–'} | {ws} |")


def e11():
    d = load("e11_valid_inference")
    s = load("e11_size_check")
    print("(a) Rejection rate at 5% (GARCH none = size); the restricted bootstrap rejects when p* <= 0.05 (B = 99)\n")
    print("| design | chi2(2) LR | restricted bootstrap LR | ARCH-LM(4) | series |")
    print("|---|---|---|---|---|")
    for k, v in d["a"].items():
        print(f"| {k.replace(' | ', ', ')} | {v['chi2']:.3f} | {v['boot']:.3f} | {v['lm']:.3f} | {v['R']} |")
    print(f"\nFresh-seed size check (AR1, no GARCH, T = 120, {s['n']} series): restricted bootstrap {s['boot_size']:.3f}, "
          f"chi2(2) {s['chi2_size']:.3f}; LR null distribution true DGP vs bootstrap DGP: q95 {s['q95'][0]:.2f} / "
          f"{s['q95'][1]:.2f}, KS p {s['ks_p']:.2f}")
    print("\n(b) 95% interval coverage for x1 / ar.L1 after selecting GARCH-or-not by the LR test\n")
    print("| design | GARCH selected | always SARIMAX opg | always SARIMAX robust | always GARCH | select, then defaults | "
          "select, then robust | select-then-robust given GARCH / given SARIMAX |")
    print("|---|---|---|---|---|---|---|---|")
    for k, v in d["b"].items():
        c, cc = v["coverage"], v["conditional"]["post_robust"]
        f = lambda dd: "/".join("–" if x is None else f"{x:.3f}" for x in dd.values())
        print(f"| {k.replace(' | ', ', ')} | {v['share_garch_selected']:.2f} | {f(c['sm_opg'])} | {f(c['sm_rob'])} | "
              f"{f(c['garch'])} | {f(c['post_default'])} | {f(c['post_robust'])} | "
              f"{f(cc['given GARCH selected'])} · {f(cc['given SARIMAX selected'])} |")


def e12():
    d = load("e12_power_source")
    models = ["oracle", "mean", "OLS", "AR(1)", "ARMA(1,1)", "MA(1) .9", "ARIMA(1,1,1)", "airline"]
    for test in ("LR one-stage", "LR two-stage", "ARCH-LM(4)", "ARCH-LM(12)"):
        print(f"\n**{test}** — share rejecting at 5% (GARCH none = size)\n")
        print("| design | " + " | ".join(models) + " |")
        print("|---|" + "---|" * len(models))
        for key, v in d.items():
            print(f"| {key.replace(' | ', ', ')} | " + " | ".join(f"{v[m][test]:.2f}" if test in v[m] else "–"
                                                         for m in models) + " |")


if __name__ == "__main__":
    for name, fn in (("E1", e1), ("E2", e2), ("E3", e3), ("E4", e4), ("E5", e5), ("E6", e6), ("E8", e8), ("E10", e10),
                     ("E11", e11), ("E12", e12)):
        print(f"\n### {name}\n")
        fn()
