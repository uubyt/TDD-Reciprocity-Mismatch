import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt


def load_results(json_path):
    with open(json_path, "r") as f:
        return json.load(f)


def get_default_json_path(results_dir="./results_e2e"):
    json_files = [
        f for f in os.listdir(results_dir)
        if f.endswith(".json") and f.startswith("e2e_link_")
    ]
    if not json_files:
        raise FileNotFoundError(f"No e2e_link_*.json found in {results_dir}")
    json_files = sorted(
        json_files,
        key=lambda x: os.path.getmtime(os.path.join(results_dir, x)),
        reverse=True,
    )
    return os.path.join(results_dir, json_files[0])


def clean_metric_values(values):
    out = []
    for v in values:
        out.append(np.nan if v is None else float(v))
    return np.asarray(out, dtype=float)


def build_scheme_label(scheme_name, results):
    """Build readable curve labels."""
    cfg = results.get("config", {})
    mismatch = cfg.get("test_mismatch", {})

    amp = mismatch.get("std_amp_db", None)
    phase = mismatch.get("std_phase_deg", None)

    if amp is not None and phase is not None:
        mismatch_text = f"{amp:.1f} dB, {phase:.1f} deg"
    else:
        mismatch_text = "unknown mismatch"

    label_map = {
        # Reference curves
        "ideal_svd": "Perfect-CSI SVD",
        "ideal_svd_ml": "Perfect-CSI SVD + ML Detector",
        "error_svd": f"Naive Reciprocity ({mismatch_text})",

        # Traditional / optimization baselines
        "ls_rc_svd": f"LS-RC ({mismatch_text})",
        "mc_sgd": f"MC-SGD ({mismatch_text})",

        # Learning baselines
        "nn_calib_svd": f"DNN-Calib SVD ({mismatch_text})",
        "moe_calib_svd": f"MoE-Calib SVD ({mismatch_text})",
        "moe_refiner_calib_svd": f"MoE+Refiner SVD ({mismatch_text})",
        "moe_preco_refiner_svd": f"MoE+W-Refiner SVD ({mismatch_text})",
        "moe_preco_refiner_svd_ml": f"MoE+W-Refiner SVD + ML ({mismatch_text})",

        # Old optional baseline
        "sa_rescal_svd": f"SA-ResCalNet SVD ({mismatch_text})",

        # SA-Mult ablation variants
        "sa_mult_nores_nocond_svd": f"SA-Mult w/o Cond. & Res. ({mismatch_text})",
        "sa_mult_nores_svd": f"SA-Mult w/o Res. ({mismatch_text})",
        "sa_mult_full_svd": f"Proposed SA-Mult ({mismatch_text})",

        # Genie references
        "genie_preco_refiner_svd": "Genie W-Refiner",
        "genie_direct_precoder": "Genie Direct Precoder",
    }

    return label_map.get(scheme_name, scheme_name)


def get_style_map():
    return {
        "ideal_svd": {
            "marker": "o", "linestyle": "-", "linewidth": 2.2,
            "markersize": 7, "fillstyle": "full", "zorder": 3,
        },
        "ideal_svd_ml": {
            "marker": "o", "linestyle": "--", "linewidth": 2.2,
            "markersize": 7, "fillstyle": "full", "zorder": 3,
        },
        "error_svd": {
            "marker": "s", "linestyle": "-", "linewidth": 2.0,
            "markersize": 7, "fillstyle": "none", "zorder": 4,
        },
        "ls_rc_svd": {
            "marker": "h", "linestyle": "-", "linewidth": 2.2,
            "markersize": 7, "fillstyle": "none", "zorder": 5,
        },
        "mc_sgd": {
            "marker": "^", "linestyle": "-", "linewidth": 2.0,
            "markersize": 7, "fillstyle": "none", "zorder": 4,
        },
        "nn_calib_svd": {
            "marker": "d", "linestyle": "-", "linewidth": 2.0,
            "markersize": 7, "fillstyle": "none", "zorder": 5,
        },
        "moe_calib_svd": {
            "marker": "D", "linestyle": "-", "linewidth": 2.0,
            "markersize": 7, "fillstyle": "none", "zorder": 6,
        },
        "moe_refiner_calib_svd": {
            "marker": "X", "linestyle": "-", "linewidth": 2.0,
            "markersize": 7, "fillstyle": "none", "zorder": 7,
        },
        "moe_preco_refiner_svd": {
            "marker": "P", "linestyle": "-", "linewidth": 2.0,
            "markersize": 7, "fillstyle": "none", "zorder": 8,
        },
        "moe_preco_refiner_svd_ml": {
            "marker": "P", "linestyle": "--", "linewidth": 2.0,
            "markersize": 7, "fillstyle": "none", "zorder": 8,
        },
        "sa_rescal_svd": {
            "marker": "X", "linestyle": "-", "linewidth": 2.0,
            "markersize": 7, "fillstyle": "none", "zorder": 6,
        },
        "sa_mult_nores_nocond_svd": {
            "marker": "v", "linestyle": "-", "linewidth": 2.0,
            "markersize": 7, "fillstyle": "none", "zorder": 6,
        },
        "sa_mult_nores_svd": {
            "marker": "X", "linestyle": "-", "linewidth": 2.2,
            "markersize": 7, "fillstyle": "none", "zorder": 7,
        },
        "sa_mult_full_svd": {
            "marker": "P", "linestyle": "-", "linewidth": 2.6,
            "markersize": 8, "fillstyle": "none", "zorder": 9,
        },
        "genie_direct_precoder": {
            "marker": "*", "linestyle": "-", "linewidth": 2.0,
            "markersize": 9, "fillstyle": "none", "zorder": 10,
        },
    }


def get_plot_order(plot_mode="all"):
    """
    plot_mode:
      - main:     main comparison curves
      - ablation: SA-Mult ablation curves
      - all:      all known schemes
    """
    main_order = [
        "ideal_svd",
        "error_svd",
        "ls_rc_svd",
        "mc_sgd",
        "nn_calib_svd",
        "moe_calib_svd",
        "moe_refiner_calib_svd",
        "sa_mult_full_svd",
        "genie_direct_precoder",
    ]

    ablation_order = [
        "ideal_svd",
        "error_svd",
        "ls_rc_svd",
        "sa_mult_nores_nocond_svd",
        "sa_mult_nores_svd",
        "sa_mult_full_svd",
    ]

    all_order = [
        "ideal_svd",
        "error_svd",
        "ls_rc_svd",
        "mc_sgd",
        "nn_calib_svd",
        "sa_rescal_svd",
        "moe_calib_svd",
        "moe_refiner_calib_svd",
        "moe_preco_refiner_svd",
        "sa_mult_nores_nocond_svd",
        "sa_mult_nores_svd",
        "sa_mult_full_svd",
        "genie_direct_precoder",
        "ideal_svd_ml",
        "moe_preco_refiner_svd_ml",
    ]

    if plot_mode == "main":
        return main_order
    if plot_mode == "ablation":
        return ablation_order
    if plot_mode == "all":
        return all_order
    raise ValueError(f"Unknown plot_mode: {plot_mode}")


def plot_metric(results, metric_name, ylabel, save_path, use_log=False, plot_mode="all"):
    snr_db = np.asarray(results["snr_db"], dtype=float)
    schemes = results["schemes"]
    style_map = get_style_map()

    plt.figure(figsize=(7.6, 5.4))

    plot_order = get_plot_order(plot_mode)
    # Append any unseen schemes to avoid silently missing curves.
    plot_order = plot_order + [s for s in schemes.keys() if s not in plot_order]

    for scheme_name in plot_order:
        if scheme_name not in schemes:
            continue

        scheme_data = schemes[scheme_name]
        if metric_name not in scheme_data:
            continue

        y = clean_metric_values(scheme_data[metric_name])
        if np.all(np.isnan(y)):
            continue

        style = style_map.get(
            scheme_name,
            {
                "marker": "o",
                "linestyle": "-",
                "linewidth": 2.0,
                "markersize": 6,
                "fillstyle": "full",
                "zorder": 1,
            },
        )

        label = build_scheme_label(scheme_name, results)
        plot_func = plt.semilogy if use_log else plt.plot

        plot_func(
            snr_db,
            np.maximum(y, 1e-12) if use_log else y,
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
            markersize=style["markersize"],
            fillstyle=style["fillstyle"],
            markeredgewidth=1.8,
            label=label,
            zorder=style["zorder"],
        )

    plt.xlabel("DL SNR (dB)", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.xticks(snr_db)
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend(fontsize=9)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved plot to: {save_path}")


def plot_all(results, output_dir, base_name, plot_mode="all"):
    os.makedirs(output_dir, exist_ok=True)
    suffix = "" if plot_mode == "all" else f"_{plot_mode}"

    plot_metric(
        results=results,
        metric_name="cap_link",
        ylabel="Link Capacity (bits/s/Hz)",
        save_path=os.path.join(output_dir, f"{base_name}{suffix}_cap_link.png"),
        use_log=False,
        plot_mode=plot_mode,
    )
    plot_metric(
        results=results,
        metric_name="effse",
        ylabel="Effective Spectral Efficiency (bits/s/Hz)",
        save_path=os.path.join(output_dir, f"{base_name}{suffix}_effse.png"),
        use_log=False,
        plot_mode=plot_mode,
    )
    plot_metric(
        results=results,
        metric_name="ber",
        ylabel="BER",
        save_path=os.path.join(output_dir, f"{base_name}{suffix}_ber.png"),
        use_log=True,
        plot_mode=plot_mode,
    )
    plot_metric(
        results=results,
        metric_name="bler",
        ylabel="BLER",
        save_path=os.path.join(output_dir, f"{base_name}{suffix}_bler.png"),
        use_log=True,
        plot_mode=plot_mode,
    )
    plot_metric(
        results=results,
        metric_name="ser",
        ylabel="SER",
        save_path=os.path.join(output_dir, f"{base_name}{suffix}_ser.png"),
        use_log=True,
        plot_mode=plot_mode,
    )
    plot_metric(
        results=results,
        metric_name="evm",
        ylabel="EVM",
        save_path=os.path.join(output_dir, f"{base_name}{suffix}_evm.png"),
        use_log=False,
        plot_mode=plot_mode,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", type=str, default=None)
    parser.add_argument("--results_dir", type=str, default="./results_e2e")
    parser.add_argument("--output_dir", type=str, default="./results_e2e/figures")
    parser.add_argument(
        "--plot_mode",
        type=str,
        default="all",
        choices=["all", "main", "ablation"],
        help="Which curves to plot: all, main, or ablation.",
    )
    args = parser.parse_args()

    if args.json_path is None:
        json_path = get_default_json_path(args.results_dir)
    else:
        json_path = args.json_path

    print(f"Loading results from: {json_path}")
    results = load_results(json_path)
    base_name = os.path.splitext(os.path.basename(json_path))[0]

    plot_all(results, args.output_dir, base_name, plot_mode=args.plot_mode)
