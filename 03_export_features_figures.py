import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

FIGURES_ROOT = Path("figures")
THRESHOLD_DIR = FIGURES_ROOT / "05_threshold_vs_metricas"
GINI_DIR = FIGURES_ROOT / "06_importancias_gini"
SUMMARY_DIR = FIGURES_ROOT / "07_resumen_features"


def discover_feature_selection_configs():
    configs = []
    for threshold_path in Path(".").glob(
        "features_*/features_extracted_*/features_benchmark/*/gini_threshold_results_no_leak_*.csv"
    ):
        parts = threshold_path.parts
        overlap_mode = parts[0].replace("features_", "")
        device = parts[1].replace("features_extracted_", "")
        window = parts[3]
        smote_name = threshold_path.stem.replace("gini_threshold_results_no_leak_", "")
        output_dir = threshold_path.parent
        gini_path = output_dir / f"gini_importance_no_leak_{smote_name}.csv"
        features_path = output_dir / f"gini_best_features_no_leak_{smote_name}.json"

        configs.append(
            {
                "overlap_mode": overlap_mode,
                "device": device,
                "window": window,
                "smote_name": smote_name,
                "output_dir": output_dir,
                "threshold_path": threshold_path,
                "gini_path": gini_path,
                "features_path": features_path,
            }
        )

    return sorted(
        configs,
        key=lambda item: (
            item["overlap_mode"],
            item["device"],
            int(item["window"]),
            item["smote_name"],
        ),
    )


def ensure_dirs(configs):
    for base_dir in [THRESHOLD_DIR, GINI_DIR, SUMMARY_DIR]:
        base_dir.mkdir(parents=True, exist_ok=True)

    for config in configs:
        for base_dir in [THRESHOLD_DIR, GINI_DIR]:
            (base_dir / config["overlap_mode"] / config["device"]).mkdir(
                parents=True,
                exist_ok=True,
            )


def export_threshold_plot(config):
    df_results = pd.read_csv(config["threshold_path"])
    best_row = df_results.iloc[0]
    best_threshold = float(best_row["threshold"])
    best_f1_mean = float(best_row["f1_mean"])

    plot_df = df_results.sort_values("threshold").reset_index(drop=True)

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(
        plot_df["threshold"],
        plot_df["f1_mean"],
        marker="o",
        markersize=4,
        linewidth=1.6,
        color="tab:blue",
        label="F1 weighted mean",
    )
    ax1.axvline(
        best_threshold,
        color="tab:blue",
        linestyle="--",
        linewidth=1.4,
        alpha=0.9,
    )
    ax1.scatter(
        [best_threshold],
        [best_f1_mean],
        color="tab:red",
        s=45,
        zorder=4,
        label=f"Best threshold = {best_threshold:.4f}",
    )
    ax1.annotate(
        f"{best_f1_mean:.4f}",
        (best_threshold, best_f1_mean),
        textcoords="offset points",
        xytext=(8, 8),
        fontsize=9,
    )
    ax1.set_xlabel("Threshold")
    ax1.set_ylabel("F1 weighted mean", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(True, linestyle="--", alpha=0.6)

    ax2 = ax1.twinx()
    ax2.plot(
        plot_df["threshold"],
        plot_df["n_features_mean"],
        marker="s",
        markersize=3.5,
        color="tab:orange",
        linewidth=1.3,
        alpha=0.9,
        label="Nº medio de features",
    )
    ax2.set_ylabel("Nº medio de features", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="best", frameon=True)

    title = (
        f"{config['device']} | {config['overlap_mode']} | "
        f"window={config['window']} | SMOTE={config['smote_name']}"
    )
    ax1.set_title(title)
    fig.tight_layout()

    output_path = (
        THRESHOLD_DIR
        / config["overlap_mode"]
        / config["device"]
        / f"threshold_f1_features_{config['device']}_window_{config['window']}_{config['overlap_mode']}_{config['smote_name']}.pdf"
    )
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Guardado: {output_path}")


def export_gini_plot(config):
    if not config["gini_path"].exists() or not config["features_path"].exists():
        print(f"Saltando Gini, faltan archivos: {config['output_dir']}")
        return

    gini_importance = pd.read_csv(config["gini_path"], index_col=0).iloc[:, 0]
    gini_importance.name = "gini_importance"

    with open(config["features_path"], "r", encoding="utf-8") as f:
        features_payload = json.load(f)

    threshold = float(features_payload["best_threshold"])
    gsr = gini_importance[gini_importance.index.str.startswith("Gsr__")].sort_values(
        ascending=False
    )
    hr = gini_importance[gini_importance.index.str.startswith("Hr__")].sort_values(
        ascending=False
    )

    n_gsr_selected = int((gsr >= threshold).sum())
    n_hr_selected = int((hr >= threshold).sum())

    fig, ax = plt.subplots(figsize=(8, 4))
    x_gsr = np.arange(len(gsr))
    x_hr = np.arange(len(hr))

    ax.bar(
        x_gsr,
        gsr.values,
        width=1.0,
        align="edge",
        color="tab:blue",
        alpha=0.62,
        edgecolor="none",
        linewidth=0,
        label="EDA",
    )
    ax.bar(
        x_hr,
        hr.values,
        width=1.0,
        align="edge",
        color="tab:orange",
        alpha=0.62,
        edgecolor="none",
        linewidth=0,
        label="HR",
    )

    ax.axhline(
        threshold,
        color="black",
        linestyle="--",
        linewidth=1.4,
        alpha=0.9,
    )

    if n_gsr_selected > 0:
        ax.axvline(
            n_gsr_selected, color="tab:blue", linestyle="-", linewidth=2.2, alpha=0.95
        )
        ax.axvspan(0, n_gsr_selected, color="tab:blue", alpha=0.08)

    if n_hr_selected > 0:
        ax.axvline(
            n_hr_selected, color="tab:orange", linestyle="-", linewidth=2.2, alpha=0.95
        )
        ax.axvspan(0, n_hr_selected, color="tab:orange", alpha=0.08)

    ymax = max(gsr.max() if len(gsr) else 0, hr.max() if len(hr) else 0)
    ax.set_ylim(0, ymax * 1.18)

    ax.text(
        0.99,
        threshold,
        f"{threshold:.4f}",
        transform=ax.get_yaxis_transform(),
        ha="right",
        va="bottom",
        fontsize=9,
        color="black",
    )

    if n_gsr_selected > 0:
        ax.text(
            n_gsr_selected,
            ymax * 1.10,
            f"{n_gsr_selected} EDA",
            ha="left",
            va="bottom",
            fontsize=9,
            color="tab:blue",
            bbox=dict(
                boxstyle="round,pad=0.25",
                facecolor="white",
                edgecolor="tab:blue",
                alpha=0.9,
            ),
        )

    if n_hr_selected > 0:
        ax.text(
            n_hr_selected,
            ymax * 1.02,
            f"{n_hr_selected} HR",
            ha="left",
            va="bottom",
            fontsize=9,
            color="tab:orange",
            bbox=dict(
                boxstyle="round,pad=0.25",
                facecolor="white",
                edgecolor="tab:orange",
                alpha=0.9,
            ),
        )

    max_len = max(len(gsr), len(hr))
    ax.set_xlim(0, max_len)
    ax.margins(x=0)
    ax.set_xticks([])
    ax.set_ylabel("Importancia (Gini)")
    ax.set_xlabel("Característica")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title(
        f"{config['device']} | {config['overlap_mode']} | "
        f"window={config['window']} | SMOTE={config['smote_name']}"
    )

    legend_elements = [
        Patch(facecolor="tab:blue", alpha=0.62, label="EDA"),
        Patch(facecolor="tab:orange", alpha=0.62, label="HR"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", frameon=True)

    fig.tight_layout()
    output_path = (
        GINI_DIR
        / config["overlap_mode"]
        / config["device"]
        / f"gini_features_{config['device']}_window_{config['window']}_{config['overlap_mode']}_{config['smote_name']}.pdf"
    )
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Guardado: {output_path}")


def export_summary(configs):
    rows = []
    for config in configs:
        df_results = pd.read_csv(config["threshold_path"])
        best_row = df_results.iloc[0].to_dict()
        row = {
            "overlap_mode": config["overlap_mode"],
            "device": config["device"],
            "window": config["window"],
            "smote_name": config["smote_name"],
        }
        row.update(best_row)

        if config["features_path"].exists():
            with open(config["features_path"], "r", encoding="utf-8") as f:
                payload = json.load(f)
            row["n_final_features"] = len(payload.get("best_features", []))
            row["best_threshold_payload"] = payload.get("best_threshold")

        rows.append(row)

    summary_df = pd.DataFrame(rows)
    csv_path = SUMMARY_DIR / "feature_selection_best_rows.csv"
    json_path = SUMMARY_DIR / "feature_selection_best_rows.json"
    summary_df.to_csv(csv_path, index=False)
    summary_df.to_json(json_path, orient="records", indent=2)
    print(f"Guardado: {csv_path}")
    print(f"Guardado: {json_path}")


if __name__ == "__main__":
    configs = discover_feature_selection_configs()
    ensure_dirs(configs)

    for config in configs:
        export_threshold_plot(config)
        export_gini_plot(config)

    export_summary(configs)
