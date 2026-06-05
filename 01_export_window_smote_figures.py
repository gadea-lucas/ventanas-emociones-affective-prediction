import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import get_features_root

FIGURES_ROOT = "figures"

DEVICES = ["emotibit_volar"]
OVERLAP_MODES = ["sin_solapamiento", "con_solapamiento"]
SMOTE_METHODS = [False, "random", "smote", "smoteenn", "smotetomek"]
SMOTE_LABELS = {
    False: "Original",
    "random": "ROS",
    "smote": "SMOTE",
    "smoteenn": "SMOTEENN",
    "smotetomek": "SMOTETomek",
}
METRIC = "f1_mean"
METRIC_LABEL = "F1 weighted"
MAX_WINDOW = 10
SUMMARY_MODEL = "RandomForest"


def ensure_dirs():
    for section in [
        "01_metricas_por_ventana",
        "02_comparativa_smote",
        "03_mejores_ventanas",
    ]:
        os.makedirs(os.path.join(FIGURES_ROOT, section), exist_ok=True)

    for overlap_mode in OVERLAP_MODES:
        os.makedirs(
            os.path.join(FIGURES_ROOT, "01_metricas_por_ventana", overlap_mode),
            exist_ok=True,
        )
        os.makedirs(
            os.path.join(FIGURES_ROOT, "02_comparativa_smote", overlap_mode),
            exist_ok=True,
        )


def load_results(device, overlap_mode, smote_method):
    features_folder = os.path.join(
        get_features_root(overlap_mode=overlap_mode),
        f"features_extracted_{device}",
    )
    json_path = os.path.join(
        features_folder, f"results_all_windows_{smote_method}.json"
    )

    if not os.path.exists(json_path):
        print(f"No se encontro {json_path}")
        return None

    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_metric_series(results, model_name, metric=METRIC):
    windows = []
    means = []
    stds = []

    for window in range(1, MAX_WINDOW + 1):
        key = f"window_{window}"
        if key not in results:
            continue
        if model_name not in results[key]:
            continue

        model_metrics = results[key][model_name]
        if metric not in model_metrics:
            continue
        if model_metrics[metric].get("mean") is None:
            continue

        windows.append(window)
        means.append(float(model_metrics[metric]["mean"]))
        stds.append(float(model_metrics[metric].get("std", 0)))

    return np.asarray(windows), np.asarray(means), np.asarray(stds)


def annotate_peaks(ax, windows, values, color):
    values = np.asarray(values)
    windows = np.asarray(windows)

    if len(values) == 0:
        return

    if len(values) < 3:
        peak_idx = np.arange(len(values))
    else:
        peak_idx = []
        for i in range(len(values)):
            left_ok = (i == 0) or (values[i] > values[i - 1])
            right_ok = (i == len(values) - 1) or (values[i] > values[i + 1])
            if left_ok and right_ok:
                peak_idx.append(i)
        peak_idx = np.array(peak_idx, dtype=int)

    for i in peak_idx:
        ax.annotate(
            f"{values[i]:.3f}",
            (windows[i], values[i]),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8,
            color=color,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color, alpha=0.7),
        )


def export_metricas_por_ventana():
    section_dir = os.path.join(FIGURES_ROOT, "01_metricas_por_ventana")

    for overlap_mode in OVERLAP_MODES:
        for smote_method in SMOTE_METHODS:
            fig, axes = plt.subplots(
                1,
                len(DEVICES),
                figsize=(7 * len(DEVICES), 5),
                sharey=True,
                squeeze=False,
            )
            axes = axes.ravel()
            found_any = False

            for ax, device in zip(axes, DEVICES):
                results = load_results(device, overlap_mode, smote_method)
                if results is None:
                    ax.set_title(f"{device}\nJSON no encontrado")
                    ax.axis("off")
                    continue

                model_names = sorted(
                    {
                        model_name
                        for window_payload in results.values()
                        if isinstance(window_payload, dict)
                        for model_name in window_payload.keys()
                        if not model_name.startswith("_")
                    }
                )

                for model_name in model_names:
                    windows, means, stds = get_metric_series(results, model_name)
                    if len(windows) == 0:
                        continue
                    found_any = True
                    line = ax.errorbar(
                        windows,
                        means,
                        yerr=stds,
                        marker="o",
                        capsize=4,
                        linewidth=1.8,
                        label=model_name,
                    )
                    annotate_peaks(ax, windows, means, line.lines[0].get_color())

                ax.set_title(device)
                ax.set_xlabel("Ventana (s)")
                ax.set_xticks(range(1, MAX_WINDOW + 1))
                ax.grid(True, alpha=0.25)

            axes[0].set_ylabel(METRIC_LABEL)
            if found_any:
                axes[-1].legend(loc="lower right", fontsize=9, frameon=True)

            fig.suptitle(
                f"{METRIC_LABEL} por ventana | {overlap_mode} | {SMOTE_LABELS[smote_method]}",
                fontsize=14,
            )
            fig.tight_layout()

            output_path = os.path.join(
                section_dir,
                overlap_mode,
                f"{METRIC}_side_by_side_{SMOTE_LABELS[smote_method]}.pdf",
            )
            fig.savefig(output_path, bbox_inches="tight")
            plt.close(fig)
            print(f"Guardado: {output_path}")


def export_comparativa_smote():
    section_dir = os.path.join(FIGURES_ROOT, "02_comparativa_smote")

    for overlap_mode in OVERLAP_MODES:
        fig, axes = plt.subplots(
            1,
            len(DEVICES),
            figsize=(7 * len(DEVICES), 5),
            sharey=True,
            squeeze=False,
        )
        axes = axes.ravel()
        found_any = False

        for ax, device in zip(axes, DEVICES):
            for smote_method in SMOTE_METHODS:
                results = load_results(device, overlap_mode, smote_method)
                if results is None:
                    continue

                windows, means, stds = get_metric_series(results, SUMMARY_MODEL)
                if len(windows) == 0:
                    continue

                found_any = True
                label = SMOTE_LABELS[smote_method]
                line = ax.errorbar(
                    windows,
                    means,
                    yerr=stds,
                    marker="o",
                    capsize=4,
                    linewidth=1.8,
                    label=label,
                )
                annotate_peaks(ax, windows, means, line.lines[0].get_color())

            ax.set_title(device)
            ax.set_xlabel("Ventana (s)")
            ax.set_xticks(range(1, MAX_WINDOW + 1))
            ax.grid(True, alpha=0.25)

        axes[0].set_ylabel(METRIC_LABEL)
        if found_any:
            axes[-1].legend(loc="lower right", fontsize=9, frameon=True)

        fig.suptitle(
            f"{SUMMARY_MODEL} | comparativa balanceo | {METRIC_LABEL} | {overlap_mode}",
            fontsize=14,
        )
        fig.tight_layout()

        output_path = os.path.join(
            section_dir,
            overlap_mode,
            f"{SUMMARY_MODEL}_{METRIC}_comparativa_smote.pdf",
        )
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)
        print(f"Guardado: {output_path}")


def export_best_window_summaries():
    rows = []

    for overlap_mode in OVERLAP_MODES:
        for device in DEVICES:
            for smote_method in SMOTE_METHODS:
                results = load_results(device, overlap_mode, smote_method)
                if results is None:
                    continue

                best_by_model = {}
                for window in range(1, MAX_WINDOW + 1):
                    key = f"window_{window}"
                    if key not in results:
                        continue

                    for model_name, model_results in results[key].items():
                        if model_name.startswith("_"):
                            continue
                        if METRIC not in model_results:
                            continue
                        mean_value = model_results[METRIC].get("mean")
                        if mean_value is None:
                            continue

                        mean_value = float(mean_value)
                        std_value = float(model_results[METRIC].get("std", 0))
                        current = best_by_model.get(model_name)
                        if current is None or mean_value > current["best_metric_mean"]:
                            best_by_model[model_name] = {
                                "best_window": window,
                                "best_metric_mean": mean_value,
                                "best_metric_std": std_value,
                            }

                for model_name, info in best_by_model.items():
                    rows.append(
                        {
                            "overlap_mode": overlap_mode,
                            "device": device,
                            "smote_method": SMOTE_LABELS[smote_method],
                            "model": model_name,
                            "metric": METRIC,
                            **info,
                        }
                    )

    summary_df = pd.DataFrame(rows).sort_values(
        ["overlap_mode", "device", "smote_method", "model"]
    )
    section_dir = os.path.join(FIGURES_ROOT, "03_mejores_ventanas")
    csv_path = os.path.join(section_dir, "best_windows_by_model_all.csv")
    json_path = os.path.join(section_dir, "best_windows_by_model_all.json")
    summary_df.to_csv(csv_path, index=False)
    summary_df.to_json(json_path, orient="records", indent=2)
    print(f"Guardado: {csv_path}")
    print(f"Guardado: {json_path}")


if __name__ == "__main__":
    ensure_dirs()
    export_metricas_por_ventana()
    export_comparativa_smote()
    export_best_window_summaries()
