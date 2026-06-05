import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
)
from sklearn.model_selection import StratifiedGroupKFold

from utils import apply_smote_per_group, load_data

# Un PDF por combinacion de overlap + dispositivo + SMOTE.
# Cada pagina del PDF corresponde a una ventana y contiene todas las matrices por usuario/split.
pdf_confusion_devices = ["emotibit_volar"]
pdf_confusion_windows = range(1, 16)
pdf_confusion_overlap_modes = ["sin_solapamiento", "con_solapamiento"]
pdf_confusion_smote_methods = [False, "random", "smote", "smoteenn", "smotetomek"]

pdf_confusion_n_splits = 10
pdf_confusion_model = RandomForestClassifier(
    n_estimators=500,
    n_jobs=1,
    random_state=42,
)

pdf_output_dir = "./figures/04_matrices_confusion"
os.makedirs(pdf_output_dir, exist_ok=True)

smote_display = {
    False: "No_SMOTE",
    "random": "ROS",
    "smote": "SMOTE",
    "smoteenn": "SMOTEENN",
    "smotetomek": "SMOTETomek",
}


def compute_confusion_matrices_for_pdf(device, window, overlap_mode, smote_method):
    X, y, groups = load_data(device=device, window=window, overlap_mode=overlap_mode)
    labels = sorted(y.astype(str).unique())
    cv = StratifiedGroupKFold(
        n_splits=pdf_confusion_n_splits,
        shuffle=True,
        random_state=42,
    )

    split_matrices = []
    fold_rows = []

    for split, (train_idx, test_idx) in enumerate(cv.split(X, y, groups), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        groups_train = groups.iloc[train_idx]
        groups_test = groups.iloc[test_idx]

        if smote_method:
            X_train_fit, y_train_fit = apply_smote_per_group(
                X_train,
                y_train,
                groups_train,
                method=smote_method,
            )
        else:
            X_train_fit, y_train_fit = X_train, y_train

        model = clone(pdf_confusion_model)
        model.fit(X_train_fit, y_train_fit)
        y_pred = model.predict(X_test)

        cm = confusion_matrix(
            y_test.astype(str),
            pd.Series(y_pred).astype(str),
            labels=labels,
        )

        test_users = sorted(pd.Series(groups_test).astype(str).unique())
        split_matrices.append(
            {
                "split": split,
                "test_users": test_users,
                "cm": cm,
            }
        )
        fold_rows.append(
            {
                "split": split,
                "test_users": ";".join(test_users),
                "n_test": int(len(y_test)),
                "accuracy": accuracy_score(y_test, y_pred),
                "f1_weighted": f1_score(
                    y_test,
                    y_pred,
                    average="weighted",
                    zero_division=0,
                ),
                "f1_macro": f1_score(
                    y_test,
                    y_pred,
                    average="macro",
                    zero_division=0,
                ),
                "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
                "mcc": matthews_corrcoef(y_test, y_pred),
            }
        )

    return labels, split_matrices, pd.DataFrame(fold_rows)


def plot_window_page(pdf, labels, split_matrices, fold_df, title):
    n_items = len(split_matrices)
    n_cols = 5 if n_items > 6 else 3
    n_rows = math.ceil(n_items / n_cols)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(n_cols * 3.0, n_rows * 3.1 + 0.7),
        squeeze=False,
    )
    fig.suptitle(title, fontsize=14, y=0.995)

    max_value = (
        max(int(item["cm"].max()) for item in split_matrices) if split_matrices else 1
    )
    im = None

    for ax, item in zip(axes.ravel(), split_matrices):
        cm = item["cm"]
        split = item["split"]
        users = ", ".join(item["test_users"])
        metrics = fold_df.loc[fold_df["split"] == split].iloc[0]

        im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=max_value)
        ax.set_title(
            f"Split {split} | User(s) {users}\n"
            f"F1w={metrics['f1_weighted']:.3f} | Acc={metrics['accuracy']:.3f}",
            fontsize=8,
        )
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_xlabel("Prediccion", fontsize=8)
        ax.set_ylabel("Real", fontsize=8)

        threshold = max_value / 2 if max_value else 0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j,
                    i,
                    int(cm[i, j]),
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if cm[i, j] > threshold else "black",
                )

    for ax in axes.ravel()[n_items:]:
        ax.axis("off")

    fig.subplots_adjust(top=0.88, right=0.93, hspace=0.75, wspace=0.45)
    if im is not None:
        cbar_ax = fig.add_axes([0.945, 0.12, 0.012, 0.72])
        fig.colorbar(im, cax=cbar_ax)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


for overlap_mode in pdf_confusion_overlap_modes:
    for device in pdf_confusion_devices:
        for smote_method in pdf_confusion_smote_methods:
            smote_name = smote_display[smote_method]
            combo_dir = os.path.join(pdf_output_dir, overlap_mode, device, smote_name)
            os.makedirs(combo_dir, exist_ok=True)

            pdf_path = os.path.join(
                combo_dir,
                f"confusion_matrices_{device}_{overlap_mode}_{smote_name}.pdf",
            )

            print(f"Generando PDF: {pdf_path}")
            with PdfPages(pdf_path) as pdf:
                for window in pdf_confusion_windows:
                    print(f"  window={window}")
                    labels, split_matrices, fold_df = (
                        compute_confusion_matrices_for_pdf(
                            device=device,
                            window=window,
                            overlap_mode=overlap_mode,
                            smote_method=smote_method,
                        )
                    )
                    title = (
                        f"{device} | {overlap_mode} | {smote_name} | "
                        f"window={window} | F1w medio={fold_df['f1_weighted'].mean():.3f}"
                    )
                    plot_window_page(pdf, labels, split_matrices, fold_df, title)

            print(f"PDF guardado en: {pdf_path}")
