from sklearn.base import clone
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.over_sampling import RandomOverSampler, SMOTE


def _safe_counts(values):
    return {str(k): int(v) for k, v in values.value_counts().to_dict().items()}


def get_features_root(overlap_mode="sin_solapamiento"):
    valid_modes = {"sin_solapamiento", "con_solapamiento"}
    if overlap_mode not in valid_modes:
        raise ValueError(
            f"overlap_mode no soportado: {overlap_mode}. "
            "Usa 'sin_solapamiento' o 'con_solapamiento'."
        )
    return f"features_{overlap_mode}"


def apply_smote_per_group(
    X_train_sel,
    y_train,
    groups_train,
    method="smote",
    random_state=42,
    return_flags=False,
):
    """
    Aplica oversampling por cada grupo (por ejemplo, usuario) únicamente en el set de entrenamiento.
    Mantiene exactamente el mismo comportamiento que el bloque original.
    """

    X_resampled = []
    y_resampled = []
    smote_flags = []
    for group in groups_train.unique():

        mask = groups_train == group

        X_u = X_train_sel.loc[mask]
        y_u = y_train.loc[mask]

        min_class = y_u.value_counts().min()
        can_resample = (method == "random" and min_class >= 1) or (
            method != "random" and min_class > 1
        )

        # print(   f"Conteo de clases para usuario {group} antes de oversampling: {y_u.value_counts().to_dict()}")
        if can_resample:
            if method == "random":
                sm = RandomOverSampler(random_state=random_state)
            elif method == "smote":
                k_neighbors = max(1, min_class - 1)
                sm = SMOTE(k_neighbors=k_neighbors, random_state=random_state)
            elif method == "smoteenn":
                k_neighbors = max(1, min_class - 1)
                sm = SMOTEENN(
                    smote=SMOTE(k_neighbors=k_neighbors, random_state=random_state),
                    random_state=random_state,
                )
            elif method == "smotetomek":
                k_neighbors = max(1, min_class - 1)
                sm = SMOTETomek(
                    smote=SMOTE(k_neighbors=k_neighbors, random_state=random_state),
                    random_state=random_state,
                )
            else:
                raise ValueError(
                    f"Método de oversampling no soportado: {method}. "
                    "Usa 'random', 'smote', 'smoteenn' o 'smotetomek'."
                )
            X_u_res, y_u_res = sm.fit_resample(X_u, y_u)
        else:
            # print(   f"Usuario {group}: oversampling skipped (minority class = 1)")
            X_u_res, y_u_res = X_u, y_u
            if return_flags:
                smote_flags.append(
                    {
                        "group": (
                            int(group)
                            if isinstance(group, (int, np.integer))
                            else str(group)
                        ),
                        "reason": "minority_class_le_1",
                        "class_counts": _safe_counts(y_u),
                    }
                )

        X_resampled.append(pd.DataFrame(X_u_res, columns=X_train_sel.columns))
        y_resampled.append(pd.Series(y_u_res))
        # print(
        #    f"Conteo de clases para usuario {group} después de oversampling: {y_resampled[-1].value_counts().to_dict()}")

    X_train_res = pd.concat(X_resampled, axis=0)
    y_train_res = pd.concat(y_resampled, axis=0)

    if return_flags:
        return X_train_res, y_train_res, smote_flags

    return X_train_res, y_train_res


# =========================
# 1. Carga de datos
# =========================
def load_data(device, window, overlap_mode="sin_solapamiento"):
    features_root = get_features_root(overlap_mode=overlap_mode)
    df = pd.read_csv(
        f"{features_root}/features_extracted_{device}/features_window_{window}.csv"
    )
    y = df["Class"].astype("category")
    groups = df["User"]
    X = df.drop(columns=["Class", "User"])
    return X, y, groups


# =========================
# 3. Evaluación de modelo
# =========================
def evaluate_model(
    model,
    X,
    y,
    groups,
    n_splits=10,
    n_significant=1,
    features=None,
    smote=True,
    return_flags=False,
):

    smote_method = None
    if isinstance(smote, str):
        smote_method = smote.lower()
        smote = True

    kf = StratifiedGroupKFold(n_splits=n_splits)

    metrics = {
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1": [],
        "f1_mean": [],
        "precision_macro": [],
        "recall_macro": [],
        "f1_macro": [],
    }
    fold_flags = []
    all_classes = set(map(str, y.astype(str).unique()))

    for i, (train_idx, test_idx) in enumerate(kf.split(X, y, groups=groups)):
        print(f"\nFold {i+1}/{n_splits}")

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        groups_train = groups.iloc[train_idx]
        current_flags = []
        fold_info = {
            "fold": int(i + 1),
            "flags": current_flags,
        }

        # =========================
        # Feature selection
        # =========================
        if isinstance(features, list):
            relevant_features = features
        else:
            relevant_features = X.columns

        if len(relevant_features) == 0:
            print("No se encontraron features relevantes.")
            current_flags.append("no_relevant_features")
            if return_flags:
                fold_flags.append(fold_info)
            continue

        X_train_sel = X_train[relevant_features]
        X_test_sel = X_test[relevant_features]

        # =========================
        # SMOTE por usuario - Solo en TRAIN
        # =========================
        if smote:
            if return_flags:
                X_train_sel, y_train, smote_skipped = apply_smote_per_group(
                    X_train_sel,
                    y_train,
                    groups_train,
                    method=smote_method,
                    return_flags=True,
                )
                if smote_skipped:
                    current_flags.append("smote_skipped_for_some_groups")
                    fold_info["smote_skipped_groups"] = smote_skipped
            else:
                X_train_sel, y_train = apply_smote_per_group(
                    X_train_sel,
                    y_train,
                    groups_train,
                    method=smote_method,
                )

        # =========================
        # Train
        # =========================
        model_fold = clone(model)  # <- nuevo modelo en cada fold
        model_fold.fit(X_train_sel, y_train)

        y_pred = model_fold.predict(X_test_sel)
        test_classes = set(map(str, y_test.astype(str).unique()))
        pred_classes = set(map(str, pd.Series(y_pred).astype(str).unique()))

        if test_classes != all_classes:
            current_flags.append("missing_classes_in_test")
            fold_info["missing_test_classes"] = sorted(all_classes - test_classes)

        if pred_classes != all_classes:
            current_flags.append("missing_classes_in_prediction")
            fold_info["missing_prediction_classes"] = sorted(all_classes - pred_classes)

        metrics["accuracy"].append(accuracy_score(y_test, y_pred))
        metrics["precision"].append(
            precision_score(y_test, y_pred, average="weighted", zero_division=0)
        )
        metrics["recall"].append(
            recall_score(y_test, y_pred, average="weighted", zero_division=0)
        )
        metrics["f1"].append(
            f1_score(y_test, y_pred, average="weighted", zero_division=0)
        )
        metrics["f1_mean"].append(
            f1_score(y_test, y_pred, average="weighted", zero_division=0)
        )
        metrics["precision_macro"].append(
            precision_score(y_test, y_pred, average="macro", zero_division=0)
        )
        metrics["recall_macro"].append(
            recall_score(y_test, y_pred, average="macro", zero_division=0)
        )
        metrics["f1_macro"].append(
            f1_score(y_test, y_pred, average="macro", zero_division=0)
        )
        if return_flags:
            fold_flags.append(fold_info)

    results = {
        m: {
            "folds": values,
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
        }
        for m, values in metrics.items()
    }

    if return_flags:
        flag_counts = {}
        folds_with_incident = []
        for fold_info in fold_flags:
            if fold_info["flags"]:
                folds_with_incident.append(fold_info["fold"])
            for flag in fold_info["flags"]:
                flag_counts[flag] = flag_counts.get(flag, 0) + 1

        results["_flags"] = {
            "has_incident": len(folds_with_incident) > 0,
            "flag_counts": flag_counts,
            "folds_with_incident": folds_with_incident,
            "per_fold": fold_flags,
        }

    return results


def evaluate_multiple_models(
    models_dict,
    X,
    y,
    groups,
    n_splits=10,
    features=None,
    smote=True,
    return_flags=False,
):

    results = {}

    for name, model in models_dict.items():
        # print(f"\n===== Evaluando {name} =====")
        results[name] = evaluate_model(
            model,
            X,
            y,
            groups,
            n_splits=n_splits,
            features=features,
            smote=smote,
            return_flags=return_flags,
        )

    return results
