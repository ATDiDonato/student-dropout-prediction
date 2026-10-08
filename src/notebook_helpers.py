from __future__ import annotations

import math
import os
import random
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import shap
import tensorflow as tf
from keras.layers import Dense, Dropout, Input
from keras.models import Sequential
from keras.optimizers import Adam, RMSprop
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    recall_score,
    r2_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.regularizers import l2 as l2_reg

from src.config import TARGET_COLUMN


def plot_distributions_and_qq(data_to_plot: pd.DataFrame, cols: list[str]) -> None:
    def add_line_subplot(
        ax: Any,
        x_values: Any,
        y_values: Any,
        color: str = "blue",
        title: str | None = None,
        linestyle: str = "-",
    ) -> None:
        sns.lineplot(
            x=x_values,
            y=y_values,
            color=color,
            ax=ax,
            linestyle=linestyle,
        )
        ax.set_title(title)

    def add_histogram_subplot(ax: Any, values: Any, color: str = "blue", title: str | None = None) -> None:
        sns.histplot(values, kde=True, color=color, ax=ax)
        ax.set_title(title)

    def add_boxplot_subplot(ax: Any, values: Any, color: str = "blue", title: str | None = None) -> None:
        sns.boxplot(values, color=color, ax=ax)
        ax.set_title(title)

    def add_scatterplot_subplot(
        ax: Any,
        x_values: Any,
        y_values: Any,
        color: str = "blue",
        title: str | None = None,
    ) -> None:
        sns.scatterplot(
            x=x_values,
            y=y_values,
            color=color,
            ax=ax,
            edgecolor=None,
            s=8,
        )
        ax.set_title(title)

    n = len(cols) * 2
    ncols = 4
    nrows = math.ceil(n / ncols)

    _, axes = plt.subplots(nrows, ncols, figsize=(20, nrows * 4))
    axes = np.asarray(axes).ravel()
    cmap = plt.colormaps.get_cmap("tab10")

    for i, col in enumerate(cols):
        add_histogram_subplot(axes[2 * i], data_to_plot[col], cmap(i % 20))
        add_boxplot_subplot(axes[(2 * i) + 1], data_to_plot[col], cmap(i % 20))

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.show()

    n = len(cols)
    _, axes = plt.subplots(nrows, ncols, figsize=(20, nrows * 4))
    axes = np.asarray(axes).ravel()

    for i, col in enumerate(cols):
        (theoretical, ordered), (slope, intercept, _) = stats.probplot(
            data_to_plot[col].to_numpy(),
            dist="norm",
            plot=None,
        )
        ax = axes[i]
        add_scatterplot_subplot(ax, theoretical, ordered, cmap(i % 20))
        add_line_subplot(
            ax,
            theoretical,
            slope * theoretical + intercept,
            color="red",
            linestyle="--",
        )
        ax.set_title(f"{col} Q-Q Plot")

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(cm: np.ndarray) -> None:
    labels = [["TN", "FP"], ["FN", "TP"]]
    annot = [[f"{labels[i][j]}\n{cm[i, j]}" for j in range(2)] for i in range(2)]

    _, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes = np.asarray(axes).ravel()

    sns.heatmap(
        cm,
        annot=annot,
        fmt="",
        cmap="Blues",
        xticklabels=["Predicted 0", "Predicted 1"],
        yticklabels=["Actual 0", "Actual 1"],
        ax=axes[0],
    )
    axes[0].set_xlabel("Predicted Label")
    axes[0].set_ylabel("Actual Label")
    axes[0].set_title("Confusion Matrix")

    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=["Predicted 0", "Predicted 1"],
        yticklabels=["Actual 0", "Actual 1"],
        ax=axes[1],
    )
    axes[1].set_title("Normalised Confusion Matrix (Row-wise)")
    plt.show()


def plot_hyperparameter_relationships(
    results_df: pd.DataFrame,
    hyperparams: list[str],
    showfliers: bool = True,
) -> None:
    n_cols = 4
    n_rows = math.ceil(len(hyperparams) / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = np.asarray(axes).flatten()

    for i, param in enumerate(hyperparams):
        sns.boxplot(
            x=param,
            y="val_auc",
            data=results_df,
            ax=axes[i],
            showfliers=showfliers,
        )
        axes[i].set_title(f"{param} vs Validation AUC")
        axes[i].tick_params(axis="x", rotation=45)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


def plot_roc_and_pr_curves(
    models: list[dict[str, Any]],
    figsize: tuple[int, int] = (12, 4),
    title_prefix: str = "Model",
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    roc_ax, pr_ax = axes

    for i, model in enumerate(models):
        name = model.get(
            "name",
            f"{title_prefix} {i + 1}",
        )
        if model.get("type") and model.get("stage"):
            name = f"{model['stage']} {model['type']} ({name})"

        y_true = np.asarray(model["y_true"])
        y_score = np.asarray(model["y_score"])

        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc = roc_auc_score(y_true, y_score)
        roc_ax.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")

        precision, recall, _ = precision_recall_curve(y_true, y_score)
        ap = average_precision_score(y_true, y_score)
        pr_ax.plot(recall, precision, label=f"{name} (AP = {ap:.3f})")

        if i == max(0, len(models) - 1):
            pos_rate = y_true.mean()
            pr_ax.axhline(
                pos_rate,
                linestyle="--",
                label=f"Baseline (Pos rate = {pos_rate:.3f})",
            )

    roc_ax.plot([0, 1], [0, 1], linestyle="--", label="Random (AUC = 0.5)")
    roc_ax.set_xlabel("False Positive Rate")
    roc_ax.set_ylabel("True Positive Rate")
    roc_ax.set_title("ROC Curve")
    roc_ax.legend()

    pr_ax.set_xlabel("Recall")
    pr_ax.set_ylabel("Precision")
    pr_ax.set_title("Precision–Recall Curve")
    pr_ax.legend()

    plt.tight_layout()
    plt.show()


def compute_confusion_matrix_elements(y_true: Any, y_pred: Any) -> tuple[int, int, int, int, np.ndarray]:
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return tn, fp, fn, tp, cm


def compute_performance_metrics(y_true: Any, y_pred: Any, y_prob: Any) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "specificity": recall_score(y_true, y_pred, pos_label=0),
        "auc": roc_auc_score(y_true, y_prob),
    }


def evaluate_and_store_model(
    results: dict[int, dict[str, Any]],
    m_id: int,
    model_name: str,
    model_type: str,
    stage: str,
    y_true: Any,
    y_pred: Any,
    y_prob: Any,
    hyperparameters: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[int, dict[str, Any]]:
    metrics = compute_performance_metrics(y_true, y_pred, y_prob)
    cm = confusion_matrix(y_true, y_pred)

    results[m_id] = {
        "model_name": model_name,
        "model_type": model_type,
        "stage": stage,
        "hyperparameters": hyperparameters or {},
        "metrics": metrics,
        "confusion_matrix": cm,
        "predictions": {
            "y_true": np.asarray(y_true),
            "y_pred": np.asarray(y_pred),
            "y_prob": np.asarray(y_prob),
        },
        "metadata": metadata or {},
    }
    return results


def print_model_metrics(results: dict[int, dict[str, Any]], model_id: int) -> None:
    m = results[model_id]["metrics"]
    model_type = results[model_id]["model_type"]
    model_name = results[model_id]["model_name"]
    model_stage = results[model_id]["stage"]

    print(f"{model_stage} {model_type} - {model_name} Performance Metrics:")
    print(f"Accuracy:    {float(m['accuracy']):.4f}")
    print(f"Precision:   {float(m['precision']):.4f}")
    print(f"Recall:      {float(m['recall']):.4f}")
    print(f"Specificity: {float(m['specificity']):.4f}")
    print(f"AUC:         {float(m['auc']):.4f}")


def plot_model_confusion_matrix(results: dict[int, dict[str, Any]], model_id: int) -> None:
    plot_confusion_matrix(results[model_id]["confusion_matrix"])


def plot_dropout_rate_by_category(
    data: pd.DataFrame,
    category_col: str,
    figsize: tuple[int, int] = (10, 7),
    min_count: int | None = None,
) -> None:
    if min_count is not None:
        data = data[data["count"] >= min_count]

    num_categories = data[category_col].nunique()
    max_label_length = data[category_col].apply(lambda x: len(str(x))).max()

    if num_categories > 15:
        figsize = (figsize[0], min(figsize[1] + max_label_length * 0.3, figsize[1] * 1.5))
    elif num_categories > 5:
        figsize = (figsize[0], min(figsize[1] + max_label_length * 0.15, figsize[1] * 1.2))

    plt.figure(figsize=figsize)
    ax = sns.barplot(
        data=data,
        x=category_col,
        y="dropout_rate",
        order=data[category_col],
    )

    ax.set_title(f"Dropout Rate by {category_col}", pad=20)
    ax.set_ylabel("Dropout Rate")
    ax.set_xlabel(category_col)
    ax.set_ylim(0, 0.7)

    annotate_rotation = 0
    if num_categories > 15:
        ax.tick_params(axis="x", rotation=90)
        annotate_rotation = 90
    elif num_categories > 5:
        ax.tick_params(axis="x", rotation=45)

    for p, (_, row) in zip(ax.patches, data.iterrows()):
        ax.annotate(
            f"n={row['count']}",
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="bottom",
            xytext=(0, 3),
            textcoords="offset points",
            fontsize=9,
            rotation=annotate_rotation,
        )

    plt.subplots_adjust(top=0.25)
    plt.tight_layout()
    plt.show()


def build_binary_classifier(
    input_dim: int,
    optimizer: str = "adam",
    units: int = 64,
    layers: int = 1,
    activation: str = "relu",
    dropout: float = 0.2,
    l2_strength: float = 1e-4,
    lr: float = 1e-3,
) -> Sequential:
    model = Sequential()
    model.add(Input(shape=(input_dim,)))

    reg = l2_reg(l2_strength) if l2_strength is not None and l2_strength > 0 else None
    for _ in range(layers):
        model.add(Dense(units, activation=activation, kernel_regularizer=reg))
        if dropout and dropout > 0:
            model.add(Dropout(dropout))

    model.add(Dense(1, activation="sigmoid"))

    opt_name = optimizer.lower() if isinstance(optimizer, str) else optimizer
    if isinstance(opt_name, str):
        if opt_name == "adam":
            opt = Adam(learning_rate=lr)
        elif opt_name == "rmsprop":
            opt = RMSprop(learning_rate=lr)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer}")
    else:
        opt = opt_name

    model.compile(
        loss="binary_crossentropy",
        optimizer=opt,
        metrics=[
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
        ],
    )
    return model


def choose_n_jobs() -> int:
    cores = os.cpu_count() or 1
    if cores <= 2:
        return cores
    return min(8, int(cores * 0.5))


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def create_train_val_test_split_and_scale(
    data_encoded: pd.DataFrame,
    stratify: bool = False,
    seed: int = 42,
    target_col: str = TARGET_COLUMN,
) -> tuple[Any, ...]:
    scaler = StandardScaler().set_output(transform="pandas")

    X = data_encoded.drop(columns=[target_col])
    y = data_encoded[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=seed,
        stratify=y if stratify else None,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.1,
        random_state=seed,
        stratify=y_train if stratify else None,
    )

    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    return (
        X_train,
        X_train_s,
        X_val,
        X_val_s,
        X_test,
        X_test_s,
        y_train,
        y_val,
        y_test,
        scaler,
    )


def build_models_to_plot(results: dict[int, dict[str, Any]], model_ids: list[int]) -> list[dict[str, Any]]:
    return [
        {
            "name": results[m_id]["model_name"],
            "type": results[m_id]["model_type"],
            "stage": results[m_id]["stage"],
            "y_true": results[m_id]["predictions"]["y_true"],
            "y_score": results[m_id]["predictions"]["y_prob"],
        }
        for m_id in model_ids
    ]


def get_default_feature_groups(columns: pd.Index) -> dict[str, list[str]]:
    return {
        "Nationality": [c for c in columns if c.startswith("Nationality_")],
        "Centre": [c for c in columns if c.startswith("CentreName_")],
        "ProgressionUniversity": [
            c for c in columns if c.startswith("ProgressionUniversity_")
        ],
        "SubjectCluster": [c for c in columns if c.startswith("subject_cluster_")],
        "DegreeLevel": [
            c for c in columns if c.startswith("progression_degree_level_")
        ],
        "AcademicStage": [c for c in columns if c.startswith("academic_stage_")],
        "DeliveryVariant": [c for c in columns if c.startswith("delivery_variant_")],
        "LeadSource": [c for c in columns if c.startswith("LeadSource_")],
        "DiscountType": [c for c in columns if c.startswith("DiscountType_")],
        "BookingType": [c for c in columns if c.startswith("BookingType_")],
        "Demographics": [c for c in columns if c in ["Age", "Gender", "IsFirstIntake"]],
        "Absence": [
            c
            for c in columns
            if c in [
                "AuthorisedAbsenceCount",
                "UnauthorisedAbsenceCount",
                "absence_values_missing",
            ]
        ],
        "Assessment": [
            c
            for c in columns
            if c in [
                "AssessedModules",
                "FailedModules",
                "PassedModules",
                "module_values_missing",
            ]
        ],
    }


def plot_grouped_feature_importance(
    feature_importance: pd.Series,
    shap_values: Any,
    X_val: pd.DataFrame,
    groups: dict[str, list[str]] | None = None,
) -> None:
    groups = groups or get_default_feature_groups(X_val.columns)

    feature_to_group = {}
    for group, cols in groups.items():
        for col in cols:
            feature_to_group[col] = group

    imp_df = feature_importance.reset_index()
    imp_df.columns = ["feature", "importance"]
    imp_df["group"] = imp_df["feature"].map(lambda feature_name: feature_to_group.get(feature_name, "Other"))
    grouped_importance_xgb = imp_df.groupby("group")["importance"].sum().sort_values(ascending=False)

    group_importance = {}
    shap_array = np.asarray(shap_values)
    for group, cols in groups.items():
        idx = [X_val.columns.get_loc(c) for c in cols if c in X_val.columns]
        if idx:
            group_importance[group] = np.abs(shap_array[:, idx]).sum(axis=1).mean()

    other_cols = [c for c in X_val.columns if c not in feature_to_group]
    other_idx = [X_val.columns.get_loc(c) for c in other_cols]
    if other_idx:
        group_importance["Other"] = np.abs(shap_array[:, other_idx]).sum(axis=1).mean()

    group_importance_shap = pd.Series(group_importance).sort_values(ascending=False)

    plt.figure(figsize=(12, 6))

    ax = plt.subplot(1, 2, 1)
    group_importance_shap.plot(kind="barh", ax=ax)
    ax.invert_yaxis()
    ax.set_title("Grouped SHAP Feature Importance")
    ax.set_xlabel("Mean |SHAP value|")

    ax = plt.subplot(1, 2, 2)
    grouped_importance_xgb.plot(kind="barh", ax=ax)
    ax.invert_yaxis()
    ax.set_title("Grouped XGBoost Feature Importance")
    ax.set_xlabel("Sum of feature importance")

    plt.tight_layout()
    plt.show()


def analyze_hp_importance(
    results_df: pd.DataFrame,
    hyperparams: list[str],
    target: str = "val_auc",
    n_estimators: int = 200,
    seed: int = 42,
    do_shap: bool = True,
) -> tuple[RandomForestRegressor, pd.Series, float, float]:
    df = results_df.copy()
    X_hp = pd.get_dummies(df[hyperparams], drop_first=False)
    y_hp = df[target]

    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=seed,
    )
    rf.fit(X_hp, y_hp)

    importances = pd.Series(rf.feature_importances_, index=X_hp.columns).sort_values(
        ascending=False
    )
    print("\nFeature importance:")
    print(importances)

    y_pred = rf.predict(X_hp)
    r2 = r2_score(y_hp, y_pred)
    rmse = np.sqrt(mean_squared_error(y_hp, y_pred))

    print("\nSurrogate performance:")
    print("R2:", r2)
    print("RMSE:", rmse)

    if do_shap:
        explainer = shap.TreeExplainer(rf)
        shap_values = explainer.shap_values(X_hp)
        shap.summary_plot(
            shap_values,
            X_hp,
            feature_names=X_hp.columns,
        )

    return rf, importances, r2, rmse
