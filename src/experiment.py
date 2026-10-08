"""Reproducible staged experiments for the student-dropout portfolio project."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier


RANDOM_SEED = 42
TARGET = "dropout"
GROUP = "LearnerCode"

STAGE_FILES = {
    "Stage 1": "Stage_1_public.csv",
    "Stage 2": "Stage_2_public.csv",
    "Stage 3": "Stage_3_public.csv",
}

# CompletedCourse is deliberately excluded: it is exactly the inverse of the
# target. LearnerCode is retained only for grouped splitting and reporting.
BASE_FEATURES = [
    "CentreName",
    "BookingType",
    "LeadSource",
    "DiscountType",
    "Gender",
    "Age",
    "Nationality",
    "academic_stage",
    "subject_cluster",
    "delivery_variant",
    "IsFirstIntake",
    "progression_degree_level",
    "ProgressionUniversity",
]

STAGE_FEATURES = {
    "Stage 1": BASE_FEATURES,
    "Stage 2": BASE_FEATURES
    + ["AuthorisedAbsenceCount", "UnauthorisedAbsenceCount"],
    "Stage 3": BASE_FEATURES
    + [
        "AuthorisedAbsenceCount",
        "UnauthorisedAbsenceCount",
        "AssessedModules",
        "FailedModules",
        "PassedModules",
    ],
}

STAGE_3_ASSESSMENT_FEATURES = [
    "AssessedModules",
    "FailedModules",
    "PassedModules",
]


@dataclass
class ExperimentResult:
    stage: str
    model_name: str
    metrics: dict[str, float | int | str]
    predictions: pd.DataFrame
    feature_importance: pd.DataFrame | None


def load_stage_data(data_dir: Path) -> dict[str, pd.DataFrame]:
    """Load and validate the three aligned public stage datasets."""
    datasets = {
        stage: pd.read_csv(data_dir / filename)
        for stage, filename in STAGE_FILES.items()
    }

    reference = datasets["Stage 1"][[GROUP, TARGET]].reset_index(drop=True)
    for stage, frame in datasets.items():
        missing = set(STAGE_FEATURES[stage] + [GROUP, TARGET]) - set(frame.columns)
        if missing:
            raise ValueError(f"{stage} is missing required columns: {sorted(missing)}")
        current = frame[[GROUP, TARGET]].reset_index(drop=True)
        if not current.equals(reference):
            raise ValueError(
                f"{stage} row order or learner/target values differ from Stage 1. "
                "Align by a true enrolment key before continuing."
            )

    return datasets


def make_grouped_split(reference: pd.DataFrame, seed: int = RANDOM_SEED) -> pd.Series:
    """Create one shared 60/20/20 split with learners isolated by group."""
    y = reference[TARGET].astype(int).to_numpy()
    groups = reference[GROUP].astype(str).to_numpy()
    placeholder = np.zeros((len(reference), 1), dtype=np.int8)

    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    train_val_idx, test_idx = next(outer.split(placeholder, y, groups))

    inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed + 1)
    inner_train, inner_val = next(
        inner.split(
            placeholder[train_val_idx],
            y[train_val_idx],
            groups[train_val_idx],
        )
    )
    train_idx = train_val_idx[inner_train]
    val_idx = train_val_idx[inner_val]

    split = pd.Series("", index=reference.index, dtype="object")
    split.iloc[train_idx] = "train"
    split.iloc[val_idx] = "validation"
    split.iloc[test_idx] = "test"

    group_sets = {
        name: set(reference.loc[split.eq(name), GROUP].astype(str))
        for name in ("train", "validation", "test")
    }
    if group_sets["train"] & group_sets["validation"]:
        raise AssertionError("Learner overlap between train and validation splits")
    if group_sets["train"] & group_sets["test"]:
        raise AssertionError("Learner overlap between train and test splits")
    if group_sets["validation"] & group_sets["test"]:
        raise AssertionError("Learner overlap between validation and test splits")
    if split.eq("").any():
        raise AssertionError("Some rows were not assigned to a split")

    return split


def _feature_types(frame: pd.DataFrame, features: list[str]) -> tuple[list[str], list[str]]:
    categorical = [
        column
        for column in features
        if not pd.api.types.is_numeric_dtype(frame[column])
        or pd.api.types.is_bool_dtype(frame[column])
    ]
    numeric = [column for column in features if column not in categorical]
    return categorical, numeric


def make_preprocessor(
    frame: pd.DataFrame,
    features: list[str],
    *,
    dense: bool,
) -> ColumnTransformer:
    categorical, numeric = _feature_types(frame, features)

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="constant", fill_value="__MISSING__"),
            ),
            (
                "one_hot",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=20,
                    sparse_output=not dense,
                ),
            ),
        ]
    )
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler(with_mean=dense)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("categorical", categorical_pipeline, categorical),
            ("numeric", numeric_pipeline, numeric),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def choose_f2_threshold(y_true: pd.Series, probabilities: np.ndarray) -> float:
    """Choose a validation threshold that weights recall twice as highly as precision."""
    thresholds = np.linspace(0.05, 0.95, 181)
    scores = [
        fbeta_score(y_true, probabilities >= threshold, beta=2, zero_division=0)
        for threshold in thresholds
    ]
    return float(thresholds[int(np.argmax(scores))])


def classification_metrics(
    y_true: pd.Series,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions).ravel()
    return {
        "threshold": threshold,
        "roc_auc": roc_auc_score(y_true, probabilities),
        "average_precision": average_precision_score(y_true, probabilities),
        "accuracy": accuracy_score(y_true, predictions),
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "specificity": tn / (tn + fp),
        "f1": f1_score(y_true, predictions, zero_division=0),
        "f2": fbeta_score(y_true, predictions, beta=2, zero_division=0),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def _make_xgboost(y_train: pd.Series, seed: int) -> XGBClassifier:
    positives = int(y_train.sum())
    negatives = int(len(y_train) - positives)
    return XGBClassifier(
        n_estimators=450,
        learning_rate=0.05,
        max_depth=5,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.05,
        reg_lambda=2.0,
        scale_pos_weight=negatives / positives,
        objective="binary:logistic",
        eval_metric="aucpr",
        random_state=seed,
        n_jobs=4,
        tree_method="hist",
    )


def _make_neural_network(seed: int) -> MLPClassifier:
    return MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=256,
        learning_rate_init=1e-3,
        max_iter=80,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=8,
        random_state=seed,
        verbose=False,
    )


def run_stage_model(
    frame: pd.DataFrame,
    split: pd.Series,
    stage: str,
    model_name: str,
    output_dir: Path,
    seed: int = RANDOM_SEED,
) -> ExperimentResult:
    features = STAGE_FEATURES[stage]
    eligible = pd.Series(True, index=frame.index)
    population = "all enrolment records"
    if stage == "Stage 3":
        # All rows without assessment values are known dropouts. Treating this
        # outcome-adjacent absence as a predictor would create severe leakage.
        eligible = frame[STAGE_3_ASSESSMENT_FEATURES].notna().all(axis=1)
        population = "records with complete assessment data"

    train_mask = split.eq("train") & eligible
    validation_mask = split.eq("validation") & eligible
    test_mask = split.eq("test") & eligible

    X_train = frame.loc[train_mask, features]
    y_train = frame.loc[train_mask, TARGET].astype(int)
    X_validation = frame.loc[validation_mask, features]
    y_validation = frame.loc[validation_mask, TARGET].astype(int)
    X_test = frame.loc[test_mask, features]
    y_test = frame.loc[test_mask, TARGET].astype(int)

    dense = model_name == "Neural network"
    preprocessor = make_preprocessor(frame, features, dense=dense)
    X_train_ready = preprocessor.fit_transform(X_train)
    X_validation_ready = preprocessor.transform(X_validation)
    X_test_ready = preprocessor.transform(X_test)

    if model_name == "XGBoost":
        model: Any = _make_xgboost(y_train, seed)
        model.fit(X_train_ready, y_train)
    elif model_name == "Neural network":
        model = _make_neural_network(seed)
        sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
        model.fit(X_train_ready, y_train, sample_weight=sample_weight)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    validation_probabilities = model.predict_proba(X_validation_ready)[:, 1]
    threshold = choose_f2_threshold(y_validation, validation_probabilities)
    test_probabilities = model.predict_proba(X_test_ready)[:, 1]
    test_predictions = (test_probabilities >= threshold).astype(int)

    metrics = {
        "stage": stage,
        "model": model_name,
        "evaluation_population": population,
        "eligible_rows_total": int(eligible.sum()),
        "population_coverage": float(eligible.mean()),
        "excluded_dropouts": int(frame.loc[~eligible, TARGET].sum()),
        "feature_count_raw": len(features),
        "feature_count_encoded": int(X_train_ready.shape[1]),
        "test_rows": int(test_mask.sum()),
        **classification_metrics(y_test, test_probabilities, threshold),
    }

    predictions = pd.DataFrame(
        {
            "row_id": frame.index[test_mask],
            GROUP: frame.loc[test_mask, GROUP].to_numpy(),
            "stage": stage,
            "model": model_name,
            "y_true": y_test.to_numpy(),
            "probability": test_probabilities,
            "prediction": test_predictions,
        }
    )

    feature_importance = None
    if model_name == "XGBoost":
        feature_importance = (
            pd.DataFrame(
                {
                    "stage": stage,
                    "feature": preprocessor.get_feature_names_out(),
                    "importance": model.feature_importances_,
                }
            )
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    artifact = {
        "stage": stage,
        "model_name": model_name,
        "features": features,
        "threshold": threshold,
        "preprocessor": preprocessor,
        "model": model,
    }
    model_filename = f"{stage.lower().replace(' ', '_')}_{model_name.lower().replace(' ', '_')}.joblib"
    joblib.dump(artifact, output_dir / "models" / model_filename)

    return ExperimentResult(
        stage=stage,
        model_name=model_name,
        metrics=metrics,
        predictions=predictions,
        feature_importance=feature_importance,
    )


def data_quality_summary(datasets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for stage, frame in datasets.items():
        duplicate_group_mask = frame.duplicated(GROUP, keep=False)
        conflicting_groups = (
            frame.loc[duplicate_group_mask].groupby(GROUP)[TARGET].nunique().gt(1).sum()
        )
        rows.append(
            {
                "stage": stage,
                "rows": len(frame),
                "columns": len(frame.columns),
                "learners": frame[GROUP].nunique(),
                "dropouts": int(frame[TARGET].sum()),
                "dropout_rate": frame[TARGET].mean(),
                "repeated_learners": int(
                    frame.loc[duplicate_group_mask, GROUP].nunique()
                ),
                "repeated_learners_with_conflicting_outcomes": int(conflicting_groups),
                "exact_duplicate_rows": int(frame.duplicated().sum()),
            }
        )
    return pd.DataFrame(rows)


def split_summary(reference: pd.DataFrame, split: pd.Series) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "split": name,
                "rows": int(split.eq(name).sum()),
                "learners": int(reference.loc[split.eq(name), GROUP].nunique()),
                "dropouts": int(reference.loc[split.eq(name), TARGET].sum()),
                "dropout_rate": reference.loc[split.eq(name), TARGET].mean(),
            }
            for name in ("train", "validation", "test")
        ]
    )


def save_figures(
    metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    importance: pd.DataFrame,
    figures_dir: Path,
) -> None:
    sns.set_theme(style="whitegrid")

    comparison = metrics.melt(
        id_vars=["stage", "model"],
        value_vars=["roc_auc", "average_precision", "precision", "recall", "f2"],
        var_name="metric",
        value_name="score",
    )
    plot = sns.catplot(
        data=comparison,
        x="stage",
        y="score",
        hue="model",
        col="metric",
        col_wrap=3,
        kind="bar",
        height=3.6,
        aspect=1.2,
        sharey=False,
    )
    plot.set_titles("{col_name}")
    plot.set_axis_labels("", "Score")
    plot.figure.suptitle("Held-out test performance by available information stage", y=1.02)
    plot.savefig(figures_dir / "stage_model_comparison.png", dpi=160, bbox_inches="tight")
    plt.close(plot.figure)

    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for ax, ((stage, model), frame) in zip(
        axes.ravel(), predictions.groupby(["stage", "model"], sort=False)
    ):
        cm = confusion_matrix(frame["y_true"], frame["prediction"])
        sns.heatmap(cm, annot=True, fmt=",d", cmap="Blues", cbar=False, ax=ax)
        ax.set_title(f"{stage} — {model}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    fig.tight_layout()
    fig.savefig(figures_dir / "confusion_matrices.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    stage_3 = importance.loc[importance["stage"].eq("Stage 3")].head(20).copy()
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.barplot(data=stage_3, x="importance", y="feature", color="#2f6f9f", ax=ax)
    ax.set_title("Stage 3 XGBoost — top feature importances")
    ax.set_xlabel("Gain-based importance")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(
        figures_dir / "stage_3_xgboost_feature_importance.png",
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(fig)


def common_stage_3_cohort_metrics(
    metrics: pd.DataFrame,
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Re-score every model on the exact test rows eligible for Stage 3."""
    common_rows = set(
        predictions.loc[predictions["stage"].eq("Stage 3"), "row_id"]
    )
    rows: list[dict[str, Any]] = []
    for (stage, model_name), frame in predictions.groupby(["stage", "model"]):
        frame = frame.loc[frame["row_id"].isin(common_rows)]
        threshold = float(
            metrics.loc[
                metrics["stage"].eq(stage) & metrics["model"].eq(model_name),
                "threshold",
            ].iloc[0]
        )
        rows.append(
            {
                "stage": stage,
                "model": model_name,
                "test_rows": len(frame),
                **classification_metrics(
                    frame["y_true"], frame["probability"].to_numpy(), threshold
                ),
            }
        )
    return pd.DataFrame(rows)

def run_all(project_root: Path) -> dict[str, pd.DataFrame]:
    """Run all six experiments and write verified portfolio artefacts."""
    project_root = project_root.resolve()
    data_dir = project_root / "data" / "processed"
    reports_dir = project_root / "reports"
    figures_dir = reports_dir / "figures"
    models_dir = project_root / "models"
    for directory in (reports_dir, figures_dir, models_dir):
        directory.mkdir(parents=True, exist_ok=True)

    datasets = load_stage_data(data_dir)
    reference = datasets["Stage 1"][[GROUP, TARGET]].copy()
    split = make_grouped_split(reference)

    results: list[ExperimentResult] = []
    for stage in STAGE_FILES:
        for model_name in ("XGBoost", "Neural network"):
            print(f"Running {stage} — {model_name}")
            results.append(
                run_stage_model(
                    datasets[stage],
                    split,
                    stage,
                    model_name,
                    project_root,
                )
            )

    metrics = pd.DataFrame([result.metrics for result in results])
    predictions = pd.concat([result.predictions for result in results], ignore_index=True)
    importance = pd.concat(
        [
            result.feature_importance
            for result in results
            if result.feature_importance is not None
        ],
        ignore_index=True,
    )
    quality = data_quality_summary(datasets)
    splits = split_summary(reference, split)
    common_cohort = common_stage_3_cohort_metrics(metrics, predictions)

    metrics.to_csv(reports_dir / "model_metrics.csv", index=False)
    predictions.to_csv(reports_dir / "test_predictions.csv", index=False)
    importance.to_csv(reports_dir / "xgboost_feature_importance.csv", index=False)
    quality.to_csv(reports_dir / "data_quality_summary.csv", index=False)
    splits.to_csv(reports_dir / "split_summary.csv", index=False)
    common_cohort.to_csv(
        reports_dir / "stage_3_eligible_cohort_metrics.csv", index=False
    )
    pd.DataFrame({"row_id": reference.index, "split": split}).to_csv(
        reports_dir / "split_assignments.csv", index=False
    )
    save_figures(metrics, predictions, importance, figures_dir)

    run_summary = {
        "random_seed": RANDOM_SEED,
        "split_strategy": "shared 60/20/20 stratified learner-grouped split",
        "threshold_strategy": "validation-set F2 maximum",
        "models": ["XGBoost", "scikit-learn MLPClassifier"],
    }
    (reports_dir / "run_summary.json").write_text(
        json.dumps(run_summary, indent=2), encoding="utf-8"
    )

    return {
        "metrics": metrics,
        "predictions": predictions,
        "importance": importance,
        "quality": quality,
        "splits": splits,
        "common_cohort": common_cohort,
    }

