from __future__ import annotations

import contextlib
import io
import json
import os
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from tensorflow.keras.models import load_model

from src.config import TUNING_ARTIFACT_FILENAMES


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_saved_best_params(path: str | Path) -> dict[str, Any] | None:
    path = Path(path)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload.get("best_params", payload)


def best_params_record(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def tuner_trials_to_dataframe(tuner: Any) -> pd.DataFrame:
    rows = []
    for trial in tuner.oracle.trials.values():
        row = trial.hyperparameters.values.copy()
        row["trial_id"] = trial.trial_id
        row["status"] = str(trial.status).split(".")[-1]
        for metric_name in [
            "val_auc",
            "val_precision",
            "val_recall",
            "val_accuracy",
            "val_loss",
        ]:
            try:
                row[metric_name] = trial.metrics.get_best_value(metric_name)
            except Exception:
                row[metric_name] = None
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("val_auc", ascending=False)


def count_optuna_completed_trials(study: optuna.Study | None) -> int:
    if study is None:
        return 0
    return sum(
        1 for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE
    )


def count_keras_tuner_completed_trials(tuner: Any) -> int:
    return sum(
        1
        for trial in tuner.oracle.trials.values()
        if str(trial.status).split(".")[-1].upper() == "COMPLETED"
    )


def load_model_weights(
    model: Any, weights_path: str | Path
) -> tuple[Any, bool]:
    weights_path = Path(weights_path)
    if weights_path.exists():
        if weights_path.suffix == ".keras":
            loaded_model = load_model(weights_path)
            print(f"Keras model loaded from {weights_path}")
            return loaded_model, True
        model.load_weights(weights_path)
        print(f"Model weights loaded from {weights_path}")
        return model, True

    print(
        f"No weights file found at {weights_path}. "
        "Model will be trained from scratch."
    )
    return model, False


def keras_tuner_paths(
    artifact_dir: str | Path,
    project_name: str,
    artifact_prefix: str = "",
) -> dict[str, Path]:
    artifact_dir = Path(artifact_dir)
    tuner_root = artifact_dir / "keras_tuner"
    tuner_project_dir = tuner_root / project_name
    prefix = f"{artifact_prefix}_" if artifact_prefix else ""
    return {
        "artifact_dir": artifact_dir,
        "tuner_root": tuner_root,
        "tuner_project_dir": tuner_project_dir,
        "oracle_path": tuner_project_dir / "oracle.json",
        "best_model_path": artifact_dir
        / f"{prefix}{TUNING_ARTIFACT_FILENAMES['best_model_training']}",
        "trials_path": artifact_dir
        / f"{prefix}{TUNING_ARTIFACT_FILENAMES['trials']}",
        "search_summary_path": artifact_dir
        / f"{prefix}{TUNING_ARTIFACT_FILENAMES['search_summary']}",
        "best_params_path": artifact_dir
        / f"{prefix}{TUNING_ARTIFACT_FILENAMES['best_params']}",
    }


def optuna_artifact_paths(
    artifact_dir: str | Path, model_dir: str | Path | None = None
) -> dict[str, Path | str]:
    artifact_dir = Path(artifact_dir)
    model_dir = Path(model_dir) if model_dir is not None else artifact_dir
    storage_path = artifact_dir / TUNING_ARTIFACT_FILENAMES["optuna_study"]
    return {
        "artifact_dir": artifact_dir,
        "model_dir": model_dir,
        "storage_path": storage_path,
        "storage_url": f"sqlite:///{storage_path}",
        "best_params_path": artifact_dir
        / TUNING_ARTIFACT_FILENAMES["best_params"],
        "model_path": model_dir / TUNING_ARTIFACT_FILENAMES["best_model_joblib"],
        "trials_path": artifact_dir / TUNING_ARTIFACT_FILENAMES["trials"],
    }


def run_keras_tuner(
    *,
    kt_module: Any,
    build_model: Callable[[Any], Any],
    max_trials: int,
    project_name: str,
    X_train: Any,
    y_train: Any,
    X_val: Any,
    y_val: Any,
    early_stopping_callback: Any,
    seed: int,
    artifact_dir: str | Path,
    executions_per_trial: int = 1,
    overwrite: bool = False,
    hp_bs: int = 64,
    run_search: bool = True,
    load_saved: bool = True,
    resume_search: bool = True,
    artifact_prefix: str = "",
) -> tuple[Any | None, Any | None]:
    paths = keras_tuner_paths(artifact_dir, project_name, artifact_prefix)
    Path(paths["tuner_root"]).mkdir(parents=True, exist_ok=True)

    tuner = kt_module.RandomSearch(
        build_model,
        objective=kt_module.Objective("val_auc", direction="max"),
        max_trials=max_trials,
        executions_per_trial=executions_per_trial,
        directory=os.fspath(paths["tuner_root"]),
        project_name=project_name,
        overwrite=overwrite,
        seed=seed,
    )

    if not overwrite and Path(paths["oracle_path"]).exists():
        tuner.reload()

    saved_exists = Path(paths["oracle_path"]).exists()
    completed_trials = count_keras_tuner_completed_trials(tuner)

    if run_search:
        if completed_trials >= max_trials:
            print(
                "Skipping NN tuning rerun because "
                f"{completed_trials} completed trials already exist in "
                f"{paths['tuner_project_dir']} (requested total: {max_trials})."
            )
        elif completed_trials > 0 and resume_search:
            print(
                "Resuming NN tuning from "
                f"{paths['tuner_project_dir']} with {completed_trials} completed "
                f"trials. Running until {max_trials} total trials are reached."
            )
            tuner.search(
                X_train,
                y_train,
                validation_data=(X_val, y_val),
                epochs=50,
                batch_size=hp_bs,
                callbacks=[early_stopping_callback],
                verbose=1,
            )
        elif completed_trials > 0 and not resume_search:
            print(
                "NN tuning results already exist in "
                f"{paths['tuner_project_dir']} with {completed_trials} completed "
                "trials. Resume is disabled, so no additional trials will run."
            )
        else:
            print(
                "Starting a new NN tuning run in "
                f"{paths['tuner_project_dir']} for up to {max_trials} trials."
            )
            tuner.search(
                X_train,
                y_train,
                validation_data=(X_val, y_val),
                epochs=50,
                batch_size=hp_bs,
                callbacks=[early_stopping_callback],
                verbose=1,
            )
    elif load_saved and saved_exists:
        print(
            "Loading saved NN tuning results from "
            f"{paths['tuner_project_dir']} without rerunning tuning."
        )
    else:
        print(
            f"No saved NN tuning artefacts found at {paths['tuner_project_dir']}. "
            "RUN_NN_TUNING is disabled, so tuning was skipped."
        )
        return None, None

    if not tuner.oracle.trials:
        print(
            f"No NN tuning trials are available in {paths['tuner_project_dir']}. "
            "Set RUN_NN_TUNING = True to generate them."
        )
        return None, None

    best_hp = tuner.get_best_hyperparameters(1)[0]
    best_model = tuner.get_best_models(1)[0]
    best_model.save(paths["best_model_path"])

    trials_df = tuner_trials_to_dataframe(tuner)
    if not trials_df.empty:
        trials_df.to_csv(paths["trials_path"], index=False)

    summary_buffer = io.StringIO()
    with contextlib.redirect_stdout(summary_buffer):
        tuner.results_summary()
    Path(paths["search_summary_path"]).write_text(
        summary_buffer.getvalue(),
        encoding="utf-8",
    )

    save_json(
        paths["best_params_path"],
        {"project_name": project_name, "best_params": best_hp.values},
    )

    return best_hp, tuner


def run_optuna_xgb(
    *,
    study_name: str,
    search_space: dict[str, dict[str, Any]],
    X_train: Any,
    y_train: Any,
    X_val: Any,
    y_val: Any,
    artifact_dir: str | Path,
    model_dir: str | Path | None = None,
    seed: int = 42,
    overwrite: bool = False,
    n_trials: int = 25,
    n_estimators: int = 3000,
    early_stopping_rounds: int = 30,
    n_jobs: int = -1,
    run_search: bool = True,
    load_saved: bool = True,
    resume_search: bool = True,
    early_stopping_callback_factory: Callable[[int], Any] | None = None,
) -> tuple[Any | None, dict[str, Any] | None, float | None, Any]:
    paths = optuna_artifact_paths(artifact_dir, model_dir=model_dir)
    Path(paths["artifact_dir"]).mkdir(parents=True, exist_ok=True)
    Path(paths["model_dir"]).mkdir(parents=True, exist_ok=True)
    saved_study_exists = Path(paths["storage_path"]).exists()

    if overwrite and Path(paths["storage_path"]).exists():
        try:
            optuna.delete_study(
                study_name=study_name, storage=paths["storage_url"]
            )
        except KeyError:
            pass

    X_train_np = np.asarray(X_train, dtype=np.float32)
    X_val_np = np.asarray(X_val, dtype=np.float32)
    y_train_np = np.asarray(y_train)
    y_val_np = np.asarray(y_val)

    def build_callbacks() -> list[Any]:
        if early_stopping_callback_factory is None:
            return []
        return [early_stopping_callback_factory(early_stopping_rounds)]

    def objective(trial: optuna.Trial) -> float:
        params = {
            "learning_rate": trial.suggest_float(
                "learning_rate",
                search_space["learning_rate"]["low"],
                search_space["learning_rate"]["high"],
                log=search_space["learning_rate"]["log"],
            ),
            "max_depth": trial.suggest_int(
                "max_depth",
                search_space["max_depth"]["low"],
                search_space["max_depth"]["high"],
            ),
            "min_child_weight": trial.suggest_float(
                "min_child_weight",
                search_space["min_child_weight"]["low"],
                search_space["min_child_weight"]["high"],
                log=search_space["min_child_weight"]["log"],
            ),
            "gamma": trial.suggest_float(
                "gamma",
                search_space["gamma"]["low"],
                search_space["gamma"]["high"],
            ),
            "subsample": trial.suggest_float(
                "subsample",
                search_space["subsample"]["low"],
                search_space["subsample"]["high"],
            ),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree",
                search_space["colsample_bytree"]["low"],
                search_space["colsample_bytree"]["high"],
            ),
            "reg_alpha": trial.suggest_float(
                "reg_alpha",
                search_space["reg_alpha"]["low"],
                search_space["reg_alpha"]["high"],
                log=search_space["reg_alpha"]["log"],
            ),
            "reg_lambda": trial.suggest_float(
                "reg_lambda",
                search_space["reg_lambda"]["low"],
                search_space["reg_lambda"]["high"],
                log=search_space["reg_lambda"]["log"],
            ),
        }

        model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            objective="binary:logistic",
            eval_metric="auc",
            random_state=seed,
            n_jobs=n_jobs,
            tree_method="hist",
            callbacks=build_callbacks(),
            **params,
        )
        model.fit(
            X_train_np,
            y_train_np,
            eval_set=[(X_val_np, y_val_np)],
            verbose=False,
        )

        best_iter = getattr(model, "best_iteration", None)
        if best_iter is not None:
            y_val_prob = model.predict_proba(
                X_val,
                iteration_range=(0, best_iter + 1),
            )[:, 1]
        else:
            y_val_prob = model.predict_proba(X_val_np)[:, 1]
        return roc_auc_score(y_val_np, y_val_prob)

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        storage=paths["storage_url"],
        load_if_exists=True,
        sampler=sampler,
    )
    completed_trials = count_optuna_completed_trials(study)

    if not run_search:
        if load_saved and completed_trials > 0:
            print(
                "Loading saved XGBoost tuning results from "
                f"{paths['artifact_dir']} without rerunning tuning."
            )
        else:
            print(
                f"No saved XGBoost tuning artefacts found in {paths['artifact_dir']}. "
                "RUN_XGB_TUNING is disabled, so tuning was skipped."
            )
            return None, None, None, study
    else:
        if completed_trials >= n_trials:
            print(
                "Skipping XGBoost tuning rerun because "
                f"{completed_trials} completed trials already exist in "
                f"{paths['artifact_dir']} (requested total: {n_trials})."
            )
        elif completed_trials > 0 and resume_search:
            remaining_trials = n_trials - completed_trials
            print(
                "Resuming XGBoost tuning from "
                f"{paths['artifact_dir']} with {completed_trials} completed "
                f"trials. Running {remaining_trials} additional trials."
            )
            study.optimize(
                objective, n_trials=remaining_trials, show_progress_bar=True
            )
        elif completed_trials > 0 and not resume_search:
            print(
                "XGBoost tuning results already exist in "
                f"{paths['artifact_dir']} with {completed_trials} completed "
                "trials. Resume is disabled, so no additional trials will run."
            )
        else:
            if saved_study_exists:
                print(
                    "Starting XGBoost tuning from the existing study in "
                    f"{paths['artifact_dir']}."
                )
            else:
                print(
                    "Starting a new XGBoost tuning run in "
                    f"{paths['artifact_dir']} for {n_trials} trials."
                )
            study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    completed_trials = count_optuna_completed_trials(study)
    if completed_trials == 0:
        print(
            f"No completed XGBoost tuning trials are available in {paths['artifact_dir']}."
        )
        return None, None, None, study

    best_params = study.best_params
    best_auc = study.best_value
    best_model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        objective="binary:logistic",
        eval_metric="auc",
        random_state=seed,
        n_jobs=n_jobs,
        tree_method="hist",
        callbacks=build_callbacks(),
        **best_params,
    )
    best_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    save_json(
        paths["best_params_path"],
        {"best_val_auc": float(best_auc), "best_params": best_params},
    )
    joblib.dump(best_model, paths["model_path"])
    study.trials_dataframe().to_csv(paths["trials_path"], index=False)

    return best_model, best_params, best_auc, study
