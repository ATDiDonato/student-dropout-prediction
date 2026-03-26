from __future__ import annotations

DEFAULT_RANDOM_SEED = 42
STABILITY_SEEDS = (42, 123, 999)

TARGET_COLUMN = "dropout"

STAGE_DATA_FILENAMES = {
    "stage_1": "Stage_1_public.csv",
    "stage_2": "Stage_2_public.csv",
    "stage_3": "Stage_3_public.csv",
}

RUNTIME_DEFAULTS = {
    "LOAD_SAVED_TUNING": True,
    "RUN_XGB_TUNING": False,
    "RUN_NN_TUNING": False,
    "RELOAD_DATA_CACHE": False,
}

MANUAL_REVIEW_DEFAULT = True

TUNING_ARTIFACT_FILENAMES = {
    "best_params": "best_params.json",
    "trials": "trials.csv",
    "search_summary": "search_summary.txt",
    "best_model_training": "best_model_training.keras",
    "optuna_study": "optuna_study.sqlite3",
    "best_model_joblib": "best_model.joblib",
    "stability_data": "stability_data.h5",
    "initial_trials": "initial_trials.csv",
}
