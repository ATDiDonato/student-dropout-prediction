from __future__ import annotations

from pathlib import Path

from src.paths import ensure_directories, resolve_project_root


PROJECT_ROOT = resolve_project_root()

DEMO_ARTIFACT_DIR = PROJECT_ROOT / "demo_artifacts"

DEMO_DATASET_DIR = DEMO_ARTIFACT_DIR / "datasets"

DEMO_MODEL_DIR = DEMO_ARTIFACT_DIR / "models"
DEMO_XGB_MODEL_DIR = DEMO_MODEL_DIR / "xgboost"
DEMO_NN_MODEL_DIR = DEMO_MODEL_DIR / "neural_network"

DEMO_TUNING_DIR = DEMO_ARTIFACT_DIR / "tuning"
DEMO_GRIDSEARCH_DIR = DEMO_TUNING_DIR / "gridsearch"
DEMO_XGB_TUNING_DIR = DEMO_TUNING_DIR / "xgboost"
DEMO_KERAS_TUNER_DIR = DEMO_TUNING_DIR / "keras_tuner"
DEMO_NN_GRID_SEARCH_DIR = DEMO_TUNING_DIR / "nn_grid_search"
DEMO_NN_GRID_SEARCH_HISTORY_DIR = DEMO_NN_GRID_SEARCH_DIR / "histories"
DEMO_NN_GRID_SEARCH_MODEL_DIR = DEMO_NN_GRID_SEARCH_DIR / "models"

DEMO_FIGURES_DIR = DEMO_ARTIFACT_DIR / "figures"
DEMO_MPLCONFIG_DIR = DEMO_ARTIFACT_DIR / ".mplconfig"


def ensure_demo_directories() -> None:
    ensure_directories(
        DEMO_ARTIFACT_DIR,
        DEMO_DATASET_DIR,
        DEMO_MODEL_DIR,
        DEMO_XGB_MODEL_DIR,
        DEMO_NN_MODEL_DIR,
        DEMO_TUNING_DIR,
        DEMO_GRIDSEARCH_DIR,
        DEMO_XGB_TUNING_DIR,
        DEMO_KERAS_TUNER_DIR,
        DEMO_NN_GRID_SEARCH_DIR,
        DEMO_NN_GRID_SEARCH_HISTORY_DIR,
        DEMO_NN_GRID_SEARCH_MODEL_DIR,
        DEMO_FIGURES_DIR,
        DEMO_MPLCONFIG_DIR,
    )
