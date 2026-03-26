from __future__ import annotations

from pathlib import Path

from src.config import STAGE_DATA_FILENAMES


def resolve_project_root(start_path: str | Path | None = None) -> Path:
    start = Path(start_path or Path.cwd()).resolve()
    for candidate in [start, *start.parents]:
        if (candidate / "data").exists() and (candidate / "notebooks").exists():
            return candidate
    raise FileNotFoundError(
        "Could not locate the repository root from the current working directory."
    )


PROJECT_ROOT = resolve_project_root()
DATA_DIR = PROJECT_ROOT / "data" / "processed"
TUNING_DIR = PROJECT_ROOT / "tuning"
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"


def get_stage_paths(stage_name: str, previous_stage: str | None = None) -> dict[str, Path]:
    data_cache_dir = PROJECT_ROOT / "data" / "cache"
    stage_tuning_dir = TUNING_DIR / stage_name
    stage_model_dir = MODELS_DIR / stage_name

    paths = {
        "DATA_CACHE_DIR": data_cache_dir,
        "STAGE_TUNING_DIR": stage_tuning_dir,
        "XGB_TUNING_DIR": stage_tuning_dir / "xgboost",
        "NN_TUNING_DIR": stage_tuning_dir / "neural_network",
        "STAGE_MODEL_DIR": stage_model_dir,
        "XGB_MODEL_DIR": stage_model_dir / "xgboost",
        "NN_MODEL_DIR": stage_model_dir / "neural_network",
    }

    if previous_stage:
        paths["PREV_XGB_TUNING_DIR"] = TUNING_DIR / previous_stage / "xgboost"
        paths["PREV_NN_TUNING_DIR"] = TUNING_DIR / previous_stage / "neural_network"

    return paths


def get_stage_data_path(stage_name: str) -> Path:
    return DATA_DIR / STAGE_DATA_FILENAMES[stage_name]


def ensure_directories(*directories: Path) -> None:
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
