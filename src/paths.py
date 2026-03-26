from __future__ import annotations

import os
import sys
from pathlib import Path

from src.config import STAGE_DATA_FILENAMES


REPO_MARKERS = ("src", "data", "notebooks")
DEFAULT_COLAB_REPO_PARENT = Path("/content")


def is_colab() -> bool:
    return "google.colab" in sys.modules or "COLAB_RELEASE_TAG" in os.environ


def _is_project_root(candidate: Path) -> bool:
    return all((candidate / marker).exists() for marker in REPO_MARKERS)


def resolve_project_root(start_path: str | Path | None = None) -> Path:
    search_roots: list[Path] = []

    if start_path is not None:
        search_roots.append(Path(start_path).resolve())
    else:
        search_roots.extend(
            [
                Path.cwd().resolve(),
                Path(__file__).resolve().parent,
                Path(__file__).resolve().parent.parent,
            ]
        )

    colab_repo_name = os.environ.get("COLAB_PROJECT_REPO", "student-dropout-prediction")
    colab_repo_root = DEFAULT_COLAB_REPO_PARENT / colab_repo_name
    if is_colab():
        search_roots.extend([colab_repo_root, DEFAULT_COLAB_REPO_PARENT])

    seen: set[Path] = set()
    for root in search_roots:
        for candidate in [root, *root.parents]:
            if candidate in seen:
                continue
            seen.add(candidate)
            if _is_project_root(candidate):
                return candidate

    if _is_project_root(colab_repo_root):
        return colab_repo_root

    if is_colab():
        raise FileNotFoundError(
            "Could not locate the repository root. In Colab, clone the repo into "
            f"{colab_repo_root} or set COLAB_PROJECT_REPO to the cloned folder name."
        )

    raise FileNotFoundError(
        "Could not locate the repository root from the current working directory."
    )


def get_notebook_root(start_path: str | Path | None = None) -> Path:
    project_root = resolve_project_root(start_path=start_path)
    notebook_root = project_root / "notebooks"
    if notebook_root.exists():
        return notebook_root
    return project_root


def ensure_repo_on_sys_path(project_root: str | Path | None = None) -> Path:
    resolved_root = resolve_project_root(project_root)
    root_str = str(resolved_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return resolved_root


def maybe_change_to_notebook_root(start_path: str | Path | None = None) -> Path:
    notebook_root = get_notebook_root(start_path=start_path)
    if Path.cwd().resolve() != notebook_root:
        os.chdir(notebook_root)
    return notebook_root


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
