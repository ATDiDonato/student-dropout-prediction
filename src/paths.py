from __future__ import annotations

import os
import sys
from pathlib import Path

from src.config import STAGE_DATA_FILENAMES


REPO_MARKERS = ("src", "data", "notebooks")
DEFAULT_COLAB_REPO_PARENT = Path("/content")
DEFAULT_GOOGLE_DRIVE_ARTIFACT_ROOT = Path("/content/drive/MyDrive/student_dropout_artifacts")
USE_GOOGLE_DRIVE_ARTIFACTS_ENV = "USE_GOOGLE_DRIVE_ARTIFACTS"
GOOGLE_DRIVE_ARTIFACT_ROOT_ENV = "GOOGLE_DRIVE_ARTIFACT_ROOT"


def is_colab() -> bool:
    return "google.colab" in sys.modules or "COLAB_RELEASE_TAG" in os.environ


def _is_project_root(candidate: Path) -> bool:
    return all((candidate / marker).exists() for marker in REPO_MARKERS)


def _env_flag_is_enabled(name: str) -> bool:
    value = os.environ.get(name, "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def use_google_drive_artifacts() -> bool:
    return is_colab() and _env_flag_is_enabled(USE_GOOGLE_DRIVE_ARTIFACTS_ENV)


def get_google_drive_artifact_root() -> Path:
    configured_root = os.environ.get(
        GOOGLE_DRIVE_ARTIFACT_ROOT_ENV,
        os.fspath(DEFAULT_GOOGLE_DRIVE_ARTIFACT_ROOT),
    )
    return Path(configured_root).expanduser().resolve()


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

LOCAL_MODELS_DIR = PROJECT_ROOT / "models"
LOCAL_TUNING_DIR = PROJECT_ROOT / "tuning"
LOCAL_REPORTS_DIR = PROJECT_ROOT / "reports"
LOCAL_FIGURES_DIR = LOCAL_REPORTS_DIR / "figures"

GOOGLE_DRIVE_ARTIFACTS_ENABLED = use_google_drive_artifacts()
GOOGLE_DRIVE_ARTIFACT_ROOT = get_google_drive_artifact_root()
PORTFOLIO_ARTIFACT_ROOT = (
    GOOGLE_DRIVE_ARTIFACT_ROOT / "portfolio"
    if GOOGLE_DRIVE_ARTIFACTS_ENABLED
    else PROJECT_ROOT
)

MODELS_DIR = PORTFOLIO_ARTIFACT_ROOT / "models"
TUNING_DIR = PORTFOLIO_ARTIFACT_ROOT / "tuning"
REPORTS_DIR = PORTFOLIO_ARTIFACT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"


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


def display_path(path: str | Path) -> str:
    resolved_path = Path(path).resolve()

    base_candidates = (
        (
            GOOGLE_DRIVE_ARTIFACT_ROOT,
            PROJECT_ROOT,
            PORTFOLIO_ARTIFACT_ROOT,
        )
        if GOOGLE_DRIVE_ARTIFACTS_ENABLED
        else (
            PROJECT_ROOT,
            PORTFOLIO_ARTIFACT_ROOT,
            GOOGLE_DRIVE_ARTIFACT_ROOT,
        )
    )

    for base in base_candidates:
        try:
            return resolved_path.relative_to(base).as_posix()
        except ValueError:
            continue

    return resolved_path.as_posix()


def resolve_saved_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()

    candidate_parts = candidate.parts
    if not candidate_parts:
        return PROJECT_ROOT

    head = candidate_parts[0]
    tail = Path(*candidate_parts[1:]) if len(candidate_parts) > 1 else Path()

    if head == "portfolio":
        preferred_base = (
            GOOGLE_DRIVE_ARTIFACT_ROOT / "portfolio"
            if GOOGLE_DRIVE_ARTIFACTS_ENABLED
            else PROJECT_ROOT
        )
        return (preferred_base / tail).resolve()
    elif head == "demo":
        preferred_base = (
            GOOGLE_DRIVE_ARTIFACT_ROOT / "demo"
            if GOOGLE_DRIVE_ARTIFACTS_ENABLED
            else PROJECT_ROOT / "demo_artifacts"
        )
        return (preferred_base / tail).resolve()
    elif head == "demo_artifacts":
        preferred_base = (
            PROJECT_ROOT / "demo_artifacts"
            if not GOOGLE_DRIVE_ARTIFACTS_ENABLED
            else GOOGLE_DRIVE_ARTIFACT_ROOT / "demo"
        )
        return (preferred_base / tail).resolve()
    elif head in {"models", "tuning", "reports"}:
        return (PORTFOLIO_ARTIFACT_ROOT / candidate).resolve()
    else:
        base_candidates = (
            (GOOGLE_DRIVE_ARTIFACT_ROOT, PROJECT_ROOT)
            if GOOGLE_DRIVE_ARTIFACTS_ENABLED
            else (PROJECT_ROOT, GOOGLE_DRIVE_ARTIFACT_ROOT)
        )

    for base in base_candidates:
        resolved = (base / candidate).resolve()
        if resolved.exists():
            return resolved

    return (base_candidates[0] / candidate).resolve()
