"""Resolve saved XGBoost parameter evidence without training or model loading."""
from __future__ import annotations

import csv as _csv
import json as _json
import math as _math
import sqlite3 as _sqlite3
from dataclasses import dataclass as _dataclass
from pathlib import Path as _Path
from types import MappingProxyType as _MappingProxyType
from typing import Literal as _Literal, Mapping as _Mapping

__all__ = ["SavedParameterResult", "resolve_xgb_results"]

_NAMES = frozenset({
    "learning_rate", "max_depth", "min_child_weight", "gamma", "subsample",
    "colsample_bytree", "reg_alpha", "reg_lambda",
})


@_dataclass(frozen=True)
class SavedParameterResult:
    parameters: _Mapping[str, str | int | float | bool]
    source_kind: _Literal["optuna", "csv", "json", "unavailable"]
    evidence_kind: _Literal["completed_trial", "parameter_summary", "unavailable"]
    source_path: _Path | None
    historical_score: float | None
    score_name: str | None
    study_name: str | None
    trial_id: str | None
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "parameters", _MappingProxyType(dict(self.parameters)))


class _StudyProvenanceError(ValueError):
    """An existing database rules out unscoped fallback for this study."""


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _loads(text):
    return _json.loads(text, object_pairs_hook=_object)


def _parameters(values, *, csv=False):
    if not isinstance(values, dict) or set(values) != _NAMES:
        raise ValueError("parameters must contain exactly the eight supported XGBoost names")
    result = {}
    for name, value in values.items():
        if csv:
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"missing parameter {name}")
            if name == "max_depth":
                value = int(value)
            else:
                value = float(value)
        if name == "max_depth":
            if type(value) is not int:
                raise ValueError("max_depth must be an integer")
        elif type(value) not in (int, float) or not _math.isfinite(value):
            raise ValueError(f"{name} must be finite numeric data, not a boolean")
        result[name] = value
    return result


def _score(value):
    if isinstance(value, str):
        value = float(value)
    if type(value) not in (int, float) or not _math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("validation AUC must be finite and within [0, 1]")
    return float(value)


def _number(value):
    if isinstance(value, str):
        value = int(value)
    if type(value) is not int or value < 0:
        raise ValueError("trial number must be a nonnegative integer")
    return value


def _result(params, kind, evidence, path, score, study, trial, diagnostics):
    return SavedParameterResult(
        params, kind, evidence, path, score,
        "val_auc" if score is not None else None, study, trial, tuple(diagnostics),
    )


def _sqlite(path, study_name, diagnostics):
    connection = _sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        connection.execute("PRAGMA query_only=ON")
        connection.execute("BEGIN")
        required = {
            "studies": {"study_id", "study_name"},
            "study_directions": {"study_id", "objective", "direction"},
            "trials": {"trial_id", "study_id", "number", "state"},
            "trial_values": {"trial_id", "objective", "value", "value_type"},
            "trial_params": {"trial_id", "param_name", "param_value", "distribution_json"},
        }
        for table, columns in required.items():
            present = {row[1] for row in connection.execute(f'PRAGMA table_info("{table}")')}
            if not columns <= present:
                raise ValueError(f"unsupported SQLite schema: {table}")
        studies = connection.execute(
            "SELECT study_id FROM studies WHERE study_name=?", (study_name,)
        ).fetchall()
        if len(studies) != 1:
            raise _StudyProvenanceError(f"expected exactly one study named {study_name!r}")
        study_id = studies[0][0]
        directions = connection.execute(
            "SELECT objective, direction FROM study_directions WHERE study_id=?", (study_id,)
        ).fetchall()
        if directions != [(0, "MAXIMIZE")]:
            raise ValueError("only one MAXIMIZE objective (0) is supported")
        candidates = []
        rows = connection.execute(
            "SELECT t.trial_id,t.number,v.value,v.value_type,v.objective "
            "FROM trials t JOIN trial_values v ON t.trial_id=v.trial_id "
            "WHERE t.study_id=? AND t.state='COMPLETE'", (study_id,)
        ).fetchall()
        for trial_id, number, score, value_type, objective in rows:
            try:
                if value_type != "FINITE" or objective != 0:
                    raise ValueError("non-finite or unsupported objective")
                number = _number(number)
                score = _score(score)
                params = {}
                for name, value, distribution_text in connection.execute(
                    "SELECT param_name,param_value,distribution_json FROM trial_params WHERE trial_id=?",
                    (trial_id,),
                ):
                    if name in params:
                        raise ValueError(f"duplicate parameter {name}")
                    distribution = _loads(distribution_text)
                    kind = distribution.get("name")
                    if type(value) not in (int, float) or not _math.isfinite(value):
                        raise ValueError(f"invalid internal parameter {name}")
                    if kind == "IntDistribution":
                        if value != int(value):
                            raise ValueError(f"nonintegral integer parameter {name}")
                        value = int(value)
                    elif kind != "FloatDistribution":
                        raise ValueError(f"unsupported distribution for {name}: {kind}")
                    params[name] = value
                candidates.append((score, number, _parameters(params)))
            except (ValueError, TypeError, AttributeError, OverflowError) as exc:
                diagnostics.append(f"{path}: rejected trial {number}: {exc}")
        if not candidates:
            raise _StudyProvenanceError(f"no valid completed finite trials in study {study_name!r}")
        score, number, params = min(candidates, key=lambda row: (-row[0], row[1]))
        return _result(params, "optuna", "completed_trial", path, score, study_name, str(number), diagnostics)
    finally:
        connection.close()


def _csv_result(path, diagnostics):
    with path.open(newline="", encoding="utf-8") as stream:
        reader = _csv.DictReader(stream, strict=True)
        headers = reader.fieldnames
        if not headers or len(headers) != len(set(headers)):
            raise ValueError("missing or duplicate CSV headers")
        param_columns = {"params_" + name for name in _NAMES}
        if not param_columns <= set(headers) or {h for h in headers if h.startswith("params_")} != param_columns:
            raise ValueError("unsupported CSV parameter schema")
        rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError("malformed CSV row")
    if "state" not in headers:
        if set(headers) != param_columns | {"number", "value"} or len(rows) != 1:
            raise ValueError("status-free CSV is not a recognized single-row parameter summary")
        row = rows[0]
        if row["number"] != "0" or row["value"].strip():
            raise ValueError("unsupported parameter-summary identity or score")
        params = _parameters({name: row["params_" + name] for name in _NAMES}, csv=True)
        diagnostics.append(f"{path}: parameter summary; completion and score are unknown")
        return _result(params, "csv", "parameter_summary", path, None, None, None, diagnostics)
    if not {"number", "value"} <= set(headers):
        raise ValueError("missing CSV trial identity or score")
    candidates = []
    for position, row in enumerate(rows):
        if row["state"] != "COMPLETE":
            continue
        try:
            params = _parameters({name: row["params_" + name] for name in _NAMES}, csv=True)
            candidates.append((_score(row["value"]), _number(row["number"]), position, params))
        except (ValueError, TypeError, OverflowError) as exc:
            diagnostics.append(f"{path}: rejected CSV row {position + 2}: {exc}")
    if not candidates:
        raise ValueError("no valid completed finite CSV trials")
    score, number, _, params = min(candidates, key=lambda row: (-row[0], row[1], row[2]))
    diagnostics.append(f"{path}: CSV does not establish the originating study")
    return _result(params, "csv", "completed_trial", path, score, None, str(number), diagnostics)


def _json_result(path, diagnostics):
    payload = _loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "best_params" not in payload:
        raise ValueError("expected a nested best_params object")
    params = _parameters(payload["best_params"])
    score = None
    if "best_val_auc" in payload:
        try:
            score = _score(payload["best_val_auc"])
        except (ValueError, TypeError, OverflowError) as exc:
            diagnostics.append(f"{path}: ignored invalid summary score: {exc}")
    return _result(params, "json", "parameter_summary", path, score, None, None, diagnostics)


def resolve_xgb_results(artifact_dir: str | _Path, *, study_name: str) -> SavedParameterResult:
    """Read SQLite, CSV, then JSON; return parameter evidence, never a model.

    An existing SQLite file is authoritative for every outcome. Unscoped
    CSV/JSON fallback is allowed only when the SQLite file does not exist.
    """
    if not isinstance(study_name, str) or not study_name.strip():
        raise ValueError("study_name must be an explicit nonempty string")
    directory = _Path(artifact_dir)
    diagnostics = []
    sqlite_exists = (directory / "optuna_study.sqlite3").exists()
    for filename, reader in (
        ("optuna_study.sqlite3", lambda path: _sqlite(path, study_name, diagnostics)),
        ("trials.csv", lambda path: _csv_result(path, diagnostics)),
        ("best_params.json", lambda path: _json_result(path, diagnostics)),
    ):
        path = directory / filename
        try:
            return reader(path)
        except (OSError, ValueError, TypeError, OverflowError, _sqlite3.Error, _csv.Error) as exc:
            diagnostics.append(f"{path}: source rejected: {exc}")
            if filename == "optuna_study.sqlite3" and sqlite_exists:
                diagnostics.append(
                    f"{path}: unscoped trials.csv and best_params.json fallback rejected "
                    "because the existing SQLite database establishes study-specific provenance"
                )
                return _result({}, "unavailable", "unavailable", None, None, None, None, diagnostics)
    return _result({}, "unavailable", "unavailable", None, None, None, None, diagnostics)
