"""Explicit Stage 2 fixed-configuration rebuilding and verified, read-only reuse.

Only rebuild_xgb_pair fits models or writes artifacts. No tuning helper is used.
"""
from __future__ import annotations

import hashlib
import json
import platform
import warnings
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split


_PARAMETERS = {
    'learning_rate', 'max_depth', 'min_child_weight', 'gamma', 'subsample',
    'colsample_bytree', 'reg_alpha', 'reg_lambda',
}


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def _read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def verify_stage_alignment(stage_1, stage_2):
    """Compare observations in place; duplicated learner identifiers are allowed."""
    if len(stage_1) != len(stage_2) or not stage_1.index.equals(stage_2.index):
        raise ValueError('Stage 1/2 row counts or indices differ; refusing to combine.')
    for frame in (stage_1, stage_2):
        if not frame.columns.is_unique or not frame.index.is_unique:
            raise ValueError('Stage 1/2 columns and row indices must be unique.')
        if 'LearnerCode' not in frame or frame['LearnerCode'].isna().any():
            raise ValueError('Stage 1/2 require nonmissing LearnerCode values.')
    columns = stage_1.columns.intersection(stage_2.columns)
    a, b = stage_1[columns], stage_2[columns]
    equal = a.eq(b) | (a.isna() & b.isna())
    if not equal.to_numpy().all():
        rows, cols = np.where(~equal.to_numpy())
        examples = [(int(r), str(columns[c])) for r, c in zip(rows[:10], cols[:10])]
        raise ValueError(f'Stage 1/2 learner/order/shared-value mismatch: {examples}')


def _evidence(record, project_root):
    params = dict(record.parameters)
    if set(params) != _PARAMETERS or record.source_path is None:
        raise ValueError('Required saved XGBoost parameter evidence is unavailable.')
    source = record.source_path.resolve()
    try:
        source_name = source.relative_to(Path(project_root).resolve()).as_posix()
    except ValueError:
        source_name = str(source)
    return {
        'parameters': params,
        'source_path': source_name,
        'source_sha256': _sha(record.source_path),
        'source_kind': record.source_kind,
        'evidence_kind': record.evidence_kind,
        'study_name': record.study_name,
        'trial_id': record.trial_id,
        'historical_score': record.historical_score,
        'score_name': record.score_name,
        'diagnostics': list(record.diagnostics),
    }


def _settings(role, n_jobs):
    return {
        'n_estimators': 100 if role == 'baseline' else 3000,
        'objective': 'binary:logistic', 'eval_metric': 'auc',
        'random_state': 42, 'n_jobs': n_jobs,
        'tree_method': None if role == 'baseline' else 'hist',
        'early_stopping': None if role == 'baseline' else {
            'rounds': 30, 'metric_name': 'auc', 'data_name': 'validation_0',
            'maximize': True, 'save_best': False,
        },
        'input_scaling': 'none', 'sample_weight': None, 'class_weight': None,
        'validation_set': 'validation', 'verbose': False,
    }


def make_rebuild_context(*, project_root, stage_1_raw, stage_2_raw,
                         X_train, X_val, X_test, y_train, y_val, y_test,
                         baseline_evidence, final_evidence, seed, n_jobs):
    if seed != 42:
        raise ValueError('Stage 2 XGBoost rebuilding requires seed 42.')
    verify_stage_alignment(stage_1_raw, stage_2_raw)
    datasets = {}
    for name, raw in [('stage_1', stage_1_raw), ('stage_2', stage_2_raw)]:
        path = Path(project_root) / 'data' / 'processed' / f'{name.title()}_public.csv'
        disk = pd.read_csv(path)
        pd.testing.assert_frame_equal(raw, disk)
        datasets[name] = {'path': str(path.relative_to(project_root)), 'sha256': _sha(path)}
    indices = stage_1_raw.index
    if not indices.equals(pd.RangeIndex(len(stage_1_raw))):
        raise ValueError('Original CSV row positions must be retained.')
    target = stage_1_raw['dropout']
    expected_train, expected_test = train_test_split(
        indices, test_size=0.2, random_state=42, stratify=target,
    )
    expected_train, expected_val = train_test_split(
        expected_train, test_size=0.1, random_state=42,
        stratify=target.loc[expected_train],
    )
    features = list(X_train.columns)
    if not X_train.columns.is_unique or not all(isinstance(c, str) for c in features):
        raise ValueError('Feature names must be unique strings.')
    splits, prepared = {}, {}
    for name, X, y, expected in [
        ('train', X_train, y_train, expected_train),
        ('validation', X_val, y_val, expected_val),
        ('test', X_test, y_test, expected_test),
    ]:
        if not X.index.equals(expected) or not y.index.equals(expected):
            raise ValueError(f'{name}: rows differ from the original stratified split.')
        pd.testing.assert_series_equal(y, target.loc[expected])
        if list(X.columns) != features or list(X.dtypes.astype(str)) != list(X_train.dtypes.astype(str)):
            raise ValueError(f'{name}: feature names/order/dtypes differ.')
        values = X.to_numpy(dtype=float)
        if np.isinf(values).any() or not y.isin([0, 1]).all():
            raise ValueError(f'{name}: infinite features or nonbinary labels.')
        splits[name] = {
            'row_positions': [int(i) for i in expected],
            'learner_codes': stage_1_raw.loc[expected, 'LearnerCode'].astype(str).tolist(),
        }
        prepared[name] = {
            'shape': list(X.shape),
            'feature_sha256': hashlib.sha256(pd.util.hash_pandas_object(X, index=True).values.tobytes()).hexdigest(),
            'label_sha256': hashlib.sha256(pd.util.hash_pandas_object(y, index=True).values.tobytes()).hexdigest(),
            'missing_values': X.isna().sum().astype(int).tolist(),
        }
    return {
        'schema_version': 1, 'datasets': datasets, 'ordered_features': features,
        'feature_dtypes': X_train.dtypes.astype(str).tolist(), 'prepared': prepared,
        'split_settings': {'seed': 42, 'stratify': True, 'test_size': 0.2, 'validation_fraction_of_remaining': 0.1},
        'splits': splits,
        'evidence': {'baseline': _evidence(baseline_evidence, project_root), 'final': _evidence(final_evidence, project_root)},
        'training_settings': {r: _settings(r, n_jobs) for r in ('baseline', 'final')},
        'library_versions': {name: version(name) for name in ('numpy', 'pandas', 'scikit-learn', 'xgboost', 'joblib')},
        'python_version': platform.python_version(),
    }


def _metrics(model, X, y):
    predictions, probabilities = model.predict(X), model.predict_proba(X)
    if predictions.shape != (len(y),) or probabilities.shape != (len(y), 2):
        raise ValueError('Unexpected prediction/probability dimensions.')
    if (not np.isfinite(probabilities).all() or
            not ((probabilities >= 0) & (probabilities <= 1)).all() or
            not np.allclose(probabilities.sum(axis=1), 1)):
        raise ValueError('Invalid model probabilities.')
    return {
        'accuracy': float(accuracy_score(y, predictions)),
        'precision': float(precision_score(y, predictions, zero_division=0)),
        'recall': float(recall_score(y, predictions, zero_division=0)),
        'specificity': float(recall_score(y, predictions, pos_label=0)),
        'auc': float(roc_auc_score(y, probabilities[:, 1])),
    }


def _verify_model(model, role, context, record):
    if not isinstance(model, xgb.XGBClassifier) or list(model.classes_) != [0, 1]:
        raise ValueError(f'{role}: expected a binary XGBClassifier.')
    booster = model.get_booster()
    if booster.feature_names != context['ordered_features'] or model.n_features_in_ != len(context['ordered_features']):
        raise ValueError(f'{role}: saved feature schema differs.')
    settings = context['training_settings'][role]
    params = model.get_params()
    expected = dict(context['evidence'][role]['parameters'])
    expected.update({k: settings[k] for k in ('n_estimators', 'objective', 'eval_metric', 'random_state', 'n_jobs', 'tree_method')})
    if any(params.get(k) != v for k, v in expected.items()):
        raise ValueError(f'{role}: saved model parameters differ.')
    callbacks = params.get('callbacks') or []
    early = settings['early_stopping']
    if (early is None and callbacks) or (early is not None and (
        len(callbacks) != 1 or not isinstance(callbacks[0], xgb.callback.EarlyStopping)
        or any(getattr(callbacks[0], 'data' if k == 'data_name' else k, None) != v
               for k, v in early.items())
    )):
        raise ValueError(f'{role}: saved early-stopping settings differ.')
    if params.get('early_stopping_rounds') is not None:
        raise ValueError(f'{role}: unexpected estimator early stopping.')
    rounds = booster.num_boosted_rounds()
    if rounds != record['boosting_rounds'] or not 1 <= rounds <= settings['n_estimators']:
        raise ValueError(f'{role}: invalid boosting-round count.')
    if role == 'baseline' and rounds != 100:
        raise ValueError('Baseline must contain 100 boosting rounds.')
    if role == 'final' and (
        model.best_iteration != record['best_iteration'] or
        float(model.best_score) != record['best_score']
    ):
        raise ValueError('Final best-iteration metadata differs.')


def _reuse_context(context):
    """Compare substantive inputs without changing the recorded training context."""
    result = {
        k: v for k, v in context.items()
        if k not in ('splits', 'python_version', 'library_versions')
    }
    result['training_settings'] = {
        role: {k: v for k, v in settings.items() if k != 'n_jobs'}
        for role, settings in context['training_settings'].items()
    }
    return result


def _warn_version_differences(recorded, current, role):
    saved_versions = {'Python': recorded['python_version'], **recorded['library_versions']}
    current_versions = {'Python': current['python_version'], **current['library_versions']}
    for name in sorted(saved_versions.keys() | current_versions.keys()):
        saved = saved_versions.get(name, '<not recorded>')
        active = current_versions.get(name, '<not installed>')
        if saved != active:
            warnings.warn(
                f'{role}: compatibility warning: {name} version differs '
                f'(training={saved}, current={active}). '
                'Model loading and substantive verification remain required.',
                RuntimeWarning,
                stacklevel=3,
            )


def _load_run(run_dir, context, expected_manifest_hash):
    if _sha(run_dir / 'manifest.json') != expected_manifest_hash:
        raise ValueError('Rebuilt manifest checksum mismatch.')
    manifest = _read_json(run_dir / 'manifest.json')
    names = {'splits.json', 'baseline.joblib', 'final.joblib', 'baseline.metadata.json', 'final.metadata.json'}
    if manifest['run_id'] != run_dir.name or set(manifest['sha256']) != names:
        raise ValueError('Invalid rebuilt manifest.')
    for name, digest in manifest['sha256'].items():
        if _sha(run_dir / name) != digest:
            raise ValueError(f'Rebuilt artifact checksum mismatch: {name}')
    if _read_json(run_dir / 'splits.json') != context['splits']:
        raise ValueError('Saved split identity differs.')
    models, records = {}, {}
    metadata_context = _reuse_context(context)
    for role in ('baseline', 'final'):
        record = _read_json(run_dir / f'{role}.metadata.json')
        if (_reuse_context(record['context']) != metadata_context or record['artifact_kind'] != 'rebuilt'
                or record['run_id'] != run_dir.name or record['role'] != role
                or record['model_path'] != f'{role}.joblib'
                or record['model_sha256'] != manifest['sha256'][f'{role}.joblib']
                or record['splits_sha256'] != manifest['sha256']['splits.json']):
            raise ValueError(f'{role}: rebuilt metadata/context mismatch.')
        _warn_version_differences(record['context'], context, role)
        model = joblib.load(run_dir / f'{role}.joblib')
        _verify_model(model, role, record['context'], record)
        models[role], records[role] = model, record
    return models['baseline'], models['final'], records


def load_rebuilt_xgb_pair(root, *, context):
    """Require a complete verified run. Never fit, save, or substitute a model."""
    root = Path(root)
    pointer = _read_json(root / 'current.json')
    run_id = pointer['run_id']
    if not isinstance(run_id, str) or not run_id or Path(run_id).name != run_id or run_id in ('.', '..'):
        raise ValueError('Invalid rebuilt run identifier.')
    return _load_run(root / run_id, context, pointer['manifest_sha256'])


def rebuild_xgb_pair(root, *, context, X_train, X_val, X_test, y_train, y_val, y_test):
    """Fit once per role, save a new run, verify readback, then publish its pointer."""
    root = Path(root)
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid4().hex
    run_dir = root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    _write_json(run_dir / 'splits.json', context['splits'])
    for role in ('baseline', 'final'):
        settings = context['training_settings'][role]
        params = dict(context['evidence'][role]['parameters'])
        params.update({k: settings[k] for k in ('n_estimators', 'objective', 'eval_metric', 'random_state', 'n_jobs')})
        if role == 'final':
            params.update(tree_method='hist', callbacks=[xgb.callback.EarlyStopping(**settings['early_stopping'])])
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        metrics = {name: _metrics(model, X, y) for name, X, y in (
            ('validation', X_val, y_val), ('test', X_test, y_test),
        )}
        model_path = run_dir / f'{role}.joblib'
        joblib.dump(model, model_path)
        record = {
            'artifact_kind': 'rebuilt', 'run_id': run_id, 'role': role,
            'created_at': datetime.now(timezone.utc).isoformat(),
            'model_path': model_path.name, 'model_sha256': _sha(model_path),
            'splits_sha256': _sha(run_dir / 'splits.json'),
            'context': {k: v for k, v in context.items() if k != 'splits'},
            'boosting_rounds': model.get_booster().num_boosted_rounds(),
            'best_iteration': model.best_iteration if role == 'final' else None,
            'best_score': float(model.best_score) if role == 'final' else None,
            'validation_history': model.evals_result(),
            'new_metrics': metrics,
            'historical_evidence': context['evidence'][role],
        }
        _verify_model(model, role, context, record)
        _write_json(run_dir / f'{role}.metadata.json', record)
    names = ('splits.json', 'baseline.joblib', 'final.joblib', 'baseline.metadata.json', 'final.metadata.json')
    _write_json(run_dir / 'manifest.json', {
        'run_id': run_id, 'sha256': {name: _sha(run_dir / name) for name in names},
    })
    manifest_hash = _sha(run_dir / 'manifest.json')
    baseline, final, records = _load_run(run_dir, context, manifest_hash)
    for role, model in [('baseline', baseline), ('final', final)]:
        for name, X, y in [('validation', X_val, y_val), ('test', X_test, y_test)]:
            measured = _metrics(model, X, y)
            if any(not np.isclose(value, records[role]['new_metrics'][name][key], rtol=0, atol=1e-12)
                   for key, value in measured.items()):
                raise ValueError(f'{role}: saved-model metric readback differs.')
    temporary_pointer = root / f'.current-{uuid4().hex}.json'
    _write_json(temporary_pointer, {'run_id': run_id, 'manifest_sha256': manifest_hash})
    temporary_pointer.replace(root / 'current.json')
    return baseline, final, records
