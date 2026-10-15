"""New fixed-configuration Stage 3 XGBoost runs and verified saved-pair reuse.

These runs use saved rounded parameters, not historical trial 1798. Shared
Stage 2 checks retain their original behavior; no Stage 2 files are changed.
"""
from __future__ import annotations

import hashlib
import platform
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split

from src import stage_2_xgb_rebuild as shared
from src.saved_results import resolve_xgb_results


_FILES = {
    'baseline.joblib', 'final.joblib', 'baseline.metadata.json',
    'final.metadata.json', 'features.json', 'splits.json',
}
_PURPOSE = 'new_fixed_configuration_rebuild_not_historical_trial_reproduction'


def verify_stage_alignment(stage_2, stage_3):
    """Compare original observations in order; never deduplicate learner codes."""
    try:
        shared.verify_stage_alignment(stage_2, stage_3)
    except ValueError as exc:
        raise ValueError(f'Stage 2/3 alignment failed: {exc}') from exc


def resolve_parameter_evidence(artifact_dir, *, study_name, project_root):
    """Require agreement between the saved JSON and resolved read-only evidence."""
    if study_name not in ('xgb_stage_2_tabular_auc', 'xgb_stage_3_tabular_auc'):
        raise ValueError('Expected the Stage 2 or Stage 3 XGBoost study identity.')
    directory = Path(artifact_dir)
    record = resolve_xgb_results(directory, study_name=study_name)
    evidence = shared._evidence(record, project_root)
    path = directory / 'best_params.json'
    if not path.is_file():
        raise FileNotFoundError(f'Required fixed-configuration parameter summary is missing: {path}')
    params = shared._read_json(path)['best_params']
    if set(params) != shared._PARAMETERS:
        raise ValueError(f'Expected exactly eight fixed XGBoost parameters: {path}')
    for name, value in params.items():
        if (type(value) not in (int, float) or not np.isfinite(value)
                or (name == 'max_depth' and type(value) is not int)):
            raise ValueError(f'Invalid fixed XGBoost parameter {name}: {path}')
    if params != evidence['parameters']:
        raise ValueError(f'Saved JSON and resolved XGBoost configuration evidence conflict: {path}')
    evidence['requested_study_name'] = study_name
    evidence['configuration_identity'] = (
        'stage_2_saved_final' if study_name == 'xgb_stage_2_tabular_auc'
        else 'stage_3_saved_rounded_final'
    )
    evidence['summary_sources'] = [
        {'path': str(source.resolve().relative_to(Path(project_root).resolve())),
         'sha256': shared._sha(source)}
        for source in (path, directory / 'trials.csv') if source.is_file()
    ]
    return evidence


def _settings(role, n_jobs):
    settings = shared._settings(role, n_jobs)
    if role == 'final':
        # Preserve the original Stage 3 callback's omitted/default arguments.
        settings['early_stopping'].update(data_name=None, min_delta=0.0)
    return settings


def _prepared(X, y):
    return {
        'shape': list(X.shape),
        'feature_sha256': hashlib.sha256(pd.util.hash_pandas_object(X, index=True).values.tobytes()).hexdigest(),
        'label_sha256': hashlib.sha256(pd.util.hash_pandas_object(y, index=True).values.tobytes()).hexdigest(),
        'missing_values': X.isna().sum().astype(int).tolist(),
    }


def make_rebuild_context(*, project_root, stage_2_raw, stage_3_raw,
                         X_train, X_val, X_test, y_train, y_val, y_test,
                         baseline_evidence, final_evidence, seed, n_jobs):
    if seed != 42:
        raise ValueError('Stage 3 XGBoost rebuilding requires seed 42.')
    verify_stage_alignment(stage_2_raw, stage_3_raw)
    root = Path(project_root)
    datasets = {}
    for stage, raw in [('stage_2', stage_2_raw), ('stage_3', stage_3_raw)]:
        path = root / 'data' / 'processed' / f'{stage.title()}_public.csv'
        pd.testing.assert_frame_equal(raw, pd.read_csv(path))
        datasets[stage] = {'path': str(path.relative_to(root)), 'sha256': shared._sha(path)}
    index, target = stage_3_raw.index, stage_3_raw['dropout']
    if not index.equals(pd.RangeIndex(len(stage_3_raw))):
        raise ValueError('Original CSV row positions must be retained.')
    train, test = train_test_split(index, test_size=0.2, random_state=42, stratify=target)
    train, val = train_test_split(train, test_size=0.1, random_state=42, stratify=target.loc[train])
    features, dtypes = list(X_train.columns), X_train.dtypes.astype(str).tolist()
    if not X_train.columns.is_unique or not all(isinstance(c, str) for c in features):
        raise ValueError('Feature names must be unique strings.')
    splits, prepared = {}, {}
    for name, X, y, expected in [('train', X_train, y_train, train),
                               ('validation', X_val, y_val, val), ('test', X_test, y_test, test)]:
        if not X.index.equals(expected) or not y.index.equals(expected):
            raise ValueError(f'{name}: original stratified split identity differs.')
        pd.testing.assert_series_equal(y, target.loc[expected])
        if list(X.columns) != features or X.dtypes.astype(str).tolist() != dtypes:
            raise ValueError(f'{name}: feature names/order/dtypes differ.')
        if np.isinf(X.to_numpy(dtype=float)).any() or not y.isin([0, 1]).all():
            raise ValueError(f'{name}: infinite features or nonbinary labels.')
        splits[name] = {
            'row_positions': list(map(int, expected)),
            'learner_codes': stage_3_raw.loc[expected, 'LearnerCode'].astype(str).tolist(),
        }
        prepared[name] = _prepared(X, y)
    historical_cells = shared._read_json(root / 'notebooks' / '03_stage_3_data_and_modelling.ipynb')['cells']
    historical = [
        {'cell_id': historical_cells[i]['id'], 'source': ''.join(historical_cells[i]['source']),
         'source_sha256': hashlib.sha256(''.join(historical_cells[i]['source']).encode()).hexdigest()}
        for i in (25, 29, 31, 54)
    ]
    return {
        'schema_version': 1, 'stage': 'stage_3', 'run_purpose': _PURPOSE,
        'datasets': datasets, 'ordered_features': features, 'feature_dtypes': dtypes,
        'prepared': prepared, 'splits': splits,
        'split_settings': dict(seed=42, stratify=True, test_size=0.2, validation_fraction_of_remaining=0.1),
        'evidence': {'baseline': baseline_evidence, 'final': final_evidence},
        'training_settings': {r: _settings(r, n_jobs) for r in ('baseline', 'final')},
        'python_version': platform.python_version(),
        'library_versions': {name: version(name) for name in ('numpy', 'pandas', 'scikit-learn', 'xgboost', 'joblib')},
        'historical_reporting': historical,
    }


def _reuse_context(context):
    return {k: v for k, v in shared._reuse_context(context).items() if k != 'historical_reporting'}


def _schema(context):
    return {k: context[k] for k in ('ordered_features', 'feature_dtypes')}


def _verify_model(model, role, context, record):
    verification_context = context
    if role == 'final':
        callbacks = model.get_params().get('callbacks') or []
        early = context['training_settings'][role]['early_stopping']
        if len(callbacks) != 1:
            raise ValueError('Final Stage 3 model requires one early-stopping callback.')
        callback = callbacks[0]
        # XGBoost stores its constructor's min_delta as _min_delta.
        actual_delta = getattr(callback, 'min_delta', getattr(callback, '_min_delta', None))
        if actual_delta != early['min_delta']:
            raise ValueError('Final Stage 3 early-stopping min_delta differs.')
        verification_context = {
            **context,
            'training_settings': {
                **context['training_settings'],
                role: {**context['training_settings'][role],
                       'early_stopping': {k: v for k, v in early.items() if k != 'min_delta'}},
            },
        }
    shared._verify_model(model, role, verification_context, record)


def _load_run(run, context, manifest_hash):
    if shared._sha(run / 'manifest.json') != manifest_hash:
        raise ValueError('Stage 3 rebuilt manifest checksum differs.')
    manifest = shared._read_json(run / 'manifest.json')
    if (manifest['run_id'] != run.name or manifest['artifact_kind'] != 'rebuilt'
            or manifest['stage'] != 'stage_3' or set(manifest['sha256']) != _FILES):
        raise ValueError('Invalid Stage 3 rebuilt manifest.')
    for name, digest in manifest['sha256'].items():
        if shared._sha(run / name) != digest:
            raise ValueError(f'Stage 3 rebuilt artifact checksum differs: {name}')
    if shared._read_json(run / 'splits.json') != context['splits']:
        raise ValueError('Stage 3 saved split identity differs.')
    if shared._read_json(run / 'features.json') != _schema(context):
        raise ValueError('Stage 3 saved feature schema differs.')
    models, records = {}, {}
    for role in ('baseline', 'final'):
        record = shared._read_json(run / f'{role}.metadata.json')
        if (record['artifact_kind'] != 'rebuilt' or record['run_id'] != run.name
                or record['role'] != role or record['stage'] != 'stage_3'
                or record['run_purpose'] != _PURPOSE or record['model_path'] != f'{role}.joblib'
                or _reuse_context(record['context']) != _reuse_context(context)
                or record['model_sha256'] != manifest['sha256'][f'{role}.joblib']
                or record['splits_sha256'] != manifest['sha256']['splits.json']
                or record['features_sha256'] != manifest['sha256']['features.json']):
            raise ValueError(f'{role}: Stage 3 rebuilt metadata/context differs.')
        shared._warn_version_differences(record['context'], context, role)
        model = joblib.load(run / f'{role}.joblib')
        # Verify against recorded training settings, including the original n_jobs.
        _verify_model(model, role, record['context'], record)
        models[role], records[role] = model, record
    return models['baseline'], models['final'], records


def load_rebuilt_xgb_pair(root, *, context):
    """Missing/incomplete/incompatible runs raise; no fitting or substitution."""
    root = Path(root)
    try:
        pointer = shared._read_json(root / 'current.json')
        identifier = pointer['run_id']
        if (not isinstance(identifier, str) or not identifier or Path(identifier).name != identifier
                or identifier in ('.', '..')):
            raise ValueError('Invalid Stage 3 rebuilt run identifier.')
        return _load_run(root / identifier, context, pointer['manifest_sha256'])
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f'Required rebuilt Stage 3 XGBoost pair is missing or incomplete at {root}. '
            'Set RUN_XGB_REBUILD=True explicitly to create it; no automatic fitting is performed.'
        ) from exc


def _evaluate(model, X, y):
    metrics = shared._metrics(model, X, y)
    return metrics, model.predict(X), model.predict_proba(X)


def rebuild_xgb_pair(root, *, context, X_train, X_val, X_test, y_train, y_val, y_test):
    """Fit the fixed pair once; publish only after checksum/model/prediction readback."""
    if context['stage'] != 'stage_3' or context['run_purpose'] != _PURPOSE:
        raise ValueError('Expected a new fixed-configuration Stage 3 training context.')
    for role in ('baseline', 'final'):
        settings = context['training_settings'][role]
        if settings != _settings(role, settings['n_jobs']):
            raise ValueError(f'{role}: fixed Stage 3 training settings differ.')
    for name, X, y in [('train', X_train, y_train), ('validation', X_val, y_val), ('test', X_test, y_test)]:
        if (_schema(context) != {'ordered_features': list(X.columns), 'feature_dtypes': X.dtypes.astype(str).tolist()}
                or _prepared(X, y) != context['prepared'][name]
                or X.index.tolist() != context['splits'][name]['row_positions']
                or not y.index.equals(X.index)):
            raise ValueError(f'{name}: prepared inputs differ from the verified context.')
    root = Path(root)
    run = root / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid4().hex)
    run.mkdir(parents=True, exist_ok=False)
    shared._write_json(run / 'features.json', _schema(context))
    shared._write_json(run / 'splits.json', context['splits'])
    expected_outputs = {}
    for role in ('baseline', 'final'):
        settings = context['training_settings'][role]
        params = dict(context['evidence'][role]['parameters'])
        params.update({k: settings[k] for k in ('n_estimators', 'objective', 'eval_metric', 'random_state', 'n_jobs')})
        if role == 'final':
            params.update(tree_method='hist', callbacks=[xgb.callback.EarlyStopping(**settings['early_stopping'])])
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        metrics = {}
        for name, X, y in [('validation', X_val, y_val), ('test', X_test, y_test)]:
            measured, predictions, probabilities = _evaluate(model, X, y)
            metrics[name] = measured
            expected_outputs[role, name] = predictions, probabilities
        path = run / f'{role}.joblib'
        joblib.dump(model, path)
        record = {
            'artifact_kind': 'rebuilt', 'stage': 'stage_3', 'run_purpose': _PURPOSE,
            'run_id': run.name, 'role': role, 'created_at': datetime.now(timezone.utc).isoformat(),
            'model_path': path.name, 'model_sha256': shared._sha(path),
            'features_sha256': shared._sha(run / 'features.json'),
            'splits_sha256': shared._sha(run / 'splits.json'),
            'context': {k: v for k, v in context.items() if k != 'splits'},
            'boosting_rounds': model.get_booster().num_boosted_rounds(),
            'best_iteration': model.best_iteration if role == 'final' else None,
            'best_score': float(model.best_score) if role == 'final' else None,
            'validation_history': model.evals_result(), 'new_metrics': metrics,
            'historical_evidence': {
                'interpretation': 'Other historical runs; not new evaluation metrics or trial reproduction.',
                'parameter_evidence': context['evidence'][role],
                'markdown_sources': context['historical_reporting'],
            },
        }
        _verify_model(model, role, context, record)
        shared._write_json(run / f'{role}.metadata.json', record)
    shared._write_json(run / 'manifest.json', {
        'run_id': run.name, 'artifact_kind': 'rebuilt', 'stage': 'stage_3',
        'sha256': {name: shared._sha(run / name) for name in sorted(_FILES)},
    })
    digest = shared._sha(run / 'manifest.json')
    baseline, final, records = _load_run(run, context, digest)
    for role, model in [('baseline', baseline), ('final', final)]:
        for name, X, y in [('validation', X_val, y_val), ('test', X_test, y_test)]:
            metrics, predictions, probabilities = _evaluate(model, X, y)
            expected_predictions, expected_probabilities = expected_outputs[role, name]
            if (metrics != records[role]['new_metrics'][name]
                    or not np.array_equal(predictions, expected_predictions)
                    or not np.array_equal(probabilities, expected_probabilities)):
                raise ValueError(f'{role}/{name}: Stage 3 saved-model readback differs.')
    temporary = root / f'.current-{uuid4().hex}.json'
    shared._write_json(temporary, {'run_id': run.name, 'manifest_sha256': digest})
    temporary.replace(root / 'current.json')
    return baseline, final, records
