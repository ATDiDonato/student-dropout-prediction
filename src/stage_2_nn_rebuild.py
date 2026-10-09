"""Stage 2 fixed NN pair and separately identified optional analyses.

Reuse never constructs/fits a model or fits a scaler. Publishing requires readback.
"""
from __future__ import annotations

import hashlib
import json
import platform
import warnings
import zipfile
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.stage_2_xgb_rebuild import verify_stage_alignment


PROJECT = 'nn_tabular_auc_refined_stage_2'
PARAMETERS = ['units', 'layers', 'activation', 'optimizer', 'lr', 'dropout', 'l2_strength']
TRIAL_COLUMNS = ['trial_id', 'status', 'val_auc', *PARAMETERS]
BASELINE = dict(units=240, layers=3, activation='relu', optimizer='adam',
                lr=0.0003, dropout=0.3, l2_strength=0.0001)
FINAL = dict(units=160, layers=3, activation='relu', optimizer='adam',
             lr=0.001, dropout=0.3, l2_strength=0.0)
BASELINE_CORRECTION = (
    'Stage 2 formerly read Stage 1 refined best_params.json (48 units, 1 layer). '
    'This rebuilt baseline uses Stage 1 selected final stability configuration 1 '
    '(240 units, 3 ReLU layers), checked against initial_best_params.json.'
)
TRAINING = {
    'seed': 42, 'epochs': 50, 'batch_size': 64, 'shuffle': True,
    'loss': 'binary_crossentropy', 'output': 'sigmoid', 'threshold': 0.5,
    'class_weight': None, 'sample_weight': None,
    'early_stopping': dict(monitor='val_auc', mode='max', patience=5,
                           min_delta=0.0001, restore_best_weights=True),
}


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def _json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _write(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _frame_hash(frame):
    return hashlib.sha256(pd.util.hash_pandas_object(frame, index=True).values.tobytes()).hexdigest()


def _parameters(values, expected=None):
    if not isinstance(values, dict) or not set(PARAMETERS).issubset(values):
        raise ValueError('Missing NN configuration settings.')
    result = {k: values[k] for k in PARAMETERS}
    for key in ('units', 'layers'):
        if isinstance(result[key], (bool, np.bool_)) or int(result[key]) != result[key]:
            raise ValueError(f'Invalid integer setting: {key}')
        result[key] = int(result[key])
    for key in ('lr', 'dropout', 'l2_strength'):
        if isinstance(result[key], (bool, np.bool_)) or not np.isfinite(result[key]):
            raise ValueError(f'Invalid numeric setting: {key}')
        result[key] = float(result[key])
    if expected is not None:
        for key in PARAMETERS:
            equal = (np.isclose(result[key], expected[key], rtol=1e-7, atol=1e-12)
                     if key in ('lr', 'dropout', 'l2_strength') else result[key] == expected[key])
            if not equal:
                raise ValueError(f'NN configuration differs at {key}.')
        return dict(expected)
    return result


def _markdown_rows(text):
    """Ignore table spacing and inline emphasis/code formatting."""
    return [
        [' '.join(value.strip().strip('*_`').split())
         for value in line.strip().strip('|').split('|')]
        for line in text.splitlines() if line.strip().startswith('|')
    ]


def _historical_number(value):
    """Unavailable reporting values must not become configuration failures."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def resolve_nn_rebuild_evidence(*, project_root, previous_tuning_dir, stage_tuning_dir):
    root = Path(project_root)
    initial_path = Path(previous_tuning_dir) / 'initial_best_params.json'
    stability_path = Path(previous_tuning_dir) / 'stability_data.h5'
    final_path = Path(stage_tuning_dir) / 'best_params.json'
    for path in (initial_path, stability_path, final_path):
        if not path.is_file():
            raise FileNotFoundError(f'Required NN configuration evidence is missing: {path}')
    initial, final = _json(initial_path), _json(final_path)
    if initial.get('project_name') != 'nn_tabular_auc':
        raise ValueError('Stage 1 baseline evidence belongs to the wrong initial search.')
    _parameters(initial['best_params'], BASELINE)
    with pd.HDFStore(stability_path, mode='r') as store:
        rows = store['stability']
    selected = rows.loc[rows['config_idx'] == 1]
    if len(selected) != 1:
        raise ValueError('Stage 1 stability evidence must contain exactly one configuration 1.')
    _parameters(selected.iloc[0].to_dict(), BASELINE)
    _parameters(final['best_params'], FINAL)
    notebook_path = root / 'notebooks' / '02_stage_2_data_and_modelling.ipynb'
    cells = _json(notebook_path)['cells']
    selected_cells = [c for c in cells if c.get('id') == 'b6b68dde']
    if len(selected_cells) != 1:
        raise ValueError('Documented Stage 2 NN configuration cell is missing or ambiguous.')
    text = ''.join(selected_cells[0]['source'])
    rows = _markdown_rows(text)
    documented = [row for row in rows if row[0].casefold() == 'final stage 2 nn']
    if len(documented) != 1:
        raise ValueError('Documented final Stage 2 NN configuration is missing or ambiguous.')
    values = documented[0][1:]
    if len(values) != 7:
        raise ValueError('Unexpected documented NN configuration format.')
    _parameters(dict(zip(PARAMETERS, [int(values[0]), int(values[1]), values[2], values[3],
                                     float(values[4]), float(values[5]), float(values[6])])), FINAL)
    stability_rows = [row for row in rows if row[0].casefold() == 'config_idx 3']
    if len(stability_rows) != 1:
        raise ValueError('Historical Stage 2 configuration 3 stability summary is ambiguous.')
    fields = stability_rows[0]
    if len(fields) < 8:
        raise ValueError('Documented configuration 3 settings are incomplete.')
    _parameters(dict(zip(PARAMETERS, [int(fields[1]), int(fields[2]), fields[3], fields[4],
                                     float(fields[5]), float(fields[6]), float(fields[7])])), FINAL)
    metrics_cells = [c for c in cells if c.get('id') == 'e808b9de']
    metric_text = '\n'.join(''.join(c['source']) for c in metrics_cells)
    metric_rows = _markdown_rows(metric_text) if len(metrics_cells) == 1 else []
    historical_tests = {'baseline': {}, 'final': {}}
    raw_test_rows = {}
    for metric in ('Accuracy', 'Precision', 'Recall', 'Specificity', 'AUC'):
        matches = [row for row in metric_rows if row[0].casefold() == metric.casefold()]
        row = matches[0] if len(matches) == 1 else []
        raw_test_rows[metric.lower()] = matches
        for position, role in enumerate(('baseline', 'final'), 1):
            historical_tests[role][metric.lower()] = _historical_number(
                row[position] if len(row) > position else None
            )
    historical_source = {'path': 'notebooks/02_stage_2_data_and_modelling.ipynb', 'cell_id': 'e808b9de',
                         'source_sha256': hashlib.sha256(metric_text.encode()).hexdigest()}
    def source(path):
        path = Path(path).resolve()
        name = path.relative_to(root.resolve()).as_posix() if path.is_relative_to(root.resolve()) else str(path)
        return {'path': name, 'sha256': _sha(path)}
    return {
        'baseline': {'parameters': dict(BASELINE), 'correction': BASELINE_CORRECTION,
                     'configuration_index': 1, 'sources': [source(initial_path), source(stability_path)],
                     'historical_scores': {'test_metrics': historical_tests['baseline'], 'source': historical_source,
                         'raw_test_rows': raw_test_rows,
                         'configuration_correspondence': 'Historical baseline identity is uncertain; baseline has been corrected.'}},
        'final': {'parameters': dict(FINAL), 'configuration_index': 3,
                  'sources': [source(final_path), {'path': 'notebooks/02_stage_2_data_and_modelling.ipynb',
                              'cell_id': 'b6b68dde', 'source_sha256': hashlib.sha256(text.encode()).hexdigest()}],
                  'evidence_kind': 'documented_configuration_and_parameter_summary',
                  'historical_scores': {'stability_mean_val_auc': _historical_number(fields[8] if len(fields) > 8 else None),
                      'stability_std_val_auc': _historical_number(fields[9] if len(fields) > 9 else None),
                      'raw_stability_row': fields, 'raw_test_rows': raw_test_rows,
                      'stability_source_cell_id': 'b6b68dde',
                      'test_metrics': historical_tests['final'], 'test_source': historical_source}},
    }


def make_nn_rebuild_context(*, project_root, stage_1_raw, stage_2_raw,
                            X_train, X_val, X_test, y_train, y_val, y_test, evidence, seed):
    if seed != 42:
        raise ValueError('Stage 2 NN requires seed 42.')
    verify_stage_alignment(stage_1_raw, stage_2_raw)
    datasets = {}
    for name, raw in [('Stage_1', stage_1_raw), ('Stage_2', stage_2_raw)]:
        path = Path(project_root) / 'data' / 'processed' / f'{name}_public.csv'
        pd.testing.assert_frame_equal(raw, pd.read_csv(path))
        datasets[name] = {'path': str(path.relative_to(project_root)), 'sha256': _sha(path)}
    index, target = stage_1_raw.index, stage_1_raw['dropout']
    if not index.equals(pd.RangeIndex(len(stage_1_raw))):
        raise ValueError('Original CSV row positions must be retained.')
    train, test = train_test_split(index, test_size=0.2, random_state=42, stratify=target)
    train, val = train_test_split(train, test_size=0.1, random_state=42, stratify=target.loc[train])
    columns, dtypes = list(X_train.columns), X_train.dtypes.astype(str).tolist()
    if not X_train.columns.is_unique or not all(isinstance(c, str) for c in columns):
        raise ValueError('NN feature names must be unique strings.')
    splits, prepared = {}, {}
    for name, X, y, expected in [('train', X_train, y_train, train),
                               ('validation', X_val, y_val, val), ('test', X_test, y_test, test)]:
        if not X.index.equals(expected) or not y.index.equals(expected):
            raise ValueError(f'{name}: original split identity differs.')
        pd.testing.assert_series_equal(y, target.loc[expected])
        if list(X.columns) != columns or X.dtypes.astype(str).tolist() != dtypes:
            raise ValueError(f'{name}: feature names/order/dtypes differ.')
        if not np.isfinite(X.to_numpy(dtype=float)).all() or not y.isin([0, 1]).all():
            raise ValueError(f'{name}: nonfinite inputs or nonbinary labels.')
        splits[name] = {'row_positions': list(map(int, expected)),
                        'learner_codes': stage_1_raw.loc[expected, 'LearnerCode'].astype(str).tolist()}
        prepared[name] = {'shape': list(X.shape), 'features_sha256': _frame_hash(X), 'labels_sha256': _frame_hash(y)}
    return {
        'schema_version': 1, 'stage': 'stage_2', 'datasets': datasets,
        'ordered_features': columns, 'feature_dtypes': dtypes, 'prepared': prepared,
        'split_settings': dict(seed=42, stratify=True, test_size=0.2, validation_fraction_of_remaining=0.1),
        'splits': splits, 'evidence': evidence, 'training_settings': TRAINING.copy(),
        'python_version': platform.python_version(),
        'library_versions': {name: version(name) for name in
                             ('tensorflow', 'keras', 'numpy', 'pandas', 'scikit-learn', 'joblib')},
    }


def _substantive(context):
    result = {k: v for k, v in context.items() if k not in ('python_version', 'library_versions')}
    # Keep JSON/HDF file hashes, configuration identities and values strict.
    # Markdown is parsed above; its raw hash and historical metrics are reporting.
    result['evidence'] = {}
    for role, evidence in context['evidence'].items():
        compatible = {k: v for k, v in evidence.items() if k not in ('historical_scores', 'correction')}
        compatible['sources'] = [
            {k: v for k, v in source.items()
             if not (k == 'source_sha256' and 'cell_id' in source)}
            for source in evidence['sources']
        ]
        result['evidence'][role] = compatible
    return result


def _warn_versions(saved, current):
    a = {'Python': saved['python_version'], **saved['library_versions']}
    b = {'Python': current['python_version'], **current['library_versions']}
    for name in sorted(a.keys() | b.keys()):
        if a.get(name) != b.get(name):
            warnings.warn(f'NN compatibility warning: {name} training={a.get(name)}, current={b.get(name)}. '
                          'Loading/schema verification remains required.', RuntimeWarning, stacklevel=3)


def _scaler_state(scaler):
    if not isinstance(scaler, StandardScaler) or not scaler.with_mean or not scaler.with_std:
        raise ValueError('Expected fitted original StandardScaler with centering and scaling.')
    return {'feature_names': scaler.feature_names_in_.tolist(), 'n_features': int(scaler.n_features_in_),
            'mean': scaler.mean_.tolist(), 'variance': scaler.var_.tolist(), 'scale': scaler.scale_.tolist(),
            'samples_seen': np.asarray(scaler.n_samples_seen_).tolist()}


def _scaled(scaler, matrices, context):
    if _scaler_state(scaler)['feature_names'] != context['ordered_features']:
        raise ValueError('Scaler feature names/order differ.')
    result = []
    for name, X in zip(('train', 'validation', 'test'), matrices):
        if (list(X.columns) != context['ordered_features'] or
                _frame_hash(X) != context['prepared'][name]['features_sha256']):
            raise ValueError(f'{name}: unscaled features differ from recorded context.')
        transformed = scaler.transform(X)
        if not isinstance(transformed, pd.DataFrame):
            raise ValueError('Scaler must preserve pandas feature names and index.')
        if not transformed.index.equals(X.index) or list(transformed.columns) != list(X.columns):
            raise ValueError(f'{name}: scaled feature schema differs.')
        if not np.isfinite(transformed.to_numpy()).all():
            raise ValueError(f'{name}: scaled inputs are nonfinite.')
        result.append(transformed)
    return tuple(result)


def transform_with_saved_scaler(scaler, *, X_train, X_val, X_test, records):
    record = records['baseline']
    if _scaler_state(scaler) != record['scaler_state']:
        raise ValueError('Saved scaler statistics differ from metadata.')
    values = _scaled(scaler, (X_train, X_val, X_test), record['context'])
    hashes = dict(zip(('train', 'validation', 'test'), map(_frame_hash, values)))
    if any(r['scaled_sha256'] != hashes or r['scaler_state'] != record['scaler_state'] for r in records.values()):
        raise ValueError('Scaled matrix hashes or scaler evidence differ.')
    return values


def _verify_model(model, config, width):
    from tensorflow import keras
    if tuple(model.input_shape) != (None, width) or tuple(model.output_shape) != (None, 1):
        raise ValueError('NN input/output shape differs.')
    expected = [kind for _ in range(config['layers']) for kind in (keras.layers.Dense, keras.layers.Dropout)]
    expected.append(keras.layers.Dense)
    if len(model.layers) != len(expected) or any(not isinstance(layer, kind) for layer, kind in zip(model.layers, expected)):
        raise ValueError('NN hidden-layer/dropout structure differs.')
    for layer in model.layers[:-1]:
        if isinstance(layer, keras.layers.Dense):
            if layer.units != config['units'] or layer.activation.__name__ != config['activation']:
                raise ValueError('NN hidden units/activation differ.')
            reg = layer.kernel_regularizer
            if config['l2_strength']:
                if not isinstance(reg, keras.regularizers.L2) or not np.isclose(float(reg.l2), config['l2_strength']):
                    raise ValueError('NN hidden-layer L2 differs.')
            elif reg is not None:
                raise ValueError('Unexpected NN hidden-layer regularization.')
        elif not np.isclose(layer.rate, config['dropout']):
            raise ValueError('NN dropout differs.')
    output = model.layers[-1]
    if output.units != 1 or output.activation.__name__ != 'sigmoid' or output.kernel_regularizer is not None:
        raise ValueError('NN output differs from one unregularized sigmoid unit.')


def _archive_compile(path, config):
    with zipfile.ZipFile(path) as archive:
        compile_config = json.loads(archive.read('config.json'))['compile_config']
    optimizer = compile_config['optimizer']
    if (compile_config['loss'] != 'binary_crossentropy' or optimizer['class_name'].lower() != 'adam'
            or not np.isclose(optimizer['config']['learning_rate'], config['lr'], rtol=1e-6, atol=1e-10)):
        raise ValueError('Saved NN loss/optimizer/learning-rate metadata differs.')
    return compile_config


def _metrics(model, X, y):
    probabilities = np.asarray(model.predict(X, verbose=0))
    if probabilities.shape != (len(y), 1) or not np.isfinite(probabilities).all():
        raise ValueError('NN probability dimensions or finiteness differ.')
    probabilities = probabilities.ravel()
    if not ((probabilities >= 0) & (probabilities <= 1)).all():
        raise ValueError('NN probabilities are outside [0, 1].')
    predictions = (probabilities >= 0.5).astype(int)
    return {'accuracy': float(accuracy_score(y, predictions)),
            'precision': float(precision_score(y, predictions, zero_division=0)),
            'recall': float(recall_score(y, predictions, zero_division=0)),
            'specificity': float(recall_score(y, predictions, pos_label=0)),
            'auc': float(roc_auc_score(y, probabilities))}, probabilities


def new_nn_analysis_run(root):
    """Create an isolated directory only when an explicit writing workflow calls it."""
    run = Path(root) / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid4().hex)
    run.mkdir(parents=True, exist_ok=False)
    return run


def _publish(run, names, extra):
    manifest = {'run_id': run.name, 'sha256': {name: _sha(run / name) for name in names}, **extra}
    _write(run / 'manifest.json', manifest)
    return _sha(run / 'manifest.json')


def _pointer(root, run_id, digest):
    temporary = Path(root) / f'.current-{uuid4().hex}.json'
    _write(temporary, {'run_id': run_id, 'manifest_sha256': digest})
    temporary.replace(Path(root) / 'current.json')


def _verify_files(run, manifest, names):
    if manifest['run_id'] != run.name or set(manifest['sha256']) != set(names):
        raise ValueError('NN manifest identity/file set differs.')
    for name in names:
        if _sha(run / name) != manifest['sha256'][name]:
            raise ValueError(f'NN artifact checksum differs: {name}')


def _verified_run(root, names, *, required):
    root = Path(root)
    if not (root / 'current.json').is_file():
        if not required:
            return None
        raise FileNotFoundError(f'Required rebuilt NN pair is missing at {root}. '
                                'Set RUN_NN_REBUILD=True explicitly to create it; no automatic fitting is performed.')
    pointer = _json(root / 'current.json')
    identifier = pointer['run_id']
    if not isinstance(identifier, str) or not identifier or Path(identifier).name != identifier or identifier in ('.', '..'):
        raise ValueError('Invalid NN run identifier.')
    run = root / identifier
    try:
        if _sha(run / 'manifest.json') != pointer['manifest_sha256']:
            raise ValueError('NN manifest checksum differs.')
        manifest = _json(run / 'manifest.json')
        _verify_files(run, manifest, names)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f'NN saved run is incomplete at {run}; an explicit rebuild is required. {exc}') from exc
    return run, manifest


_PAIR_FILES = ['baseline.keras', 'final_config_3.keras', 'scaler.joblib', 'features.json', 'splits.json',
               'baseline.history.json', 'final.history.json', 'baseline.metadata.json', 'final.metadata.json']


def load_rebuilt_nn_pair(root, *, context):
    run, manifest = _verified_run(root, _PAIR_FILES, required=True)
    return _load_pair_run(run, manifest, context)


def _load_pair_run(run, manifest, context):
    from tensorflow import keras
    _verify_files(run, manifest, _PAIR_FILES)
    if manifest['artifact_kind'] != 'rebuilt':
        raise ValueError('Expected an explicitly rebuilt NN pair.')
    _parameters(context['evidence']['baseline']['parameters'], BASELINE)
    _parameters(context['evidence']['final']['parameters'], FINAL)
    features = _json(run / 'features.json')
    if (features['ordered_features'] != context['ordered_features'] or
            features['feature_dtypes'] != context['feature_dtypes'] or _json(run / 'splits.json') != context['splits']):
        raise ValueError('NN saved feature schema or split identity differs.')
    models, records = {}, {}
    for role, filename in [('baseline', 'baseline.keras'), ('final', 'final_config_3.keras')]:
        record = _json(run / f'{role}.metadata.json')
        if (record['artifact_kind'] != 'rebuilt' or record['run_id'] != run.name or record['role'] != role
                or _substantive(record['context']) != _substantive(context)
                or record['model_sha256'] != manifest['sha256'][filename]
                or record['scaler_sha256'] != manifest['sha256']['scaler.joblib']):
            raise ValueError(f'{role}: rebuilt NN metadata/context differs.')
        history = _json(run / f'{role}.history.json')
        epoch = record['restored_epoch_zero_based']
        if (type(epoch) is not int or not 0 <= epoch < record['epochs_run'] <= 50
                or len(history['val_auc']) != record['epochs_run']
                or record['restored_epoch_one_based'] != epoch + 1
                or record['selected_val_auc'] != history['val_auc'][epoch]
                or record['selection_method'] != 'EarlyStopping.best_epoch after on_train_end'):
            raise ValueError(f'{role}: restored-epoch/history metadata differs.')
        _warn_versions(record['context'], context)
        compile_config = _archive_compile(run / filename, record['context']['evidence'][role]['parameters'])
        if compile_config != record['saved_compile_config']:
            raise ValueError(f'{role}: saved compile configuration differs.')
        model = keras.models.load_model(run / filename, compile=False)
        _verify_model(model, record['context']['evidence'][role]['parameters'], len(context['ordered_features']))
        models[role], records[role] = model, record
    scaler = joblib.load(run / 'scaler.joblib')
    state = _scaler_state(scaler)
    if features['scaler_state'] != state or any(r['scaler_state'] != state for r in records.values()):
        raise ValueError('Saved NN scaler state differs.')
    return models['baseline'], models['final'], scaler, records


def rebuild_nn_pair(root, *, context, scaler, X_train, X_val, X_test, y_train, y_val, y_test):
    from tensorflow import keras
    from src.notebook_helpers import build_binary_classifier, set_seed
    for name, y in [('train', y_train), ('validation', y_val), ('test', y_test)]:
        if _frame_hash(y) != context['prepared'][name]['labels_sha256']:
            raise ValueError(f'{name}: labels differ from recorded context.')
    _parameters(context['evidence']['baseline']['parameters'], BASELINE)
    _parameters(context['evidence']['final']['parameters'], FINAL)
    if context['training_settings'] != TRAINING:
        raise ValueError('Fixed NN training settings differ.')
    state = _scaler_state(scaler)
    values = X_train.to_numpy(dtype=float)
    if (state['samples_seen'] != len(X_train) or not np.allclose(scaler.mean_, values.mean(axis=0), rtol=1e-10, atol=1e-12)
            or not np.allclose(scaler.var_, values.var(axis=0), rtol=1e-10, atol=1e-12)):
        raise ValueError('Scaler does not match a fit on the training partition only.')
    scaled = _scaled(scaler, (X_train, X_val, X_test), context)
    run = new_nn_analysis_run(root)
    joblib.dump(scaler, run / 'scaler.joblib')
    _write(run / 'features.json', {'ordered_features': context['ordered_features'],
                                  'feature_dtypes': context['feature_dtypes'], 'scaler_state': state})
    _write(run / 'splits.json', context['splits'])
    expected_probabilities = {}
    for role, filename in [('baseline', 'baseline.keras'), ('final', 'final_config_3.keras')]:
        keras.backend.clear_session()
        set_seed(42)
        config = context['evidence'][role]['parameters']
        model = build_binary_classifier(input_dim=X_train.shape[1], **config)
        early_stop = keras.callbacks.EarlyStopping(**TRAINING['early_stopping'])
        history = model.fit(scaled[0], y_train, validation_data=(scaled[1], y_val),
                            epochs=50, batch_size=64, shuffle=True, callbacks=[early_stop], verbose=1)
        # The callback restored its selected weights in on_train_end, including min_delta.
        if early_stop.best_weights is None or not hasattr(early_stop, 'best_epoch'):
            raise RuntimeError('EarlyStopping did not expose the epoch/weights actually restored.')
        if (len(model.get_weights()) != len(early_stop.best_weights) or
                any(not np.array_equal(a, b) for a, b in zip(model.get_weights(), early_stop.best_weights))):
            raise RuntimeError('Final model weights differ from EarlyStopping restored weights.')
        restored_epoch = int(early_stop.best_epoch)
        metrics = {}
        for name, X, y in [('validation', scaled[1], y_val), ('test', scaled[2], y_test)]:
            metrics[name], expected_probabilities[role, name] = _metrics(model, X, y)
        model.save(run / filename)
        _write(run / f'{role}.history.json', {k: list(map(float, v)) for k, v in history.history.items()})
        record = {'artifact_kind': 'rebuilt', 'run_id': run.name, 'role': role,
                  'created_at': datetime.now(timezone.utc).isoformat(), 'context': context,
                  'model_sha256': _sha(run / filename), 'scaler_sha256': _sha(run / 'scaler.joblib'),
                  'scaler_state': state, 'scaled_sha256': dict(zip(('train', 'validation', 'test'), map(_frame_hash, scaled))),
                  'epochs_run': len(history.history['val_auc']), 'restored_epoch_zero_based': restored_epoch,
                  'restored_epoch_one_based': restored_epoch + 1, 'stopped_epoch_zero_based': int(early_stop.stopped_epoch),
                  'selection_method': 'EarlyStopping.best_epoch after on_train_end',
                  'selected_val_auc': float(history.history['val_auc'][restored_epoch]),
                  'saved_compile_config': _archive_compile(run / filename, config), 'new_metrics': metrics,
                  'historical_evidence': {
                      'interpretation': 'Other historical training runs; not new evaluation metrics.',
                      **context['evidence'][role]['historical_scores']}}
        _write(run / f'{role}.metadata.json', record)
    digest = _publish(run, _PAIR_FILES, {'artifact_kind': 'rebuilt'})
    # Read the complete immutable run before publishing its public pointer.
    baseline, final, saved_scaler, records = _load_pair_run(run, _json(run / 'manifest.json'), context)
    readback = transform_with_saved_scaler(saved_scaler, X_train=X_train, X_val=X_val, X_test=X_test, records=records)
    for role, model in [('baseline', baseline), ('final', final)]:
        for name, X, y in [('validation', readback[1], y_val), ('test', readback[2], y_test)]:
            measured, probabilities = _metrics(model, X, y)
            if not np.array_equal(probabilities, expected_probabilities[role, name]) or measured != records[role]['new_metrics'][name]:
                raise ValueError(f'{role}/{name}: saved NN readback predictions or metrics differ.')
    _pointer(root, run.name, digest)
    return baseline, final, saved_scaler, records


def _genuine_trials(frame):
    if not set(TRIAL_COLUMNS).issubset(frame.columns) or frame.empty:
        raise ValueError('Genuine completed Stage 2 search trials are required, not a parameter summary.')
    completed = frame.loc[frame['status'].eq('COMPLETED')].copy()
    ids = completed['trial_id'].astype(str)
    if (completed.empty or completed['trial_id'].isna().any() or ids.eq('saved_best_params').any()
            or ids.str.strip().eq('').any() or not ids.is_unique):
        raise ValueError('Invalid completed-trial identities.')
    scores = pd.to_numeric(completed['val_auc'], errors='raise')
    if not np.isfinite(scores).all() or not scores.between(0, 1).all():
        raise ValueError('Invalid completed-trial AUC values.')
    for _, row in completed.iterrows():
        p = _parameters(row.to_dict())
        if (p['units'] not in range(32, 193, 16) or p['layers'] not in range(1, 5)
                or p['activation'] not in ('relu', 'tanh') or p['optimizer'] != 'adam'
                or not any(np.isclose(p['lr'], v) for v in (0.0003, 0.001))
                or not any(np.isclose(p['dropout'], v) for v in (0, 0.1, 0.2, 0.3, 0.4))
                or p['l2_strength'] not in (0, 1e-6, 1e-5, 1e-4)):
            raise ValueError('Trial configuration is outside the intended Stage 2 refined search.')
    completed['val_auc'] = scores
    return completed.sort_values('val_auc', ascending=False)


def publish_nn_analysis(run, *, kind, project_name, context, trials, stability=None):
    """Publish only real trials from the explicitly invoked Stage 2 tuner."""
    if project_name != PROJECT:
        raise ValueError('Optional NN analysis belongs to a different search project.')
    if kind == 'stability' and trials.attrs.get('provenance') != {
        'stage': 'stage_2', 'project_name': PROJECT, 'context_sha256': _digest(_substantive(context)),
    }:
        raise ValueError('Stability fitting requires verified Stage 2 search-history provenance.')
    run = Path(run)
    genuine = _genuine_trials(trials)
    genuine.to_csv(run / 'trials.csv', index=False)
    names = ['trials.csv']
    if kind == 'stability':
        _validate_stability(stability, genuine)
        stability.to_hdf(run / 'stability.h5', key='stability', mode='w')
        names.append('stability.h5')
    elif kind != 'search':
        raise ValueError('Unknown NN analysis kind.')
    digest = _publish(run, names, {
        'kind': kind, 'stage': 'stage_2', 'project_name': PROJECT, 'context': context,
        'stability_score_definition': 'maximum validation AUC per seed, separate from restored epoch',
    })
    # Read and validate before publishing the optional-analysis pointer.
    _verify_files(run, _json(run / 'manifest.json'), names)
    _read_analysis(run, _json(run / 'manifest.json'), kind, context)
    _pointer(run.parent, run.name, digest)


def _validate_stability(frame, trials):
    if frame is None or len(frame) != min(5, len(trials)):
        raise ValueError('Stability rows must correspond to the genuine top search trials.')
    for index, (_, trial) in enumerate(trials.head(5).iterrows(), 1):
        row = frame.loc[frame['config_idx'] == index]
        if len(row) != 1:
            raise ValueError('Stability configuration identity differs.')
        row = row.iloc[0]
        _parameters(row.to_dict(), _parameters(trial.to_dict()))
        scores = row['seed_scores']
        epochs, histories = row['restored_epochs_zero_based'], row['histories']
        if len(epochs) != 3 or len(histories) != 3:
            raise ValueError('Stability restored-epoch/history records differ.')
        for epoch, history, score in zip(epochs, histories, scores):
            if (type(epoch) is not int or not 0 <= epoch < len(history['val_auc']) <= 50
                    or not np.isfinite(history['val_auc']).all() or score != max(history['val_auc'])):
                raise ValueError('Stability history/restored-epoch evidence differs.')
        if (list(row['seeds']) != [42, 123, 999] or len(scores) != 3 or not np.isfinite(scores).all()
                or not np.all((np.asarray(scores) >= 0) & (np.asarray(scores) <= 1))
                or not np.isclose(row['mean_val_auc'], np.mean(scores))
                or not np.isclose(row['std_val_auc'], np.std(scores))):
            raise ValueError('Stability seed-score evidence differs.')


def _read_analysis(run, manifest, kind, context):
    if (manifest['kind'] != kind or manifest['stage'] != 'stage_2' or manifest['project_name'] != PROJECT
            or _substantive(manifest['context']) != _substantive(context)):
        raise ValueError('NN analysis stage/search/data/split provenance differs.')
    _warn_versions(manifest['context'], context)
    trials = _genuine_trials(pd.read_csv(run / 'trials.csv', dtype={'trial_id': str}))
    if kind == 'search':
        trials.attrs['provenance'] = {'stage': 'stage_2', 'project_name': PROJECT,
                                     'context_sha256': _digest(_substantive(context))}
        return trials
    with pd.HDFStore(run / 'stability.h5', mode='r') as store:
        stability = store['stability']
    _validate_stability(stability, trials)
    return stability


def load_nn_analysis(root, *, kind, context):
    """Unproven legacy summary CSVs/HDFs are never used as trial provenance."""
    names = ['trials.csv'] + (['stability.h5'] if kind == 'stability' else [])
    result = _verified_run(root, names, required=False)
    if result is None:
        columns = TRIAL_COLUMNS if kind == 'search' else ['config_idx', *PARAMETERS, 'seeds', 'seed_scores', 'mean_val_auc', 'std_val_auc']
        return pd.DataFrame(columns=columns)
    run, manifest = result
    return _read_analysis(run, manifest, kind, context)
