"""Fixed Stage 3 NN pair, read-only reuse, and separately identified explicit search."""
from __future__ import annotations

import copy
import hashlib
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src import stage_2_nn_rebuild as shared
from src import stage_3_xgb_rebuild as data_checks


PROJECT = 'nn_tabular_auc_refined_stage_3'
BASELINE = dict(units=160, layers=3, activation='relu', optimizer='adam',
                lr=0.001, dropout=0.3, l2_strength=0.0)
FINAL = dict(units=160, layers=2, activation='tanh', optimizer='adam',
             lr=0.001, dropout=0.1, l2_strength=0.00001)
TRAINING = copy.deepcopy(shared.TRAINING)
_FILES = ['baseline.keras', 'final.keras', 'scaler.joblib', 'features.json', 'splits.json',
          'baseline.history.json', 'final.history.json', 'baseline.metadata.json', 'final.metadata.json']
_PURPOSE = 'new_fixed_stage_3_nn_pair_not_historical_trial_reproduction'


def split_unscaled(data_encoded, *, seed):
    """Original split operations, with scaling deferred until the NN branch."""
    if seed != 42:
        raise ValueError('Stage 3 NN requires seed 42.')
    X, y = data_encoded.drop(columns=['dropout']), data_encoded['dropout']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=seed, stratify=y_train,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def _summary(directory, expected):
    path = Path(directory) / 'best_params.json'
    if not path.is_file():
        raise FileNotFoundError(f'Required NN parameter evidence is missing: {path}')
    record = shared._json(path)
    if set(record['best_params']) != set(shared.PARAMETERS):
        raise ValueError(f'Expected exactly seven NN parameters: {path}')
    params = shared._parameters(record['best_params'], expected)
    sources = [{'path': str(path.resolve()), 'sha256': shared._sha(path)}]
    csv_path = Path(directory) / 'trials.csv'
    if csv_path.is_file():
        rows = pd.read_csv(csv_path, dtype={'trial_id': str})
        if len(rows) != 1 or rows.iloc[0].get('trial_id') != 'saved_best_params':
            raise ValueError(f'Expected the legacy parameter summary, not search history: {csv_path}')
        shared._parameters(rows.iloc[0].to_dict(), expected)
        sources.append({'path': str(csv_path.resolve()), 'sha256': shared._sha(csv_path)})
    return {'parameters': params, 'sources': sources,
            'historical_scores': {'summary_val_auc': record.get('best_val_auc'),
                                  'interpretation': 'Summary of another historical run; not a genuine trial record.'}}


def resolve_nn_evidence(*, project_root, previous_tuning_dir, stage_tuning_dir, require_final=True):
    root = Path(project_root).resolve()
    baseline = _summary(previous_tuning_dir, BASELINE)
    baseline['configuration_identity'] = 'stage_2_selected_final_configuration_3'
    evidence = {'baseline': baseline}
    if require_final:
        final = _summary(stage_tuning_dir, FINAL)
        notebook = root / 'notebooks' / '03_stage_3_data_and_modelling.ipynb'
        cells = shared._json(notebook)['cells']
        selected = [c for c in cells if c.get('id') == '24573ac8']
        if len(selected) != 1:
            raise ValueError('Documented Stage 3 NN configuration cell is missing or ambiguous.')
        text = ''.join(selected[0]['source'])
        rows = [r for r in shared._markdown_rows(text) if r[0].casefold() == 'trial 053']
        if len(rows) != 1 or len(rows[0]) != 9:
            raise ValueError('Documented trial 053 configuration is missing or ambiguous.')
        values = rows[0][1:8]
        shared._parameters(dict(zip(shared.PARAMETERS, [int(values[0]), int(values[1]), values[2],
                           values[3], float(values[4]), float(values[5]), float(values[6])])), FINAL)
        final['configuration_identity'] = 'documented_stage_3_final_parameters'
        final['sources'].append({'path': str(notebook.relative_to(root)), 'cell_id': '24573ac8',
                                 'source_sha256': hashlib.sha256(text.encode()).hexdigest()})
        final['historical_scores'].update(documented_trial_id='053', trial_identity_verified=False,
                                          documented_val_auc=shared._historical_number(rows[0][8]))
        evidence['final'] = final
    else:
        # Explicit search is independent of fixed-final evidence and rebuilt files.
        evidence['final'] = {'parameters': {}, 'sources': [],
                             'configuration_identity': 'independent_explicit_stage_3_search'}
    for record in evidence.values():
        for source in record['sources']:
            path = Path(source['path'])
            if path.is_absolute():
                source['path'] = str(path.relative_to(root))
    return evidence


def make_nn_context(*, project_root, stage_2_raw, stage_3_raw, X_train, X_val, X_test,
                    y_train, y_val, y_test, evidence, seed):
    # Reuse the verified Stage 3 data/alignment/split checks, without training.
    context = data_checks.make_rebuild_context(
        project_root=project_root, stage_2_raw=stage_2_raw, stage_3_raw=stage_3_raw,
        X_train=X_train, X_val=X_val, X_test=X_test, y_train=y_train, y_val=y_val, y_test=y_test,
        baseline_evidence=evidence['baseline'], final_evidence=evidence['final'], seed=seed, n_jobs=1,
    )
    context['run_purpose'] = _PURPOSE
    context['training_settings'] = copy.deepcopy(TRAINING)
    for prepared in context['prepared'].values():
        prepared['features_sha256'] = prepared.pop('feature_sha256')
        prepared['labels_sha256'] = prepared.pop('label_sha256')
    context['library_versions'] = {name: version(name) for name in
                                  ('tensorflow', 'keras', 'numpy', 'pandas', 'scikit-learn', 'joblib')}
    cells = shared._json(Path(project_root) / 'notebooks' / '03_stage_3_data_and_modelling.ipynb')['cells']
    context['historical_reporting'] = [
        {'cell_id': cells[i]['id'], 'source': ''.join(cells[i]['source']),
         'source_sha256': hashlib.sha256(''.join(cells[i]['source']).encode()).hexdigest()} for i in (41, 46, 51, 54)
    ]
    if not all(np.isfinite(X.to_numpy(dtype=float)).all() for X in (X_train, X_val, X_test)):
        raise ValueError('NN inputs must be finite.')
    return context


def _substantive(context):
    return {k: v for k, v in shared._substantive(context).items() if k != 'historical_reporting'}


def _search_context(context):
    compatible = _substantive(context)
    compatible['evidence'] = {'baseline': compatible['evidence']['baseline']}
    return compatible


def _verify_inputs(context, matrices, labels):
    for name, X, y in zip(('train', 'validation', 'test'), matrices, labels):
        if (list(X.columns) != context['ordered_features']
                or X.dtypes.astype(str).tolist() != context['feature_dtypes']
                or X.index.tolist() != context['splits'][name]['row_positions']
                or not X.index.equals(y.index)
                or shared._frame_hash(X) != context['prepared'][name]['features_sha256']
                or shared._frame_hash(y) != context['prepared'][name]['labels_sha256']):
            raise ValueError(f'{name}: NN inputs differ from the verified context.')


def _fit_scaler(context, matrices, labels):
    _verify_inputs(context, matrices, labels)
    scaler = StandardScaler().set_output(transform='pandas')
    scaler.fit(matrices[0])
    return scaler, shared._scaled(scaler, matrices, context)


def _verify_model(model, config, width):
    if config['dropout'] > 0:
        shared._verify_model(model, config, width)
        return
    # The explicit search also allows dropout=0, for which the builder omits Dropout.
    from tensorflow import keras
    if tuple(model.input_shape) != (None, width) or tuple(model.output_shape) != (None, 1):
        raise ValueError('NN input/output shape differs.')
    if len(model.layers) != config['layers'] + 1 or any(not isinstance(l, keras.layers.Dense) for l in model.layers):
        raise ValueError('Unexpected zero-dropout NN architecture.')
    for layer in model.layers[:-1]:
        if layer.units != config['units'] or layer.activation.__name__ != config['activation']:
            raise ValueError('NN hidden units/activation differ.')
        reg = layer.kernel_regularizer
        if config['l2_strength']:
            if not isinstance(reg, keras.regularizers.L2) or not np.isclose(float(reg.l2), config['l2_strength']):
                raise ValueError('NN hidden L2 differs.')
        elif reg is not None:
            raise ValueError('Unexpected NN hidden L2.')
    output = model.layers[-1]
    if output.units != 1 or output.activation.__name__ != 'sigmoid' or output.kernel_regularizer is not None:
        raise ValueError('NN output differs.')


def _save_preparation(run, context, scaler):
    joblib.dump(scaler, run / 'scaler.joblib')
    shared._write(run / 'features.json', {
        'ordered_features': context['ordered_features'], 'feature_dtypes': context['feature_dtypes'],
        'scaler_state': shared._scaler_state(scaler),
    })
    shared._write(run / 'splits.json', context['splits'])


def _train_model(run, role, config, context, scaler, scaled, labels, artifact_kind):
    from tensorflow import keras
    from src.notebook_helpers import build_binary_classifier, set_seed
    if context['training_settings'] != TRAINING:
        raise ValueError('Original NN training settings differ.')
    keras.backend.clear_session()
    set_seed(42)
    model = build_binary_classifier(input_dim=len(context['ordered_features']), **config)
    callback = keras.callbacks.EarlyStopping(**TRAINING['early_stopping'])
    history = model.fit(scaled[0], labels[0], validation_data=(scaled[1], labels[1]),
                        epochs=50, batch_size=64, shuffle=True, callbacks=[callback], verbose=1)
    if callback.best_weights is None or not hasattr(callback, 'best_epoch'):
        raise RuntimeError('EarlyStopping did not expose its actual restored epoch/weights.')
    if (len(model.get_weights()) != len(callback.best_weights)
            or any(not np.array_equal(a, b) for a, b in zip(model.get_weights(), callback.best_weights))):
        raise RuntimeError('NN weights differ from the callback-restored weights.')
    epoch = int(callback.best_epoch)
    metrics, probabilities = {}, {}
    for name, X, y in [('validation', scaled[1], labels[1]), ('test', scaled[2], labels[2])]:
        metrics[name], probabilities[name] = shared._metrics(model, X, y)
    path = run / f'{role}.keras'
    model.save(path)
    shared._write(run / f'{role}.history.json', {k: list(map(float, v)) for k, v in history.history.items()})
    record = {
        'artifact_kind': artifact_kind, 'stage': 'stage_3', 'run_id': run.name, 'role': role,
        'created_at': datetime.now(timezone.utc).isoformat(), 'context': context, 'parameters': config,
        'model_sha256': shared._sha(path), 'scaler_sha256': shared._sha(run / 'scaler.joblib'),
        'scaler_state': shared._scaler_state(scaler),
        'scaled_sha256': dict(zip(('train', 'validation', 'test'), map(shared._frame_hash, scaled))),
        'epochs_run': len(history.history['val_auc']), 'restored_epoch_zero_based': epoch,
        'restored_epoch_one_based': epoch + 1, 'stopped_epoch_zero_based': int(callback.stopped_epoch),
        'selection_method': 'EarlyStopping.best_epoch after on_train_end',
        'selected_val_auc': float(history.history['val_auc'][epoch]),
        'saved_compile_config': shared._archive_compile(path, config), 'new_metrics': metrics,
        'historical_trial_reproduction': False,
        'historical_evidence': context['evidence'][role].get('historical_scores', {}),
    }
    _verify_model(model, config, len(context['ordered_features']))
    shared._write(run / f'{role}.metadata.json', record)
    return model, record, probabilities


def _load_pair(run, manifest, context):
    from tensorflow import keras
    shared._verify_files(run, manifest, _FILES)
    if manifest['stage'] != 'stage_3' or manifest['artifact_kind'] != 'rebuilt':
        raise ValueError('Expected an explicitly rebuilt Stage 3 NN pair.')
    shared._parameters(context['evidence']['baseline']['parameters'], BASELINE)
    shared._parameters(context['evidence']['final']['parameters'], FINAL)
    features = shared._json(run / 'features.json')
    if (features['ordered_features'] != context['ordered_features']
            or features['feature_dtypes'] != context['feature_dtypes']
            or shared._json(run / 'splits.json') != context['splits']):
        raise ValueError('Saved NN feature schema or split identity differs.')
    models, records = {}, {}
    for role, expected in [('baseline', BASELINE), ('final', FINAL)]:
        record = shared._json(run / f'{role}.metadata.json')
        if (record['artifact_kind'] != 'rebuilt' or record['stage'] != 'stage_3'
                or record['run_id'] != run.name or record['role'] != role
                or _substantive(record['context']) != _substantive(context)
                or record['model_sha256'] != manifest['sha256'][f'{role}.keras']
                or record['scaler_sha256'] != manifest['sha256']['scaler.joblib']):
            raise ValueError(f'{role}: Stage 3 NN metadata/context differs.')
        shared._parameters(record['parameters'], expected)
        history = shared._json(run / f'{role}.history.json')
        epoch = record['restored_epoch_zero_based']
        if (type(epoch) is not int or not 0 <= epoch < record['epochs_run'] <= 50
                or len(history['val_auc']) != record['epochs_run']
                or record['restored_epoch_one_based'] != epoch + 1
                or record['selected_val_auc'] != history['val_auc'][epoch]
                or record['selection_method'] != 'EarlyStopping.best_epoch after on_train_end'):
            raise ValueError(f'{role}: actual restored-epoch/history evidence differs.')
        shared._warn_versions(record['context'], context)
        if shared._archive_compile(run / f'{role}.keras', expected) != record['saved_compile_config']:
            raise ValueError(f'{role}: saved optimizer/loss metadata differs.')
        model = keras.models.load_model(run / f'{role}.keras', compile=False)
        _verify_model(model, expected, len(context['ordered_features']))
        models[role], records[role] = model, record
    scaler = joblib.load(run / 'scaler.joblib')
    state = shared._scaler_state(scaler)
    if features['scaler_state'] != state or any(r['scaler_state'] != state for r in records.values()):
        raise ValueError('Saved NN scaler statistics differ.')
    return models['baseline'], models['final'], scaler, records


def load_rebuilt_nn_pair(root, *, context):
    run, manifest = shared._verified_run(root, _FILES, required=True)
    return _load_pair(run, manifest, context)


def transform_with_saved_scaler(scaler, *, X_train, X_val, X_test, records):
    return shared.transform_with_saved_scaler(
        scaler, X_train=X_train, X_val=X_val, X_test=X_test, records=records,
    )


def rebuild_nn_pair(root, *, context, X_train, X_val, X_test, y_train, y_val, y_test):
    shared._parameters(context['evidence']['baseline']['parameters'], BASELINE)
    shared._parameters(context['evidence']['final']['parameters'], FINAL)
    matrices, labels = (X_train, X_val, X_test), (y_train, y_val, y_test)
    scaler, scaled = _fit_scaler(context, matrices, labels)
    run = shared.new_nn_analysis_run(root)
    _save_preparation(run, context, scaler)
    expected_probabilities = {}
    for role, config in [('baseline', BASELINE), ('final', FINAL)]:
        _, _, expected_probabilities[role] = _train_model(run, role, config, context, scaler, scaled, labels, 'rebuilt')
    digest = shared._publish(run, _FILES, {'stage': 'stage_3', 'artifact_kind': 'rebuilt'})
    baseline, final, saved_scaler, records = _load_pair(run, shared._json(run / 'manifest.json'), context)
    readback = transform_with_saved_scaler(saved_scaler, X_train=X_train, X_val=X_val, X_test=X_test, records=records)
    for role, model in [('baseline', baseline), ('final', final)]:
        for name, X, y in [('validation', readback[1], y_val), ('test', readback[2], y_test)]:
            metrics, probabilities = shared._metrics(model, X, y)
            expected = expected_probabilities[role][name]
            if (metrics != records[role]['new_metrics'][name] or not np.array_equal(probabilities, expected)
                    or not np.array_equal(probabilities >= 0.5, expected >= 0.5)):
                raise ValueError(f'{role}/{name}: saved NN predictions/probabilities/metrics differ.')
    shared._pointer(root, run.name, digest)
    return baseline, final, saved_scaler, records


def begin_explicit_search(root, *, context, X_train, X_val, X_test, y_train, y_val, y_test):
    """Only the explicit tuning branch may fit its baseline and scaler."""
    matrices, labels = (X_train, X_val, X_test), (y_train, y_val, y_test)
    scaler, scaled = _fit_scaler(context, matrices, labels)
    run = shared.new_nn_analysis_run(root)
    _save_preparation(run, context, scaler)
    model, record, _ = _train_model(run, 'baseline', BASELINE, context, scaler, scaled, labels, 'explicit_search_baseline')
    return run, model, scaler, {'baseline': record}, scaled


def completed_search_trials(frame):
    """Summary rows and unscored/non-completed trials never become search history."""
    if not set(shared.TRIAL_COLUMNS).issubset(frame.columns):
        return pd.DataFrame(columns=shared.TRIAL_COLUMNS)
    scores = pd.to_numeric(frame['val_auc'], errors='coerce')
    ids = frame['trial_id'].astype('string')
    keep = (frame['status'].eq('COMPLETED') & ids.notna() & ids.str.strip().ne('')
            & ids.ne('saved_best_params') & scores.notna() & np.isfinite(scores) & scores.between(0, 1))
    completed = frame.loc[keep].copy()
    completed['val_auc'] = scores.loc[completed.index]
    # Stage 2 and Stage 3 use the same refined parameter search space.
    return shared._genuine_trials(completed) if not completed.empty else completed


_SEARCH_FILES = ['baseline.keras', 'baseline.history.json', 'baseline.metadata.json',
                 'scaler.joblib', 'features.json', 'splits.json', 'best_model_training.keras',
                 'best_params.json', 'search_summary.txt', 'trials.csv', 'search.metadata.json']


def _read_search(run, manifest, context):
    if (manifest['stage'] != 'stage_3' or manifest['project_name'] != PROJECT
            or manifest['artifact_kind'] != 'explicit_search'
            or _search_context(manifest['context']) != _search_context(context)):
        raise ValueError('Search history belongs to a different stage/search/data/split context.')
    shared._verify_files(run, manifest, _SEARCH_FILES)
    trials = completed_search_trials(pd.read_csv(run / 'trials.csv', dtype={'trial_id': str}))
    if trials.empty:
        raise ValueError('Published Stage 3 search has no genuine completed trials.')
    shared._warn_versions(manifest['context'], context)
    return trials


def load_search_history(root, *, context):
    result = shared._verified_run(root, _SEARCH_FILES, required=False)
    if result is None:
        return pd.DataFrame(columns=shared.TRIAL_COLUMNS)
    return _read_search(*result, context)


def finish_explicit_search(run, *, tuner, context, X_val_s, X_test_s, y_val, y_test):
    """Verify the real tuner-selected checkpoint; never label it historical trial 053."""
    from tensorflow import keras
    from src.tuning_utils import tuner_trials_to_dataframe
    if tuner is None or tuner.project_name != PROJECT:
        raise ValueError('Expected the explicitly requested Stage 3 tuner.')
    trials = completed_search_trials(tuner_trials_to_dataframe(tuner))
    if trials.empty:
        raise ValueError('Explicit Stage 3 search produced no genuine completed trials.')
    selected = tuner.oracle.get_best_trials(1)[0]
    matches = trials.loc[trials['trial_id'].eq(selected.trial_id)]
    if len(matches) != 1:
        raise ValueError('Tuner-selected trial is absent from genuine completed history.')
    config = shared._parameters(matches.iloc[0].to_dict())
    run = Path(run)
    path = run / 'best_model_training.keras'
    compile_config = shared._archive_compile(path, config)
    model = keras.models.load_model(path, compile=False)
    _verify_model(model, config, len(context['ordered_features']))
    metrics = {name: shared._metrics(model, X, y)[0] for name, X, y in
               [('validation', X_val_s, y_val), ('test', X_test_s, y_test)]}
    record = {
        'artifact_kind': 'explicit_search', 'stage': 'stage_3', 'project_name': PROJECT,
        'run_id': run.name, 'trial_id': selected.trial_id, 'parameters': config,
        'model_sha256': shared._sha(path), 'new_metrics': metrics, 'saved_compile_config': compile_config,
        'historical_trial_reproduction': False,
        'epoch_selection': 'Keras Tuner selected checkpoint; actual restored epoch is not available from this API.',
    }
    trials.to_csv(run / 'trials.csv', index=False)
    shared._write(run / 'search.metadata.json', record)
    digest = shared._publish(run, _SEARCH_FILES, {
        'stage': 'stage_3', 'project_name': PROJECT, 'artifact_kind': 'explicit_search', 'context': context,
    })
    _read_search(run, shared._json(run / 'manifest.json'), context)
    shared._pointer(run.parent, run.name, digest)
    return model, config, record
