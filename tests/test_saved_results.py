import csv
import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from src.saved_results import SavedParameterResult, resolve_xgb_results


PARAMS = dict(learning_rate=0.1, max_depth=8, min_child_weight=1.1,
              gamma=0.2, subsample=0.8, colsample_bytree=0.7,
              reg_alpha=0.01, reg_lambda=1.0)


class SavedResultsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def resolve(self, **kwargs):
        return resolve_xgb_results(self.root, study_name=kwargs.get('study_name', 'main'))

    def json(self, params=None, **extra):
        (self.root / 'best_params.json').write_text(
            json.dumps({'best_params': PARAMS if params is None else params, **extra}))

    def csv(self, rows, summary=False, headers=None):
        fields = headers or ['number', 'value', *('params_' + p for p in PARAMS)] + ([] if summary else ['state'])
        with (self.root / 'trials.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    def row(self, number=0, value=0.9, state='COMPLETE', params=None):
        return {'number': number, 'value': value, 'state': state,
                **{'params_' + k: v for k, v in (PARAMS if params is None else params).items()}}

    def db(self, trials=None):
        db = sqlite3.connect(self.root / 'optuna_study.sqlite3')
        db.executescript('''
            CREATE TABLE studies(study_id INTEGER, study_name TEXT);
            CREATE TABLE study_directions(study_id INTEGER, objective INTEGER, direction TEXT);
            CREATE TABLE trials(trial_id INTEGER, study_id INTEGER, number INTEGER, state TEXT);
            CREATE TABLE trial_values(trial_id INTEGER, objective INTEGER, value REAL, value_type TEXT);
            CREATE TABLE trial_params(trial_id INTEGER, param_name TEXT, param_value REAL, distribution_json TEXT);
            INSERT INTO studies VALUES(1,'main'),(2,'empty');
            INSERT INTO study_directions VALUES(1,0,'MAXIMIZE'),(2,0,'MAXIMIZE');
        ''')
        for ident, number, state, score, value_type in (trials or [(1, 1546, 'COMPLETE', 0.95, 'FINITE')]):
            db.execute('INSERT INTO trials VALUES(?,1,?,?)', (ident, number, state))
            db.execute('INSERT INTO trial_values VALUES(?,0,?,?)', (ident, score, value_type))
            for key, value in PARAMS.items():
                distribution = {'name': 'IntDistribution' if key == 'max_depth' else 'FloatDistribution', 'attributes': {}}
                db.execute('INSERT INTO trial_params VALUES(?,?,?,?)',
                           (ident, key, value, json.dumps(distribution)))
        db.commit()
        db.close()

    def mutate_db(self, sql):
        with sqlite3.connect(self.root / 'optuna_study.sqlite3') as db:
            db.executescript(sql)

    def test_precedence_and_integer_decoding(self):
        self.db()
        self.csv([self.row(value=0.99)])
        self.json(best_val_auc=1.0)
        result = self.resolve()
        self.assertEqual((result.source_kind, result.evidence_kind, result.trial_id),
                         ('optuna', 'completed_trial', '1546'))
        self.assertEqual(result.historical_score, 0.95)
        self.assertIs(type(result.parameters['max_depth']), int)
        self.assertEqual(result.study_name, 'main')

    def test_empty_or_missing_study_blocks_unscoped_fallback(self):
        self.db()
        self.csv([self.row()])
        self.json(best_val_auc=.99)
        for study in ('empty', 'missing'):
            for csv_present in (True, False):
                with self.subTest(study=study, csv_present=csv_present):
                    if csv_present:
                        self.csv([self.row()])
                    else:
                        (self.root / 'trials.csv').unlink()
                    before = self.snapshot()
                    result = self.resolve(study_name=study)
                    self.assertEqual(result.source_kind, 'unavailable')
                    self.assertEqual(result.evidence_kind, 'unavailable')
                    self.assertIsNone(result.historical_score)
                    self.assertIsNone(result.source_path)
                    self.assertFalse(result.parameters)
                    self.assertTrue(any('unscoped' in d and 'study-specific provenance' in d
                                        for d in result.diagnostics))
                    self.assertEqual(before, self.snapshot())

    def snapshot(self):
        return {str(p.relative_to(self.root)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in self.root.rglob('*') if p.is_file()}

    def test_without_sqlite_summary_fallback_is_read_only(self):
        for stage in (2, 3):
            with self.subTest(stage=stage):
                row = self.row(value='')
                del row['state']
                self.csv([row], summary=True)
                self.json()
                before = self.snapshot()
                result = self.resolve()
                self.assertEqual((result.source_kind, result.evidence_kind),
                                 ('csv', 'parameter_summary'))
                self.assertIsNone(result.historical_score)
                self.assertEqual(before, self.snapshot())
                (self.root / 'trials.csv').unlink()
                before = self.snapshot()
                result = self.resolve()
                self.assertEqual((result.source_kind, result.evidence_kind),
                                 ('json', 'parameter_summary'))
                self.assertIsNone(result.historical_score)
                self.assertEqual(before, self.snapshot())

    def test_sqlite_filters_and_ties(self):
        self.db([(1, 0, 'FAIL', 1., 'FINITE'), (2, 1, 'RUNNING', 1., 'FINITE'),
                 (3, 2, 'PRUNED', 1., 'FINITE'), (4, 3, 'COMPLETE', float('inf'), 'INF_POS'),
                 (5, 4, 'COMPLETE', None, 'FINITE'), (6, 6, 'COMPLETE', .8, 'FINITE'),
                 (7, 5, 'COMPLETE', .8, 'FINITE'), (8, 7, 'COMPLETE', .99, 'FINITE')])
        self.mutate_db("UPDATE trial_params SET param_value=8.5 WHERE trial_id=8 AND param_name='max_depth'")
        self.assertEqual(self.resolve().trial_id, '5')

    def test_corrupt_empty_or_incompatible_sqlite_blocks_fallback(self):
        self.json()
        self.csv([self.row()])
        path = self.root / 'optuna_study.sqlite3'
        for content in (b'not sqlite', b'', None):
            with self.subTest(content=content):
                if path.exists():
                    path.unlink()
                if content is None:
                    with sqlite3.connect(path) as db:
                        db.execute('CREATE TABLE studies(unexpected_column TEXT)')
                else:
                    path.write_bytes(content)
                before = self.snapshot()
                result = self.resolve()
                self.assertEqual(result.source_kind, 'unavailable')
                self.assertIsNone(result.historical_score)
                self.assertTrue(any('unscoped' in d for d in result.diagnostics))
                self.assertEqual(before, self.snapshot())

    def test_sqlite_unsupported_distribution_or_objective(self):
        for mutation in [
            "UPDATE trial_params SET distribution_json='{\"name\":\"CategoricalDistribution\"}'",
            "UPDATE study_directions SET direction='MINIMIZE'",
            "INSERT INTO study_directions VALUES(1,1,'MAXIMIZE')",
        ]:
            with self.subTest(mutation=mutation):
                path = self.root / 'optuna_study.sqlite3'
                if path.exists():
                    path.unlink()
                self.db()
                self.mutate_db(mutation)
                self.json()
                self.csv([self.row()])
                before = self.snapshot()
                result = self.resolve()
                self.assertEqual(result.source_kind, 'unavailable')
                self.assertIsNone(result.historical_score)
                self.assertTrue(any('unscoped' in d for d in result.diagnostics))
                self.assertEqual(before, self.snapshot())

    def test_csv_filters_and_ties(self):
        self.csv([self.row(0, 1., 'FAIL'), self.row(1, 'nan'), self.row(2, 'inf'),
                  self.row(3, 1.1), self.row(4, .99, params={**PARAMS, 'max_depth': '8.5'}),
                  self.row(6, .8), self.row(5, .8)])
        self.assertEqual(self.resolve().trial_id, '5')

    def test_parameter_summary_has_unknown_score_and_identity(self):
        row = self.row(value='')
        del row['state']
        self.csv([row], summary=True)
        self.json(best_val_auc=.99)
        result = self.resolve()
        self.assertEqual((result.source_kind, result.evidence_kind), ('csv', 'parameter_summary'))
        self.assertIsNone(result.historical_score)
        self.assertIsNone(result.trial_id)
        self.assertIsNone(result.score_name)

    def test_status_free_csv_rejected_unless_exact_summary(self):
        row = self.row(value=.9)
        del row['state']
        for rows in [[row], [{**row, 'value': ''}, {**row, 'value': ''}]]:
            self.csv(rows, summary=True)
            self.json()
            self.assertEqual(self.resolve().source_kind, 'json')

    def test_bad_csv_and_json_fallback(self):
        self.json()
        for content in ['number,number\n0,0\n', 'number,value,state\n0,.9,COMPLETE\n',
                        'number,value\n0,.9,extra\n', '"unterminated']:
            (self.root / 'trials.csv').write_text(content)
            self.assertEqual(self.resolve().source_kind, 'json')

    def test_json_scores_and_invalid_parameters(self):
        self.json(best_val_auc=.91)
        self.assertEqual(self.resolve().historical_score, .91)
        for score in [None, float('nan'), float('inf'), True, 1.1, 'bad']:
            self.json(best_val_auc=score)
            result = self.resolve()
            self.assertEqual(result.source_kind, 'json')
            self.assertIsNone(result.historical_score)
            self.assertTrue(any('invalid summary score' in d for d in result.diagnostics))
        for params in [{}, {**PARAMS, 'extra': 1}, {**PARAMS, 'max_depth': True},
                       {**PARAMS, 'max_depth': 8.0}, {**PARAMS, 'gamma': None},
                       {**PARAMS, 'gamma': []}, {**PARAMS, 'gamma': {}},
                       {**PARAMS, 'gamma': ''}, {**PARAMS, 'gamma': float('nan')}]:
            self.json(params=params)
            self.assertEqual(self.resolve().source_kind, 'unavailable')

    def test_duplicate_or_malformed_json(self):
        for content in ['{"best_params":{},"best_params":{}}', '[1,2]', '{broken']:
            (self.root / 'best_params.json').write_text(content)
            self.assertEqual(self.resolve().source_kind, 'unavailable')

    def test_missing_sources_do_not_create_files(self):
        result = self.resolve()
        self.assertEqual(result.source_kind, 'unavailable')
        self.assertEqual(len(result.diagnostics), 3)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_immutable_result(self):
        self.json()
        result = self.resolve()
        self.assertIsInstance(result, SavedParameterResult)
        with self.assertRaises(TypeError):
            result.parameters['max_depth'] = 9

    def test_no_ml_imports(self):
        script = """
import builtins, tempfile
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'optuna', 'xgboost', 'tensorflow', 'keras', 'keras_tuner', 'numpy', 'pandas'} or name == 'src.tuning_utils':
        raise AssertionError('forbidden import: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from src.saved_results import resolve_xgb_results
with tempfile.TemporaryDirectory() as directory:
    assert resolve_xgb_results(directory, study_name='main').source_kind == 'unavailable'
"""
        subprocess.run([sys.executable, '-B', '-c', script], check=True,
                       cwd=Path(__file__).resolve().parents[1])

    def test_explicit_study_name(self):
        with self.assertRaises(TypeError):
            resolve_xgb_results(self.root)
        with self.assertRaises(ValueError):
            self.resolve(study_name=' ')

    def test_read_only_uri_query_only_and_unchanged_files(self):
        self.db()
        (self.root / 'best_model.joblib').write_bytes(b'not a model')
        def snapshot():
            return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in self.root.iterdir()}
        before = snapshot()
        real_connect = sqlite3.connect
        observed = []
        class Connection(sqlite3.Connection):
            def execute(self, sql, *args):
                if sql == 'PRAGMA query_only=ON':
                    result = super().execute(sql, *args)
                    with self_test.assertRaises(sqlite3.OperationalError):
                        super().execute('CREATE TABLE forbidden(value)')
                    observed.append('query_only')
                    return result
                return super().execute(sql, *args)
        self_test = self
        def connect(database_uri, **kwargs):
            self.assertTrue(database_uri.endswith('?mode=ro'))
            self.assertTrue(kwargs['uri'])
            connection = real_connect(database_uri, factory=Connection, **kwargs)
            with self.assertRaises(sqlite3.OperationalError):
                connection.execute('CREATE TABLE forbidden(value)')
            observed.append('mode_ro')
            return connection
        with patch('src.saved_results._sqlite3.connect', side_effect=connect):
            self.assertEqual(self.resolve().source_kind, 'optuna')
        self.assertEqual(observed, ['mode_ro', 'query_only'])
        self.assertEqual(before, snapshot())


if __name__ == '__main__':
    unittest.main()
