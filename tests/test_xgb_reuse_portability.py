import copy
import json
from contextlib import ExitStack
from pathlib import Path
import unittest
from unittest.mock import patch

from src import stage_2_xgb_rebuild as stage_2
from src import stage_3_xgb_rebuild as stage_3


ROOT = Path(__file__).resolve().parents[1]


class XgbReusePortabilityTests(unittest.TestCase):
    def saved_run(self, stage):
        """Read existing fixtures, then serve all loader inputs from memory."""
        original = ROOT / 'models' / stage / 'xgboost' / 'rebuilt'
        pointer = json.loads((original / 'current.json').read_text())
        run = original / pointer['run_id']
        manifest = json.loads((run / 'manifest.json').read_text())
        root = Path('/different/checkout/models') / stage / 'xgboost' / 'rebuilt'
        relocated = root / pointer['run_id']
        documents = {root / 'current.json': pointer, relocated / 'manifest.json': manifest}
        hashes = {relocated / 'manifest.json': pointer['manifest_sha256']}
        for name, digest in manifest['sha256'].items():
            hashes[relocated / name] = digest
            if name.endswith('.json'):
                documents[relocated / name] = json.loads((run / name).read_text())
        context = copy.deepcopy(documents[relocated / 'final.metadata.json']['context'])
        context['splits'] = copy.deepcopy(documents[relocated / 'splits.json'])
        for evidence in context['evidence'].values():
            evidence['diagnostics'] = ['/different/checkout/tuning: diagnostic text']
        return root, documents, hashes, context

    def load(self, module, root, documents, hashes, context):
        """Use the public pointer/manifest/context path; mock filesystem and models."""
        with ExitStack() as stack:
            stack.enter_context(patch.object(stage_2, '_read_json', side_effect=lambda p: copy.deepcopy(documents[p])))
            stack.enter_context(patch.object(stage_2, '_sha', side_effect=lambda p: hashes[p]))
            model_load = stack.enter_context(patch.object(stage_2.joblib, 'load', side_effect=[object(), object()]))
            # Architecture validation is independent of context compatibility.
            verify = stack.enter_context(patch.object(module, '_verify_model'))
            try:
                result = module.load_rebuilt_xgb_pair(root, context=context)
            except ValueError:
                model_load.assert_not_called()
                verify.assert_not_called()
                raise
            self.assertEqual(model_load.call_count, 2)
            self.assertEqual(verify.call_count, 2)
            return result

    def test_different_diagnostic_paths_allow_reuse_without_mutation(self):
        for name, module in [('stage_2', stage_2), ('stage_3', stage_3)]:
            with self.subTest(stage=name):
                root, documents, hashes, context = self.saved_run(name)
                original_documents, original_context = copy.deepcopy(documents), copy.deepcopy(context)
                self.load(module, root, documents, hashes, context)
                self.assertEqual(documents, original_documents)
                self.assertEqual(context, original_context)

    def test_substantive_changes_reject_reuse_before_model_loading(self):
        mutations = [
            ('evidence hash', ('evidence', 'final', 'source_sha256'), 'changed-hash'),
            ('baseline evidence hash', ('evidence', 'baseline', 'source_sha256'), 'changed-hash'),
            ('parameters', ('evidence', 'final', 'parameters', 'max_depth'), 99),
            ('relative evidence path', ('evidence', 'final', 'source_path'), 'tuning/other/trials.csv'),
            ('evidence kind', ('evidence', 'final', 'evidence_kind'), 'different-kind'),
            ('trial identity', ('evidence', 'final', 'trial_id'), 'different-trial'),
            ('study identity', ('evidence', 'final', 'study_name'), 'different-study'),
            ('dataset hash', ('datasets', 'stage_2', 'sha256'), 'changed-hash'),
            ('dataset path', ('datasets', 'stage_2', 'path'), 'data/processed/other.csv'),
            ('schema names', ('ordered_features',), ['different-feature']),
            ('schema dtypes', ('feature_dtypes',), ['different-dtype']),
            ('splits', ('splits', 'train', 'row_positions'), [999999]),
        ]
        for name, module in [('stage_2', stage_2), ('stage_3', stage_3)]:
            root, documents, hashes, original = self.saved_run(name)
            for label, keys, value in mutations:
                with self.subTest(stage=name, change=label):
                    context = copy.deepcopy(original)
                    target = context
                    for key in keys[:-1]:
                        target = target[key]
                    target[keys[-1]] = value
                    with self.assertRaisesRegex(ValueError, 'metadata/context|split identity|feature schema'):
                        self.load(module, root, documents, hashes, context)


if __name__ == '__main__':
    unittest.main()
