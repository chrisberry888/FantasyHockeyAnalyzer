import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, call, patch

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils.validation import check_is_fitted

import my_module_v2 as mx


class TrainingResumeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        working_directory = patch.object(mx.os, 'getcwd', return_value=temporary.name)
        working_directory.start()
        self.addCleanup(working_directory.stop)
        progress = patch.object(mx, 'print', create=True)
        progress.start()
        self.addCleanup(progress.stop)
        self.X = pd.DataFrame({'Shots': np.arange(24) + 40, 'Hits': np.arange(24) * 2})
        self.y = (self.X['Shots'] * 2 + self.X['Hits']).tolist()
        self.model = RandomForestRegressor(n_estimators=3, max_depth=2,
                                           random_state=42, n_jobs=1)
        self.folder = self.root / 'models' / '1_year' / 'random_forests'

    def train(self, count=3, resume=True, X=None, y=None, model=None):
        mx.create_models(self.X if X is None else X, self.y if y is None else y,
                         self.model if model is None else model, count,
                         '1_year', 'random_forests', resume_training=resume)

    def model_bytes(self):
        return {path.name: path.read_bytes() for path in self.folder.glob('model_*.joblib')}

    def test_restart_after_interruption_reuses_completed_models(self):
        original_fit = RandomForestRegressor.fit
        calls = []

        def interrupted_fit(model, X, y):
            calls.append(model)
            if len(calls) == 3:
                raise KeyboardInterrupt('simulated shutdown during training')
            return original_fit(model, X, y)

        with patch.object(RandomForestRegressor, 'fit', new=interrupted_fit):
            with self.assertRaises(KeyboardInterrupt):
                self.train()
        before = self.model_bytes()
        self.assertEqual(set(before), {'model_0.joblib', 'model_1.joblib'})

        calls.clear()

        def counted_fit(model, X, y):
            calls.append(model)
            return original_fit(model, X, y)

        with patch.object(RandomForestRegressor, 'fit', new=counted_fit):
            self.train()
        self.assertEqual(len(calls), 1)
        after = self.model_bytes()
        for name, contents in before.items():
            self.assertEqual(after[name], contents)
        self.assertEqual(len(after), 3)
        with patch.object(RandomForestRegressor, 'fit', side_effect=AssertionError('unexpected refit')):
            self.train()

    def test_missing_and_corrupt_models_are_retrained_without_replacing_valid_models(self):
        self.train(count=4)
        before = self.model_bytes()
        (self.folder / 'model_1.joblib').unlink()
        (self.folder / 'model_2.joblib').write_bytes(b'incomplete model')
        (self.folder / '.model_3.joblib.abandoned.tmp').write_bytes(b'partial write')
        original_fit = RandomForestRegressor.fit
        with patch.object(RandomForestRegressor, 'fit', autospec=True,
                          side_effect=original_fit) as fit:
            self.train(count=4)
        self.assertEqual(fit.call_count, 2)
        after = self.model_bytes()
        self.assertEqual(after['model_0.joblib'], before['model_0.joblib'])
        self.assertEqual(after['model_3.joblib'], before['model_3.joblib'])
        for path in self.folder.glob('model_*.joblib'):
            check_is_fitted(joblib.load(path))

    def test_incompatible_data_features_and_settings_leave_saved_progress_unchanged(self):
        self.train()
        before = {path.name: path.read_bytes() for path in self.folder.iterdir()}
        variants = [
            {'X': self.X.rename(columns={'Shots': 'Attempts'})},
            {'X': self.X + 1},
            {'y': [value + 1 for value in self.y]},
            {'model': RandomForestRegressor(n_estimators=5, max_depth=2,
                                            random_state=42, n_jobs=1)},
        ]
        for changes in variants:
            with self.subTest(changes=changes), \
                    patch.object(RandomForestRegressor, 'fit', side_effect=AssertionError('unexpected fit')):
                with self.assertRaisesRegex(ValueError, 'resume_training=False'):
                    self.train(**changes)
            self.assertEqual({path.name: path.read_bytes() for path in self.folder.iterdir()}, before)

    def test_models_without_metadata_are_not_silently_reused_or_deleted(self):
        self.train()
        (self.folder / 'training_metadata.json').unlink()
        before = self.model_bytes()
        with self.assertRaisesRegex(ValueError, 'no training metadata'):
            self.train()
        self.assertEqual(self.model_bytes(), before)

    def test_equivalent_rebuilt_feature_table_reuses_saved_models(self):
        self.train()
        before = self.model_bytes()
        rebuilt = pd.concat([self.X[['Shots']], self.X[['Hits']]], axis=1)
        pd.testing.assert_frame_equal(self.X, rebuilt)
        with patch.object(RandomForestRegressor, 'fit', side_effect=AssertionError('unexpected refit')):
            self.train(X=rebuilt)
        self.assertEqual(self.model_bytes(), before)

    def test_fresh_training_clears_only_the_existing_model_group(self):
        self.train()
        other_group = self.root / 'models' / '2_year' / 'random_forests'
        other_group.mkdir(parents=True)
        other_file = other_group / 'model_0.joblib'
        other_file.write_bytes(b'preserve another group')
        root_file = self.root / 'models' / 'notes.txt'
        root_file.write_text('preserve root file')
        replacement = RandomForestRegressor(n_estimators=5, random_state=7, n_jobs=1)
        self.train(count=2, resume=False, model=replacement)
        self.assertEqual(set(self.model_bytes()), {'model_0.joblib', 'model_1.joblib'})
        self.assertEqual(len(joblib.load(self.folder / 'model_0.joblib').estimators_), 5)
        self.assertEqual(other_file.read_bytes(), b'preserve another group')
        self.assertEqual(root_file.read_text(), 'preserve root file')
        self.train(count=2, model=replacement)

    def test_interrupted_save_does_not_publish_partial_model(self):
        original_fit = RandomForestRegressor.fit
        calls = []

        def interrupted_fit(model, X, y):
            calls.append(model)
            if len(calls) == 2:
                raise KeyboardInterrupt('simulated shutdown during training')
            return original_fit(model, X, y)

        with patch.object(RandomForestRegressor, 'fit', new=interrupted_fit):
            with self.assertRaises(KeyboardInterrupt):
                self.train(count=2)
        before = self.model_bytes()

        def interrupted_dump(model, output):
            output.write(b'incomplete model')
            raise KeyboardInterrupt('simulated shutdown during saving')

        with patch.object(mx.joblib, 'dump', side_effect=interrupted_dump):
            with self.assertRaises(KeyboardInterrupt):
                self.train(count=2)
        self.assertEqual(self.model_bytes(), before)
        self.assertFalse((self.folder / 'model_1.joblib').exists())
        self.assertEqual(list(self.folder.glob('*.tmp')), [])
        self.train(count=2)
        check_is_fitted(joblib.load(self.folder / 'model_1.joblib'))

    def test_predictions_ignore_temporary_files_and_training_metadata(self):
        self.train(count=2)
        (self.folder / '.model_2.joblib.abandoned.tmp').write_bytes(b'partial write')
        (self.folder / 'notes.joblib').write_bytes(b'not a model checkpoint')
        current = self.X.iloc[[5, 1, 3]].copy()
        current['playerId'] = [8470005, 8470001, 8470003]
        players = pd.DataFrame({'playerId': current['playerId'], 'name': ['A', 'B', 'C']})
        mx.generate_predictions((current, current, current), players)
        paths = list((self.root / 'predictions').rglob('*.parquet'))
        self.assertEqual(len(paths), 2)
        for path in paths:
            result = pd.read_parquet(path)
            self.assertEqual(set(result['playerId']), set(current['playerId']))
            self.assertEqual(set(result['name']), {'A', 'B', 'C'})

    def test_reset_clears_both_output_directories_and_preserves_other_project_files(self):
        self.train()
        predictions = self.root / 'predictions'
        for directory in (self.root / 'models', predictions):
            nested = directory / 'obsolete_group' / '.hidden_directory'
            nested.mkdir(parents=True)
            (nested / 'stale_file').write_text('stale')
            (directory / '.root_file').write_text('stale')
        data = self.root / 'data'
        data.mkdir()
        (data / 'players.csv').write_text('keep data')
        final = self.root / 'final_prediction.csv'
        final.write_text('keep root CSV')

        mx.reset_model_and_prediction_directories()

        for directory in (self.root / 'models', predictions):
            self.assertTrue(directory.is_dir())
            self.assertEqual(list(directory.iterdir()), [])
        self.assertEqual((data / 'players.csv').read_text(), 'keep data')
        self.assertEqual(final.read_text(), 'keep root CSV')

    def test_reset_creates_missing_output_directories(self):
        mx.reset_model_and_prediction_directories()
        for name in ('models', 'predictions'):
            directory = self.root / name
            self.assertTrue(directory.is_dir())
            self.assertEqual(list(directory.iterdir()), [])

    def test_reset_removes_nested_links_without_deleting_their_targets(self):
        target = self.root / 'data'
        target.mkdir()
        player_data = target / 'players.csv'
        player_data.write_text('keep data')
        for name in ('models', 'predictions'):
            directory = self.root / name
            directory.mkdir()
            (directory / 'linked_directory').symlink_to(target, target_is_directory=True)
            (directory / 'linked_file').symlink_to(player_data)
            (directory / 'broken_link').symlink_to(self.root / 'missing')

        mx.reset_model_and_prediction_directories()

        self.assertEqual(player_data.read_text(), 'keep data')
        for name in ('models', 'predictions'):
            self.assertEqual(list((self.root / name).iterdir()), [])

    def test_reset_rejects_linked_output_roots_before_deleting_anything(self):
        models = self.root / 'models'
        models.mkdir()
        saved = models / 'saved_model'
        saved.write_text('keep progress')
        target = self.root / 'data'
        target.mkdir()
        data = target / 'players.csv'
        data.write_text('keep data')
        (self.root / 'predictions').symlink_to(target, target_is_directory=True)

        with self.assertRaisesRegex(ValueError, 'linked output directory'):
            mx.reset_model_and_prediction_directories()

        self.assertEqual(saved.read_text(), 'keep progress')
        self.assertEqual(data.read_text(), 'keep data')

    def test_notebook_flags_select_full_reset_resume_or_prediction_only(self):
        path = Path(__file__).resolve().parents[1] / 'notebooks' / 'V2 Fantasy Hockey Analyzer.ipynb'
        notebook = json.loads(path.read_text())
        source = next(''.join(cell['source']) for cell in notebook['cells']
                      if cell.get('id') == 'ba27186b')
        actions = Mock()
        create = actions.create_models
        reset = actions.reset_model_and_prediction_directories
        scope = {'mx': actions}
        for prefix in ('one_year', 'two_year', 'three_year'):
            for family in ('neural_net', 'random_forest', 'boosted_tree'):
                name = f'{prefix}_{family}_args'
                scope[name] = (name,)
        for enabled in (True, False):
            for resume in (True, False):
                with self.subTest(create_new_models=enabled, resume_training=resume):
                    actions.reset_mock()
                    scope.update(create_new_models=enabled, resume_training=resume)
                    exec(source, scope)
                    self.assertEqual(reset.call_count, 1 if enabled else 0)
                    self.assertEqual(create.call_count, 9 if enabled or resume else 0)
                    if enabled:
                        self.assertEqual(actions.mock_calls[0],
                                         call.reset_model_and_prediction_directories())
                    for invocation in create.call_args_list:
                        self.assertEqual(invocation.kwargs,
                                         {'resume_training': resume and not enabled})


if __name__ == '__main__':
    unittest.main()
