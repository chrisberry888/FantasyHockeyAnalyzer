import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import my_module_v2 as mx


class ModelPreprocessingTests(unittest.TestCase):
    def test_saved_models_use_training_scaling_and_preserve_player_identity(self):
        notebook_path = Path(__file__).resolve().parents[1] / 'notebooks' / 'V2 Fantasy Hockey Analyzer.ipynb'
        notebook = json.loads(notebook_path.read_text())
        sources = {cell['id']: ''.join(cell.get('source', [])) for cell in notebook['cells']}
        scope = {}
        exec(sources['caaf64fa'], scope)
        # The notebook's working-directory magic is unnecessary in a Python test.
        exec('\n'.join(line for line in sources['e41adee7'].splitlines()
                       if not line.startswith('%')), scope)

        ids = np.arange(48) + 8470000
        players = pd.DataFrame({'playerId': ids, 'name': [f'Player {i}' for i in ids]})
        current_tables = []
        prefixes = ('one_year', 'two_year', 'three_year')
        for years, prefix in enumerate(prefixes, start=1):
            training = pd.DataFrame({
                'playerId': ids,
                'Goals': np.arange(48, dtype=float) + 1,
                'Hits': np.arange(48, dtype=float) * 3 + 5,
                'team_COL': np.arange(48) % 2 == 0,
            })
            for history in range(1, years):
                training[f'Goals_{history}'] = np.arange(48, dtype=float) + history * 2
            training['Fantasy_Points'] = training['Goals'] * 5 + training['Hits'] * 0.5
            # These held-out outliers must not influence the fitted scaler.
            training.loc[36:, 'Hits'] += 10000
            X, y = mx.separate_fantasy_points(training)
            self.assertNotIn('playerId', X.columns)
            self.assertNotIn('Fantasy_Points', X.columns)
            self.assertIn('playerId', training.columns)
            scope[f'{prefix}_X'] = X
            scope[f'{prefix}_y'] = y
            current_tables.append(training.drop(columns=['Fantasy_Points']).iloc[
                [15, 1, 12, 3, 9, 4, 8][:8 - years]
            ].copy())

        exec(sources['50f9a5ec'], scope)
        configurations = []
        for prefix in prefixes:
            for family in ('neural_net', 'random_forest', 'boosted_tree'):
                configurations.append(scope[f'{prefix}_{family}_args'])
        self.assertEqual(sum(args[3] for args in configurations), 270)

        def fixed_split(X, y):
            return X.iloc[:36], X.iloc[36:], y[:36], y[36:]

        with tempfile.TemporaryDirectory() as directory, \
                patch.object(mx.os, 'getcwd', return_value=directory), \
                patch.object(mx, 'train_test_split', side_effect=fixed_split):
            saved_models = []
            for args in configurations:
                X, y, template, _, year_folder, model_folder = args
                model = clone(template)
                if model_folder == 'neural_nets':
                    self.assertIsInstance(model, Pipeline)
                    self.assertIsInstance(model.named_steps['standardscaler'], StandardScaler)
                    self.assertEqual(model.named_steps['mlpregressor'].max_iter, 300)
                    # Limit test epochs while exercising validation-based early stopping.
                    model.set_params(mlpregressor__max_iter=30,
                                     mlpregressor__tol=0.5,
                                     mlpregressor__random_state=7)
                elif model_folder == 'random_forests':
                    self.assertIsInstance(model, RandomForestRegressor)
                    model.set_params(n_jobs=1, random_state=7)
                else:
                    self.assertIsInstance(model, HistGradientBoostingRegressor)
                    model.set_params(max_iter=5, min_samples_leaf=2, random_state=7)

                mx.create_models(X, y, model, 1, year_folder, model_folder)
                model_path = Path(directory) / 'models' / year_folder / model_folder / 'model_0.joblib'
                saved_bytes = model_path.read_bytes()
                with patch.object(type(model), 'fit', side_effect=AssertionError('unexpected refit')):
                    mx.create_models(X, y, model, 1, year_folder, model_folder)
                self.assertEqual(model_path.read_bytes(), saved_bytes)
                saved = joblib.load(model_path)
                self.assertEqual(saved.feature_names_in_.tolist(), X.columns.tolist())
                self.assertNotIn('playerId', saved.feature_names_in_)
                if model_folder == 'neural_nets':
                    scaler = saved.named_steps['standardscaler']
                    np.testing.assert_allclose(scaler.mean_, X.iloc[:36].mean().to_numpy())
                    np.testing.assert_allclose(scaler.var_, X.iloc[:36].var(ddof=0).to_numpy())
                    neural_net = saved.named_steps['mlpregressor']
                    self.assertEqual(neural_net.coefs_[0].shape[1], 50)
                    self.assertIsNotNone(neural_net.validation_scores_)
                    self.assertLess(neural_net.n_iter_, 30)
                elif model_folder == 'random_forests':
                    self.assertEqual(len(saved.estimators_), 50)
                    for tree in saved.estimators_:
                        self.assertLessEqual(tree.get_depth(), 10)
                        leaves = tree.tree_.children_left == -1
                        self.assertTrue((tree.tree_.n_node_samples[leaves] >= 5).all())
                else:
                    self.assertTrue(saved.do_early_stopping_)
                    self.assertGreater(len(saved.validation_score_), 0)
                saved_models.append((year_folder, model_folder, saved))

            mx.generate_predictions(tuple(current_tables), players)
            self.assertEqual(len(list((Path(directory) / 'predictions').rglob('*.parquet'))), 9)
            for year_folder, model_folder, saved in saved_models:
                with self.subTest(year=year_folder, model=model_folder):
                    current = current_tables[int(year_folder[0]) - 1]
                    result = pd.read_parquet(Path(directory) / 'predictions' / year_folder /
                                             model_folder / 'prediction_0.parquet')
                    self.assertEqual(set(result['playerId']), set(current['playerId']))
                    expected = saved.predict(current.drop(columns=['playerId']))
                    np.testing.assert_allclose(
                        result.set_index('playerId').loc[current['playerId'], 'prediction'],
                        expected,
                    )
                    self.assertEqual(result.set_index('playerId')['name'].to_dict(),
                                     players.set_index('playerId').loc[current['playerId'], 'name'].to_dict())

                    direct = mx.get_prediction_table([saved], current, players)
                    pd.testing.assert_frame_equal(direct, result)
                    changed_current = current.copy()
                    changed_current['playerId'] += 1000000
                    changed_players = players.copy()
                    changed_players['playerId'] += 1000000
                    changed = mx.get_prediction_table([saved], changed_current, changed_players)
                    self.assertEqual(set(changed['playerId']), set(changed_current['playerId']))
                    pd.testing.assert_series_equal(
                        direct.set_index('name')['prediction'],
                        changed.set_index('name')['prediction'],
                    )


if __name__ == '__main__':
    unittest.main()
