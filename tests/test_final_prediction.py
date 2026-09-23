import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import my_module_v2 as mx


class FinalPredictionTests(unittest.TestCase):
    def test_output_uses_api_metadata_and_preserves_available_prediction_averages(self):
        model_rows = [
            [(1, 'Nathan MacKinnon_COL_2024_C', 300),
             (2, 'Elias Pettersson_VAN_2024_C', 100),
             (3, 'Elias Pettersson_VAN_2024_D', 60),
             (4, 'Sidney Crosby_PIT_2024_C', 600)],
            [(1, 'Nathan MacKinnon_COL_2024_C', 500),
             (2, 'Elias Pettersson_VAN_2024_C', 300)],
            [(1, 'Nathan MacKinnon_COL_2024_C', 400),
             (3, 'Elias Pettersson_VAN_2024_D', 140)],
        ]
        api_players = pd.DataFrame([
            (3, 'LAK', 'D'), (1, 'NJD', 'RW'), (2, 'VAN', 'C'), (5, 'BOS', 'G'),
        ], columns=['player_id', 'team_abbrev', 'position'])

        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / 'predictions'
            folder.mkdir()
            for index, rows in enumerate(model_rows):
                pd.DataFrame(rows, columns=['playerId', 'name', 'prediction']).to_parquet(
                    folder / f'prediction_{index}.parquet', index=False
                )
            with patch.object(mx.os, 'getcwd', return_value=directory), \
                    patch.object(mx, 'get_nhl_players', return_value=api_players) as get_players:
                mx.generate_final_table()
                result = mx.get_final_table()

        get_players.assert_called_once_with()
        self.assertEqual(result.columns.tolist(), [
            'playerID', 'Player Name', 'Team', 'Position', 'Prediction', 'model_coverage_percent'
        ])
        self.assertEqual(result['playerID'].tolist(), [4, 1, 2, 3])
        by_player = result.set_index('playerID')
        self.assertEqual(by_player['Player Name'].to_dict(), {
            1: 'Nathan MacKinnon', 2: 'Elias Pettersson',
            3: 'Elias Pettersson', 4: 'Sidney Crosby',
        })
        self.assertEqual(by_player['Prediction'].to_dict(), {1: 400, 2: 200, 3: 100, 4: 600})
        self.assertEqual(by_player['model_coverage_percent'].to_dict(), {
            1: 100.0, 2: 66.7, 3: 66.7, 4: 33.3,
        })
        self.assertEqual(by_player['Team'].to_dict(), {1: 'NJD', 2: 'VAN', 3: 'LAK', 4: 'UNKNOWN'})
        self.assertEqual(by_player['Position'].to_dict(), {1: 'RW', 2: 'C', 3: 'D', 4: 'UNKNOWN'})

    def test_empty_api_results_keep_player_with_unknown_metadata(self):
        api_players = pd.DataFrame(columns=['player_id', 'team_abbrev', 'position'])
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / 'predictions'
            folder.mkdir()
            pd.DataFrame([(1, 'A_Player_COL_2024_C', 500)],
                         columns=['playerId', 'name', 'prediction']).to_parquet(
                folder / 'prediction_0.parquet', index=False
            )
            with patch.object(mx.os, 'getcwd', return_value=directory), \
                    patch.object(mx, 'get_nhl_players', return_value=api_players):
                mx.generate_final_table()
                result = mx.get_final_table()

        self.assertEqual(len(result), 1)
        self.assertEqual(result.loc[0, 'Player Name'], 'A_Player')
        self.assertEqual(result.loc[0, 'Team'], 'UNKNOWN')
        self.assertEqual(result.loc[0, 'Position'], 'UNKNOWN')


if __name__ == '__main__':
    unittest.main()
