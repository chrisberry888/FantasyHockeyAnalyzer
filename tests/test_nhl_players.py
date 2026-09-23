import unittest
from unittest.mock import Mock, patch

import pandas as pd

import my_module_v2 as mx


class NhlPlayersTests(unittest.TestCase):
    def test_complete_rosters_include_players_beyond_first_five_and_all_positions(self):
        base = 'https://api-web.nhle.com/v1'
        payloads = {
            f'{base}/standings/now': {'standings': [
                {'teamAbbrev': {'default': 'NJD'}},
                {'teamAbbrev': {'default': 'COL'}},
                {'teamAbbrev': {'default': 'COL'}},
            ]},
            f'{base}/roster/COL/current': {
                'forwards': [{'id': player_id, 'positionCode': 'C'}
                             for player_id in range(1, 8)],
                'defensemen': [{'id': 8, 'positionCode': 'D'}],
                'goalies': [{'id': 9, 'positionCode': 'G'}],
            },
            f'{base}/roster/NJD/current': {
                'forwards': [{'id': 10, 'positionCode': 'RW'}],
                'defensemen': [],
                'goalies': [],
            },
        }

        def get_response(url, timeout):
            self.assertEqual(timeout, 30)
            return Mock(json=Mock(return_value=payloads[url]))

        with patch.object(mx.requests, 'get', side_effect=get_response) as get:
            result = mx.get_nhl_players()

        expected = pd.DataFrame(
            [(player_id, 'COL', 'C') for player_id in range(1, 8)]
            + [(8, 'COL', 'D'), (9, 'COL', 'G'), (10, 'NJD', 'RW')],
            columns=['player_id', 'team_abbrev', 'position'],
        )
        pd.testing.assert_frame_equal(result, expected)
        self.assertEqual(get.call_count, 3)


if __name__ == '__main__':
    unittest.main()
