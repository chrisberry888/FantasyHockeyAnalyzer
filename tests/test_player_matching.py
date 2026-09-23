import unittest

import pandas as pd

import my_module_v2 as mx


def moneypuck_player(player_id, name, position, team):
    return {
        'playerId': player_id, 'season': 2023, 'name': name,
        'position': position, 'team': team, 'games_played': 82,
        'all_I_F_shotsOnGoal': 278, 'all_I_F_hits': 101,
    }


def rotowire_player(name, position, team, assists):
    return {
        'Player Name': name, 'Pos': position, 'Team': team,
        'Games': 81, 'SOG': 277, 'Hits': 100, '+/-': 1,
        'A': assists, 'G.1': 2, 'A.1': 3, 'G.2': 4, 'A.2': 5,
    }


class PlayerMatchingTests(unittest.TestCase):
    def test_stat_disagreements_preserve_player_and_moneypuck_statistics(self):
        mp = pd.DataFrame([moneypuck_player(8471675, 'Sidney Crosby', 'C', 'PIT')])
        rw = mx.preformat_rotowire_df(pd.DataFrame([
            rotowire_player('  SIDNEY   CROSBY  ', 'C', 'PIT', 52)
        ]))
        result, unmatched = mx.merge_dataframes(mp, rw, return_unmatched=True)

        self.assertEqual(result['playerId'].tolist(), [8471675])
        self.assertEqual(result.loc[0, 'games_played'], 82)
        self.assertEqual(result.loc[0, 'all_I_F_shotsOnGoal'], 278)
        self.assertEqual(result.loc[0, 'all_I_F_hits'], 101)
        self.assertEqual(result.loc[0, 'Assists'], 52)
        self.assertTrue(unmatched.empty)

    def test_same_name_players_use_position_groups_and_ignore_team_differences(self):
        mp = pd.DataFrame([
            moneypuck_player(8480012, 'Elias Pettersson', 'C', 'VAN'),
            moneypuck_player(8483678, 'Elias Pettersson', 'D', 'VAN'),
            moneypuck_player(8478427, 'Sebastian Aho', 'C', 'CAR'),
            moneypuck_player(8480222, 'Sebastian Aho', 'D', 'NYI'),
        ])
        rw = mx.preformat_rotowire_df(pd.DataFrame([
            rotowire_player('Elias Pettersson', 'D', 'VAN', 3),
            rotowire_player('Sebastian Aho', 'D', 'PIT', 6),
            rotowire_player('Elias Pettersson', 'C', 'VAN', 30),
            rotowire_player('Sebastian Aho', 'LW', 'CAR', 31),
        ]))
        result, unmatched = mx.merge_dataframes(mp, rw, return_unmatched=True)

        self.assertEqual(len(result), 4)
        self.assertEqual(result.set_index('playerId')['Assists'].to_dict(), {
            8480012: 30, 8483678: 3, 8478427: 31, 8480222: 6,
        })
        self.assertTrue(unmatched.empty)

    def test_verified_aliases_and_accents_match_without_changing_output_names(self):
        mp = pd.DataFrame([
            moneypuck_player(8475690, 'Christopher Tanev', 'D', 'VAN'),
            moneypuck_player(8482116, 'Tim Sttzle', 'C', 'OTT'),
        ])
        rw = mx.preformat_rotowire_df(pd.DataFrame([
            rotowire_player('Chris Tanev', 'D', 'TOR', 10),
            rotowire_player('Tim Stützle', 'C', 'OTT', 20),
        ]))
        result, unmatched = mx.merge_dataframes(mp, rw, return_unmatched=True)

        self.assertEqual(result['name'].tolist(), ['Christopher Tanev', 'Tim Sttzle'])
        self.assertTrue(unmatched.empty)
        self.assertIsInstance(mx.merge_dataframes(mp, rw), pd.DataFrame)

    def test_missing_and_ambiguous_players_are_reported_from_both_sources(self):
        mp = pd.DataFrame([
            moneypuck_player(1, 'Alex Smith', 'C', 'AAA'),
            moneypuck_player(2, 'Alex Smith', 'LW', 'BBB'),
            moneypuck_player(3, 'MoneyPuck Only', 'D', 'CCC'),
        ])
        rw = mx.preformat_rotowire_df(pd.DataFrame([
            rotowire_player('Alex Smith', 'C', 'AAA', 10),
            rotowire_player('Alex Smith', 'RW', 'BBB', 20),
            rotowire_player('RotoWire Only', 'D', 'DDD', 30),
        ]))
        result, unmatched = mx.merge_dataframes(mp, rw, return_unmatched=True)

        self.assertTrue(result.empty)
        self.assertEqual(len(unmatched), 6)
        self.assertEqual(unmatched['season'].tolist(), [2023] * 6)
        self.assertEqual(unmatched['reason'].value_counts().to_dict(), {
            'ambiguous_name_or_position': 4, 'no_matching_name': 2,
        })
        self.assertEqual(unmatched.loc[unmatched['source'].eq('moneypuck'), 'playerId'].tolist(), [1, 2, 3])
        self.assertTrue(unmatched.loc[unmatched['source'].eq('rotowire'), 'playerId'].isna().all())


if __name__ == '__main__':
    unittest.main()
