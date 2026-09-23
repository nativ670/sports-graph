from pathlib import Path

import pytest

from sportsgraph.adapters.metrica import load_metrica_match, load_metrica_tracking
from sportsgraph.contract.errors import ContractError


def test_metrica_drops_both_substitution_duplicates(tmp_path: Path):
    """
    Test that when two players share the exact same coordinates in a single frame
    (as happens during Metrica substitutions, e.g., frame 46706), BOTH rows are dropped.
    """
    csv_data = """,,,Home,,Home,,,
,,,1,,12,,,
Period,Frame,Time [s],Player1,,Player12,,Ball,
2,46706,1868.24,0.49667,0.98615,0.49667,0.98615,NaN,NaN
"""
    file_path = tmp_path / "test_frame.csv"
    file_path.write_text(csv_data)

    df = load_metrica_tracking(file_path, "home")

    # Both players have identical coordinates, so both rows should be dropped (keep=False)
    # resulting in 0 rows surviving.
    assert len(df) == 0


def test_load_metrica_match_raises_on_track_id_collision(tmp_path: Path):
    """
    Test that loading a match where home and away have overlapping track_ids
    in the same frame raises a ContractError from the final validation step.
    """
    home_csv = """,,,Home,,,
,,,1,,,
Period,Frame,Time [s],Player1,,Ball,
1,1,0.04,0.5,0.5,NaN,NaN
"""
    away_csv = """,,,Away,,,
,,,1,,,
Period,Frame,Time [s],Player1,,Ball,
1,1,0.04,0.2,0.2,NaN,NaN
"""
    home_file = tmp_path / "home.csv"
    home_file.write_text(home_csv)

    away_file = tmp_path / "away.csv"
    away_file.write_text(away_csv)

    with pytest.raises(ContractError, match="(?i)duplicate"):
        load_metrica_match(home_file, away_file)
