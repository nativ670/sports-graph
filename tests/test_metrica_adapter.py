from pathlib import Path

from sportsgraph.adapters.metrica import load_metrica_tracking


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
