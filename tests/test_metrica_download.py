from sportsgraph.adapters.metrica_download import game_files


def test_game_1_local_names():
    files = game_files(1)
    assert len(files) == 3
    assert set(files.keys()) == {
        "Sample_Game_1_RawEventsData.csv",
        "Sample_Game_1_RawTrackingData_Away_Team.csv",
        "Sample_Game_1_RawTrackingData_Home_Team.csv",
    }


def test_urls_format():
    files = game_files(1)
    for url in files.values():
        assert url.startswith("https://raw.githubusercontent.com")
        assert "/tree/" not in url


def test_game_1_exact_urls():
    files = game_files(1)
    base = "https://raw.githubusercontent.com/metrica-sports/sample-data/master/data/Sample_Game_1"

    assert (
        files["Sample_Game_1_RawEventsData.csv"]
        == f"{base}/Sample_Game_1_RawEventsData.csv"
    )
    assert (
        files["Sample_Game_1_RawTrackingData_Away_Team.csv"]
        == f"{base}/Sample_Game_1_RawTrackingData_Away_Team.csv"
    )
    assert (
        files["Sample_Game_1_RawTrackingData_Home_Team.csv"]
        == f"{base}/Sample_Game_1_RawTrackingData_Home_Team.csv"
    )


def test_game_2_names_and_substitution():
    files = game_files(2)

    # Check that game 2 produces Sample_Game_2_... names
    assert len(files) == 3
    assert set(files.keys()) == {
        "Sample_Game_2_RawEventsData.csv",
        "Sample_Game_2_RawTrackingData_Away_Team.csv",
        "Sample_Game_2_RawTrackingData_Home_Team.csv",
    }

    # Assert that game 2 substitutes its number
    base = "https://raw.githubusercontent.com/metrica-sports/sample-data/master/data/Sample_Game_2"
    assert (
        files["Sample_Game_2_RawEventsData.csv"]
        == f"{base}/Sample_Game_2_RawEventsData.csv"
    )
    assert (
        files["Sample_Game_2_RawTrackingData_Away_Team.csv"]
        == f"{base}/Sample_Game_2_RawTrackingData_Away_Team.csv"
    )
    assert (
        files["Sample_Game_2_RawTrackingData_Home_Team.csv"]
        == f"{base}/Sample_Game_2_RawTrackingData_Home_Team.csv"
    )
