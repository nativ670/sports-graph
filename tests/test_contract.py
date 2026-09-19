"""Tests for the tracking-table contract: the players table and the ball table.

These tests are the executable version of the contract section in AGENTS.md.

Every rejection test is pinned to a reason: it expects ContractError AND a keyword
in the message's label (the text before the first colon). Matching only the label
means a rule cannot pass because the raw library details happen to mention the same
word. The keywords are listed in KEYWORDS below, so error labels must contain them.
"""

import pandas as pd
import pytest

from sportsgraph.contract import (
    OUT_OF_BOUNDS_MARGIN_M,
    ContractError,
    MatchMeta,
    validate_ball,
    validate_players,
)

# --------------------------------------------------------------------------- #
# Setup: match metadata, limits, and small table builders
# --------------------------------------------------------------------------- #

META = MatchMeta(fps=25.0, pitch_length_m=105.0, pitch_width_m=68.0)
SMALL_META = MatchMeta(fps=25.0, pitch_length_m=90.0, pitch_width_m=45.0)

# Largest valid |x| and |y| on META: half the pitch plus the tolerated margin.
MAX_X = META.pitch_length_m / 2 + OUT_OF_BOUNDS_MARGIN_M
MAX_Y = META.pitch_width_m / 2 + OUT_OF_BOUNDS_MARGIN_M
EPS = 0.01  # "just beyond" a limit


def kw(word):
    """Match `word` only in the message's label, the text before the first colon."""
    return f"(?i)^[^:]*{word}"


KEYWORDS = {
    "duplicate": kw("duplicate"),
    "bounds": kw("bounds"),
    "null": kw("null"),
    "negative": kw("negative"),
    "team": kw("unknown team"),
    "too_many": kw("more than 11"),
    "column": kw("column"),
    "dtype": kw("dtype"),
}


def make_players(rows):
    """rows are (frame, track_id, team, x_m, y_m) tuples."""
    return pd.DataFrame(rows, columns=["frame", "track_id", "team", "x_m", "y_m"])


def make_ball(rows):
    """rows are (frame, x_m, y_m) tuples."""
    return pd.DataFrame(rows, columns=["frame", "x_m", "y_m"])


def valid_players():
    """3 frames, 2 home + 2 away players, positions on both sides of the origin.

    The same track_id appears in every frame, and several players share each frame.
    Both are normal, so this table must be valid.
    """
    rows = []
    for frame in range(3):
        rows += [
            (frame, 1, "home", -30.0 + frame, 10.0),
            (frame, 2, "home", 40.0, -20.0),
            (frame, 101, "away", 5.0, 0.0),
            (frame, 102, "away", -45.0, 25.0),
        ]
    return make_players(rows)


def valid_ball():
    return make_ball([(0, 0.0, 0.0), (1, 12.5, -3.0), (2, -20.0, 30.0)])


def with_value(df, row, column, value):
    """Return a copy of df with one cell changed."""
    df = df.copy()
    df.loc[row, column] = value
    return df


def with_null(df, column):
    """Return a copy of df with a null in row 0 of `column` (ints become floats first)."""
    df = df.copy()
    if column != "team":
        df[column] = df[column].astype("float64")
    df.loc[0, column] = None
    return df


def test_float32_coordinates_are_rejected():
    # The contract is float64; adapters must cast explicitly.
    df = valid_players().astype({"x_m": "float32"})
    with pytest.raises(ContractError, match=KEYWORDS["dtype"]):
        validate_players(df, META)


# --------------------------------------------------------------------------- #
# Players: valid tables
# --------------------------------------------------------------------------- #


class TestValidPlayers:
    def test_valid_table_passes_and_is_returned_unchanged(self):
        result = validate_players(valid_players(), META)
        pd.testing.assert_frame_equal(result, valid_players())

    def test_empty_table_with_correct_columns_is_valid(self):
        # A deliberate decision: "no players yet" is not a contract violation.
        empty = valid_players().iloc[0:0]
        assert len(validate_players(empty, META)) == 0

    def test_extra_columns_are_allowed(self):
        # A deliberate decision: the detector may later add columns such as confidence.
        validate_players(valid_players().assign(confidence=0.9), META)

    def test_eleven_players_for_one_team_in_a_frame_is_allowed(self):
        rows = [(0, i, "home", float(i), 0.0) for i in range(1, 12)]
        validate_players(make_players(rows), META)

    @pytest.mark.parametrize(
        "x, y", [(MAX_X, 0.0), (-MAX_X, 0.0), (0.0, MAX_Y), (0.0, -MAX_Y)]
    )
    def test_position_exactly_at_the_margin_is_allowed(self, x, y):
        df = valid_players()
        df.loc[0, ["x_m", "y_m"]] = [x, y]
        validate_players(df, META)

    def test_bounds_depend_on_the_pitch_metadata(self):
        # The same table is valid on a big pitch and invalid on a small one.
        x = SMALL_META.pitch_length_m / 2 + OUT_OF_BOUNDS_MARGIN_M + 1.0
        df = with_value(valid_players(), 0, "x_m", x)
        validate_players(df, META)
        with pytest.raises(ContractError, match=KEYWORDS["bounds"]):
            validate_players(df, SMALL_META)


# --------------------------------------------------------------------------- #
# Players: rejected tables
# --------------------------------------------------------------------------- #

INVALID_PLAYERS = {
    "unknown_team": (lambda: with_value(valid_players(), 0, "team", "referee"), "team"),
    "team_wrong_case": (lambda: with_value(valid_players(), 0, "team", "Home"), "team"),
    "negative_frame": (lambda: with_value(valid_players(), 0, "frame", -1), "negative"),
    "duplicate_frame_and_track_id": (
        lambda: pd.concat(
            [valid_players(), valid_players().iloc[[0]]], ignore_index=True
        ),
        "duplicate",
    ),
    "x_beyond_margin": (
        lambda: with_value(valid_players(), 0, "x_m", MAX_X + EPS),
        "bounds",
    ),
    "x_beyond_margin_negative": (
        lambda: with_value(valid_players(), 0, "x_m", -MAX_X - EPS),
        "bounds",
    ),
    "y_beyond_margin": (
        lambda: with_value(valid_players(), 0, "y_m", MAX_Y + EPS),
        "bounds",
    ),
    "y_beyond_margin_negative": (
        lambda: with_value(valid_players(), 0, "y_m", -MAX_Y - EPS),
        "bounds",
    ),
    "twelve_home_players_in_one_frame": (
        lambda: make_players([(0, i, "home", float(i), 0.0) for i in range(1, 13)]),
        "too_many",
    ),
    "null_team": (lambda: with_null(valid_players(), "team"), "null"),
    "null_x": (lambda: with_null(valid_players(), "x_m"), "null"),
    "null_y": (lambda: with_null(valid_players(), "y_m"), "null"),
    "null_frame": (lambda: with_null(valid_players(), "frame"), "null"),
    "null_track_id": (lambda: with_null(valid_players(), "track_id"), "null"),
    "missing_column": (lambda: valid_players().drop(columns=["team"]), "column"),
    "renamed_column": (lambda: valid_players().rename(columns={"x_m": "x"}), "column"),
    # A float frame WITHOUT nulls must be a dtype error, not a null error.
    "frame_not_integer": (
        lambda: valid_players().astype({"frame": "float64"}),
        "dtype",
    ),
    "x_as_text": (lambda: valid_players().astype({"x_m": "str"}), "dtype"),
}


@pytest.mark.parametrize(
    "build, reason", list(INVALID_PLAYERS.values()), ids=list(INVALID_PLAYERS)
)
def test_invalid_players_are_rejected_for_the_right_reason(build, reason):
    with pytest.raises(ContractError, match=KEYWORDS[reason]):
        validate_players(build(), META)


def test_invalid_players_are_not_repaired_in_place():
    # The contract layer must never impute: a NaN stays a NaN, and the call fails.
    df = with_null(valid_players(), "x_m")
    with pytest.raises(ContractError):
        validate_players(df, META)
    assert pd.isna(df.loc[0, "x_m"])


# --------------------------------------------------------------------------- #
# Ball: valid tables
# --------------------------------------------------------------------------- #


class TestValidBall:
    def test_valid_table_passes_and_is_returned_unchanged(self):
        result = validate_ball(valid_ball(), META)
        pd.testing.assert_frame_equal(result, valid_ball())

    def test_frames_without_a_ball_row_are_allowed(self):
        # No row means "not observed", exactly as for players.
        validate_ball(make_ball([(0, 0.0, 0.0), (5, 1.0, 1.0), (9, 2.0, 2.0)]), META)

    def test_empty_table_with_correct_columns_is_valid(self):
        assert len(validate_ball(valid_ball().iloc[0:0], META)) == 0

    @pytest.mark.parametrize(
        "x, y", [(MAX_X, 0.0), (-MAX_X, 0.0), (0.0, MAX_Y), (0.0, -MAX_Y)]
    )
    def test_position_exactly_at_the_margin_is_allowed(self, x, y):
        df = valid_ball()
        df.loc[0, ["x_m", "y_m"]] = [x, y]
        validate_ball(df, META)


# --------------------------------------------------------------------------- #
# Ball: rejected tables
# --------------------------------------------------------------------------- #

INVALID_BALL = {
    "duplicate_frame": (
        lambda: pd.concat([valid_ball(), valid_ball().iloc[[0]]], ignore_index=True),
        "duplicate",
    ),
    "negative_frame": (lambda: with_value(valid_ball(), 0, "frame", -1), "negative"),
    "x_beyond_margin": (
        lambda: with_value(valid_ball(), 0, "x_m", MAX_X + EPS),
        "bounds",
    ),
    "y_beyond_margin": (
        lambda: with_value(valid_ball(), 0, "y_m", -MAX_Y - EPS),
        "bounds",
    ),
    "null_x": (lambda: with_null(valid_ball(), "x_m"), "null"),
    "null_y": (lambda: with_null(valid_ball(), "y_m"), "null"),
    "null_frame": (lambda: with_null(valid_ball(), "frame"), "null"),
    "missing_column": (lambda: valid_ball().drop(columns=["y_m"]), "column"),
    "frame_not_integer": (lambda: valid_ball().astype({"frame": "float64"}), "dtype"),
    "x_as_text": (lambda: valid_ball().astype({"x_m": "str"}), "dtype"),
}


@pytest.mark.parametrize(
    "build, reason", list(INVALID_BALL.values()), ids=list(INVALID_BALL)
)
def test_invalid_ball_is_rejected_for_the_right_reason(build, reason):
    with pytest.raises(ContractError, match=KEYWORDS[reason]):
        validate_ball(build(), META)


# --------------------------------------------------------------------------- #
# The exception type
# --------------------------------------------------------------------------- #


def test_contract_error_is_a_value_error():
    # Callers who only catch ValueError still catch contract violations.
    assert issubclass(ContractError, ValueError)
