import functools

import pandas as pd
import pandera.pandas as pa
from pandera.errors import SchemaErrorReason
from pandera.pandas import Check, Column, DataFrameSchema

from sportsgraph.contract.errors import ContractError
from sportsgraph.contract.meta import MatchMeta

OUT_OF_BOUNDS_MARGIN_M = 5.0


def _to_contract_error(e: pa.errors.SchemaError) -> ContractError:
    if e.reason_code == SchemaErrorReason.COLUMN_NOT_IN_DATAFRAME:
        kind = "missing column"
    elif e.reason_code == SchemaErrorReason.SERIES_CONTAINS_NULLS:
        kind = "null values"
    elif e.reason_code == SchemaErrorReason.WRONG_DATATYPE:
        kind = "wrong dtype"
    else:
        kind = getattr(e.check, "name", None) or "contract violation"
    return ContractError(f"{kind}: {e}")


def handle_pandera_errors(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except pa.errors.SchemaError as e:
            raise _to_contract_error(e) from e

    return wrapper


@handle_pandera_errors
def validate_players(df: pd.DataFrame, meta: MatchMeta) -> pd.DataFrame:
    max_x = meta.pitch_length_m / 2 + OUT_OF_BOUNDS_MARGIN_M
    max_y = meta.pitch_width_m / 2 + OUT_OF_BOUNDS_MARGIN_M

    schema = DataFrameSchema(
        {
            "frame": Column(int, checks=[Check(lambda s: s >= 0, name="negative")], nullable=False),
            "track_id": Column(int, nullable=False),
            "team": Column(
                str,
                checks=[Check.isin(["home", "away"], name="unknown team")],
                nullable=False,
            ),
            "x_m": Column(
                float,
                checks=[Check(lambda s: s.abs() <= max_x, name="bounds")],
                nullable=False,
            ),
            "y_m": Column(
                float,
                checks=[Check(lambda s: s.abs() <= max_y, name="bounds")],
                nullable=False,
            ),
        },
        checks=[
            Check(
                lambda df: ~df.duplicated(subset=["frame", "track_id"], keep=False),
                name="duplicate",
            ),
            Check(
                lambda df: df.groupby(["frame", "team"])["frame"].transform("size") <= 11,
                name="more than 11",
            ),
        ],
    )

    return schema.validate(df)


@handle_pandera_errors
def validate_ball(df: pd.DataFrame, meta: MatchMeta) -> pd.DataFrame:
    max_x = meta.pitch_length_m / 2 + OUT_OF_BOUNDS_MARGIN_M
    max_y = meta.pitch_width_m / 2 + OUT_OF_BOUNDS_MARGIN_M

    schema = DataFrameSchema(
        {
            "frame": Column(int, checks=[Check(lambda s: s >= 0, name="negative")], nullable=False),
            "x_m": Column(
                float,
                checks=[Check(lambda s: s.abs() <= max_x, name="bounds")],
                nullable=False,
            ),
            "y_m": Column(
                float,
                checks=[Check(lambda s: s.abs() <= max_y, name="bounds")],
                nullable=False,
            ),
        },
        checks=[
            Check(
                lambda df: ~df.duplicated(subset=["frame"], keep=False),
                name="duplicate",
            )
        ],
    )

    return schema.validate(df)
