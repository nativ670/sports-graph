# sportsgraph
Real-time player tracking -> live player graph -> tactical metrics (soccer).

## Architecture rules
- vision/ and analytics/ communicate only through the tracking table
  (frame, track_id, team, x_m, y_m). Neither imports the other.
- No video, weights, or datasets in git. Data lives in data/ (gitignored).

## Commands
- Install: `uv sync`   Test: `uv run pytest`
- Lint: `uv run ruff check . && uv run ruff format .`

## Working rules
- Propose a plan before editing more than 2 files.
- Never commit to main. Never force-push. Ask before adding a dependency.
- Conventional commits: feat:, fix:, docs:, test:, chore:

## Tracking table contract
- Players table: frame (int, >= 0), track_id (int, unique per match), team ("home" | "away"), x_m, y_m (float, non-null); (frame, track_id) is unique
- At most 11 rows per team per frame
- Coordinates: meters, origin at pitch center, x along the length, y up
- Match metadata, stored beside the tables: fps, pitch_length_m, pitch_width_m (pitches vary; only Metrica-specific code may hard-code 105 x 68)
- Positions may fall slightly outside the pitch; bounds checks use the metadata plus a margin
- No row for a player means not observed in that frame; never impute inside the contract layer (gap filling is an explicit analytics step)
- Ball: separate table (frame, x_m, y_m), same coordinates, at most one row per frame
- Attacking direction is not normalized in the table; use a helper
- Adapters convert each data source into this contract; validate with the Pandera schema at every module boundary
