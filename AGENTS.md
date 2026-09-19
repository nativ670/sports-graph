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
