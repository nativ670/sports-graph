# Project status

## Phase 0 — Tooling
- [x] Repo skeleton, uv, ruff, pytest, pre-commit, GitHub Actions CI
- [x] Branch protection on `main` (PR + passing `ci / test` required)
- [x] `AGENTS.md` (shared agent instructions), imported by `CLAUDE.md` and `GEMINI.md`
- [x] README (architecture diagram, getting started, roadmap)

**Merged.**

## Phase 1 — Analytics on open tracking data

- [x] **Step 1: Tracking table contract** — Pandera schema (`contract/schema.py`), `ContractError`, `MatchMeta` (carries `fps`, `pitch_length_m`, `pitch_width_m` — pitch size is data, not hard-coded). Contract: players table (`frame, track_id, team, x_m, y_m`), ball as a separate table, no row = not observed, center-origin meters, y up. *Merged.*
- [x] **Step 2: Metrica download script** — `adapters/metrica_download.py`, downloads Game 1/2 CSVs into `data/raw/metrica/` (git-ignored), idempotent, network tests marked separately from the default `pytest` run. README Data section credits Metrica Sports. *Merged.* Ruff/pytest config restored in a follow-up PR. *Merged.*
- [x] **Step 3: Metrica adapter** — `adapters/metrica.py`. *Merged.*
  - `load_metrica_tracking(filepath, team, meta=None)`: wide-to-long reshape, unit conversion (0–1 → meters, center origin, y-flip), drops NaN (unobserved) rows, validates against the contract.
  - Handles a real data issue: 3 frames (of 145,006) where a substitution briefly produces 12 players with duplicate coordinates — both rows dropped. Covered by a unit test.
  - `load_metrica_match(home_path, away_path)` — combines both teams into one validated table; checked against real Game 1 data (3.18M rows).
  - `MatchMeta.period_boundary_frames` added, derived from the raw `Period` column via `_derive_period_boundaries()`. Needed because frame numbers don't reset at half-time but also don't reflect the real elapsed time across the break — anything computing time deltas has to treat that one frame specially.
  - `analytics/quality.py::flag_implausible_speed()` — flags per-player frame-to-frame speed above a physically implausible threshold (12 m/s), excluding the period-boundary frame. Found 433 genuine flagged transitions in Game 1, scattered across 427 frames — real, isolated tracking glitches, unrelated to the half-time jump.
  - Ball loader — separate table, deliberately deferred; nothing downstream needs it yet.
- [x] **Step 4: Pitch drawing** — `analytics/pitch.py`: `draw_pitch()` (regulation dimensions: 16.5/40.32m penalty area, 9.15m circle, correct penalty-arc angle), `plot_trail()`. Verified against real data: kickoff formation looks correct, and each team's mean x position flips sign at half-time as expected. *Merged.*
- [x] **Step 5: Graph builder** — `analytics/graph.py::build_frame_graph()`: Delaunay and radius (15m) edge rules, handles 0/1/2/collinear-point edge cases without crashing. `scripts/compare_graphs.py` renders a side-by-side comparison (`docs/graph_comparison.png`), referenced from the README. Confirmed real difference: Delaunay always connects the whole team (e.g. reaches the goalkeeper); the radius graph can leave players isolated. *Merged.*
- [ ] **Step 6: Metrics vs. baselines** — Up next.
  - Metrics (per frame, per team, from `build_frame_graph` output): `n_nodes`, `n_edges`, `density`, `mean_degree`, `largest_component_frac`, `algebraic_connectivity`. Geometric baseline computed independently of the graph: mean pairwise distance (or convex hull area).
  - Null-model design: considered two shuffles — across frames (replace a player's position with their own position from a random other frame, tests whether a metric depends on real-time coordination) vs. within-frame identity shuffle (swap which player is at which position, positions held fixed, tests whether a metric depends on who's where vs. just the shape). Concluded the within-frame identity shuffle is a no-op for every metric currently planned, since none of them use player identity, only positions — so only the across-frames shuffle is useful for this step.
  - Plan: `analytics/metrics.py` (`compute_frame_metrics`, `compute_geometric_baseline`), a surrogate-frame builder (`build_shuffled_frame`, using an explicit seeded `np.random.Generator` for reproducibility), and a comparison runner (`compare_real_vs_shuffled`) producing a long table of real vs. shuffled metric values, compared via `scipy.stats.mannwhitneyu` per metric. Framed as "detectably different from chance," not as effect size or causal claims.
  - Explicitly out of scope for this step (noted for later): player-identity-aware analysis (e.g. a specific fast winger vs. a specific defender), ball possession, goals, and moments where a defensive line breaks — all real and important, but need player identity and ball/event data neither of which any current metric uses. Revisit once the ball loader and event data are in place.
- [ ] **Step 7: Wrap-up** — update `AGENTS.md` contract section, tick Phase 1 in the README, write `docs/phase1-findings.md` (should include the data-quality findings below).

### Data-quality findings so far (for the Step 7 write-up)
- Substitution frames produce brief duplicate-coordinate rows (handled in the adapter).
- The half-time period boundary is a real elapsed-time gap the frame counter doesn't represent (handled via `period_boundary_frames`).
- 433 isolated, genuine implausible-speed transitions exist across the match — real tracking noise to account for before trusting per-frame metrics near those frames.
- One anomalous event (frame 71269–71278): two players from different teams (home #3, away #22) land on identical off-pitch coordinates at the exact half-time frame and drift together for several frames before separating — likely a shared real-world object (staff, sub) picked up independently by each team's tracker. Rare (10 frames total) and self-correcting; noted, not specially handled.
**Definition of done for Phase 1:** metrics compute on a full match, and baseline comparisons show which ones are meaningful.

## Later phases (unstarted)
- Phase 2 — Detection (pretrained detector, fine-tuning, match-level split eval)
- Phase 3 — Tracking & homography (fixed camera first; per-frame calibration via pitch keypoints noted as a harder follow-on, not a new phase)
- Phase 4 — Streaming (replay-as-stream, latency budget, live dashboard)
- Phase 5 — LLM layer (MCP server over metrics; interactive natural-language queries, not just a scheduled narrator)
- Phase 6 — Hardening (Dockerfile, model card, release tag)
- Possible future work (post–Phase 6, not scheduled): pitch control / space-control models, player trajectory prediction ("ghosting")
- Player- and event-aware tactical analysis (individual matchups, possession, goals, defensive-line breaks) — needs the ball loader and event data; not just a graph-shape question like the current metrics.

## Decisions worth remembering
- Sport: soccer. Compared against basketball, hockey, volleyball, American football — soccer is the only one with both open tracking data and open detection footage.
- Contract stays sport-agnostic where possible; pitch dimensions are metadata, not constants.
- No paid tools. Heavy compute (training, large video) goes through free hosted environments (Colab/Kaggle) when Phase 2 starts, not the local machine.
- Large files never go into git — `.gitignore` + (planned, once video enters the picture in Phase 2) DVC with a free remote.
