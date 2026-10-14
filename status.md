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
- [x] **Step 6: Metrics vs. baselines** — Completed.
  - Metrics: `n_nodes`, `n_edges`, `density`, `mean_degree`, `largest_component_frac`, `algebraic_connectivity`. Geometric baseline: mean pairwise distance.
  - Null-model evaluation on 10,000 samples comparing real vs. shuffled surrogates via `scipy.stats.mannwhitneyu`. Statistically significant team coordination detected.
  - Goalkeeper identification via within-period positional variance (`analytics/goalkeeper.py`), separating half-time side swap. Robustness check evaluated on 10 outfield players. *Merged in PR #16.*
- [x] **Step 7: Wrap-up** — `docs/phase-1 findings.md` written and merged with statistical caveats, findings, and kickoff plots. *Merged in PR #17 & #18.*

**Phase 1 Complete & Merged.**

---

## Phase 2 — Detection

- [x] **Step 1: Baseline evaluation framework & dataset integration**
  - Building blocks: IoU (`iou.py`), greedy bipartite matching (`evaluate.py`), mAP / AP calculation, time-based split (`split.py`), YOLO label parsing (`yolo_labels.py`).
  - Integrated 4-class soccer dataset (`adapters/unpack_roboflow.py`): 372 frames (298 train, 49 valid, 25 test), 19.1 MB disk footprint.
  - Classes: `ball`, `goalkeeper`, `player`, `referee`.
  - Baseline benchmark established on off-the-shelf COCO YOLO (`yolo11n.pt` / `yolov8n.pt`):
    - Macro mAP: 0.1767 (player: 0.5542, ball: 0.1053, goalkeeper: 0.0138, referee: 0.0334).
- [x] **Step 2: Fine-tuning & Regularization (Role Specialist)** — Completed.
  - Training pipeline built: `src/sportsgraph/detection/train.py` with transfer learning, regularizing augmentations (Mosaic 1.0, MixUp 0.15, HSV jitter, flips).
  - Preserved Roboflow native `train/`, `valid/`, and `test/` splits to guarantee zero data leakage.
  - Remote execution runner: `notebooks/kaggle_train.ipynb` executed on free Kaggle T4 GPU.
  - Ingested weights: `models/fine_tuned/weights/best.pt` (5.66 MB). (Previous v8 weights preserved as `best8.pt`).
  - Empirical verification via `run_baseline_eval.py`:
    - Native `test` set (25 frames): Player **0.9883 AP**, Goalkeeper **0.9237 AP**, Referee **0.9498 AP**, Ball **0.5179 AP** (macro mAP: **0.8449**).
    - Native `valid` set (49 frames): Player **0.9886 AP**, Goalkeeper **0.9209 AP**, Referee **0.9380 AP**, Ball **0.6479 AP** (macro mAP: **0.8738**).
  - Exceeded target benchmark (>0.85 mAP) on held-out match frames.
- [x] **Step 3: Dual-Detector Architecture & Ball Specialist** — Completed.
  - Adapter created: `src/sportsgraph/adapters/ball_dataset.py` with unit tests to isolate ball labels and generate background frames.
  - Training runner executed on Kaggle: `notebooks/kaggle_train_ball.ipynb` trained on 4,254 combined frames (3,956 ball `train` + 298 player `train` ball-filtered).
  - Ingested weights: `models/ball_specialist/weights/best.pt` (5.66 MB).
  - Empirical verification via `run_baseline_eval.py`:
    - Independent 500-frame test set (`data/raw/ball_test/`): **0.9301 AP** (vs 0.5211 from 4-class model, +78.5% relative gain).
    - Match `test` set (25 frames): **0.6859 AP** (vs 0.5179 from 4-class model, +32.4% relative gain).
- [x] **Step 4: Upgrade to YOLO11 Architecture** — Completed.
  - Upgraded base models across codebase, evaluators, and Kaggle runners to Ultralytics YOLO11 (`yolo11n.pt`).
  - Weight sizes reduced from 6.66 MB to 5.66 MB (-15% footprint).
  - Validated all 4 benchmarks with zero leakage.

**Phase 2 Complete & Verified.** Ready for Phase 3 (Tracking and Homography).

## Later phases (unstarted)
- Phase 3 — Tracking & homography (fixed camera first; per-frame calibration via pitch keypoints noted as a harder follow-on, not a new phase)
  - Tracking: temporal state estimation via Kalman filter, association via Hungarian algorithm.
  - Homography: projection of image bounding boxes into metric pitch coordinates `(x_m, y_m)` to produce the tracking table contract.
- Phase 4 — Streaming (replay-as-stream, latency budget, live dashboard)
- Phase 5 — LLM layer (MCP server over metrics; interactive natural-language queries, not just a scheduled narrator)
- Phase 6 — Hardening (Dockerfile, model card, release tag)
- Possible future work (post–Phase 6, not scheduled): pitch control / space-control models, player trajectory prediction ("ghosting")
- Analytics enhancements: centrality beyond degree (betweenness, closeness, eigenvector centrality) once tracking table is populated from vision.
- Player- and event-aware tactical analysis (individual matchups, possession, goals, defensive-line breaks) —
          needs the ball loader and event data; not just a graph-shape question like the current metrics.
- Final Plan - Dashboard containing home team data and metrics, and scouting system of away teams. Possibly be used by analytics teams        using broadcast data or team's footage.

## Decisions worth remembering
- Sport: soccer. Compared against basketball, hockey, volleyball, American football — soccer is the only one with both open tracking data and open detection footage.
- Contract stays sport-agnostic where possible; pitch dimensions are metadata, not constants.
- No paid tools. Heavy compute (training, large video) goes through free hosted environments (Colab/Kaggle) when Phase 2 starts, not the local machine.
- Large files never go into git — `.gitignore` + (planned, once video enters the picture in Phase 2) DVC with a free remote.
