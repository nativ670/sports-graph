# Project status

## Phase 0 — Tooling
- [x] Repo skeleton, uv, ruff, pytest, pre-commit, GitHub Actions CI
- [x] Branch protection on `main` (PR + passing `ci / test` required)
- [x] `AGENTS.md` (shared agent instructions), imported by `CLAUDE.md` and `GEMINI.md`
- [x] README (architecture diagram, getting started, roadmap)

**Merged.**

## Phase 1 — Analytics on open tracking data

- [x] **Step 1: Tracking table contract** — Pandera schema (`contract/schema.py`), `ContractError`, `MatchMeta` (carries `fps`, `pitch_length_m`, `pitch_width_m` — pitch size is data, not hard-coded), 46 tests. Contract: players table (`frame, track_id, team, x_m, y_m`), ball as a separate table, no row = not observed, center-origin meters, y up. *Merged.*
- [x] **Step 2: Metrica download script** — `adapters/metrica_download.py`, downloads Game 1/2 CSVs into `data/raw/metrica/` (git-ignored), idempotent, network tests marked separately from the default `pytest` run. README Data section credits Metrica Sports. *Merged.* Ruff/pytest config restored in a follow-up PR. *Merged.*
- [ ] **Step 3: Metrica adapter** — `adapters/metrica.py`.
  - [x] `load_metrica_tracking(filepath, team)`: wide-to-long reshape, unit conversion (0–1 → meters, center origin, y-flip), drops NaN (unobserved) rows, validates against the contract.
  - [x] Handles a real data issue found while testing: 3 frames (of 145,006) where a substitution briefly produces 12 players with duplicate coordinates — both rows are dropped. Covered by a unit test with a synthetic fixture.
  - [ ] `load_metrica_match(home_path, away_path)` — combine both teams into one validated table. **In progress.**
  - [ ] Not yet committed/pushed (branch `feat/metrica-adapter` created off `main`, work is local).
  - [ ] Ball loader — separate table, not started; deliberately deferred to its own step.
- [ ] **Step 4: Pitch drawing** — `draw_pitch()`, player trails, visual sanity checks (kickoff positions, half-time side swap). Not started.
- [ ] **Step 5: Graph builder** — proximity and Delaunay edge rules on a sliding window. Not started.
- [ ] **Step 6: Metrics vs. baselines** — density, degree, largest-component fraction, algebraic connectivity; null-model comparisons; framed as hypothesis testing, not causal claims. Not started.
- [ ] **Step 7: Wrap-up** — update `AGENTS.md` contract section, tick Phase 1 in the README, write `docs/phase1-findings.md`. Not started.

**Definition of done for Phase 1:** metrics compute on a full match, and baseline comparisons show which ones are meaningful.

## Later phases (unstarted)
- Phase 2 — Detection (pretrained detector, fine-tuning, match-level split eval)
- Phase 3 — Tracking & homography (fixed camera first; per-frame calibration via pitch keypoints noted as a harder follow-on, not a new phase)
- Phase 4 — Streaming (replay-as-stream, latency budget, live dashboard)
- Phase 5 — LLM layer (MCP server over metrics; interactive natural-language queries, not just a scheduled narrator)
- Phase 6 — Hardening (Dockerfile, model card, release tag)
- Possible future work (post–Phase 6, not scheduled): pitch control / space-control models, player trajectory prediction ("ghosting")

## Decisions worth remembering
- Sport: soccer. Compared against basketball, hockey, volleyball, American football — soccer is the only one with both open tracking data and open detection footage.
- Contract stays sport-agnostic where possible; pitch dimensions are metadata, not constants.
- No paid tools. Heavy compute (training, large video) goes through free hosted environments (Colab/Kaggle) when Phase 2 starts, not the local machine.
- Large files never go into git — `.gitignore` + (planned, once video enters the picture in Phase 2) DVC with a free remote.
