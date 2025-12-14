<!-- Repository-specific Copilot instructions for AI coding agents -->

# Copilot Instructions — nflplaycaller

Purpose: give an AI coding agent the minimum, high-value context to be productive in this repo.

- **Big picture**
  - This repo contains a small ML project predicting run vs pass in NFL plays. Major components:
    - `src/nfl_run_pass/` — core library: data loading, preprocessing, feature engineering, model training and pipeline.
    - `run_pipeline.py` — runnable training script. It inserts `src/` on `sys.path` and invokes `nfl_run_pass.pipeline.run_training_pipeline`.
    - `api/main.py` — FastAPI service serving the trained model(s) from `models/` (loads `log_reg_model.pkl`, `scaler.pkl`, and `feature_cols.json`).
    - `streamlit/` — Streamlit demo app using the same features; keeps a copy of `feature_cols.json` for the UI.
    - `webapp/nfl-playcall-ui/` — Next.js frontend (separate JS app) used for deployment.

- **Single source of configuration**
  - `src/nfl_run_pass/config.py` is the canonical project configuration: data paths, feature lists, training hyperparams, and artifact filenames. Tests monkeypatch `CONFIG` frequently — prefer editing `CONFIG` only when intentionally changing defaults.

- **Where artifacts live**
  - Trained artifacts are saved to `models/` (script `run_pipeline.py` writes here by default). Key filenames are defined in `CONFIG.artifacts` and mirrored in `api/main.py`.
  - `models/feature_cols.json` must match the API/Streamlit expected ordering; if you change features update this JSON and re-save models.

- **How to run locally** (assume repo root)
  - Install requirements: `pip install -r requirements.txt`.
  - Run training pipeline: `python run_pipeline.py` (this script ensures `src/` is importable).
  - Run API (development): `uvicorn api.main:app --reload --host 0.0.0.0 --port 8000` and POST to `/predict`.
    - Example request body (JSON): `{ "down":1, "ydstogo":10, "yardline_100":75, "offense_score":14, "defense_score":7, "qtr":2, "seconds_remaining_half":300, "shotgun":false, "no_huddle":false, "is_home_offense":true }`
  - Run Streamlit UI: `streamlit run streamlit/streamlit_app.py`.
  - Web frontend (Next.js): `cd webapp/nfl-playcall-ui && npm install && npm run dev`.

- **Testing & quick checks**
  - Run unit tests: `pytest -q` (tests live in `tests/`; they use `monkeypatch` to override `CONFIG` paths).
  - Linting/formatting not enforced here — follow existing style in `src/` (simple, functional, dataclass-based `CONFIG`).

- **Project-specific patterns & gotchas**
  - `src/` is not a top-level package by default; scripts like `run_pipeline.py` add `src/` to `sys.path`. When running modules directly, ensure `PYTHONPATH` includes `src/` or run via the provided scripts.
  - The model-serving API expects artifacts with specific filenames. If you rename artifacts update `src/nfl_run_pass/config.py` and `api/main.py` accordingly.
  - Tests often rely on `CONFIG` defaults — prefer `monkeypatch.setattr(CONFIG.paths, 'raw_data', csv_path, raising=False)` in tests rather than editing CONFIG globally.

- **When editing features or training**
  - Update feature engineering in `src/nfl_run_pass/features.py` and ensure the saved `feature_cols.json` (artifact) matches the order used by `api/main.py` and Streamlit UI.
  - Re-run `python run_pipeline.py` to regenerate `models/` artifacts after changing features or model code.

- **Key files to inspect before coding**
  - `src/nfl_run_pass/config.py` — configuration and defaults
  - `src/nfl_run_pass/data_loading.py` — raw data ingestion and filtering
  - `src/nfl_run_pass/features.py` — feature-engineering logic used by both training and API
  - `src/nfl_run_pass/pipeline.py` — orchestration of train/validate/save
  - `api/main.py` — model serving; shows exact expected inputs/feature building

If anything above is unclear or you'd like the instructions to emphasize CI/CD, code ownership, or preferred dev commands, tell me which area to expand and I will iterate.
