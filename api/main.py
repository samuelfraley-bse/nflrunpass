# api/main.py
from pathlib import Path
import json
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"

app = FastAPI()

# Load artifacts once at startup
model = joblib.load(MODELS_DIR / "log_reg_model.pkl")
scaler = joblib.load(MODELS_DIR / "scaler.pkl")
with (MODELS_DIR / "feature_cols.json").open() as f:
    FEATURE_COLS = json.load(f)


class PlayRequest(BaseModel):
    down: int
    ydstogo: int
    yardline_100: float
    offense_score: int
    defense_score: int
    qtr: int
    seconds_remaining_half: int
    shotgun: bool
    no_huddle: bool
    is_home_offense: bool


@app.post("/predict")
def predict(req: PlayRequest):
    score_diff = req.offense_score - req.defense_score

    data = {
        "down": req.down,
        "ydstogo": req.ydstogo,
        "yardline_100": req.yardline_100,
        "game_seconds_remaining": req.seconds_remaining_half,
        "is_red_zone": 1 if req.yardline_100 <= 20 else 0,
        "is_goal_to_go": 1 if req.yardline_100 <= 10 else 0,
        "short_ydstogo": 1 if req.ydstogo <= 3 else 0,
        "medium_ydstogo": 1 if 4 <= req.ydstogo <= 7 else 0,
        "long_ydstogo": 1 if req.ydstogo >= 8 else 0,
        "shotgun": int(req.shotgun),
        "no_huddle": int(req.no_huddle),
        "score_differential": score_diff,
        "is_trailing": 1 if score_diff < 0 else 0,
        "is_tied": 1 if score_diff == 0 else 0,
        "is_leading": 1 if score_diff > 0 else 0,
        "is_fourth_qtr": 1 if req.qtr == 4 else 0,
        "late_half": 1 if req.seconds_remaining_half < 120 else 0,
        "is_home_offense": int(req.is_home_offense),
    }

    df = pd.DataFrame([data])
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0
    df = df[FEATURE_COLS]

    X_scaled = scaler.transform(df)
    prob_pass = float(model.predict_proba(X_scaled)[0, 1])
    prob_run = 1.0 - prob_pass
    pred_label = "PASS" if prob_pass >= 0.5 else "RUN"

    return {
        "prediction": pred_label,
        "prob_pass": prob_pass,
        "prob_run": prob_run,
    }
