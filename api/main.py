from pathlib import Path
import json

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"

app = FastAPI(
    title="NFL Play Call Prediction API",
    description="Predicts run vs pass plays based on pre-snap game situation",
    version="1.0.0"
)

# CORS
origins = [
    "https://nfl-playcaller.vercel.app",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load artifacts
model = joblib.load(MODELS_DIR / "log_reg_model.pkl")
scaler = joblib.load(MODELS_DIR / "scaler.pkl")
with (MODELS_DIR / "feature_cols.json").open("r") as f:
    FEATURE_COLS = json.load(f)


class PlayRequest(BaseModel):
    """Input schema for play prediction."""
    down: int = Field(..., ge=1, le=4, description="Current down (1-4)")
    ydstogo: int = Field(..., ge=0, le=99, description="Yards to first down")
    yardline_100: float = Field(..., ge=0, le=100, description="Yards to opponent end zone")
    offense_score: int = Field(..., ge=0, description="Offensive team score")
    defense_score: int = Field(..., ge=0, description="Defensive team score")
    qtr: int = Field(..., ge=1, le=4, description="Quarter (1-4)")
    seconds_remaining_half: int = Field(..., ge=0, le=1800, description="Seconds remaining in half")
    shotgun: bool = Field(..., description="Is offense in shotgun formation")
    no_huddle: bool = Field(..., description="Is offense running no-huddle")
    is_home_offense: bool = Field(..., description="Is offensive team the home team")

    class Config:
        schema_extra = {
            "example": {
                "down": 3,
                "ydstogo": 7,
                "yardline_100": 35,
                "offense_score": 14,
                "defense_score": 17,
                "qtr": 4,
                "seconds_remaining_half": 180,
                "shotgun": True,
                "no_huddle": False,
                "is_home_offense": True
            }
        }


class PredictionResponse(BaseModel):
    """Output schema for play prediction."""
    prediction: str = Field(..., description="Predicted play type: RUN or PASS")
    prob_pass: float = Field(..., ge=0, le=1, description="Probability of pass play")
    prob_run: float = Field(..., ge=0, le=1, description="Probability of run play")
    confidence: str = Field(..., description="Confidence level: Low, Medium, or High")
    game_situation: str = Field(..., description="Human-readable game situation")

    class Config:
        schema_extra = {
            "example": {
                "prediction": "PASS",
                "prob_pass": 0.73,
                "prob_run": 0.27,
                "confidence": "High",
                "game_situation": "3rd & 7 at OPP 35, Q4, Down by 3"
            }
        }


def build_feature_df(req: PlayRequest) -> pd.DataFrame:
    """Rebuild feature vector exactly like training."""
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
        
        # Short yardage
        "fourth_and_one": 1 if (req.down == 4 and req.ydstogo == 1) else 0,
        "fourth_and_two": 1 if (req.down == 4 and req.ydstogo == 2) else 0,
        "fourth_and_three": 1 if (req.down == 4 and req.ydstogo == 3) else 0,
        "third_and_one": 1 if (req.down == 3 and req.ydstogo == 1) else 0,
        "third_and_two": 1 if (req.down == 3 and req.ydstogo == 2) else 0,
        "third_and_three": 1 if (req.down == 3 and req.ydstogo == 3) else 0,
        
        # Goal line
        "goal_line_short": 1 if (req.yardline_100 <= 3 and req.ydstogo <= 3) else 0,
        "goal_line_one_yard": 1 if (req.yardline_100 == 1 and req.ydstogo == 1) else 0,
        
        # 2-minute drill (context-aware)
        "two_minute_drill": 1 if (req.seconds_remaining_half <= 120 and req.seconds_remaining_half > 0 and score_diff <= 3) else 0,
        "two_minute_drill_trailing": 1 if (req.seconds_remaining_half <= 120 and req.seconds_remaining_half > 0 and score_diff < 0) else 0,
        "final_minute": 1 if (req.seconds_remaining_half <= 60 and req.seconds_remaining_half > 0 and abs(score_diff) <= 8) else 0,
        "two_minute_and_long": 1 if (req.seconds_remaining_half <= 120 and req.seconds_remaining_half > 0 and req.ydstogo >= 7 and score_diff <= 3) else 0,
        
        # Score context
        "trailing_multi_score": 1 if score_diff <= -9 else 0,
        "leading_late_game": 1 if (score_diff >= 3 and req.seconds_remaining_half <= 300 and req.qtr == 4) else 0,
        "blowout_lead_late": 1 if (score_diff >= 14 and req.seconds_remaining_half <= 300 and req.qtr == 4) else 0,
        "close_game_fourth_qtr": 1 if (abs(score_diff) <= 3 and req.qtr == 4) else 0,
        
        # Down × distance
        "down_x_ydstogo": req.down * req.ydstogo,
        "third_or_fourth_and_long": 1 if (req.down in [3, 4] and req.ydstogo >= 10) else 0,
        "first_and_ten": 1 if (req.down == 1 and req.ydstogo == 10) else 0,
        
        # Advanced
        "score_time_pressure": score_diff * (1 - min(req.seconds_remaining_half / 1800, 1)),
    }

    df = pd.DataFrame([data])

    # Ensure correct order
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0
    df = df[FEATURE_COLS]

    return df


def get_confidence_level(prob: float) -> str:
    """Determine confidence level from probability."""
    if prob >= 0.7:
        return "High"
    elif prob >= 0.55:
        return "Medium"
    else:
        return "Low"


def format_game_situation(req: PlayRequest) -> str:
    """Format human-readable game situation."""
    down_text = {1: "1st", 2: "2nd", 3: "3rd", 4: "4th"}[req.down]
    qtr_text = f"Q{req.qtr}"
    
    # Field position
    if req.yardline_100 > 50:
        field_pos = f"OWN {100 - req.yardline_100:.0f}"
    else:
        field_pos = f"OPP {req.yardline_100:.0f}"
    
    # Score situation
    score_diff = req.offense_score - req.defense_score
    if score_diff > 0:
        score_text = f"Up by {score_diff}"
    elif score_diff < 0:
        score_text = f"Down by {abs(score_diff)}"
    else:
        score_text = "Tied"
    
    return f"{down_text} & {req.ydstogo} at {field_pos}, {qtr_text}, {score_text}"


@app.get("/", tags=["Health"])
def root():
    """Health check endpoint."""
    return {
        "status": "ok",
        "message": "NFL Play Call Prediction API",
        "version": "1.0.0"
    }


@app.get("/health", tags=["Health"])
def health():
    """Detailed health check with model info."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
        "num_features": len(FEATURE_COLS),
        "features": FEATURE_COLS
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(req: PlayRequest):
    """
    Predict run or pass play based on game situation.
    
    Returns prediction with probabilities and confidence level.
    """
    try:
        # Build features
        X = build_feature_df(req)
        
        # Scale features (keeping column names)
        X_scaled_array = scaler.transform(X)
        X_scaled = pd.DataFrame(X_scaled_array, columns=X.columns)
        
        # Predict
        prob_pass = float(model.predict_proba(X_scaled)[0, 1])
        prob_run = 1.0 - prob_pass
        pred_label = "PASS" if prob_pass >= 0.5 else "RUN"
        
        # Additional info
        confidence = get_confidence_level(max(prob_pass, prob_run))
        game_situation = format_game_situation(req)
        
        return {
            "prediction": pred_label,
            "prob_pass": round(prob_pass, 3),
            "prob_run": round(prob_run, 3),
            "confidence": confidence,
            "game_situation": game_situation
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.get("/examples", tags=["Examples"])
def get_examples():
    """Get example game situations for testing."""
    return {
        "obvious_pass": {
            "description": "3rd & long, down late in game",
            "input": {
                "down": 3,
                "ydstogo": 12,
                "yardline_100": 75,
                "offense_score": 17,
                "defense_score": 24,
                "qtr": 4,
                "seconds_remaining_half": 120,
                "shotgun": True,
                "no_huddle": True,
                "is_home_offense": False
            }
        },
        "obvious_run": {
            "description": "1st & 10, leading, early in game",
            "input": {
                "down": 1,
                "ydstogo": 10,
                "yardline_100": 65,
                "offense_score": 14,
                "defense_score": 7,
                "qtr": 2,
                "seconds_remaining_half": 600,
                "shotgun": False,
                "no_huddle": False,
                "is_home_offense": True
            }
        },
        "short_yardage": {
            "description": "3rd & 1, red zone",
            "input": {
                "down": 3,
                "ydstogo": 1,
                "yardline_100": 15,
                "offense_score": 10,
                "defense_score": 10,
                "qtr": 3,
                "seconds_remaining_half": 300,
                "shotgun": False,
                "no_huddle": False,
                "is_home_offense": True
            }
        }
    }