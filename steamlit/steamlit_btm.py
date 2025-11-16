import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import random
from pathlib import Path

# --------------------------------------------------------------------
# Config / paths
# --------------------------------------------------------------------
# Use your smaller sample file for now
PBP_CSV_PATH = "sample_plays.csv"   # <-- put sample_plays.csv in same folder as this script
GIF_DIR = Path("gifs")              # <-- folder with 2023_07_MIA_PHI_PRE.gif etc.

st.set_page_config(
    page_title="Can You Beat the Model?",
    layout="centered",
)


# --------------------------------------------------------------------
# Load data + model artifacts
# --------------------------------------------------------------------
@st.cache_resource
def load_artifacts_and_data():
    # 1) Model artifacts
    model = joblib.load("log_reg_model.pkl")
    scaler = joblib.load("scaler.pkl")
    with open("feature_cols.json", "r") as f:
        feature_cols = json.load(f)

    # 2) Play-by-play data (here: your curated sample plays)
    df = pd.read_csv(PBP_CSV_PATH, low_memory=False)

    # Keep only run/pass plays with needed columns present
    df = df[df["play_type"].isin(["run", "pass"])].copy()

    needed_cols = [
        "home_team",
        "away_team",
        "season",
        "week",
        "posteam",
        "defteam",
        "total_home_score",
        "total_away_score",
        "posteam_score",
        "defteam_score",
        "qtr",
        "quarter_seconds_remaining",
        "half_seconds_remaining",
        "down",
        "ydstogo",
        "yardline_100",
        "shotgun",
        "no_huddle",
        "play_type",
        "desc",
        "game_id",   # used for GIF filenames
    ]

    missing = [c for c in needed_cols if c not in df.columns]
    if missing:
        st.warning(f"Warning: missing columns in CSV: {missing}")
    existing = [c for c in needed_cols if c in df.columns]
    df = df.dropna(subset=existing)

    # Reset index so we can sample by integer position
    df = df.reset_index(drop=True)

    return model, scaler, feature_cols, df


log_reg, scaler, FEATURE_COLS, PBP_DF = load_artifacts_and_data()


# --------------------------------------------------------------------
# Helpers: scenario text + feature engineering from a row
# --------------------------------------------------------------------
def seconds_to_clock(secs: int) -> str:
    secs = max(int(secs), 0)
    m, s = divmod(secs, 60)
    return f"{m:01d}:{s:02d}"


def down_to_ordinal(down: int) -> str:
    mapping = {1: "1st", 2: "2nd", 3: "3rd", 4: "4th"}
    return mapping.get(int(down), f"{down}th")


def yardline_text(row) -> str:
    """
    Convert yardline_100 + offense/defense teams into
    'own 30' / 'opponent 25' style text.
    yardline_100 = distance to opponent end zone (nflfastR convention).
    """
    y100 = int(row["yardline_100"])
    off = row["posteam"]
    home = row["home_team"]
    away = row["away_team"]

    # Which team is offense & defense?
    if off == home:
        defense = away
    else:
        defense = home

    # 0 = at opponent goal line, 100 = at own goal line
    if y100 > 50:
        yard = 100 - y100
        return f"their own {yard}-yard line"
    else:
        yard = y100
        return f"the {defense} {yard}-yard line"


def make_scenario_text(row) -> str:
    home = row["home_team"]
    away = row["away_team"]
    season = int(row["season"])
    week = int(row["week"])

    home_score = int(row["total_home_score"])
    away_score = int(row["total_away_score"])

    off = row["posteam"]
    def_ = row["defteam"]

    qtr = int(row["qtr"])
    clock = seconds_to_clock(int(row["quarter_seconds_remaining"]))
    down = int(row["down"])
    ytg = int(row["ydstogo"])

    yard_text = yardline_text(row)

    # lead / trail / tied from offense perspective
    off_score = int(row["posteam_score"])
    def_score = int(row["defteam_score"])
    if off_score > def_score:
        score_phrase = f"{off} leads {off_score}-{def_score}"
    elif off_score < def_score:
        score_phrase = f"{off} trails {def_score}-{off_score}"
    else:
        score_phrase = f"the game is tied {off_score}-{def_score}"

    return (
        f"{home} vs {away}, Week {week} {season}. "
        f"{clock} in Q{qtr}. {score_phrase}. "
        f"{off} has the ball on {yard_text}, it's {down_to_ordinal(down)} and {ytg}."
    )


def build_features_from_row(row) -> pd.DataFrame:
    """
    Recreate the pre-snap features for a single play row,
    matching the training feature engineering as closely as possible.
    """
    down = int(row["down"])
    ydstogo = int(row["ydstogo"])
    yardline_100 = float(row["yardline_100"])
    offense_score = int(row["posteam_score"])
    defense_score = int(row["defteam_score"])
    qtr = int(row["qtr"])

    # Prefer half_seconds_remaining if available, else quarter_seconds_remaining
    if "half_seconds_remaining" in row.index and not pd.isna(row["half_seconds_remaining"]):
        seconds_remaining_half = int(row["half_seconds_remaining"])
    else:
        # approximate: 2 quarters in half, so quarter_seconds + 900 * (qtr % 2)
        seconds_remaining_half = int(row["quarter_seconds_remaining"]) + 900 * ((qtr - 1) % 2)

    shotgun = int(row.get("shotgun", 0)) == 1
    no_huddle = int(row.get("no_huddle", 0)) == 1
    is_home_offense = row["posteam"] == row["home_team"]

    score_diff = offense_score - defense_score

    data = {
        "down": down,
        "ydstogo": ydstogo,
        "yardline_100": yardline_100,
        "game_seconds_remaining": seconds_remaining_half,
        "is_red_zone": 1 if yardline_100 <= 20 else 0,
        "is_goal_to_go": 1 if yardline_100 <= 10 else 0,
        "short_ydstogo": 1 if ydstogo <= 3 else 0,
        "medium_ydstogo": 1 if 4 <= ydstogo <= 7 else 0,
        "long_ydstogo": 1 if ydstogo >= 8 else 0,
        "shotgun": int(shotgun),
        "no_huddle": int(no_huddle),
        "score_differential": score_diff,
        "is_trailing": 1 if score_diff < 0 else 0,
        "is_tied": 1 if score_diff == 0 else 0,
        "is_leading": 1 if score_diff > 0 else 0,
        "is_fourth_qtr": 1 if qtr == 4 else 0,
        "late_half": 1 if seconds_remaining_half < 120 else 0,
        "is_home_offense": int(is_home_offense),
    }

    df = pd.DataFrame([data])

    # Add any missing columns and align order to FEATURE_COLS
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0
    df = df[FEATURE_COLS]

    return df


def get_gif_paths(row):
    """
    Build PRE and FULL gif paths from game_id.
    Example filenames:
        2023_07_MIA_PHI_PRE.gif
        2023_07_MIA_PHI_FULL.gif
    """
    game_id = str(row.get("game_id", ""))
    if not game_id:
        return None, None

    pre_path = GIF_DIR / f"{game_id}_PRE.gif"
    full_path = GIF_DIR / f"{game_id}_FULL.gif"
    return pre_path, full_path


# --------------------------------------------------------------------
# Session state: scores + current play + flow control
# --------------------------------------------------------------------
if "user_correct" not in st.session_state:
    st.session_state.user_correct = 0
if "model_correct" not in st.session_state:
    st.session_state.model_correct = 0
if "rounds_played" not in st.session_state:
    st.session_state.rounds_played = 0
if "current_idx" not in st.session_state:
    st.session_state.current_idx = random.randrange(len(PBP_DF))
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "has_run_model" not in st.session_state:
    st.session_state.has_run_model = False
if "user_guess" not in st.session_state:
    st.session_state.user_guess = None


def pick_new_play():
    st.session_state.current_idx = random.randrange(len(PBP_DF))
    st.session_state.last_result = None
    st.session_state.has_run_model = False
    st.session_state.user_guess = None


def evaluate_guess(user_choice: str):
    row = PBP_DF.iloc[st.session_state.current_idx]
    X = build_features_from_row(row)
    X_scaled = scaler.transform(X)
    proba = log_reg.predict_proba(X_scaled)[0]
    prob_pass = float(proba[1])
    prob_run = float(proba[0])

    model_choice = "pass" if prob_pass >= 0.5 else "run"
    actual = row["play_type"]  # "run" or "pass"

    st.session_state.rounds_played += 1
    if user_choice == actual:
        st.session_state.user_correct += 1
    if model_choice == actual:
        st.session_state.model_correct += 1

    st.session_state.last_result = {
        "user_choice": user_choice,
        "model_choice": model_choice,
        "prob_pass": prob_pass,
        "prob_run": prob_run,
        "actual": actual,
        "desc": row.get("desc", ""),
    }
    st.session_state.has_run_model = True
    st.session_state.user_guess = user_choice


# --------------------------------------------------------------------
# UI
# --------------------------------------------------------------------
st.title("Can You Beat the Model?")

# Scoreboard
rc = st.session_state.rounds_played
user = st.session_state.user_correct
model_score = st.session_state.model_correct

if rc > 0:
    st.markdown(
        f"**Your record:** {user}/{rc}  &nbsp;&nbsp; | &nbsp;&nbsp; "
        f"**Model record:** {model_score}/{rc}"
    )
else:
    st.markdown("Play a few rounds to see who’s better: you or the model!")

st.markdown("---")

# Current scenario
current_row = PBP_DF.iloc[st.session_state.current_idx]
scenario_text = make_scenario_text(current_row)
st.write(scenario_text)

# PRE GIF
pre_gif_path, full_gif_path = get_gif_paths(current_row)
if pre_gif_path and pre_gif_path.exists():
    st.image(str(pre_gif_path), caption="Pre-snap view (PRE)", use_column_width=True)
else:
    st.info("PRE GIF not found for this play.")

st.markdown("**What do you think the offense did?**")

guess_choice = st.radio(
    "Choose run or pass, then click 'Run model':",
    ["run", "pass"],
    horizontal=True,
    key="guess_radio",
)

col_run_model, col_next = st.columns(2)
with col_run_model:
    run_model_clicked = st.button("Run model", use_container_width=True)
with col_next:
    next_play_clicked = st.button("Next play", use_container_width=True)

# Handle button logic
if next_play_clicked:
    pick_new_play()
elif run_model_clicked:
    # Evaluate the current guess
    evaluate_guess(guess_choice)

# Show result of last round (if any)
if st.session_state.last_result is not None:
    res = st.session_state.last_result

    st.markdown("---")
    st.subheader("Result & Full Play")

    # FULL GIF
    if full_gif_path and full_gif_path.exists():
        st.image(str(full_gif_path), caption="Full play (FULL)", use_column_width=True)
    else:
        st.info("FULL GIF not found for this play.")

    st.write(f"**Your guess:** {res['user_choice'].upper()}")
    st.write(
        f"**Model prediction:** {res['model_choice'].upper()} "
        f"(pass: {res['prob_pass']:.1%}, run: {res['prob_run']:.1%})"
    )
    st.write(f"**Actual play:** {res['actual'].upper()}")

    if res["desc"]:
        st.markdown("**Play description:**")
        st.write(res["desc"])
