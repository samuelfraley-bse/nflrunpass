"""
NFL Play Call Predictor - Enhanced Version with Proper Feature Engineering

This version uses the actual features.py to ensure consistency with training.
"""

import json
import sys
from pathlib import Path
import base64

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.patches import Circle, Rectangle

import streamlit as st

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="NFL Play Call Predictor",
    page_icon="🏈",
    layout="wide",
)

# =============================================================================
# SETUP PATHS AND IMPORT FEATURES
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
SRC_DIR = PROJECT_ROOT / "src"

# Add src to path
sys.path.insert(0, str(SRC_DIR))

# Import feature engineering
try:
    from nfl_run_pass.features import add_engineered_features
    FEATURES_AVAILABLE = True
except ImportError:
    st.error("⚠️ Could not import features.py - using basic features only")
    FEATURES_AVAILABLE = False

# =============================================================================
# LOAD BACKGROUND IMAGE (optional)
# =============================================================================

def get_base64_image(image_path):
    """Load background image if it exists."""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except:
        return None

img_path = Path(__file__).parent / "assets" / "stadium.png"
img_base64 = get_base64_image(img_path)

# =============================================================================
# CUSTOM CSS
# =============================================================================

background_css = f"""
    .stApp {{
        background-image: url('data:image/png;base64,{img_base64}');
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    .stApp::before {{
        content: "";
        position: fixed;
        top: 0; left: 0;
        width: 100%; height: 100%;
        background: rgba(0, 0, 0, 0.7);
        z-index: -1;
    }}
""" if img_base64 else ""

st.markdown(
    f"""
    <style>
    .block-container {{
        max-width: 1200px;
        padding-top: 1rem;
    }}

    .prediction-card {{
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 16px;
        padding: 1.5rem;
        border: 1px solid #0f3460;
        color: white;
    }}
    
    .prediction-pass {{
        background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 100%);
        border: 2px solid #4a9eff;
    }}
    
    .prediction-run {{
        background: linear-gradient(135deg, #3d1a1a 0%, #5c2e2e 100%);
        border: 2px solid #ff6b6b;
    }}

    .confidence-high {{ color: #4ade80; }}
    .confidence-medium {{ color: #fbbf24; }}
    .confidence-low {{ color: #f87171; }}

    .feature-positive {{ color: #4a9eff; }}
    .feature-negative {{ color: #ff6b6b; }}
    
    .scoreboard {{
        background: linear-gradient(135deg, #b71c1c 0%, #7f0000 100%);
        color: white;
        border-radius: 12px;
        padding: 12px 20px;
        text-align: center;
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }}

    .section-header {{
        font-size: 1.3rem;
        font-weight: 700;
        margin-bottom: 1rem;
        color: #e0e0e0;
    }}

    .prob-bar-container {{
        background-color: #2d2d2d;
        border-radius: 10px;
        height: 30px;
        overflow: hidden;
        margin: 10px 0;
    }}
    
    .prob-bar-pass {{
        background: linear-gradient(90deg, #1e3a5f, #4a9eff);
        height: 100%;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: 600;
        transition: width 0.5s ease;
    }}
    
    {background_css}
    </style>
    """,
    unsafe_allow_html=True,
)

# =============================================================================
# LOAD MODEL ARTIFACTS
# =============================================================================


@st.cache_resource
def load_artifacts():
    """Load model, scaler, and feature columns."""
    model_path = MODELS_DIR / "log_reg_model.pkl"
    scaler_path = MODELS_DIR / "scaler.pkl"
    feature_cols_path = MODELS_DIR / "feature_cols.json"

    if not model_path.exists():
        st.error(f"Model not found at: {model_path}")
        st.stop()

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    with feature_cols_path.open("r") as f:
        feature_cols = json.load(f)

    return model, scaler, feature_cols


log_reg, scaler, FEATURE_COLS = load_artifacts()

# =============================================================================
# SESSION STATE
# =============================================================================

if "off_score" not in st.session_state:
    st.session_state.off_score = 17
if "def_score" not in st.session_state:
    st.session_state.def_score = 14
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None
if "history" not in st.session_state:
    st.session_state.history = []

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def draw_field(yardline_100: float):
    """Enhanced field visualization."""
    x_ball = 100 - yardline_100

    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 10)
    ax.axis("off")
    fig.patch.set_facecolor('#0e1117')

    # Base field
    field = Rectangle((0, 0), 100, 10, facecolor="#1b5e20", edgecolor="white", linewidth=2)
    ax.add_patch(field)

    # Red zone highlights
    red_zone_left = Rectangle((0, 0), 20, 10, facecolor="#c62828", alpha=0.25)
    red_zone_right = Rectangle((80, 0), 20, 10, facecolor="#c62828", alpha=0.25)
    ax.add_patch(red_zone_left)
    ax.add_patch(red_zone_right)

    # End zones
    end_zone_left = Rectangle((0, 0), 10, 10, facecolor="#0d47a1", alpha=0.3, edgecolor="white")
    end_zone_right = Rectangle((90, 0), 10, 10, facecolor="#0d47a1", alpha=0.3, edgecolor="white")
    ax.add_patch(end_zone_left)
    ax.add_patch(end_zone_right)

    # Yard lines
    for x in range(0, 101, 10):
        lw = 2.5 if x == 50 else 1.5
        color = "yellow" if x == 50 else "white"
        ax.plot([x, x], [0, 10], color=color, linewidth=lw, alpha=0.7)
        if x in [10, 20, 30, 40, 60, 70, 80, 90]:
            label = x if x <= 50 else 100 - x
            ax.text(x, 1.8, str(label), color="white", ha="center", va="center",
                   fontsize=10, fontweight="bold", alpha=0.8)

    # 50 yard line
    ax.text(50, 1.8, "50", color="yellow", ha="center", va="center",
           fontsize=11, fontweight="bold")

    # Ball with glow
    glow = Circle((x_ball, 5), radius=2, facecolor="#ffcc80", alpha=0.3)
    ax.add_patch(glow)
    ball = Circle((x_ball, 5), radius=1.5, facecolor="#ff6f00", edgecolor="white", linewidth=2)
    ax.add_patch(ball)
    
    ax.text(x_ball, 8.8, f"Ball at {int(100 - yardline_100)}",
           color="white", ha="center", va="center", fontsize=9, fontweight="bold",
           bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.7, edgecolor="white"))

    fig.tight_layout()
    return fig


def parse_clock(text, default_seconds=600):
    """Parse 'mm:ss' or raw seconds."""
    txt = text.strip()
    if not txt:
        return default_seconds
    if ":" in txt:
        try:
            m, s = txt.split(":")
            return int(m) * 60 + int(s)
        except:
            return default_seconds
    try:
        return int(txt)
    except:
        return default_seconds


def build_scenario_dataframe(down, ydstogo, yardline_100, game_seconds_remaining,
                             score_differential, qtr, shotgun, no_huddle, is_home_offense):
    """
    Build scenario using proper feature engineering.
    This matches how the model was trained!
    """
    # Calculate half_seconds_remaining
    if qtr <= 2:
        half_seconds_remaining = game_seconds_remaining - 1800
    else:
        half_seconds_remaining = game_seconds_remaining % 1800
    
    # Auto-detect goal_to_go
    goal_to_go = 1 if yardline_100 <= ydstogo else 0
    
    scenario = pd.DataFrame([{
        'down': down,
        'ydstogo': ydstogo,
        'yardline_100': yardline_100,
        'game_seconds_remaining': game_seconds_remaining,
        'score_differential': score_differential,
        'qtr': qtr,
        'half_seconds_remaining': max(0, half_seconds_remaining),
        'shotgun': int(shotgun),
        'no_huddle': int(no_huddle),
        'goal_to_go': goal_to_go,
        'posteam': 'OFF',
        'home_team': 'OFF' if is_home_offense else 'DEF',
    }])
    
    return scenario


def get_confidence_level(prob_pass):
    """Determine confidence level."""
    max_prob = max(prob_pass, 1 - prob_pass)
    if max_prob >= 0.7:
        return "High", "confidence-high", "🟢"
    elif max_prob >= 0.55:
        return "Medium", "confidence-medium", "🟡"
    else:
        return "Low", "confidence-low", "🔴"


def get_feature_contributions(model, scaler, X_scaled, feature_cols):
    """Get top feature contributions."""
    coefficients = model.coef_[0]
    feature_values = X_scaled[0]
    
    contributions = []
    for i, col in enumerate(feature_cols):
        contrib = coefficients[i] * feature_values[i]
        if abs(contrib) > 0.05:
            contributions.append((col, contrib))
    
    contributions.sort(key=lambda x: abs(x[1]), reverse=True)
    return contributions[:6]


# Feature display names (comprehensive)
FEATURE_DISPLAY_NAMES = {
    "down": "Down",
    "ydstogo": "Yards to go",
    "yardline_100": "Field position",
    "is_red_zone": "Red zone",
    "is_goal_to_go": "Goal to go",
    "short_ydstogo": "Short yardage",
    "medium_ydstogo": "Medium yardage",
    "long_ydstogo": "Long yardage",
    "shotgun": "Shotgun",
    "no_huddle": "No huddle",
    "is_trailing": "Trailing",
    "is_tied": "Tied",
    "is_leading": "Leading",
    "is_fourth_qtr": "4th quarter",
    "late_half": "Late in half",
    "is_home_offense": "Home offense",
    # New interaction features
    "fourth_and_one": "4th & 1",
    "fourth_and_two": "4th & 2",
    "fourth_and_three": "4th & 3",
    "third_and_one": "3rd & 1",
    "third_and_two": "3rd & 2",
    "third_and_three": "3rd & 3",
    "goal_line_short": "Goal line (short)",
    "goal_line_one_yard": "Goal line (1 yd)",
    "two_minute_drill": "2-min drill",
    "two_minute_drill_trailing": "2-min (trailing)",
    "final_minute": "Final minute",
    "two_minute_and_long": "2-min & long",
    "trailing_multi_score": "Down multiple scores",
    "leading_late_game": "Leading late",
    "blowout_lead_late": "Blowout lead",
    "close_game_fourth_qtr": "Close game (Q4)",
    "score_time_pressure": "Score×Time pressure",
    "down_x_ydstogo": "Down×Distance",
    "third_or_fourth_and_long": "3rd/4th & long",
    "first_and_ten": "1st & 10",
}

# =============================================================================
# PAGE LAYOUT
# =============================================================================

st.markdown(
    """
    <div style="text-align: center; margin-bottom: 1.5rem;">
        <h1 style="margin-bottom: 0.3rem;">🏈 NFL Play Call Predictor</h1>
        <p style="color: #888; font-size: 1rem;">
            Pre-snap run vs pass prediction using contextual features
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

left_col, right_col = st.columns([1, 1.2])

# =============================================================================
# LEFT COLUMN: Inputs
# =============================================================================

with left_col:
    st.markdown('<p class="section-header">📋 Game Situation</p>', unsafe_allow_html=True)
    
    # Score
    score_cols = st.columns(2)
    with score_cols[0]:
        off_score = st.number_input("Offense score", min_value=0, max_value=99,
                                    value=st.session_state.off_score)
    with score_cols[1]:
        def_score = st.number_input("Defense score", min_value=0, max_value=99,
                                   value=st.session_state.def_score)
    
    st.session_state.off_score = off_score
    st.session_state.def_score = def_score
    
    # Scoreboard
    st.markdown(
        f"""
        <div class="scoreboard">
            <span>OFF</span> <span style="font-size: 1.4rem;">{off_score}</span>
            &nbsp;&nbsp;│&nbsp;&nbsp;
            <span style="font-size: 1.4rem;">{def_score}</span> <span>DEF</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # Down & distance
    st.markdown("**Down & Distance**")
    dd_cols = st.columns(2)
    with dd_cols[0]:
        down = st.selectbox("Down", options=[1, 2, 3, 4], index=0)
    with dd_cols[1]:
        ydstogo = st.number_input("Yards to go", min_value=1, max_value=99, value=10)

    # Clock
    st.markdown("**Clock & Context**")
    cc_cols = st.columns(2)
    with cc_cols[0]:
        qtr = st.selectbox("Quarter", options=[1, 2, 3, 4], index=2)
    with cc_cols[1]:
        clock_text = st.text_input("Game clock (mm:ss)", value="10:00")
        game_seconds_remaining = parse_clock(clock_text, default_seconds=600)

    # Formation
    st.markdown("**Formation & Situation**")
    form_cols = st.columns(3)
    with form_cols[0]:
        shotgun = st.checkbox("Shotgun", value=False)
    with form_cols[1]:
        no_huddle = st.checkbox("No huddle", value=False)
    with form_cols[2]:
        is_home_offense = st.checkbox("Home team", value=True)

    # Field position
    st.markdown("**Field Position**")
    ui_pos = st.slider(
        "Ball position (0 = own goal, 100 = opponent goal)",
        min_value=1, max_value=99, value=65
    )
    
    yardline_100 = 100 - ui_pos
    
    if ui_pos <= 50:
        position_text = f"Own {ui_pos}-yard line"
    else:
        position_text = f"Opponent {100 - ui_pos}-yard line"
    
    if yardline_100 <= 20:
        position_text += " 🔴 RED ZONE"
    
    st.caption(position_text)

# =============================================================================
# RIGHT COLUMN: Prediction
# =============================================================================

with right_col:
    st.markdown('<p class="section-header">🤖 Model Prediction</p>', unsafe_allow_html=True)
    
    predict_clicked = st.button("⚡ Predict Play Call", use_container_width=True, type="primary")
    
    if predict_clicked:
        score_diff = off_score - def_score
        
        # Build scenario
        scenario_df = build_scenario_dataframe(
            down, ydstogo, yardline_100, game_seconds_remaining,
            score_diff, qtr, shotgun, no_huddle, is_home_offense
        )
        
        # Add engineered features
        if FEATURES_AVAILABLE:
            scenario_with_features = add_engineered_features(scenario_df)
        else:
            scenario_with_features = scenario_df
        
        # Extract features and scale
        X = scenario_with_features[FEATURE_COLS]
        X_scaled = scaler.transform(X)
        
        # Predict
        prob_pass = float(log_reg.predict_proba(X_scaled)[0, 1])
        prob_run = 1.0 - prob_pass
        pred_label = "PASS" if prob_pass >= 0.5 else "RUN"
        
        # Get contributions
        contributions = get_feature_contributions(log_reg, scaler, X_scaled, FEATURE_COLS)
        
        # Confidence
        conf_text, conf_class, conf_emoji = get_confidence_level(prob_pass)
        
        # Store in session
        st.session_state.last_prediction = {
            "label": pred_label,
            "prob_pass": prob_pass,
            "prob_run": prob_run,
            "contributions": contributions,
            "confidence": (conf_text, conf_class, conf_emoji)
        }
        
        # Add to history
        down_text = {1: "1st", 2: "2nd", 3: "3rd", 4: "4th"}[down]
        if yardline_100 > 50:
            field = f"Own {100-yardline_100:.0f}"
        else:
            field = f"Opp {yardline_100:.0f}"
        situation = f"{down_text}&{ydstogo} at {field}, Q{qtr}"
        
        st.session_state.history.insert(0, {
            "Situation": situation,
            "Prediction": pred_label,
            "Pass %": f"{prob_pass:.1%}",
            "Confidence": conf_text
        })
        
        if len(st.session_state.history) > 10:
            st.session_state.history = st.session_state.history[:10]
    
    # Display prediction
    if st.session_state.last_prediction:
        pred = st.session_state.last_prediction
        pred_label = pred["label"]
        prob_pass = pred["prob_pass"]
        prob_run = pred["prob_run"]
        contributions = pred["contributions"]
        conf_text, conf_class, conf_emoji = pred["confidence"]
        
        card_class = "prediction-pass" if pred_label == "PASS" else "prediction-run"
        
        st.markdown(
            f"""
            <div class="prediction-card {card_class}">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                    <div style="font-size: 1.8rem; font-weight: 700;">
                        {"ᯓ🏈" if pred_label == "PASS" else "🏃"} {pred_label}
                    </div>
                    <div class="{conf_class}" style="font-size: 0.9rem;">
                        {conf_emoji} {conf_text} confidence
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
        # Probability bar
        st.markdown(
            f"""
            <div style="margin: 1rem 0;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 5px; font-size: 0.85rem;">
                    <span>🏃 RUN {prob_run:.1%}</span>
                    <span>PASS {prob_pass:.1%} ᯓ🏈</span>
                </div>
                <div class="prob-bar-container">
                    <div class="prob-bar-pass" style="width: {prob_pass * 100}%;">
                        {prob_pass:.0%}
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
        # Feature importance
        st.markdown("**Why this prediction?**")
        
        for feature, contrib in contributions:
            display_name = FEATURE_DISPLAY_NAMES.get(feature, feature)
            direction = "→ PASS" if contrib > 0 else "→ RUN"
            color_class = "feature-positive" if contrib > 0 else "feature-negative"
            bar_width = min(abs(contrib) * 50, 100)
            
            st.markdown(
                f"""
                <div style="display: flex; align-items: center; margin: 4px 0; font-size: 0.85rem;">
                    <span style="width: 160px; color: #ccc;">{display_name}</span>
                    <div style="flex: 1; background: #2d2d2d; height: 8px; border-radius: 4px; margin: 0 10px;">
                        <div style="width: {bar_width}%; height: 100%; background: {'#4a9eff' if contrib > 0 else '#ff6b6b'}; border-radius: 4px;"></div>
                    </div>
                    <span class="{color_class}" style="width: 70px; text-align: right;">{direction}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
    
    else:
        st.info("👆 Set up a game situation and click **Predict Play Call**")

# =============================================================================
# FIELD VISUALIZATION
# =============================================================================

st.markdown("---")
st.markdown('<p class="section-header">🏟️ Field Position</p>', unsafe_allow_html=True)

field_fig = draw_field(yardline_100)
st.pyplot(field_fig, use_container_width=True)
plt.close(field_fig)

# =============================================================================
# PREDICTION HISTORY
# =============================================================================

if st.session_state.history:
    st.markdown("---")
    st.markdown('<p class="section-header">📊 Prediction History</p>', unsafe_allow_html=True)
    
    history_df = pd.DataFrame(st.session_state.history)
    st.dataframe(history_df, use_container_width=True, hide_index=True)
    
    if st.button("🗑️ Clear History"):
        st.session_state.history = []
        st.rerun()

# =============================================================================
# FOOTER
# =============================================================================

st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #666; font-size: 0.8rem;">
        Built with Streamlit • Model: Logistic Regression with context-aware features
    </div>
    """,
    unsafe_allow_html=True,
)