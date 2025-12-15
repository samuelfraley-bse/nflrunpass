"""
NFL Play Call Predictor - Improved Version

Enhancements:
1. Persistent predictions (stays visible after other interactions)
2. Feature importance display (shows why the model decided)
3. Confidence indicator (high/medium/low)
4. Probability bar visualization
5. Cleaner visual design
"""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.patches import Circle, Rectangle

import streamlit as st

# =============================================================================
# PAGE CONFIGURATION - Must be first Streamlit command
# =============================================================================

st.set_page_config(
    page_title="NFL Play Call Predictor",
    page_icon="🏈",
    layout="wide",
)

# =============================================================================
# CUSTOM CSS
# =============================================================================

st.markdown(
    """
    <style>
    /* Main container */
    .block-container {
        max-width: 1200px;
        padding-top: 1rem;
    }

    /* Prediction card styles */
    .prediction-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 16px;
        padding: 1.5rem;
        border: 1px solid #0f3460;
        color: white;
    }
    
    .prediction-pass {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 100%);
        border: 2px solid #4a9eff;
    }
    
    .prediction-run {
        background: linear-gradient(135deg, #3d1a1a 0%, #5c2e2e 100%);
        border: 2px solid #ff6b6b;
    }

    /* Confidence badge */
    .confidence-high { color: #4ade80; }
    .confidence-medium { color: #fbbf24; }
    .confidence-low { color: #f87171; }

    /* Feature importance */
    .feature-positive { color: #4a9eff; }
    .feature-negative { color: #ff6b6b; }
    
    /* Scoreboard */
    .scoreboard {
        background: linear-gradient(135deg, #b71c1c 0%, #7f0000 100%);
        color: white;
        border-radius: 12px;
        padding: 12px 20px;
        text-align: center;
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }

    /* Section headers */
    .section-header {
        font-size: 1.3rem;
        font-weight: 700;
        margin-bottom: 1rem;
        color: #e0e0e0;
    }

    /* Probability bar */
    .prob-bar-container {
        background-color: #2d2d2d;
        border-radius: 10px;
        height: 30px;
        overflow: hidden;
        margin: 10px 0;
    }
    
    .prob-bar-pass {
        background: linear-gradient(90deg, #1e3a5f, #4a9eff);
        height: 100%;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: 600;
        transition: width 0.5s ease;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =============================================================================
# LOAD MODEL ARTIFACTS (cached)
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"


@st.cache_resource
def load_artifacts():
    """Load model, scaler, and feature columns. Cached across all sessions."""
    model_path = MODELS_DIR / "log_reg_model.pkl"
    scaler_path = MODELS_DIR / "scaler.pkl"
    feature_cols_path = MODELS_DIR / "feature_cols.json"

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    with feature_cols_path.open("r") as f:
        feature_cols = json.load(f)

    return model, scaler, feature_cols


log_reg, scaler, FEATURE_COLS = load_artifacts()

# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================

# Score state
if "off_score" not in st.session_state:
    st.session_state.off_score = 17
if "def_score" not in st.session_state:
    st.session_state.def_score = 14

# Prediction state (NEW - makes prediction persist)
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def draw_field(yardline_100: float):
    """Draw a football field with ball position marker."""
    x_ball = 100 - yardline_100

    fig, ax = plt.subplots(figsize=(12, 3))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 10)
    ax.axis("off")
    fig.patch.set_facecolor('#0e1117')  # Match Streamlit dark theme

    # Field
    field = Rectangle((0, 0), 100, 10, facecolor="#1b5e20", edgecolor="white", linewidth=2)
    ax.add_patch(field)

    # End zones
    left_endzone = Rectangle((0, 0), 0, 10, facecolor="#0d3d0d", edgecolor="white")
    right_endzone = Rectangle((100, 0), 0, 10, facecolor="#0d3d0d", edgecolor="white")
    ax.add_patch(left_endzone)
    ax.add_patch(right_endzone)

    # Major yard lines
    for x in range(0, 101, 10):
        lw = 2 if x == 50 else 1
        ax.plot([x, x], [0, 10], color="white", linewidth=lw)
    
    # Minor hashes
    for x in range(5, 100, 10):
        ax.plot([x, x], [3.5, 6.5], color="white", linewidth=0.5)

    # Yard numbers
    for x in range(10, 50, 10):
        ax.text(x, 1.5, str(x), color="white", ha="center", va="center", fontsize=9, fontweight="bold")
        ax.text(100 - x, 1.5, str(x), color="white", ha="center", va="center", fontsize=9, fontweight="bold")

    # End zone labels
    ax.text(3, 5, "OWN", color="white", ha="center", va="center", fontsize=8, fontweight="bold", alpha=0.7)
    ax.text(97, 5, "OPP", color="white", ha="center", va="center", fontsize=8, fontweight="bold", alpha=0.7)

    # Ball marker with glow effect
    glow = Circle((x_ball, 5), radius=2, facecolor="#ffcc80", alpha=0.3)
    ax.add_patch(glow)
    ball = Circle((x_ball, 5), radius=1.2, facecolor="#ffcc80", edgecolor="#8b4513", linewidth=2)
    ax.add_patch(ball)
    
    # Position label
    ax.text(x_ball, 8.5, f"{int(100 - yardline_100)}-yard line", 
            color="white", ha="center", va="center", fontsize=9, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#1b5e20", edgecolor="white", alpha=0.8))

    fig.tight_layout()
    return fig


def parse_int(text, default):
    """Safely parse integer from text input."""
    try:
        return int(text)
    except Exception:
        return default


def parse_clock(text, default_seconds=600):
    """Parse 'mm:ss' or raw seconds string into integer seconds."""
    txt = text.strip()
    if not txt:
        return default_seconds
    if ":" in txt:
        try:
            m, s = txt.split(":")
            return int(m) * 60 + int(s)
        except Exception:
            return default_seconds
    try:
        return int(txt)
    except Exception:
        return default_seconds


def build_feature_vector(
    down, ydstogo, yardline_100, offense_score, defense_score,
    qtr, seconds_remaining_half, shotgun, no_huddle, is_home_offense
):
    """Construct feature vector matching training features."""
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
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0
    df = df[FEATURE_COLS]
    return df, data  # Return both DataFrame and raw data dict


def get_feature_contributions(model, scaler, X_scaled, feature_data):
    """
    Calculate how much each feature contributed to the prediction.
    
    For logistic regression: contribution = coefficient × scaled_value
    Positive contribution → pushes toward PASS
    Negative contribution → pushes toward RUN
    """
    coefficients = model.coef_[0]
    scaled_values = X_scaled[0]
    
    contributions = {}
    for i, col in enumerate(FEATURE_COLS):
        contrib = coefficients[i] * scaled_values[i]
        # Only include features that are "active" (non-zero in raw data) or have significant contribution
        if abs(contrib) > 0.05 or feature_data.get(col, 0) != 0:
            contributions[col] = contrib
    
    # Sort by absolute contribution
    sorted_contribs = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)
    return sorted_contribs[:6]  # Top 6 contributors


def get_confidence_level(prob):
    """Categorize prediction confidence."""
    confidence = max(prob, 1 - prob)
    if confidence >= 0.70:
        return "High", "confidence-high", "🟢"
    elif confidence >= 0.55:
        return "Medium", "confidence-medium", "🟡"
    else:
        return "Low", "confidence-low", "🔴"


# Human-readable feature names
FEATURE_DISPLAY_NAMES = {
    "down": "Down",
    "ydstogo": "Yards to go",
    "yardline_100": "Field position",
    "game_seconds_remaining": "Time remaining",
    "is_red_zone": "Red zone",
    "is_goal_to_go": "Goal to go",
    "short_ydstogo": "Short yardage (≤3)",
    "medium_ydstogo": "Medium yardage (4-7)",
    "long_ydstogo": "Long yardage (8+)",
    "shotgun": "Shotgun formation",
    "no_huddle": "No huddle",
    "score_differential": "Score difference",
    "is_trailing": "Trailing",
    "is_tied": "Tied",
    "is_leading": "Leading",
    "is_fourth_qtr": "4th quarter",
    "late_half": "Late in half (<2 min)",
    "is_home_offense": "Home team offense",
}


# =============================================================================
# MAIN APP LAYOUT
# =============================================================================

# Header
st.markdown(
    """
    <div style="text-align: center; margin-bottom: 1.5rem;">
        <h1 style="margin-bottom: 0.3rem;">🏈 NFL Play Call Predictor</h1>
        <p style="color: #888; font-size: 1rem;">
            Pre-snap run vs pass prediction using 2021–2023 play-by-play data
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Two-column layout
left_col, right_col = st.columns([1, 1.2])

# =============================================================================
# LEFT COLUMN: Game Situation Inputs
# =============================================================================

with left_col:
    st.markdown('<p class="section-header">📋 Game Situation</p>', unsafe_allow_html=True)
    
    # Score inputs FIRST (so we can use their values in the scoreboard)
    score_cols = st.columns(2)
    with score_cols[0]:
        off_score = st.number_input(
            "Offense score", min_value=0, max_value=99, step=1,
            value=st.session_state.off_score, key="off_score_input"
        )
    with score_cols[1]:
        def_score = st.number_input(
            "Defense score", min_value=0, max_value=99, step=1,
            value=st.session_state.def_score, key="def_score_input"
        )
    
    # Update session state with current values
    st.session_state.off_score = off_score
    st.session_state.def_score = def_score
    
    # Scoreboard displays the CURRENT input values (no lag)
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

    # Clock & context
    st.markdown("**Clock & Context**")
    cc_cols = st.columns(2)
    with cc_cols[0]:
        qtr = st.selectbox("Quarter", options=[1, 2, 3, 4], index=0)
    with cc_cols[1]:
        clock_text = st.text_input("Game clock (mm:ss)", value="10:00")
        seconds_remaining_half = parse_clock(clock_text, default_seconds=600)

    # Formation checkboxes
    st.markdown("**Formation & Situation**")
    form_cols = st.columns(3)
    with form_cols[0]:
        shotgun = st.checkbox("Shotgun", value=True)
    with form_cols[1]:
        no_huddle = st.checkbox("No huddle", value=False)
    with form_cols[2]:
        is_home_offense = st.checkbox("Home team", value=True)

    # Field position
    st.markdown("**Field Position**")
    ui_pos = st.slider(
        "Ball position (0 = own goal, 100 = opponent goal)",
        min_value=1, max_value=99, value=65,
        help="Slide right to move toward opponent's end zone"
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
# RIGHT COLUMN: Model Output
# =============================================================================

with right_col:
    st.markdown('<p class="section-header">🤖 Model Prediction</p>', unsafe_allow_html=True)
    
    # Predict button
    predict_clicked = st.button("⚡ Predict Play Call", use_container_width=True, type="primary")
    
    # Run prediction when button clicked
    if predict_clicked:
        # Build features
        X_df, feature_data = build_feature_vector(
            down=down, ydstogo=ydstogo, yardline_100=yardline_100,
            offense_score=off_score,
            defense_score=def_score,
            qtr=qtr, seconds_remaining_half=seconds_remaining_half,
            shotgun=shotgun, no_huddle=no_huddle, is_home_offense=is_home_offense,
        )
        
        # Scale and predict
        X_scaled = scaler.transform(X_df)
        prob_pass = float(log_reg.predict_proba(X_scaled)[0, 1])
        prob_run = 1.0 - prob_pass
        pred_label = "PASS" if prob_pass >= 0.5 else "RUN"
        
        # Get feature contributions
        contributions = get_feature_contributions(log_reg, scaler, X_scaled, feature_data)
        
        # Store in session state so it persists
        st.session_state.last_prediction = {
            "label": pred_label,
            "prob_pass": prob_pass,
            "prob_run": prob_run,
            "contributions": contributions,
        }
    
    # Display prediction (from session state - persists across reruns)
    if st.session_state.last_prediction:
        pred = st.session_state.last_prediction
        pred_label = pred["label"]
        prob_pass = pred["prob_pass"]
        prob_run = pred["prob_run"]
        contributions = pred["contributions"]
        
        # Confidence level
        conf_text, conf_class, conf_emoji = get_confidence_level(prob_pass)
        
        # Prediction card
        card_class = "prediction-pass" if pred_label == "PASS" else "prediction-run"
        
        st.markdown(
            f"""
            <div class="prediction-card {card_class}">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                    <div style="font-size: 1.8rem; font-weight: 700;">
                        {"📡" if pred_label == "PASS" else "🏃"} {pred_label}
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
                    <span>PASS {prob_pass:.1%} 📡</span>
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
            bar_width = min(abs(contrib) * 50, 100)  # Scale for display
            
            st.markdown(
                f"""
                <div style="display: flex; align-items: center; margin: 4px 0; font-size: 0.85rem;">
                    <span style="width: 140px; color: #ccc;">{display_name}</span>
                    <div style="flex: 1; background: #2d2d2d; height: 8px; border-radius: 4px; margin: 0 10px;">
                        <div style="width: {bar_width}%; height: 100%; background: {'#4a9eff' if contrib > 0 else '#ff6b6b'}; border-radius: 4px;"></div>
                    </div>
                    <span class="{color_class}" style="width: 70px; text-align: right;">{direction}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
    
    else:
        # No prediction yet
        st.info("👆 Set up a game situation and click **Predict Play Call** to see the model's prediction.")


# =============================================================================
# FIELD VISUALIZATION
# =============================================================================

st.markdown("---")
st.markdown('<p class="section-header">🏟️ Field Position</p>', unsafe_allow_html=True)

field_fig = draw_field(yardline_100)
st.pyplot(field_fig, use_container_width=True)
plt.close(field_fig)  # Clean up memory


# =============================================================================
# FOOTER
# =============================================================================

st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #666; font-size: 0.8rem;">
        Built with Streamlit • Model: Logistic Regression trained on nflfastR data (2021-2023)
    </div>
    """,
    unsafe_allow_html=True,
)
