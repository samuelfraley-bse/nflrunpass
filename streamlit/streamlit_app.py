"""
NFL Play Call Predictor - Enhanced Version

Features:
1. Persistent predictions (stays visible after other interactions)
2. Feature importance display (shows why the model decided)
3. Confidence indicator (high/medium/low)
4. Probability visualization (horizontal bar)
5. Cleaner visual design
6. PREDICTION HISTORY tracker
7. ENHANCED FIELD with colored zones
8. STADIUM BACKGROUND IMAGE
"""

import json
from pathlib import Path
import base64

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
# LOAD BACKGROUND IMAGE
# =============================================================================

def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

img_path = Path(__file__).parent / "assets" / "stadium.png"
img_base64 = get_base64_image(img_path)

# =============================================================================
# CUSTOM CSS
# =============================================================================

st.markdown(
    f"""
    <style>
    /* Main container */
    .block-container {{
        max-width: 1200px;
        padding-top: 1rem;
    }}

    /* Prediction card styles */
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

    /* Confidence badge */
    .confidence-high {{ color: #4ade80; }}
    .confidence-medium {{ color: #fbbf24; }}
    .confidence-low {{ color: #f87171; }}

    /* Feature importance */
    .feature-positive {{ color: #4a9eff; }}
    .feature-negative {{ color: #ff6b6b; }}
    
    /* Scoreboard */
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

    /* Section headers */
    .section-header {{
        font-size: 1.3rem;
        font-weight: 700;
        margin-bottom: 1rem;
        color: #e0e0e0;
    }}

    /* Probability bar */
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
    
    /* Background image */
    .stApp {{
        background-image: url('data:image/png;base64,{img_base64}');
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}

    /* Dark overlay for readability */
    .stApp::before {{
        content: "";
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0, 0, 0, 0.1);
        z-index: -1;
    }}
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

# Prediction state (makes prediction persist)
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None

# Prediction history
if "history" not in st.session_state:
    st.session_state.history = []

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def draw_field(yardline_100: float):
    """Enhanced field with colored zones, better yard markers."""
    x_ball = 100 - yardline_100

    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 10)
    ax.axis("off")
    fig.patch.set_facecolor('#0e1117')

    # Base field
    field = Rectangle((0, 0), 100, 10, facecolor="#1b5e20", edgecolor="white", linewidth=2)
    ax.add_patch(field)

    # Red zone highlight (both sides)
    red_zone_left = Rectangle((0, 0), 20, 10, facecolor="#c62828", alpha=0.25, edgecolor=None)
    red_zone_right = Rectangle((80, 0), 20, 10, facecolor="#c62828", alpha=0.25, edgecolor=None)
    ax.add_patch(red_zone_left)
    ax.add_patch(red_zone_right)

    # End zones
    end_zone_left = Rectangle((0, 0), 10, 10, facecolor="#0d47a1", alpha=0.3, edgecolor="white", linewidth=1)
    end_zone_right = Rectangle((90, 0), 10, 10, facecolor="#0d47a1", alpha=0.3, edgecolor="white", linewidth=1)
    ax.add_patch(end_zone_left)
    ax.add_patch(end_zone_right)

    # Major yard lines
    for x in range(0, 101, 10):
        lw = 2.5 if x == 50 else 1.5
        color = "yellow" if x == 50 else "white"
        ax.plot([x, x], [0, 10], color=color, linewidth=lw)

    # Minor hash marks
    for x in range(5, 100, 10):
        ax.plot([x, x], [3.5, 6.5], color="white", linewidth=0.8, alpha=0.7)

    # Yard numbers
    for x in range(10, 50, 10):
        ax.text(x, 1.8, str(x), color="white", ha="center", va="center", 
                fontsize=10, fontweight="bold")
        ax.text(100 - x, 1.8, str(x), color="white", ha="center", va="center", 
                fontsize=10, fontweight="bold")

    # 50 yard line
    ax.text(50, 1.8, "50", color="yellow", ha="center", va="center", 
            fontsize=11, fontweight="bold")

    # End zone labels
    ax.text(5, 5, "END\nZONE", color="white", ha="center", va="center", 
            fontsize=8, fontweight="bold", alpha=0.8)
    ax.text(95, 5, "END\nZONE", color="white", ha="center", va="center", 
            fontsize=8, fontweight="bold", alpha=0.8)

    # Ball marker with glow
    glow = Circle((x_ball, 5), radius=2, facecolor="#ffcc80", alpha=0.3)
    ax.add_patch(glow)
    ball = Circle((x_ball, 5), radius=1.5, facecolor="#ff6f00", edgecolor="white", linewidth=2)
    ax.add_patch(ball)
    
    # Ball position label
    ax.text(x_ball, 8.8, f"Ball at {int(100 - yardline_100)}", 
            color="white", ha="center", va="center", fontsize=9, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.7, edgecolor="white"))

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


def build_feature_vector(down, ydstogo, yardline_100, offense_score, defense_score,
                        qtr, seconds_remaining_half, shotgun, no_huddle, is_home_offense):
    """Build feature DataFrame with all 18 features."""
    score_diff = offense_score - defense_score

    feature_data = {
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

    df = pd.DataFrame([feature_data])
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0
    df = df[FEATURE_COLS]
    
    return df, feature_data


def get_confidence_level(prob_pass):
    """Determine confidence level from probability."""
    max_prob = max(prob_pass, 1 - prob_pass)
    if max_prob >= 0.7:
        return "High", "confidence-high", "🟢"
    elif max_prob >= 0.55:
        return "Medium", "confidence-medium", "🟡"
    else:
        return "Low", "confidence-low", "🔴"


def get_feature_contributions(model, scaler, X_scaled, feature_data):
    """Get top feature contributions to the prediction."""
    coefficients = model.coef_[0]
    
    # Get scaled feature values
    feature_values = X_scaled[0]
    
    # Calculate contribution (coefficient * scaled_value)
    contributions = []
    for i, col in enumerate(FEATURE_COLS):
        contrib = coefficients[i] * feature_values[i]
        if abs(contrib) > 0.05:  # Only show significant contributions
            contributions.append((col, contrib))
    
    # Sort by absolute contribution
    contributions.sort(key=lambda x: abs(x[1]), reverse=True)
    return contributions[:5]  # Top 5


def format_situation(down, ydstogo, yardline_100, qtr, score_diff):
    """Format readable situation string."""
    down_text = {1: "1st", 2: "2nd", 3: "3rd", 4: "4th"}[down]
    qtr_text = f"Q{qtr}"
    
    if yardline_100 > 50:
        field = f"Own {100-yardline_100:.0f}"
    else:
        field = f"Opp {yardline_100:.0f}"
    
    if score_diff > 0:
        score = f"+{score_diff}"
    elif score_diff < 0:
        score = f"{score_diff}"
    else:
        score = "Tied"
    
    return f"{down_text}&{ydstogo} at {field}, {qtr_text}, {score}"


# Feature display names
FEATURE_DISPLAY_NAMES = {
    "down": "Down",
    "ydstogo": "Yards to go",
    "yardline_100": "Field position",
    "game_seconds_remaining": "Time remaining",
    "is_red_zone": "Red zone",
    "is_goal_to_go": "Goal to go",
    "short_ydstogo": "Short yardage",
    "medium_ydstogo": "Medium yardage",
    "long_ydstogo": "Long yardage",
    "shotgun": "Shotgun",
    "no_huddle": "No huddle",
    "score_differential": "Score differential",
    "is_trailing": "Trailing",
    "is_tied": "Tied",
    "is_leading": "Leading",
    "is_fourth_qtr": "4th quarter",
    "late_half": "Late in half",
    "is_home_offense": "Home offense",
}

# =============================================================================
# PAGE LAYOUT
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
    
    # Score inputs
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
    
    # Update session state
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
            offense_score=off_score, defense_score=def_score,
            qtr=qtr, seconds_remaining_half=seconds_remaining_half,
            shotgun=shotgun, no_huddle=no_huddle, is_home_offense=is_home_offense,
        )
        
        # Scale and predict
        X_scaled_array = scaler.transform(X_df)
        X_scaled = pd.DataFrame(X_scaled_array, columns=X_df.columns)
        
        prob_pass = float(log_reg.predict_proba(X_scaled)[0, 1])
        prob_run = 1.0 - prob_pass
        pred_label = "PASS" if prob_pass >= 0.5 else "RUN"
        
        # Get feature contributions
        contributions = get_feature_contributions(log_reg, scaler, X_scaled.values, feature_data)
        
        # Determine confidence
        conf_text, conf_class, conf_emoji = get_confidence_level(prob_pass)
        
        # Store in session state
        st.session_state.last_prediction = {
            "label": pred_label,
            "prob_pass": prob_pass,
            "prob_run": prob_run,
            "contributions": contributions,
            "confidence": (conf_text, conf_class, conf_emoji)
        }
        
        # Add to history
        score_diff = off_score - def_score
        situation = format_situation(down, ydstogo, yardline_100, qtr, score_diff)
        
        st.session_state.history.insert(0, {
            "Situation": situation,
            "Prediction": pred_label,
            "Pass %": f"{prob_pass:.1%}",
            "Confidence": conf_text
        })
        
        # Keep only last 10
        if len(st.session_state.history) > 10:
            st.session_state.history = st.session_state.history[:10]
    
    # Display prediction (from session state - persists)
    if st.session_state.last_prediction:
        pred = st.session_state.last_prediction
        pred_label = pred["label"]
        prob_pass = pred["prob_pass"]
        prob_run = pred["prob_run"]
        contributions = pred["contributions"]
        conf_text, conf_class, conf_emoji = pred["confidence"]
        
        # Prediction card
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
        
        # Probability bar (original style)
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
        st.info("👆 Set up a game situation and click **Predict Play Call** to see the model's prediction.")


# =============================================================================
# FIELD VISUALIZATION (ENHANCED)
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
        Built with Streamlit • Model: Logistic Regression trained on nflfastR data (2021-2023)
    </div>
    """,
    unsafe_allow_html=True,
)
