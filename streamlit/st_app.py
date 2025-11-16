from pathlib import Path
import json

import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle

# -----------------------
# Page config + styling
# -----------------------
st.set_page_config(
    page_title="NFL Play Call Predictor",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {
        max-width: 1150px;
        padding-top: 1.5rem;
    }

    html, body, [class*="css"] {
        font-size: 16px;
    }

    h2, h3 {
        font-weight: 700 !important;
    }

    label, .stSlider label {
        font-size: 0.95rem !important;
        font-weight: 600 !important;
    }

    .stTextInput input, .stNumberInput input {
        font-size: 1rem !important;
        padding: 0.4rem 0.6rem !important;
    }

    .prediction-card {
        background-color: #f1f8ff;
        border-radius: 12px;
        padding: 0.9rem 1.1rem;
        border: 1px solid #d0e2ff;
        margin-top: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------
# Artifact loading
# -----------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"


@st.cache_resource
def load_artifacts():
    model_path = MODELS_DIR / "log_reg_model.pkl"
    scaler_path = MODELS_DIR / "scaler.pkl"
    feature_cols_path = MODELS_DIR / "feature_cols.json"

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    with feature_cols_path.open("r") as f:
        feature_cols = json.load(f)

    return model, scaler, feature_cols


log_reg, scaler, FEATURE_COLS = load_artifacts()

# -----------------------
# Session state
# -----------------------
if "history" not in st.session_state:
    st.session_state["history"] = []
if "off_score" not in st.session_state:
    st.session_state["off_score"] = 17
if "def_score" not in st.session_state:
    st.session_state["def_score"] = 14

# -----------------------
# Helpers
# -----------------------
def draw_vertical_field(ui_pos: float):
    """
    Draw a vertical field:
    bottom = OWN goal line (0)
    top   = OPP goal line (100)
    ui_pos is 0–100 along that axis.
    """
    fig, ax = plt.subplots(figsize=(3.2, 6))  # tall vertical field

    # Field background
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 100)
    ax.axis("off")

    field = Rectangle((0, 0), 10, 100, facecolor="#1b5e20", edgecolor="white")
    ax.add_patch(field)

    # Major yard lines every 10 yards (horizontal)
    for y in range(0, 101, 10):
        ax.plot([0, 10], [y, y], color="white", linewidth=1)

    # Hash marks
    for y in range(5, 100, 10):
        ax.plot([2, 8], [y, y], color="white", linewidth=0.6)

    # Yard numbers (from each goal line perspective)
    # Own side: 10, 20, 30, 40
    for y in range(10, 50, 10):
        ax.text(
            1.2,
            y,
            str(y),
            color="white",
            ha="left",
            va="center",
            fontsize=7,
        )

    # Opponent side: 40, 30, 20, 10 (distance to opp goal)
    opp_labels = [40, 30, 20, 10]
    opp_positions = [60, 70, 80, 90]
    for lbl, y in zip(opp_labels, opp_positions):
        ax.text(
            8.8,
            y,
            str(lbl),
            color="white",
            ha="right",
            va="center",
            fontsize=7,
        )

    # End zone labels
    ax.text(5, 3, "OWN", color="white", ha="center", va="center", fontsize=8)
    ax.text(5, 97, "OPP", color="white", ha="center", va="center", fontsize=8)

    # Ball
    ball = Circle((5, ui_pos), radius=1.6, facecolor="#ffcc80", edgecolor="black")
    ax.add_patch(ball)

    # Yard line text near ball
    if ui_pos <= 50:
        side_label = "OWN"
        yard_label = ui_pos
    else:
        side_label = "OPP"
        yard_label = 100 - ui_pos

    ax.text(
        5,
        ui_pos + 5 if ui_pos < 80 else ui_pos - 5,
        f"{int(yard_label)}-yd line ({side_label})",
        color="white",
        ha="center",
        va="center",
        fontsize=8,
    )

    fig.tight_layout()
    return fig


def parse_int(text, default):
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
    down,
    ydstogo,
    yardline_100,
    offense_score,
    defense_score,
    qtr,
    seconds_remaining_half,
    shotgun,
    no_huddle,
    is_home_offense,
):
    score_diff = offense_score - defense_score

    data = {
        "down": down,
        "ydstogo": ydstogo,
        "yardline_100": yardline_100,
        "game_seconds_remaining": seconds_remaining_half,
    }

    data.update(
        {
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
    )

    df = pd.DataFrame([data])
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0
    df = df[FEATURE_COLS]
    return df


# -----------------------
# Layout
# -----------------------
st.markdown(
    "<h2 style='text-align:center; margin-bottom:0.4rem;'>🏈 NFL Play Call Predictor</h2>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align:center; margin-top:0;'>Pre-snap run vs pass prediction from 2021–2023 play-by-play.</p>",
    unsafe_allow_html=True,
)

left_col, right_col = st.columns([0.6, 0.4])

# ===== LEFT: Game situation + field =====
with left_col:
    st.markdown("### Game situation")

    # Scoreboard bar
    st.markdown(
        f"""
        <div style="
            background-color:#b71c1c;
            color:white;
            border-radius:10px;
            padding:6px 12px;
            text-align:center;
            margin-bottom:8px;
            font-size:0.95rem;
        ">
            <span style="font-weight:bold;">OFF</span> {st.session_state.off_score}
            &nbsp;&nbsp;|&nbsp;&nbsp;
            {st.session_state.def_score} <span style="font-weight:bold;">DEF</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    score_cols = st.columns(2)
    with score_cols[0]:
        st.session_state.off_score = st.number_input(
            "Offense score",
            min_value=0,
            step=1,
            value=st.session_state.off_score,
        )
    with score_cols[1]:
        st.session_state.def_score = st.number_input(
            "Defense score",
            min_value=0,
            step=1,
            value=st.session_state.def_score,
        )

    st.markdown("---")

    # Down & distance
    st.markdown("**Down & distance**")
    dd_cols = st.columns(2)
    with dd_cols[0]:
        down_text = st.text_input("Down", value="1")
    with dd_cols[1]:
        ydstogo_text = st.text_input("Yards to go", value="5")

    down = parse_int(down_text, 1)
    down = min(max(down, 1), 4)
    ydstogo = max(parse_int(ydstogo_text, 1), 1)

    # Clock & context
    st.markdown("**Clock & context**")
    cc_cols = st.columns(2)
    with cc_cols[0]:
        qtr_text = st.text_input("Quarter (1–4)", value="1")
        qtr = min(max(parse_int(qtr_text, 1), 1), 4)
    with cc_cols[1]:
        clock_text = st.text_input("Game clock (mm:ss or seconds)", value="10:00")
        seconds_remaining_half = parse_clock(clock_text, default_seconds=600)

    form_cols = st.columns(3)
    with form_cols[0]:
        shotgun = st.checkbox("Shotgun", value=True)
    with form_cols[1]:
        no_huddle = st.checkbox("No huddle", value=False)
    with form_cols[2]:
        is_home_offense = st.checkbox("Offense is home", value=True)

    st.markdown("**Field & position**")
    field_col, slider_col = st.columns([0.55, 0.45])

    with slider_col:
        ui_pos = st.slider(
            "Ball position (0–50 own, 50–100 opponent)",
            min_value=0,
            max_value=100,
            value=60,
        )

        if ui_pos <= 50:
            side_label = "OWN"
            yard_label = ui_pos
        else:
            side_label = "OPP"
            yard_label = 100 - ui_pos

        st.caption(f"Ball on {side_label} {int(yard_label)}-yard line")

        # Convert to yardline_100 (distance to opponent end zone) for the model
        yardline_100 = 100 - ui_pos

    with field_col:
        field_fig = draw_vertical_field(ui_pos)
        st.pyplot(field_fig, use_container_width=True)

# ===== RIGHT: Prediction & history =====
with right_col:
    st.markdown("### Model output")

    if st.button("Predict play call", use_container_width=True):
        X_example = build_feature_vector(
            down=down,
            ydstogo=ydstogo,
            yardline_100=yardline_100,
            offense_score=st.session_state.off_score,
            defense_score=st.session_state.def_score,
            qtr=qtr,
            seconds_remaining_half=seconds_remaining_half,
            shotgun=shotgun,
            no_huddle=no_huddle,
            is_home_offense=is_home_offense,
        )

        X_scaled = scaler.transform(X_example)
        prob_pass = float(log_reg.predict_proba(X_scaled)[0, 1])
        prob_run = 1.0 - prob_pass
        pred_label = "PASS" if prob_pass >= 0.5 else "RUN"

        st.markdown(
            f"""
            <div class="prediction-card">
                <div style="font-size:1.1rem; font-weight:700; margin-bottom:0.25rem;">
                    Prediction: {pred_label}
                </div>
                <div>Pass probability: <strong>{prob_pass:.1%}</strong></div>
                <div>Run probability: <strong>{prob_run:.1%}</strong></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Explanation bullets
        reasons = []
        if ydstogo >= 8:
            reasons.append("long yardage")
        elif ydstogo <= 3:
            reasons.append("short yardage")

        score_diff = st.session_state.off_score - st.session_state.def_score
        if score_diff < 0:
            reasons.append("trailing")
        elif score_diff > 0:
            reasons.append("leading")

        if shotgun:
            reasons.append("shotgun formation")
        if seconds_remaining_half < 120:
            reasons.append("late in the half")
        if yardline_100 <= 20:
            reasons.append("in or near the red zone")

        if reasons:
            st.markdown(
                "🧠 **Why?** Based on features like "
                + ", ".join(reasons)
                + ", the model leans toward this play type."
            )

        # Add to history
        st.session_state.history.append(
            {
                "Down": down,
                "To go": ydstogo,
                "Yardline_100": yardline_100,
                "Off score": st.session_state.off_score,
                "Def score": st.session_state.def_score,
                "Qtr": qtr,
                "Clock (s)": seconds_remaining_half,
                "Shotgun": shotgun,
                "No huddle": no_huddle,
                "Pred": pred_label,
                "P(pass)": round(prob_pass, 3),
            }
        )

        with st.expander("Show raw feature vector sent to model"):
            st.dataframe(X_example.T, use_container_width=True)
    else:
        st.markdown(
            """
            <div class="prediction-card">
                Set up a situation on the left and click
                <strong>Predict play call</strong>.
            </div>
            """,
            unsafe_allow_html=True,
        )

    if st.session_state.history:
        st.markdown("#### Prediction history (this session)")
        hist_df = pd.DataFrame(st.session_state.history)[
            [
                "Down",
                "To go",
                "Yardline_100",
                "Off score",
                "Def score",
                "Qtr",
                "Clock (s)",
                "Shotgun",
                "No huddle",
                "Pred",
                "P(pass)",
            ]
        ]
        st.dataframe(hist_df, use_container_width=True, height=260)
