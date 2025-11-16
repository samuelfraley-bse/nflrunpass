import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle

# ---------- Page config ----------
st.set_page_config(
    page_title="NFL Play Call Predictor",
    layout="wide",
)

# ---------- Load artifacts ----------
@st.cache_resource
def load_artifacts():
    model = joblib.load("log_reg_model.pkl")
    scaler = joblib.load("scaler.pkl")
    with open("feature_cols.json", "r") as f:
        feature_cols = json.load(f)
    return model, scaler, feature_cols

log_reg, scaler, FEATURE_COLS = load_artifacts()

# ---------- Session state ----------
if "history" not in st.session_state:
    st.session_state["history"] = []
if "off_score" not in st.session_state:
    st.session_state["off_score"] = 17
if "def_score" not in st.session_state:
    st.session_state["def_score"] = 14


# ---------- Helpers ----------
def draw_field(yardline_100: float):
    """Compact green field with yard numbers and a ball marker."""
    x_ball = 100 - yardline_100

    fig, ax = plt.subplots(figsize=(4.8, 1.4))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 10)
    ax.axis("off")

    field = Rectangle((0, 0), 100, 10, facecolor="#1b5e20", edgecolor="white")
    ax.add_patch(field)

    # Major yard lines
    for x in range(0, 101, 10):
        ax.plot([x, x], [0, 10], color="white", linewidth=1)
    # Minor hashes
    for x in range(5, 100, 10):
        ax.plot([x, x], [3, 7], color="white", linewidth=0.6)

    # Midfield
    ax.plot([50, 50], [0, 10], color="white", linewidth=2)

    # Yard numbers
    for x in range(10, 50, 10):
        ax.text(x, 1.3, str(x), color="white", ha="center", va="center", fontsize=7)
        ax.text(100 - x, 1.3, str(x), color="white", ha="center", va="center", fontsize=7)

    ax.text(5, 5, "OWN", color="white", ha="center", va="center", fontsize=7)
    ax.text(95, 5, "OPP", color="white", ha="center", va="center", fontsize=7)

    ball = Circle((x_ball, 5), radius=1.2, facecolor="#ffcc80", edgecolor="black")
    ax.add_patch(ball)
    ax.text(
        x_ball,
        8.4,
        f"{int(100 - yardline_100)}-yd line",
        color="white",
        ha="center",
        va="center",
        fontsize=7,
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


# ---------- Layout ----------
st.markdown(
    "<h2 style='text-align:center; margin-bottom:0.4rem;'>🏈 NFL Play Call Predictor</h2>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align:center; margin-top:0;'>Pre-snap run vs pass prediction from 2021–2023 play-by-play.</p>",
    unsafe_allow_html=True,
)

left_col, right_col = st.columns([1.0, 1.1])

# ===== LEFT: Game situation =====
with left_col:
    st.markdown("### Game situation")

    # --- Compact red scoreboard ---
    sb = st.container()
    with sb:
        st.markdown(
            f"""
            <div style="
                background-color:#b71c1c;
                color:white;
                border-radius:10px;
                padding:6px 12px;
                text-align:center;
                margin-bottom:6px;
                font-size:0.95rem;
            ">
                <span style="font-weight:bold;">OFF</span> {st.session_state.off_score}
                &nbsp;&nbsp;|&nbsp;&nbsp;
                {st.session_state.def_score} <span style="font-weight:bold;">DEF</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        btn_row = st.columns(4)
        with btn_row[0]:
            if st.button("OFF -", use_container_width=True):
                st.session_state.off_score = max(0, st.session_state.off_score - 1)
        with btn_row[1]:
            if st.button("OFF +", use_container_width=True):
                st.session_state.off_score += 1
        with btn_row[2]:
            if st.button("DEF -", use_container_width=True):
                st.session_state.def_score = max(0, st.session_state.def_score - 1)
        with btn_row[3]:
            if st.button("DEF +", use_container_width=True):
                st.session_state.def_score += 1

    st.markdown("---")

    # --- Down & distance (text boxes) ---
    st.markdown("**Down & distance**")
    dd_cols = st.columns(2)
    with dd_cols[0]:
        down_text = st.text_input("Down", value="1")
    with dd_cols[1]:
        ydstogo_text = st.text_input("Yards to go", value="5")

    down = parse_int(down_text, 1)
    down = min(max(down, 1), 4)  # clamp 1–4
    ydstogo = max(parse_int(ydstogo_text, 1), 1)

    # --- Field position ---
    st.markdown("**Field position**")
    yardline_100 = st.slider(
        "Distance to opponent end zone (0 = goal line, 100 = own goal line)",
        0,
        100,
        60,
    )
    field_fig = draw_field(yardline_100)
    st.pyplot(field_fig, use_container_width=True)

    # --- Clock & context (text for quarter + game clock) ---
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


# ===== RIGHT: Prediction & history =====
with right_col:
    st.markdown("### Model output")

    if st.button("Predict play call"):
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

        st.markdown(f"#### Prediction: **{pred_label}**")
        st.write(f"Pass probability: **{prob_pass:.1%}**")
        st.write(f"Run probability: **{prob_run:.1%}**")

        # Short explanation
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
        st.info("Set up a situation on the left and click **Predict play call**.")

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
        st.dataframe(hist_df, use_container_width=True, height=250)
