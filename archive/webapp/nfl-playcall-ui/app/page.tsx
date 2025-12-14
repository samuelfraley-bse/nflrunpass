"use client";

import { useState } from "react";
import Scoreboard from "@/components/Scoreboard";
import VerticalField from "@/components/VerticalField";

type PredictionResponse = {
  prediction: string;
  prob_pass: number;
  prob_run: number;
};

const API_BASE = (
  process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000"
).replace(/\/+$/, "");

export default function Home() {
  // ----- Form state -----
  const [offScore, setOffScore] = useState(17);
  const [defScore, setDefScore] = useState(14);

  const [down, setDown] = useState(1);
  const [ydstogo, setYdstogo] = useState(5);
  const [qtr, setQtr] = useState(1);
  const [clock, setClock] = useState("10:00");

  const [shotgun, setShotgun] = useState(true);
  const [noHuddle, setNoHuddle] = useState(false);
  const [isHomeOffense, setIsHomeOffense] = useState(true);

  // 0 = own goal line (bottom), 100 = opponent goal line (top)
  const [uiPos, setUiPos] = useState(60);

  // ----- Prediction state -----
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [predLabel, setPredLabel] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  // derived: label for ball position
  const sideLabel = uiPos <= 50 ? "OWN" : "OPP";
  const yardLabel = uiPos <= 50 ? uiPos : 100 - uiPos;

  // helpers
  const parseClock = (txt: string): number => {
    const s = txt.trim();
    if (!s) return 600;
    if (s.includes(":")) {
      const [m, sec] = s.split(":");
      const mm = parseInt(m, 10);
      const ss = parseInt(sec, 10);
      if (!Number.isNaN(mm) && !Number.isNaN(ss)) return mm * 60 + ss;
    }
    const raw = parseInt(s, 10);
    return Number.isNaN(raw) ? 600 : raw;
  };

const handlePredict = async () => {
  setLoading(true);
  setErrorMsg(null);
  setPrediction(null);
  setPredLabel(null);

  const secondsRemainingHalf = parseClock(clock);
  const yardline_100 = 100 - uiPos;

  const payload = {
    down,
    ydstogo,
    yardline_100,
    offense_score: offScore,
    defense_score: defScore,
    qtr,
    seconds_remaining_half: secondsRemainingHalf,
    shotgun,
    no_huddle: noHuddle,
    is_home_offense: isHomeOffense,
  };

  try {
    const res = await fetch(`${API_BASE}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      const text = await res.text().catch(() => "");
      console.error("API non-OK response:", res.status, text);
      throw new Error(`API error: ${res.status}`);
    }

    const data: PredictionResponse = await res.json();
    setPrediction(data);
    setPredLabel(data.prediction);
  } catch (err: any) {
    console.error("Prediction API request failed:", err);
    setErrorMsg(
      `Could not reach prediction API. Currently configured base URL: ${API_BASE}`
    );
  } finally {
    setLoading(false);
  }
};


  return (
    <main className="page">
      <header className="page-header">
        <h1>🏈 NFL Play Call Predictor</h1>
        <p>Use machine learning to predict run-pass probability, trained on 2021-2023 data & pre-snap indicators.</p>
      </header>

      <section className="layout">
        {/* LEFT: inputs + field */}
        <div className="left-column">
          <h2 className="section-title">Game situation</h2>

          {/* Scoreboard with +/- controls */}
          <Scoreboard
            offScore={offScore}
            defScore={defScore}
            onChangeOff={setOffScore}
            onChangeDef={setDefScore}
          />

          <hr className="divider" />

          {/* Down & distance */}
          <h3 className="subheading">Down &amp; distance</h3>
          <div className="two-col-row">
            <div className="field-group">
              <label>Down</label>
              <input
                type="number"
                min={1}
                max={4}
                value={down}
                onChange={(e) =>
                  setDown(
                    Math.min(
                      4,
                      Math.max(1, parseInt(e.target.value || "1", 10))
                    )
                  )
                }
              />
            </div>
            <div className="field-group">
              <label>Yards to go</label>
              <input
                type="number"
                min={1}
                value={ydstogo}
                onChange={(e) =>
                  setYdstogo(
                    Math.max(1, parseInt(e.target.value || "1", 10))
                  )
                }
              />
            </div>
          </div>

          {/* Clock & context */}
          <h3 className="subheading">Clock &amp; context</h3>
          <div className="two-col-row">
            <div className="field-group">
              <label>Quarter (1–4)</label>
              <input
                type="number"
                min={1}
                max={4}
                value={qtr}
                onChange={(e) =>
                  setQtr(
                    Math.min(
                      4,
                      Math.max(1, parseInt(e.target.value || "1", 10))
                    )
                  )
                }
              />
            </div>
            <div className="field-group">
              <label>Game clock (mm:ss or seconds)</label>
              <input
                type="text"
                value={clock}
                onChange={(e) => setClock(e.target.value)}
              />
            </div>
          </div>

          <div className="checkbox-row">
            <label className="checkbox">
              <input
                type="checkbox"
                checked={shotgun}
                onChange={(e) => setShotgun(e.target.checked)}
              />
              <span>Shotgun</span>
            </label>
            <label className="checkbox">
              <input
                type="checkbox"
                checked={noHuddle}
                onChange={(e) => setNoHuddle(e.target.checked)}
              />
              <span>No huddle</span>
            </label>
            <label className="checkbox">
              <input
                type="checkbox"
                checked={isHomeOffense}
                onChange={(e) => setIsHomeOffense(e.target.checked)}
              />
              <span>Offense is home</span>
            </label>
          </div>

          {/* Field + vertical slider */}
          <h3 className="subheading">Field &amp; position</h3>
          <div className="field-layout">
            <div className="field-graphic">
              <VerticalField uiPos={uiPos} />
            </div>
            <div className="field-slider">
              <span className="slider-label">
                Ball position (0–50 own, 50–100 opponent)
              </span>
              <div className="vertical-slider-wrapper">
                <input
                  type="range"
                  min={0}
                  max={100}
                  value={uiPos}
                  onChange={(e) => setUiPos(parseInt(e.target.value, 10))}
                  className="vertical-slider"
                />
              </div>
              <div className="slider-caption">
                Ball on {sideLabel} {Math.round(yardLabel)}-yard line
              </div>
            </div>
          </div>
        </div>

        {/* RIGHT: prediction */}
        <div className="right-column">
          <h2 className="section-title">Model output</h2>
          <button
            className="predict-button"
            onClick={handlePredict}
            disabled={loading}
          >
            {loading ? "Predicting..." : "Predict play call"}
          </button>

          {errorMsg && <div className="error-card">{errorMsg}</div>}

          {!prediction && !errorMsg && (
            <div className="prediction-card">
              Set up a situation on the left and click{" "}
              <strong>Predict play call</strong>.
            </div>
          )}

          {prediction && predLabel && (
            <div className="prediction-card">
              <div className="prediction-title">
                Prediction: <span>{predLabel}</span>
              </div>
              <div>
                Pass probability:{" "}
                <strong>
                  {(prediction.prob_pass * 100).toFixed(1)}%
                </strong>
              </div>
              <div>
                Run probability:{" "}
                <strong>
                  {(prediction.prob_run * 100).toFixed(1)}%
                </strong>
              </div>
            </div>
          )}
        </div>
      </section>
    </main>
  );
}
