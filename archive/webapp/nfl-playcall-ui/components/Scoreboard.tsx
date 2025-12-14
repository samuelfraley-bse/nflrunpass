type Props = {
  offScore: number;
  defScore: number;
  onChangeOff: (score: number) => void;
  onChangeDef: (score: number) => void;
};

export default function Scoreboard({
  offScore,
  defScore,
  onChangeOff,
  onChangeDef,
}: Props) {
  const bump = (current: number, delta: number) =>
    Math.max(0, current + delta);

  return (
    <div className="scoreboard">
      <div className="score-segment">
        <span className="score-label">OFF</span>
        <div className="score-value-row">
          <button
            type="button"
            className="score-btn"
            onClick={() => onChangeOff(bump(offScore, -1))}
          >
            –
          </button>
          <span className="score-value">{offScore}</span>
          <button
            type="button"
            className="score-btn"
            onClick={() => onChangeOff(bump(offScore, +1))}
          >
            +
          </button>
        </div>
      </div>

      <div className="score-divider" />

      <div className="score-segment">
        <span className="score-label">DEF</span>
        <div className="score-value-row">
          <button
            type="button"
            className="score-btn"
            onClick={() => onChangeDef(bump(defScore, -1))}
          >
            –
          </button>
          <span className="score-value">{defScore}</span>
          <button
            type="button"
            className="score-btn"
            onClick={() => onChangeDef(bump(defScore, +1))}
          >
            +
          </button>
        </div>
      </div>
    </div>
  );
}
