type Props = {
  offScore: number;
  defScore: number;
};

export default function Scoreboard({ offScore, defScore }: Props) {
  return (
    <div className="scoreboard">
      <span className="score-label">OFF</span> {offScore}
      <span className="score-sep">|</span>
      {defScore} <span className="score-label">DEF</span>
    </div>
  );
}
