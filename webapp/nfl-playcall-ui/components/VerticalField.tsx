type Props = {
  uiPos: number; // 0 bottom (OWN GL) → 100 top (OPP GL)
};

export default function VerticalField({ uiPos }: Props) {
  const clamped = Math.max(0, Math.min(100, uiPos));

  const sideLabel = clamped <= 50 ? "OWN" : "OPP";
  const yardLabel = clamped <= 50 ? clamped : 100 - clamped;

  return (
    <svg
      viewBox="0 0 10 100"
      className="field-svg"
      preserveAspectRatio="xMidYMid meet"
    >
      {/* Field background */}
      <rect x="0" y="0" width="10" height="100" fill="#1b5e20" stroke="white" />

      {/* Major yard lines */}
      {Array.from({ length: 11 }).map((_, i) => {
        const y = i * 10;
        return (
          <line
            key={y}
            x1="0"
            x2="10"
            y1={y}
            y2={y}
            stroke="white"
            strokeWidth="0.15"
          />
        );
      })}

      {/* Hash marks */}
      {Array.from({ length: 10 }).map((_, i) => {
        const y = 5 + i * 10;
        return (
          <line
            key={y}
            x1="2"
            x2="8"
            y1={y}
            y2={y}
            stroke="white"
            strokeWidth="0.1"
          />
        );
      })}

      {/* Endzone labels */}
      <text
        x="5"
        y="4"
        textAnchor="middle"
        alignmentBaseline="middle"
        fill="white"
        fontSize="2"
      >
        OPP
      </text>
      <text
        x="5"
        y="96"
        textAnchor="middle"
        alignmentBaseline="middle"
        fill="white"
        fontSize="2"
      >
        OWN
      </text>

      {/* Ball */}
      <ellipse
        cx="5"
        cy={clamped}
        rx="1.5"
        ry="2.2"
        fill="#ffcc80"
        stroke="black"
        strokeWidth="0.1"
      />

      {/* Label above/below ball */}
      <text
        x="5"
        y={clamped < 80 ? clamped + 5 : clamped - 5}
        textAnchor="middle"
        alignmentBaseline="middle"
        fill="white"
        fontSize="1.8"
      >
        {Math.round(yardLabel)} ({sideLabel})
      </text>
    </svg>
  );
}
