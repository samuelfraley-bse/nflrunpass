// components/VerticalField.tsx
type Props = {
  uiPos: number; // 0 bottom (OWN GL) → 100 top (OPP GL)
};

export default function VerticalField({ uiPos }: Props) {
  const clamped = Math.max(0, Math.min(100, uiPos));
  const sideLabel = clamped <= 50 ? "OWN" : "OPP";
  const yardLabel = clamped <= 50 ? clamped : 100 - clamped;

  // We'll map 0–100 (yards) into 0–40 in the SVG for a chunkier aspect ratio
  const S = 0.4; // scale factor: 100 * 0.4 = 40
  const yBall = clamped * S;

  return (
    <svg
      // Wider, shorter coordinate system: 20 (x) × 40 (y)
      viewBox="0 0 20 40"
      width={220}
      height={360}
      className="field-svg"
      preserveAspectRatio="xMidYMid meet"
    >
      {/* Field background */}
      <rect x={0} y={0} width={20} height={40} fill="#1b5e20" stroke="white" />

      {/* Major yard lines every 10 yards (scaled) */}
      {Array.from({ length: 11 }).map((_, i) => {
        const y = i * 10 * S; // 0,4,8,...,40
        return (
          <line
            key={y}
            x1={0}
            x2={20}
            y1={y}
            y2={y}
            stroke="white"
            strokeWidth={0.2}
          />
        );
      })}

      {/* Hash marks every 10 yards, offset by 5 yards */}
      {Array.from({ length: 10 }).map((_, i) => {
        const y = (5 + i * 10) * S; // 2,6,...,38
        return (
          <line
            key={y}
            x1={4}
            x2={16}
            y1={y}
            y2={y}
            stroke="white"
            strokeWidth={0.15}
          />
        );
      })}

      {/* Endzone labels */}
      <text
        x={10}
        y={2}
        textAnchor="middle"
        alignmentBaseline="middle"
        fill="white"
        fontSize={2.2}
      >
        OPP
      </text>
      <text
        x={10}
        y={38}
        textAnchor="middle"
        alignmentBaseline="middle"
        fill="white"
        fontSize={2.2}
      >
        OWN
      </text>

      {/* Ball */}
      <ellipse
        cx={10}
        cy={yBall}
        rx={2.4}
        ry={1.8}
        fill="#ffcc80"
        stroke="black"
        strokeWidth={0.18}
      />

      {/* Label above/below ball */}
      <text
        x={10}
        y={yBall < 30 ? yBall + 3 : yBall - 3}
        textAnchor="middle"
        alignmentBaseline="middle"
        fill="white"
        fontSize={1.8}
      >
        {Math.round(yardLabel)} ({sideLabel})
      </text>
    </svg>
  );
}
