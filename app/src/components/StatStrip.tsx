import "./StatStrip.css";

export interface Stat {
  value: string | number;
  label: string;
  color?: string;
}

export function StatStrip({ stats }: { stats: Stat[] }) {
  return (
    <div className="stat-strip" style={{ gridTemplateColumns: `repeat(${stats.length}, 1fr)` }}>
      {stats.map((s) => (
        <div className="stat-strip-item" key={s.label}>
          <div className="stat-strip-n" style={s.color ? { color: s.color } : undefined}>
            {s.value}
          </div>
          <div className="stat-strip-l">{s.label}</div>
        </div>
      ))}
    </div>
  );
}
