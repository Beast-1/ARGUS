import type { CSSProperties } from "react";
import type { LucideIcon } from "lucide-react";
import { Icon } from "./Icon";
import "./StatStrip.css";

export interface Stat {
  value: string | number;
  label: string;
  color?: string;
  icon?: LucideIcon;
}

export function StatStrip({ stats }: { stats: Stat[] }) {
  return (
    // The count rides in as a custom property rather than as an inline
    // grid-template-columns, because an inline declaration outranks every
    // stylesheet rule — including the media query that reflows this to 2x2 on a
    // phone. As a variable the value stays overridable.
    <div
      className="stat-strip"
      style={{ "--stat-count": stats.length } as CSSProperties}
    >
      {stats.map((s) => (
        <div className="stat-strip-item" key={s.label}>
          <div className="stat-strip-n" style={s.color ? { color: s.color } : undefined}>
            {s.value}
          </div>
          <div className="stat-strip-l">
            {s.icon && <Icon icon={s.icon} size={11} className="stat-strip-icon" />}
            {s.label}
          </div>
        </div>
      ))}
    </div>
  );
}
