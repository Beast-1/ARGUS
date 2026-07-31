import { Logo } from "./Logo";
import type { Route } from "../routes";
import "./TopNav.css";

interface TopNavProps {
  current: Route;
  onNavigate: (route: Route) => void;
}

const ITEMS: { route: Route; label: string }[] = [
  { route: "workbench", label: "Workbench" },
  { route: "dashboard", label: "Dashboard" },
];

export function TopNav({ current, onNavigate }: TopNavProps) {
  return (
    <div className="top-nav">
      <span className="top-nav-brand">
        <Logo />
        <b>ARGUS</b>
      </span>
      {ITEMS.map((item) => (
        <a
          key={item.route}
          className={item.route === current ? "current" : undefined}
          onClick={() => onNavigate(item.route)}
        >
          {item.label}
        </a>
      ))}
    </div>
  );
}
