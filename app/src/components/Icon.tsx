import type { LucideIcon } from "lucide-react";
import {
  Activity,
  AlertTriangle,
  ArrowRight,
  Box,
  Check,
  Clock,
  FolderOpen,
  Gauge,
  Grid3x3,
  Image,
  KeyRound,
  Orbit,
  Palette,
  Shapes,
  Trash2,
  Triangle,
  X,
} from "lucide-react";

/** Thin wrapper around lucide-react so every icon in the app shares one stroke
 *  weight and default size, instead of each call site picking its own. Line-style
 *  icons, not filled/emoji — matches the rest of the UI's flat, hairline-bordered
 *  chrome instead of introducing a heavier, glossier icon language. */
interface IconProps {
  icon: LucideIcon;
  size?: number;
  className?: string;
}

export function Icon({ icon: IconComponent, size = 14, className }: IconProps) {
  return (
    <IconComponent size={size} strokeWidth={1.75} aria-hidden="true" className={className} />
  );
}

export const Icons = {
  blender: Box,
  folder: FolderOpen,
  delete: Trash2,
  cancel: X,
  viewResult: ArrowRight,
  warning: AlertTriangle,
  confirm: Check,
  runsToday: Clock,
  inProgress: Activity,
  avgScore: Gauge,
  cleanTopology: Shapes,
  triangles: Triangle,
  materials: Palette,
  viewRender: Image,
  viewModel: Orbit,
  viewWireframe: Grid3x3,
  key: KeyRound,
} as const;
