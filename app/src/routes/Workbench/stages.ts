/** Canonical stage order for the inspector's pipeline rail. Numbers mirror
 *  service/pipeline_events.py's STAGE_LABELS (itself ported from desktop_app.py).
 *
 *  The numbering carries information the old table threw away. 55 and 75 are not
 *  the 55th and 75th stages — they are the visual-scoring passes wedged between
 *  5→6 and 7→8, and rendering them zero-padded in a column next to "01" made the
 *  sequence look like a numbering bug. They are marked as loop steps here and
 *  drawn as branches off the spine with no ordinal, so the run reads as what it
 *  is: eight stations, with two inspection passes that can repeat. */
export interface StageSpec {
  /** Wire number from the backend. */
  number: number;
  label: string;
  /** Position in the eight-station sequence; absent for the loop passes. */
  ordinal?: number;
}

export const STAGE_SEQUENCE: StageSpec[] = [
  { number: 1, label: "Plan", ordinal: 1 },
  { number: 2, label: "Naming", ordinal: 2 },
  { number: 3, label: "Script", ordinal: 3 },
  { number: 4, label: "Review", ordinal: 4 },
  { number: 5, label: "Export", ordinal: 5 },
  { number: 55, label: "Visual pass" },
  { number: 6, label: "Topology", ordinal: 6 },
  { number: 7, label: "Quality", ordinal: 7 },
  { number: 75, label: "Visual QA" },
  { number: 8, label: "Reference", ordinal: 8 },
];

/** Stations only — the denominator for "3 of 8". The loop passes are excluded
 *  because they can run any number of times, so counting them would make the
 *  total move around while the operator is watching it. */
export const STATION_COUNT = STAGE_SEQUENCE.filter((s) => s.ordinal !== undefined).length;

/** Compact durations for the rail's right column: seconds up to a minute, then
 *  minutes and seconds. Never milliseconds — the operator is judging whether a
 *  stage is slow, not profiling it. */
export function formatDuration(ms: number): string {
  const total = Math.max(0, Math.round(ms / 1000));
  if (total < 60) return `${total}s`;
  const m = Math.floor(total / 60);
  const s = total % 60;
  return s === 0 ? `${m}m` : `${m}m ${s}s`;
}
