/** Canonical stage order for the inspector's pipeline table. Numbers/labels mirror
 *  service/pipeline_events.py's STAGE_LABELS (itself ported from desktop_app.py). The
 *  optional loop stages (54/55/56/75/77) only appear on runs that actually reach them,
 *  so they're listed but render as "—" until seen. */
export const STAGE_SEQUENCE = [
  { number: 1, label: "PLAN" },
  { number: 2, label: "NAMING" },
  { number: 3, label: "SCRIPT" },
  { number: 4, label: "REVIEW" },
  { number: 5, label: "EXPORT" },
  { number: 55, label: "VISUAL" },
  { number: 6, label: "TOPOLOGY" },
  { number: 7, label: "QUALITY" },
  { number: 75, label: "VISUAL QA" },
  { number: 8, label: "REFERENCE" },
];
