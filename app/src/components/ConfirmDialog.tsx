import "./ConfirmDialog.css";

interface ConfirmDialogProps {
  title: string;
  body: string;
  confirmLabel?: string;
  danger?: boolean;
  pending?: boolean;
  onConfirm: () => void;
  onCancel: () => void;
}

/** Theme-agnostic — styles against the unified --bg/--text/--muted/--line tokens
 *  both themes expose, so it renders correctly whether it's opened from Dashboard
 *  (.theme-reveal) or Workbench (.theme-workbench). */
export function ConfirmDialog({
  title,
  body,
  confirmLabel = "Confirm",
  danger,
  pending,
  onConfirm,
  onCancel,
}: ConfirmDialogProps) {
  return (
    <div
      className="confirm-scrim"
      onClick={(e) => {
        // Stop here: this scrim is often rendered inside a clickable card
        // (Dashboard's AssetCard), and without this a backdrop click to
        // dismiss would also bubble up and trigger the card's own onClick.
        e.stopPropagation();
        onCancel();
      }}
    >
      <div className="confirm-box" onClick={(e) => e.stopPropagation()}>
        <h3>{title}</h3>
        <p>{body}</p>
        <div className="confirm-actions">
          <button className="confirm-btn-ghost" disabled={pending} onClick={onCancel}>
            Cancel
          </button>
          <button
            className={danger ? "confirm-btn-danger" : "confirm-btn-primary"}
            disabled={pending}
            onClick={onConfirm}
          >
            {pending ? "Working…" : confirmLabel}
          </button>
        </div>
      </div>
    </div>
  );
}
