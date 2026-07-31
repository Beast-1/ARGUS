import type { ApprovalRequest } from "../../api/useEventStream";

interface ApprovalDialogProps {
  request: ApprovalRequest;
  onDecide: (approved: boolean) => void;
  pending: boolean;
}

/** The pipeline thread is blocked waiting on this decision (see
 *  run_manager._memory_approval_callback), so the dialog is deliberately modal. */
export function ApprovalDialog({ request, onDecide, pending }: ApprovalDialogProps) {
  return (
    <div className="wb-modal-scrim">
      <div className="wb-modal">
        <h3>Save as a reference structure?</h3>
        <p className="wb-modal-body">
          This build passed the strict quality gate. Saving it lets future generations of
          similar objects learn from its structure.
        </p>
        <table className="wb-kv wb-modal-kv">
          <tbody>
            <tr>
              <td>asset</td>
              <td>{request.asset_name ?? "—"}</td>
            </tr>
            <tr>
              <td>blueprint</td>
              <td>{request.blueprint || "—"}</td>
            </tr>
            <tr>
              <td>prompt</td>
              <td>{request.prompt ?? "—"}</td>
            </tr>
          </tbody>
        </table>
        <div className="wb-modal-actions">
          <button className="wb-btn-ghost" disabled={pending} onClick={() => onDecide(false)}>
            Skip
          </button>
          <button className="wb-btn-primary" disabled={pending} onClick={() => onDecide(true)}>
            Save as reference
          </button>
        </div>
      </div>
    </div>
  );
}
