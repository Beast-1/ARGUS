import "./Logo.css";

/** The three-bar ARGUS mark. Single source of truth — it previously existed as
 *  duplicated markup and duplicated CSS in both TopNav and Workbench's menubar. */
export function Logo() {
  return (
    <span className="argus-logo" aria-hidden="true">
      <i />
      <i />
      <i />
    </span>
  );
}
