import { Suspense, lazy } from "react";
import type { ModelViewerProps } from "./ModelViewer";
import "./ModelViewer.css";

/** three.js and its three loader/control modules are ~600 KB of the bundle, and
 *  they are only reachable behind a "3D" toggle that most visits never touch.
 *  Importing ModelViewer directly put all of it in the initial payload; loading it
 *  on demand cuts first paint for everyone who just wants to look at the library.
 *
 *  The CSS is imported eagerly here (it's under a kilobyte) so the placeholder
 *  below is styled while the chunk is still downloading, rather than flashing
 *  unstyled text in the middle of the frame. */
const ModelViewer = lazy(() => import("./ModelViewer"));

export function LazyModelViewer(props: ModelViewerProps) {
  return (
    <Suspense
      fallback={
        <div className="model-viewer">
          <div className="model-viewer-status">Loading viewer…</div>
        </div>
      }
    >
      <ModelViewer {...props} />
    </Suspense>
  );
}
