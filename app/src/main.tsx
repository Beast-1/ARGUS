import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import { primeToken } from "./api/client";
import { ErrorBoundary } from "./components/ErrorBoundary";
import "./styles/tokens.css";
import "./styles/theme-workbench.css";
import "./styles/theme-reveal.css";

function mount() {
  ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
    <React.StrictMode>
      <ErrorBoundary>
        <App />
      </ErrorBoundary>
    </React.StrictMode>,
  );
}

// Resolve the API token before the first render. Thumbnails, the 3D viewer and
// download links build authenticated URLs synchronously (apiFileUrl), so if the
// token isn't cached by the time they render they emit unauthenticated URLs and
// every image 401s. Mount regardless of the outcome — an unconfigured browser
// needs to reach the Setup screen, which doesn't require a token.
primeToken().finally(mount);
