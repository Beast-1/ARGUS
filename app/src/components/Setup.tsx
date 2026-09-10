import { useState } from "react";
import { apiUrl, resetTokenCache } from "../api/client";
import {
  getApiBase,
  getStoredToken,
  setApiBase,
  setStoredToken,
} from "../api/config";
import { Icon, Icons } from "./Icon";
import "./Setup.css";

interface SetupProps {
  onDone: () => void;
  /** Shown when an existing session started failing, rather than on first run. */
  reason?: string;
}

/** Browser-only first-run screen.
 *
 *  The desktop build never sees this: Tauri hands the token over from the env
 *  var both processes inherit. In a browser there is no shared environment, so
 *  the user points the app at their own backend — a tunnel to the machine that
 *  actually runs Blender — and supplies the token it prints at startup. */
export function Setup({ onDone, reason }: SetupProps) {
  const [base, setBase] = useState(getApiBase());
  const [token, setToken] = useState(getStoredToken());
  const [testing, setTesting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function save(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setTesting(true);
    // Persist first so apiUrl() resolves against the URL being tested.
    setApiBase(base);
    setStoredToken(token);
    resetTokenCache();
    try {
      // /api/health is deliberately unauthenticated, so this separates "URL is
      // wrong / tunnel is down" from "URL is fine but the token is rejected".
      const health = await fetch(apiUrl("/api/health"));
      if (!health.ok) throw new Error(`health check returned ${health.status}`);
      const authed = await fetch(apiUrl("/api/projects"), {
        headers: { Authorization: `Bearer ${token.trim()}` },
      });
      if (authed.status === 401 || authed.status === 403) {
        setError("Backend is reachable, but rejected that token.");
        return;
      }
      if (!authed.ok) throw new Error(`projects returned ${authed.status}`);
      onDone();
    } catch {
      setError(
        `Couldn't reach a backend at ${base || "(no URL)"}. Check it's running ` +
          "and the tunnel is up.",
      );
    } finally {
      setTesting(false);
    }
  }

  return (
    // Each route carries its own theme class (Dashboard/Reveal use theme-reveal,
    // Workbench theme-workbench) — without one here the semantic tokens
    // --bg/--text/--panel never resolve and the screen renders unstyled.
    <div className="theme-reveal setup">
      <form className="setup-card" onSubmit={save}>
        <h1>Connect to your backend</h1>
        <p className="setup-lede">
          ARGUS runs Blender on your own machine — this page is just the interface.
          Point it at the backend and it'll remember for next time.
        </p>

        {reason && (
          <div className="setup-banner">
            <Icon icon={Icons.warning} size={15} />
            {reason}
          </div>
        )}

        <label>
          Backend URL
          <input
            value={base}
            onChange={(e) => setBase(e.target.value)}
            placeholder="https://argus.example.trycloudflare.com"
            spellCheck={false}
            autoComplete="off"
          />
          <small>
            A Cloudflare Tunnel to the machine running <code>run_app.bat</code>,
            or <code>http://127.0.0.1:8420</code> on the same machine.
          </small>
        </label>

        <label>
          API token
          <input
            value={token}
            onChange={(e) => setToken(e.target.value)}
            type="password"
            placeholder="ARGUS_API_TOKEN"
            spellCheck={false}
            autoComplete="off"
          />
          <small>
            Printed by the backend at startup, and set as <code>ARGUS_API_TOKEN</code>.
          </small>
        </label>

        {error && <div className="setup-error">{error}</div>}

        <button className="setup-submit" disabled={testing || !base || !token}>
          {testing ? "Checking…" : "Connect"}
        </button>

        <p className="setup-note">
          <Icon icon={Icons.warning} size={13} />
          <span>
            Exposing the backend over a tunnel means anyone with the URL and token
            can run generations on your machine. Putting Cloudflare Access in front
            of the tunnel is strongly recommended.
          </span>
        </p>
      </form>
    </div>
  );
}
