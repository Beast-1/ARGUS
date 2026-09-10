/** Where the backend lives, and how we authenticate to it.
 *
 * The app now targets two very different hosts from one build:
 *
 *   Desktop (Tauri) — the backend is on localhost and run_app.bat exports
 *   ARGUS_API_TOKEN to both processes, so the Rust side can hand the token over
 *   and there is nothing to configure.
 *
 *   Browser (Cloudflare Pages) — there is no Tauri IPC and no shared env, so the
 *   user supplies their own backend URL (a tunnel to the machine running Blender)
 *   and token once, and we keep them in localStorage.
 *
 * Tauri is tried first and silently falls through, which is what lets a single
 * bundle serve both without a build flag.
 */

const LS_BASE = "argus.apiBase";
const LS_TOKEN = "argus.apiToken";

/** Baked in at build time for Pages deploys (VITE_ARGUS_API_BASE); ignored when
 *  the user has configured a URL by hand. */
const BUILD_TIME_BASE = (import.meta.env.VITE_ARGUS_API_BASE as string | undefined) || "";

const DESKTOP_DEFAULT = "http://127.0.0.1:8420";

function read(key: string): string {
  try {
    return localStorage.getItem(key) || "";
  } catch {
    return ""; // private mode / storage disabled
  }
}

function write(key: string, value: string): void {
  try {
    if (value) localStorage.setItem(key, value);
    else localStorage.removeItem(key);
  } catch {
    /* nothing we can do; the session just won't persist */
  }
}

/** True when running inside the Tauri desktop shell rather than a browser tab. */
export function isDesktop(): boolean {
  return typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;
}

export function getApiBase(): string {
  const stored = read(LS_BASE);
  if (stored) return stored.replace(/\/$/, "");
  if (BUILD_TIME_BASE) return BUILD_TIME_BASE.replace(/\/$/, "");
  return DESKTOP_DEFAULT;
}

export function setApiBase(value: string): void {
  write(LS_BASE, value.trim().replace(/\/$/, ""));
}

export function getStoredToken(): string {
  return read(LS_TOKEN);
}

export function setStoredToken(value: string): void {
  write(LS_TOKEN, value.trim());
}

/** Whether the app has enough to talk to a backend at all. On desktop the token
 *  arrives from Tauri, so only the browser case needs configuring up front. */
export function isConfigured(): boolean {
  return isDesktop() || Boolean(getStoredToken());
}

export function clearConfig(): void {
  write(LS_BASE, "");
  write(LS_TOKEN, "");
}
