import { getApiBase, getStoredToken, isDesktop } from "./config";

// The backend requires a shared-secret token on every /api/* route except
// /api/health (service/auth.py).
//
// Desktop: run_app.bat exports ARGUS_API_TOKEN to both processes, so the Rust
// side reads its own inherited env var (get_api_token in lib.rs) and there is no
// bootstrap step. Browser: no IPC and no shared env, so the token comes from
// whatever the user entered in Setup and we keep in localStorage.
//
// Tauri's own module is imported lazily rather than at the top level, because a
// static `import { invoke } from "@tauri-apps/api/core"` executes in a plain
// browser tab too and throws before any of our code runs.
let tokenPromise: Promise<string> | null = null;
// Resolved copy, kept so URLs can be built synchronously during render — see
// apiFileUrl below.
let cachedToken = "";

async function resolveToken(): Promise<string> {
  if (isDesktop()) {
    try {
      const { invoke } = await import("@tauri-apps/api/core");
      const fromShell = await invoke<string>("get_api_token");
      if (fromShell) return fromShell;
    } catch {
      /* fall through to the stored token */
    }
  }
  return getStoredToken();
}

function getToken(): Promise<string> {
  if (!tokenPromise) {
    tokenPromise = resolveToken()
      .then((t) => {
        cachedToken = t;
        return t;
      })
      .catch(() => "");
  }
  return tokenPromise;
}

/** Resolve the token up front so apiFileUrl() works on the first render. */
export function primeToken(): Promise<string> {
  return getToken();
}

/** Called after the user saves new credentials, so the next request doesn't keep
 *  using the token cached from before they fixed it. */
export function resetTokenCache(): void {
  tokenPromise = null;
  cachedToken = "";
}

/** Synchronous authenticated URL, for the things that fetch by URL and cannot
 *  carry an Authorization header: <img src>, <a download>, and three.js's
 *  GLTFLoader. Every /api/* route requires the token, so without this the
 *  entire visual layer — thumbnails, previews, the 3D viewer, downloads —
 *  silently 401s and renders as broken images.
 *
 *  Sync because it is called during render, which is why the token is primed in
 *  main.tsx before the app mounts rather than awaited here. */
export function apiFileUrl(path: string): string {
  const url = apiUrl(path);
  if (!cachedToken) return url;
  const sep = url.includes("?") ? "&" : "?";
  return `${url}${sep}token=${encodeURIComponent(cachedToken)}`;
}

async function authHeaders(extra?: Record<string, string>): Promise<Record<string, string>> {
  const token = await getToken();
  return { Authorization: `Bearer ${token}`, ...extra };
}

export function apiUrl(path: string): string {
  return path.startsWith("http") ? path : `${getApiBase()}${path}`;
}

/** For EventSource, which can't set custom headers — the token rides in the URL
 * instead. Every other caller uses the Authorization header via authHeaders(). */
export async function apiUrlWithToken(path: string): Promise<string> {
  const token = await getToken();
  const url = apiUrl(path);
  const sep = url.includes("?") ? "&" : "?";
  return `${url}${sep}token=${encodeURIComponent(token)}`;
}

/** Distinguishes "your credentials are wrong" from "the backend is unreachable",
 *  which are the two failure modes a tunnelled setup actually hits and which
 *  need completely different fixes from the user. */
export class ApiError extends Error {
  constructor(message: string, readonly status: number) {
    super(message);
    this.name = "ApiError";
  }
  get isAuth(): boolean {
    return this.status === 401 || this.status === 403;
  }
  get isUnreachable(): boolean {
    return this.status === 0;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(apiUrl(path), init);
  } catch {
    // fetch only rejects on network-level failure: wrong URL, tunnel down, CORS.
    throw new ApiError(
      `Can't reach the backend at ${getApiBase()} — is it running, and is the URL right?`,
      0,
    );
  }
  if (!res.ok) {
    // FastAPI puts the actionable message in `detail`. Without reading it, a
    // 400 that says "add your own API key in Settings" reached the user as
    // "POST /api/generate failed: 400", which tells them nothing they can act on.
    let detail = "";
    try {
      const body = await res.json();
      if (typeof body?.detail === "string") detail = body.detail;
    } catch {
      /* not JSON, or an empty body — fall back to the status line below */
    }
    const what = `${init?.method || "GET"} ${path}`;
    throw new ApiError(
      res.status === 401 || res.status === 403
        ? "Backend rejected the API token."
        : detail || `${what} failed: ${res.status}`,
      res.status,
    );
  }
  return res.json() as Promise<T>;
}

export async function apiGet<T>(path: string): Promise<T> {
  return request<T>(path, { headers: await authHeaders() });
}

export async function apiPost<T>(path: string, body?: unknown): Promise<T> {
  return request<T>(path, {
    method: "POST",
    headers: await authHeaders({ "Content-Type": "application/json" }),
    body: body !== undefined ? JSON.stringify(body) : undefined,
  });
}

export async function apiDelete<T>(path: string): Promise<T> {
  return request<T>(path, { method: "DELETE", headers: await authHeaders() });
}
