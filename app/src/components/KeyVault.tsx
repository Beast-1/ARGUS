import { useState } from "react";
import {
  EMPTY_PROVIDER_KEYS,
  activeProviders,
  getProviderKeys,
  setProviderKeys,
  type ProviderKeys,
} from "../api/config";
import { Icon, Icons } from "./Icon";
import "./KeyVault.css";

interface KeyVaultProps {
  onClose: () => void;
  onSaved: (keys: ProviderKeys) => void;
}

/** Fields are offered for every provider, but a run only uses the ones filled
 *  in — that is the backend's gating rule, so the copy states it plainly rather
 *  than leaving the user to infer it. Gemini is listed first and marked as the
 *  one that matters: it plans, writes the Blender script and scores the render,
 *  so a run without it has nothing to drive the pipeline. */
const FIELDS: {
  key: keyof ProviderKeys;
  label: string;
  hint: string;
  multi?: boolean;
  required?: boolean;
}[] = [
  {
    key: "google",
    label: "Gemini",
    hint: "Plans the object, writes the Blender script and scores the render. One key per line — extras are rotated when one hits its rate limit.",
    multi: true,
    required: true,
  },
  { key: "groq", label: "Groq", hint: "Fast fallback if Gemini fails. One per line.", multi: true },
  { key: "openrouter", label: "OpenRouter", hint: "Last-resort fallback. One per line.", multi: true },
  { key: "deepseek", label: "DeepSeek", hint: "Another script-generation fallback." },
  { key: "huggingface", label: "HuggingFace", hint: "Reference images for the concept pass." },
  { key: "nvidia", label: "NVIDIA NIM", hint: "Alternative source of reference images." },
  { key: "cloudflare_account_id", label: "Cloudflare account ID", hint: "Needs the API token below to be usable." },
  { key: "cloudflare_api_token", label: "Cloudflare API token", hint: "Reference images via Workers AI." },
];

function toText(value: string[] | string): string {
  return Array.isArray(value) ? value.join("\n") : value;
}

export function KeyVault({ onClose, onSaved }: KeyVaultProps) {
  const [keys, setKeys] = useState<ProviderKeys>(() => getProviderKeys());

  function update(field: (typeof FIELDS)[number], raw: string) {
    setKeys((prev) => ({
      ...prev,
      [field.key]: field.multi
        ? raw.split("\n").map((s) => s.trim()).filter(Boolean)
        : raw.trim(),
    }));
  }

  const active = activeProviders(keys);

  return (
    <div className="kv-backdrop" role="dialog" aria-modal="true" aria-label="API keys">
      <div className="kv">
        <div className="kv-head">
          <h2>Your API keys</h2>
          <button className="kv-x" onClick={onClose} aria-label="Close">
            <Icon icon={Icons.cancel} size={14} />
          </button>
        </div>

        <p className="kv-lede">
          Generation runs on your own provider accounts. Keys are kept in this browser
          and sent with each build — they are never stored on the server.
        </p>

        <div className="kv-fields">
          {FIELDS.map((f) => (
            <label key={f.key} className="kv-field">
              <span className="kv-label">
                {f.label}
                {f.required && <b className="kv-req">needed to build</b>}
              </span>
              {f.multi ? (
                <textarea
                  rows={2}
                  value={toText(keys[f.key])}
                  onChange={(e) => update(f, e.currentTarget.value)}
                  autoComplete="off"
                  spellCheck={false}
                />
              ) : (
                <input
                  type="password"
                  value={toText(keys[f.key])}
                  onChange={(e) => update(f, e.currentTarget.value)}
                  autoComplete="off"
                  spellCheck={false}
                />
              )}
              <small>{f.hint}</small>
            </label>
          ))}
        </div>

        <div className="kv-active">
          {active.length === 0 ? (
            <span className="kv-dim">No providers yet — add a Gemini key to build.</span>
          ) : (
            <>
              <span className="kv-dim">This build will use</span>
              {active.map((p) => (
                <span className="kv-chip" key={p}>
                  {p}
                </span>
              ))}
            </>
          )}
        </div>

        <div className="kv-actions">
          <button
            className="kv-clear"
            onClick={() => setKeys({ ...EMPTY_PROVIDER_KEYS })}
            disabled={active.length === 0}
          >
            Remove all
          </button>
          <button
            className="kv-save"
            onClick={() => {
              setProviderKeys(keys);
              onSaved(keys);
              onClose();
            }}
          >
            Save keys
          </button>
        </div>

        <p className="kv-note">
          <Icon icon={Icons.warning} size={13} />
          <span>
            Anything stored in a browser can be read by scripts running on the page.
            Use keys you can rotate, and remove them here when you're done.
          </span>
        </p>
      </div>
    </div>
  );
}
