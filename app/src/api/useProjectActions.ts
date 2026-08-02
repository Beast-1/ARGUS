import { useState } from "react";
import { apiDelete, apiPost } from "./client";

/** Open-in-Blender / Open-folder / Delete — real actions the old Tkinter app had
 *  (desktop_app.py's `_open_in_blender` / folder action / `_delete_project`) that
 *  didn't make it into the Tauri rebuild. Shared so Dashboard's AssetCard and
 *  Reveal don't each reimplement the busy/error handling. */
export function useProjectActions(onDeleted?: (name: string) => void) {
  const [busyName, setBusyName] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function openInBlender(name: string) {
    setBusyName(name);
    setError(null);
    try {
      await apiPost(`/api/projects/${encodeURIComponent(name)}/open-in-blender`);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusyName(null);
    }
  }

  async function openFolder(name: string) {
    setBusyName(name);
    setError(null);
    try {
      await apiPost(`/api/projects/${encodeURIComponent(name)}/open-folder`);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusyName(null);
    }
  }

  async function deleteProject(name: string) {
    setBusyName(name);
    setError(null);
    try {
      await apiDelete(`/api/projects/${encodeURIComponent(name)}`);
      onDeleted?.(name);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusyName(null);
    }
  }

  return { openInBlender, openFolder, deleteProject, busyName, error };
}
