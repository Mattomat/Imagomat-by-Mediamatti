// Zugriff auf das lokale Python-Backend.
declare global {
  interface Window {
    __TAURI_INTERNALS__?: unknown;
  }
}

export const BASE = window.__TAURI_INTERNALS__ ? "http://127.0.0.1:8765" : "";

async function req<T>(method: string, path: string, body?: unknown): Promise<T> {
  const r = await fetch(BASE + path, {
    method,
    headers: body !== undefined ? { "Content-Type": "application/json" } : undefined,
    body: body !== undefined ? JSON.stringify(body) : undefined,
  });
  if (!r.ok) {
    let msg = `${r.status}`;
    try {
      const j = await r.json();
      msg = j.detail ?? msg;
    } catch {
      /* keine JSON-Antwort */
    }
    throw new Error(msg);
  }
  return r.json() as Promise<T>;
}

export const api = {
  get: <T,>(p: string) => req<T>("GET", p),
  post: <T,>(p: string, b: unknown = {}) => req<T>("POST", p, b),
  put: <T,>(p: string, b: unknown) => req<T>("PUT", p, b),
  patch: <T,>(p: string, b: unknown) => req<T>("PATCH", p, b),
  del: <T,>(p: string) => req<T>("DELETE", p),
  img: (p: string) => BASE + p,
};

export interface Shoot {
  id: number;
  name: string;
  folder: string;
  profile: string | null;
  n: number;
  kept: number | null;
  settings: string | null;
}

export interface ImageItem {
  id: number;
  filename: string;
  capture_time: number | null;
  iso: number | null;
  decision: "keep" | "reject" | null;
  rating: number | null;
  label: string | null;
  reasons: string[];
  reason_keys: string[];
  series: number | null;
  best: boolean;
  score: number | null;
  manual: boolean;
  confidence: number | null;
  denoise: number | null;
  people: string[];
  notes: string[];
  preset: string | null;
  faces: number;
}

export interface Job {
  id: number;
  kind: string;
  shoot_id: number | null;
  status: string;
  progress: number;
  total: number;
  message: string | null;
  error: string | null;
}

export interface Preset {
  key: string;
  name: string;
  description: string;
}

export interface Profile {
  name: string;
  base_preset: string | null;
  created: string;
  n: number;
  templates: { sig: string; name: string; count: number }[];
  metrics: { mae_model?: Record<string, number> };
}

export interface Person {
  id: number;
  name: string;
  team: string | null;
  number: string | null;
  keyword: string;
}

export interface Cluster {
  key: string;
  person_id: number | null;
  name: string | null;
  cluster_id: number | null;
  count: number;
  faces: { id: number; image_id: number; assigned_by: string | null }[];
}

export type JobEvent = { type: "job"; job_id: number; progress: number; total: number; message: string | null };

export function connectEvents(onEvent: (e: JobEvent) => void): () => void {
  let ws: WebSocket | null = null;
  let stop = false;
  const open = () => {
    const url = (BASE || location.origin).replace(/^http/, "ws") + "/api/ws";
    ws = new WebSocket(url);
    ws.onmessage = (m) => onEvent(JSON.parse(m.data));
    ws.onclose = () => {
      if (!stop) setTimeout(open, 1500);
    };
  };
  open();
  return () => {
    stop = true;
    ws?.close();
  };
}

// Ordnerauswahl: in Tauri nativer Dialog, im Browser Texteingabe.
export async function pickFolder(current?: string): Promise<string | null> {
  if (window.__TAURI_INTERNALS__) {
    const { open } = await import("@tauri-apps/plugin-dialog");
    const r = await open({ directory: true, multiple: false, defaultPath: current });
    return typeof r === "string" ? r : null;
  }
  return window.prompt("Ordnerpfad", current ?? "") || null;
}

export async function pickFile(ext: string[], current?: string): Promise<string | null> {
  if (window.__TAURI_INTERNALS__) {
    const { open } = await import("@tauri-apps/plugin-dialog");
    const r = await open({ multiple: false, defaultPath: current, filters: [{ name: ext.join(", "), extensions: ext }] });
    return typeof r === "string" ? r : null;
  }
  return window.prompt(`Dateipfad (${ext.join(", ")})`, current ?? "") || null;
}
