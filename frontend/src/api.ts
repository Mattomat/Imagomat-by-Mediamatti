// Zugriff auf das lokale Python-Backend.
declare global {
  interface Window {
    __TAURI_INTERNALS__?: unknown;
  }
}

export const IS_APP = !!window.__TAURI_INTERNALS__;
export const BASE = IS_APP ? "http://127.0.0.1:8765" : "";

async function req<T>(method: string, path: string, body?: unknown): Promise<T> {
  const r = await fetch(BASE + path, {
    method,
    headers: body !== undefined ? { "Content-Type": "application/json" } : undefined,
    body: body !== undefined ? JSON.stringify(body) : undefined,
  });
  if (!r.ok) {
    let msg = `Fehler ${r.status}`;
    try {
      const j = await r.json();
      msg = typeof j.detail === "string" ? j.detail : msg;
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

export interface JobInfo {
  id: number;
  kind: string;
  status: string;
  progress: number;
  total: number;
  message: string | null;
  error?: string | null;
  shoot_id?: number | null;
}

export interface Shoot {
  id: number;
  name: string;
  folder: string;
  profile: string | null;
  n: number;
  kept: number | null;
  cover: number | null;
  settings: string | null;
  job: JobInfo | null;
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
  series: number | null;
  best: boolean;
  manual: boolean;
  confidence: number | null;
  denoise: number | null;
  people: string[];
  notes: string[];
  preset: string | null;
  moment: string | null;
  action: number | null;
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
}

export interface Cluster {
  key: string;
  person_id: number | null;
  name: string | null;
  cluster_id: number | null;
  count: number;
  faces: { id: number; image_id: number; assigned_by: string | null }[];
}

export interface Overview {
  persons: number;
  persons_with_face: number;
  teams: string[];
  profiles: Profile[];
  catalogs: string[];
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

/** Wartet, bis ein Job fertig ist (für Aktionen mit Rückmeldung). */
export async function waitForJob(id: number, onProgress?: (j: JobInfo) => void): Promise<JobInfo> {
  for (;;) {
    const jobs = await api.get<JobInfo[]>("/api/jobs");
    const j = jobs.find((x) => x.id === id);
    if (j) {
      onProgress?.(j);
      if (["done", "failed", "cancelled"].includes(j.status)) return j;
    }
    await new Promise((r) => setTimeout(r, 800));
  }
}

// Ordner-/Dateiauswahl: in der App nativer Dialog, im Browser Texteingabe.
export async function pickFolder(title = "Ordner wählen"): Promise<string | null> {
  if (IS_APP) {
    const { open } = await import("@tauri-apps/plugin-dialog");
    const r = await open({ directory: true, multiple: false, title });
    return typeof r === "string" ? r : null;
  }
  return window.prompt(`${title} – Pfad eingeben`) || null;
}

export async function pickFile(ext: string[], title = "Datei wählen"): Promise<string | null> {
  if (IS_APP) {
    const { open } = await import("@tauri-apps/plugin-dialog");
    const r = await open({ multiple: false, title, filters: [{ name: ext.join(", "), extensions: ext }] });
    return typeof r === "string" ? r : null;
  }
  return window.prompt(`${title} – Pfad eingeben`) || null;
}

/** Drag & Drop von Dateien/Ordnern aus dem Finder (nur in der App liefert das Pfade). */
export async function onFileDrop(handler: (paths: string[]) => void, hover: (on: boolean) => void): Promise<() => void> {
  if (!IS_APP) return () => undefined;
  const { getCurrentWebview } = await import("@tauri-apps/api/webview");
  return getCurrentWebview().onDragDropEvent((e) => {
    const p = e.payload;
    if (p.type === "enter" || p.type === "over") hover(true);
    else if (p.type === "leave") hover(false);
    else if (p.type === "drop") {
      hover(false);
      handler(p.paths);
    }
  });
}

export async function reveal(path: string) {
  await api.post("/api/reveal", { path });
}
