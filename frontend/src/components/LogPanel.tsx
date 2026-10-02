import { useEffect, useState } from "react";
import { api, reveal } from "../api";

type Logs = {
  dir: string;
  version: string;
  platform: string;
  raw?: { rawpy?: string; libraw?: string; sample?: string; camera?: string; libraw_ok?: boolean; error?: string };
  failed_jobs: { id: number; kind: string; error: string | null; message: string | null; updated_at: number }[];
  log: string;
  setup: string;
  backend: string;
};

/** Fehlerprotokoll: zeigt fehlgeschlagene Aufgaben und das Protokoll, zum Kopieren bei Problemen. */
export default function LogPanel({ toast }: { toast: (m: string, k?: "ok" | "error") => void }) {
  const [logs, setLogs] = useState<Logs | null>(null);
  const load = () => api.get<Logs>("/api/logs").then(setLogs).catch((e) => toast(String(e), "error"));
  useEffect(() => {
    load();
  }, []);
  if (!logs) return null;

  const r = logs.raw ?? {};
  const rawLine = r.sample
    ? (r.libraw_ok ? `${r.camera}: wird mit echten RAW-Daten gelesen (${r.sample})`
      : `${r.camera}: RAW-Leser kann ${r.sample} NICHT lesen (${r.error ?? "?"})`)
    : "noch keine RAW-Datei zum Testen";
  const report = [
    `Tagmatti ${logs.version} (${logs.platform})`,
    "",
    "== RAW-Leser ==",
    `rawpy ${r.rawpy ?? "?"}, LibRaw ${r.libraw ?? "?"}`,
    rawLine,
    "",
    "== Fehlgeschlagene Aufgaben ==",
    ...logs.failed_jobs.map((j) => `#${j.id} ${j.kind} ${new Date(j.updated_at * 1000).toLocaleString()}\n${j.error ?? ""}`),
    "",
    "== Protokoll ==",
    logs.log,
    "",
    "== Einrichtung ==",
    logs.setup,
    "",
    "== Backend-Start ==",
    logs.backend,
  ].join("\n");

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(report);
      toast("Protokoll kopiert – einfach in den Chat einfügen");
    } catch {
      toast("Kopieren nicht möglich – Ordner öffnen und Datei tagmatti.log senden", "error");
    }
  };

  return (
    <div className="card logpanel">
      <div className="setting">
        <div>
          <b>Protokoll</b>
          <div className="hint">
            {logs.failed_jobs.length ? `${logs.failed_jobs.length} fehlgeschlagene Aufgabe(n)` : "keine Fehler"} · {logs.dir}
          </div>
        </div>
        <div className="row">
          <button onClick={copy}>Protokoll kopieren</button>
          <button className="ghost" onClick={() => reveal(logs.dir)}>Ordner öffnen</button>
          <button className="ghost" onClick={load}>Aktualisieren</button>
        </div>
      </div>
      <div className={r.sample && !r.libraw_ok ? "failed-job" : "hint"}>RAW-Leser: {rawLine}</div>
      {logs.failed_jobs.slice(0, 3).map((j) => (
        <div key={j.id} className="failed-job">
          <b>{j.kind}</b>: {(j.error ?? "").split("\n")[0]}
        </div>
      ))}
      <pre className="logtext">{logs.log.split("\n").slice(-60).join("\n") || "(leer)"}</pre>
    </div>
  );
}
