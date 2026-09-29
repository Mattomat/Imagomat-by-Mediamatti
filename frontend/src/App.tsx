import { useCallback, useEffect, useRef, useState } from "react";
import { api, connectEvents, IS_APP, JobInfo, onFileDrop, Shoot } from "./api";
import Splash from "./components/Splash";
import { Toasts, useToasts } from "./ui";
import HomeView from "./views/Home";
import PeopleView from "./views/People";
import SettingsView from "./views/Settings";
import ShootView from "./views/ShootView";
import StyleView from "./views/Style";

export type Page = { name: "home" } | { name: "shoot"; id: number } | { name: "people" } | { name: "style" } | { name: "settings" };

const NAV: [Page["name"], string, string][] = [
  ["home", "Start", "⌂"],
  ["people", "Personen", "☺"],
  ["style", "Mein Stil", "✦"],
  ["settings", "Einstellungen", "⚙"],
];

export interface AppCtx {
  go: (p: Page) => void;
  toast: (msg: string, kind?: "ok" | "error") => void;
  tick: number;
  jobs: JobInfo[];
  refreshJobs: () => void;
}

export default function App() {
  const [ready, setReady] = useState(false);
  const [page, setPage] = useState<Page>({ name: "home" });
  const [jobs, setJobs] = useState<JobInfo[]>([]);
  const [tick, setTick] = useState(0);
  const [recent, setRecent] = useState<Shoot[]>([]);
  const [dragging, setDragging] = useState(false);
  const { toasts, toast } = useToasts();
  const dropHandler = useRef<((paths: string[]) => void) | null>(null);

  const refreshJobs = useCallback(() => {
    api.get<JobInfo[]>("/api/jobs").then(setJobs).catch(() => undefined);
  }, []);
  const onReady = useCallback(() => setReady(true), []);

  useEffect(() => {
    if (!ready) return;
    refreshJobs();
    api.get<Shoot[]>("/api/shoots").then((s) => setRecent(s.slice(0, 6)));
    return connectEvents((e) => {
      if (e.message === "status") {
        refreshJobs();
        setTick((t) => t + 1);
        api.get<Shoot[]>("/api/shoots").then((s) => setRecent(s.slice(0, 6)));
      } else {
        setJobs((js) => js.map((j) => (j.id === e.job_id ? { ...j, progress: e.progress, total: e.total, message: e.message ?? j.message } : j)));
      }
    });
  }, [ready, refreshJobs]);

  useEffect(() => {
    if (!ready) return;
    let un: () => void = () => undefined;
    onFileDrop((paths) => dropHandler.current?.(paths), setDragging).then((u) => (un = u));
    return () => un();
  }, [ready]);

  if (!ready) return <Splash onReady={onReady} />;

  const ctx: AppCtx = { go: setPage, toast, tick, jobs, refreshJobs };
  const running = jobs.filter((j) => j.status === "running" || j.status === "queued");
  const failed = jobs.find((j) => j.status === "failed" && Date.now() > 0);

  return (
    <div className={`app ${IS_APP ? "native" : ""}`}>
      <aside className="sidebar">
        <div className="drag" data-tauri-drag-region />
        <div className="brand">Imagomat</div>
        <nav>
          {NAV.map(([key, label, icon]) => (
            <button key={key} className={page.name === key ? "active" : ""} onClick={() => setPage({ name: key } as Page)}>
              <span className="ico">{icon}</span>
              {label}
            </button>
          ))}
        </nav>
        {recent.length > 0 && (
          <div className="recent">
            <div className="section-title">Letzte Shoots</div>
            {recent.map((s) => (
              <button
                key={s.id}
                className={page.name === "shoot" && page.id === s.id ? "active" : ""}
                onClick={() => setPage({ name: "shoot", id: s.id })}
                title={s.folder}
              >
                {s.name}
              </button>
            ))}
          </div>
        )}
        <div className="activity">
          {running.map((j) => (
            <div key={j.id} className="act">
              <div className="act-msg">{j.message || "Arbeite …"}</div>
              <div className="bar"><div style={{ width: `${j.total ? Math.round((100 * j.progress) / j.total) : 5}%` }} /></div>
            </div>
          ))}
          {running.length === 0 && failed && (
            <div className="act error" title={failed.error ?? ""} onClick={() => toast(failed.error?.split("\n")[0] ?? "Fehler", "error")}>
              Letzte Aufgabe fehlgeschlagen – Details
            </div>
          )}
        </div>
      </aside>
      <main>
        {page.name === "home" && <HomeView ctx={ctx} setDrop={(h) => (dropHandler.current = h)} />}
        {page.name === "shoot" && <ShootView ctx={ctx} id={page.id} key={page.id} />}
        {page.name === "people" && <PeopleView ctx={ctx} setDrop={(h) => (dropHandler.current = h)} />}
        {page.name === "style" && <StyleView ctx={ctx} setDrop={(h) => (dropHandler.current = h)} />}
        {page.name === "settings" && <SettingsView ctx={ctx} />}
      </main>
      {dragging && <div className="drop-overlay">Loslassen zum Importieren</div>}
      <Toasts toasts={toasts} />
    </div>
  );
}
