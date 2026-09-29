import { useEffect, useState } from "react";
import { api, connectEvents, Job, JobEvent, Shoot } from "./api";
import JobBar from "./components/JobBar";
import Splash from "./components/Splash";
import ExportView from "./views/Export";
import PeopleView from "./views/People";
import ProfilesView from "./views/Profiles";
import ReviewView from "./views/Review";
import SettingsView from "./views/Settings";
import ShootsView from "./views/Shoots";

type Tab = "shoots" | "review" | "people" | "profiles" | "export" | "settings";

const TABS: [Tab, string][] = [
  ["shoots", "Import"],
  ["review", "Culling & Review"],
  ["people", "Personen"],
  ["profiles", "Stil-Profile"],
  ["export", "Export"],
  ["settings", "Einstellungen"],
];

export default function App() {
  const [tab, setTab] = useState<Tab>("shoots");
  const [shoot, setShoot] = useState<Shoot | null>(null);
  const [jobs, setJobs] = useState<Job[]>([]);
  const [tick, setTick] = useState(0);
  const [ready, setReady] = useState(false);

  const refreshJobs = () => api.get<Job[]>("/api/jobs").then(setJobs).catch(() => undefined);

  useEffect(() => {
    if (!ready) return;
    refreshJobs();
    return connectEvents((e: JobEvent) => {
      setJobs((js) => {
        const found = js.find((j) => j.id === e.job_id);
        if (!found || e.message === "status") {
          refreshJobs();
          setTick((t) => t + 1);
          return js;
        }
        return js.map((j) =>
          j.id === e.job_id ? { ...j, progress: e.progress, total: e.total, message: e.message ?? j.message } : j,
        );
      });
    });
  }, [ready]);

  const openShoot = (s: Shoot, next: Tab = "review") => {
    setShoot(s);
    setTab(next);
  };

  if (!ready) return <Splash onReady={() => setReady(true)} />;

  return (
    <div className="app">
      <header>
        <div className="brand">
          Imagomat <span>by Mediamatti</span>
        </div>
        <nav>
          {TABS.map(([k, label]) => (
            <button key={k} className={tab === k ? "active" : ""} onClick={() => setTab(k)}>
              {label}
            </button>
          ))}
        </nav>
        <div className="shoot-name">{shoot ? shoot.name : "kein Shoot gewählt"}</div>
      </header>
      <main>
        {tab === "shoots" && <ShootsView onOpen={openShoot} tick={tick} onJob={refreshJobs} />}
        {tab === "review" && <ReviewView shoot={shoot} tick={tick} />}
        {tab === "people" && <PeopleView shoot={shoot} tick={tick} onJob={refreshJobs} />}
        {tab === "profiles" && <ProfilesView shoot={shoot} onJob={refreshJobs} />}
        {tab === "export" && <ExportView shoot={shoot} onJob={refreshJobs} />}
        {tab === "settings" && <SettingsView />}
      </main>
      <JobBar jobs={jobs} onChange={refreshJobs} />
    </div>
  );
}
