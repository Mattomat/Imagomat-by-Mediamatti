import { api, Job } from "../api";

const NAMES: Record<string, string> = {
  pipeline: "Verarbeitung",
  analyze: "Analyse",
  cull: "Culling",
  people: "Personen",
  develop: "Entwicklung",
  export: "Export",
  train_profile: "Profil-Training",
  feedback: "Feedback",
  calibrate_culling: "Culling-Kalibrierung",
};

export default function JobBar({ jobs, onChange }: { jobs: Job[]; onChange: () => void }) {
  const active = jobs.filter((j) => ["running", "queued"].includes(j.status));
  const recent = jobs.filter((j) => !["running", "queued"].includes(j.status)).slice(0, 2);
  return (
    <footer className="jobbar">
      {[...active, ...recent].map((j) => {
        const pct = j.total ? Math.round((100 * j.progress) / j.total) : 0;
        return (
          <div key={j.id} className={`job ${j.status}`} title={j.error ?? ""}>
            <b>{NAMES[j.kind] ?? j.kind}</b>
            <span className="msg">{j.status === "failed" ? "Fehler" : j.message}</span>
            {j.status === "running" && (
              <>
                <div className="bar">
                  <div style={{ width: `${pct}%` }} />
                </div>
                <button onClick={() => api.post(`/api/jobs/${j.id}/cancel`).then(onChange)}>Abbrechen</button>
              </>
            )}
            {["paused", "cancelled", "failed"].includes(j.status) && (
              <button onClick={() => api.post(`/api/jobs/${j.id}/resume`).then(onChange)}>Fortsetzen</button>
            )}
          </div>
        );
      })}
    </footer>
  );
}
