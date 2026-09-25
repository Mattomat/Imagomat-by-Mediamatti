import { useEffect, useState } from "react";
import { api, pickFolder, Preset, Profile, Shoot } from "../api";

export default function ShootsView({
  onOpen,
  tick,
  onJob,
}: {
  onOpen: (s: Shoot) => void;
  tick: number;
  onJob: () => void;
}) {
  const [shoots, setShoots] = useState<Shoot[]>([]);
  const [profiles, setProfiles] = useState<Profile[]>([]);
  const [presets, setPresets] = useState<Preset[]>([]);
  const [teams, setTeams] = useState<string[]>([]);
  const [form, setForm] = useState({ folder: "", name: "", profile: "", preset: "", keep: 20, teams: [] as string[] });
  const [err, setErr] = useState("");

  useEffect(() => {
    api.get<Shoot[]>("/api/shoots").then(setShoots);
  }, [tick]);
  useEffect(() => {
    api.get<Profile[]>("/api/profiles").then(setProfiles);
    api.get<Preset[]>("/api/presets").then(setPresets);
    api.get<{ team: string | null }[]>("/api/persons").then((ps) =>
      setTeams([...new Set(ps.map((p) => p.team).filter((t): t is string => !!t))]),
    );
  }, []);

  const start = async () => {
    setErr("");
    try {
      await api.post("/api/shoots/import", {
        folder: form.folder,
        name: form.name || undefined,
        profile: form.profile || undefined,
        preset: form.preset || undefined,
        keep_ratio: form.keep / 100,
        teams: form.teams,
      });
      onJob();
      setShoots(await api.get<Shoot[]>("/api/shoots"));
    } catch (e) {
      setErr(String(e));
    }
  };

  return (
    <div className="page two-col">
      <section className="card">
        <h2>Neuer Shoot</h2>
        <label>Ordner oder Speicherkarte</label>
        <div className="row">
          <input value={form.folder} onChange={(e) => setForm({ ...form, folder: e.target.value })} placeholder="/Volumes/Untitled/DCIM" />
          <button onClick={async () => { const f = await pickFolder(form.folder); if (f) setForm({ ...form, folder: f }); }}>Wählen…</button>
        </div>
        <label>Name (optional)</label>
        <input value={form.name} onChange={(e) => setForm({ ...form, name: e.target.value })} placeholder="FCW – GCZ, 01.05.2026" />
        <label>Stil-Profil</label>
        <select value={form.profile} onChange={(e) => setForm({ ...form, profile: e.target.value })}>
          <option value="">(kein Profil, nur Preset)</option>
          {profiles.map((p) => (
            <option key={p.name} value={p.name}>{p.name} ({p.n} Bilder)</option>
          ))}
        </select>
        <label>Preset / Situation</label>
        <select value={form.preset} onChange={(e) => setForm({ ...form, preset: e.target.value })}>
          <option value="">automatisch erkennen</option>
          {presets.map((p) => (
            <option key={p.key} value={p.key} title={p.description}>{p.name}</option>
          ))}
        </select>
        <label>Strenge: behalte ca. {form.keep} %</label>
        <input type="range" min={5} max={80} value={form.keep} onChange={(e) => setForm({ ...form, keep: +e.target.value })} />
        {teams.length > 0 && (
          <>
            <label>Teams (für Rückennummern)</label>
            <div className="chips">
              {teams.map((t) => (
                <button
                  key={t}
                  className={form.teams.includes(t) ? "chip on" : "chip"}
                  onClick={() =>
                    setForm({ ...form, teams: form.teams.includes(t) ? form.teams.filter((x) => x !== t) : [...form.teams, t] })
                  }
                >
                  {t}
                </button>
              ))}
            </div>
          </>
        )}
        <button className="primary" disabled={!form.folder} onClick={start}>Importieren und verarbeiten</button>
        {err && <p className="error">{err}</p>}
        <p className="hint">Culling nutzt die eingebetteten JPEG-Vorschauen; RAWs werden nur gelesen, nie verändert.</p>
      </section>
      <section className="card">
        <h2>Shoots</h2>
        <table className="list">
          <thead><tr><th>Name</th><th>Bilder</th><th>Behalten</th><th>Profil</th><th /></tr></thead>
          <tbody>
            {shoots.map((s) => (
              <tr key={s.id}>
                <td title={s.folder}>{s.name}</td>
                <td>{s.n}</td>
                <td>{s.kept ?? "–"}</td>
                <td>{s.profile ?? "Preset"}</td>
                <td><button onClick={() => onOpen(s)}>Öffnen</button></td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>
    </div>
  );
}
