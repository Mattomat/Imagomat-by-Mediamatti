import { useEffect, useState } from "react";
import { api, pickFile, pickFolder, Preset, Profile, Shoot } from "../api";

const METRICS: [string, string, number][] = [
  ["Exposure2012", "Belichtung (EV)", 2],
  ["wb_dmired", "Weissabgleich (Mired)", 1],
  ["Contrast2012", "Kontrast", 1],
  ["Highlights2012", "Lichter", 1],
  ["Shadows2012", "Tiefen", 1],
];

export default function ProfilesView({ shoot, onJob }: { shoot: Shoot | null; onJob: () => void }) {
  const [profiles, setProfiles] = useState<Profile[]>([]);
  const [presets, setPresets] = useState<Preset[]>([]);
  const [f, setF] = useState({ name: "", catalog: "", folders: [] as string[], presets: [] as string[], base: "", minRating: 0, onlyPicked: false });
  const [msg, setMsg] = useState("");

  const load = () => api.get<Profile[]>("/api/profiles").then(setProfiles);
  useEffect(() => {
    load();
    api.get<Preset[]>("/api/presets").then(setPresets);
  }, []);

  const train = async () => {
    await api.post("/api/profiles/train", {
      name: f.name, catalog: f.catalog || null, folders: f.folders, presets: f.presets,
      base_preset: f.base || null, min_rating: f.minRating, only_picked: f.onlyPicked,
    });
    setMsg("Training gestartet (Fortschritt unten). Bei 1600 Bildern dauert die Merkmalsberechnung einmalig einige Minuten.");
    onJob();
  };

  return (
    <div className="page two-col">
      <section className="card">
        <h2>Stil lernen</h2>
        <p className="hint">
          Lade deinen Lightroom-Katalog oder einen Ordner mit Bildern + XMP. Imagomat lernt daraus, wie du jedes Bild
          individuell bearbeitest. Ohne eigene Daten greifen die mitgelieferten Presets.
        </p>
        <label>Profilname</label>
        <input value={f.name} onChange={(e) => setF({ ...f, name: e.target.value })} placeholder="Sport Nacht" />
        <label>Lightroom-Katalog (.lrcat, wird nur als Kopie gelesen)</label>
        <div className="row">
          <input value={f.catalog} onChange={(e) => setF({ ...f, catalog: e.target.value })} />
          <button onClick={async () => { const p = await pickFile(["lrcat"]); if (p) setF({ ...f, catalog: p }); }}>Wählen…</button>
        </div>
        <label>Ordner mit RAW/JPEG + XMP</label>
        {f.folders.map((d) => <div key={d} className="path">{d} <button onClick={() => setF({ ...f, folders: f.folders.filter((x) => x !== d) })}>✕</button></div>)}
        <button onClick={async () => { const p = await pickFolder(); if (p) setF({ ...f, folders: [...f.folders, p] }); }}>+ Ordner</button>
        <label>Lightroom-Presets (.xmp, optional: fester Look)</label>
        {f.presets.map((d) => <div key={d} className="path">{d}</div>)}
        <button onClick={async () => { const p = await pickFile(["xmp"]); if (p) setF({ ...f, presets: [...f.presets, p] }); }}>+ Preset</button>
        <label>Basis-Preset (Startwerte bei wenigen Beispielen)</label>
        <select value={f.base} onChange={(e) => setF({ ...f, base: e.target.value })}>
          <option value="">automatisch pro Bild</option>
          {presets.map((p) => <option key={p.key} value={p.key}>{p.name}</option>)}
        </select>
        <label>Nur Bilder mit mindestens … Sternen (oder Pick)</label>
        <select value={f.minRating} onChange={(e) => setF({ ...f, minRating: +e.target.value })}>
          {[0, 1, 2, 3, 4, 5].map((n) => <option key={n} value={n}>{n === 0 ? "alle bearbeiteten" : "★".repeat(n)}</option>)}
        </select>
        <label className="check"><input type="checkbox" checked={f.onlyPicked} onChange={(e) => setF({ ...f, onlyPicked: e.target.checked })} /> nur ausgewählte (Pick)</label>
        <button className="primary" disabled={!f.name || (!f.catalog && f.folders.length === 0)} onClick={train}>Profil trainieren</button>
        {msg && <p className="hint">{msg}</p>}
      </section>
      <section className="card">
        <h2>Profile</h2>
        {profiles.length === 0 && <p>Noch keine Profile.</p>}
        {profiles.map((p) => (
          <div key={p.name} className="profile">
            <h3>{p.name}</h3>
            <p>{p.n} Bilder · Basis {p.base_preset ?? "automatisch"} · {p.created}</p>
            <table className="list">
              <tbody>
                {METRICS.map(([k, label, d]) => (
                  <tr key={k}><td>{label}</td><td>± {p.metrics.mae_model?.[k]?.toFixed(d) ?? "–"}</td></tr>
                ))}
              </tbody>
            </table>
            {p.templates.length > 0 && <p>Masken-Vorlagen: {p.templates.map((t) => `${t.name} (${t.count}×)`).join(", ")}</p>}
            <div className="row">
              {shoot && (
                <button onClick={() => api.post(`/api/shoots/${shoot.id}/feedback`, { profile: p.name }).then(onJob)}>
                  Korrekturen aus „{shoot.name}“ lernen
                </button>
              )}
              <button onClick={async () => { const d = await pickFolder(); if (d) { const r = await api.post<{ file: string }>(`/api/profiles/${encodeURIComponent(p.name)}/export`, { target: `${d}/${p.name}` }); setMsg(`Exportiert: ${r.file}`); } }}>Teilen…</button>
            </div>
          </div>
        ))}
        <button onClick={async () => { const fl = await pickFile(["imagomat-profile"]); if (fl) { await api.post("/api/profiles/import", { file: fl }); load(); } }}>Profil importieren…</button>
        <h2>Mitgelieferte Presets</h2>
        <table className="list"><tbody>{presets.map((p) => <tr key={p.key}><td><b>{p.name}</b></td><td>{p.description}</td></tr>)}</tbody></table>
      </section>
    </div>
  );
}
