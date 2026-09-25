import { useEffect, useState } from "react";
import { api } from "../api";

type S = {
  culling: { keep_ratio: number; reject_rating: number; series_gap_seconds: number; eyes_closed_threshold: number };
  keywords: { people_root: string; culling_root: string; write_face_regions: boolean };
  denoise: { mode: string; always: boolean; default_amount: number; min_amount: number; max_amount: number };
  develop: {
    auto_straighten: boolean; auto_crop: boolean; write_masks: boolean; ai_masks: boolean; shoot_consistency: number;
    max_straighten_deg: number;
  };
  device: string; face_backend: string; embedding_backend: string; ocr_backend: string;
};

export default function SettingsView() {
  const [s, setS] = useState<S | null>(null);
  const [dialect, setDialect] = useState<Record<string, unknown> | null>(null);
  const [licenses, setLicenses] = useState<{ name: string; license: string; commercial_ok: boolean }[]>([]);
  const [saved, setSaved] = useState(false);

  useEffect(() => {
    api.get<S>("/api/settings").then(setS);
    api.get<Record<string, unknown>>("/api/dialect").then(setDialect);
    api.get<typeof licenses>("/api/licenses").then(setLicenses);
  }, []);
  if (!s) return null;

  const set = <K extends keyof S>(k: K, v: Partial<S[K]> | S[K]) => {
    setS({ ...s, [k]: typeof v === "object" ? { ...(s[k] as object), ...(v as object) } : v } as S);
    setSaved(false);
  };
  const save = () => api.put("/api/settings", s).then(() => setSaved(true));

  return (
    <div className="page two-col">
      <section className="card">
        <h2>Verarbeitung</h2>
        <label>Denoise</label>
        <select value={s.denoise.mode} onChange={(e) => set("denoise", { mode: e.target.value })}>
          <option value="lightroom">Lightroom rechnet (Wert im XMP, „KI-Einstellungen aktualisieren“)</option>
          <option value="local">Lokal entrauschen (lineare DNG)</option>
          <option value="mark">Nur markieren (Stichwort + lila Label)</option>
        </select>
        <label className="check"><input type="checkbox" checked={s.denoise.always} onChange={(e) => set("denoise", { always: e.target.checked })} /> immer entrauschen (Stärke je Bild)</label>
        <label>Denoise-Stärke min/max: {s.denoise.min_amount}–{s.denoise.max_amount}</label>
        <div className="row">
          <input type="number" value={s.denoise.min_amount} onChange={(e) => set("denoise", { min_amount: +e.target.value })} />
          <input type="number" value={s.denoise.max_amount} onChange={(e) => set("denoise", { max_amount: +e.target.value })} />
        </div>
        <label className="check"><input type="checkbox" checked={s.develop.auto_straighten} onChange={(e) => set("develop", { auto_straighten: e.target.checked })} /> automatisch begradigen</label>
        <label className="check"><input type="checkbox" checked={s.develop.auto_crop} onChange={(e) => set("develop", { auto_crop: e.target.checked })} /> zuschneiden wie gelernt</label>
        <label className="check"><input type="checkbox" checked={s.develop.write_masks} onChange={(e) => set("develop", { write_masks: e.target.checked })} /> Masken schreiben</label>
        <label className="check"><input type="checkbox" checked={s.develop.ai_masks} onChange={(e) => set("develop", { ai_masks: e.target.checked })} /> Lightroom-KI-Masken (sonst Pinsel aus eigener Segmentierung)</label>
        <label>Konsistenz im Shoot: {Math.round(s.develop.shoot_consistency * 100)} %</label>
        <input type="range" min={0} max={100} value={s.develop.shoot_consistency * 100} onChange={(e) => set("develop", { shoot_consistency: +e.target.value / 100 })} />
        <label>Culling: Sterne für Aussortierte</label>
        <select value={s.culling.reject_rating} onChange={(e) => set("culling", { reject_rating: +e.target.value })}>
          <option value={0}>keine</option><option value={1}>1 Stern</option>
        </select>
        <label>Serien: max. Abstand (Sekunden)</label>
        <input type="number" step={0.5} value={s.culling.series_gap_seconds} onChange={(e) => set("culling", { series_gap_seconds: +e.target.value })} />
        <label>Stichwort-Stamm Personen</label>
        <input value={s.keywords.people_root} onChange={(e) => set("keywords", { people_root: e.target.value })} />
        <label className="check"><input type="checkbox" checked={s.keywords.write_face_regions} onChange={(e) => set("keywords", { write_face_regions: e.target.checked })} /> Gesichtsregionen (MWG) schreiben</label>
        <h3>Modelle</h3>
        <label>Rechengerät</label>
        <select value={s.device} onChange={(e) => set("device", e.target.value)}>
          <option value="auto">automatisch (Apple MPS / CUDA / CPU)</option><option value="mps">Apple MPS</option><option value="cuda">NVIDIA CUDA</option><option value="cpu">CPU</option>
        </select>
        <label>Gesichtserkennung</label>
        <select value={s.face_backend} onChange={(e) => set("face_backend", e.target.value)}>
          <option value="auto">automatisch (InsightFace, sonst YuNet)</option>
          <option value="insightface">InsightFace (beste Qualität, nicht kommerziell)</option>
          <option value="yunet">YuNet + SFace (kommerziell nutzbar)</option>
        </select>
        <button className="primary" onClick={save}>{saved ? "Gespeichert" : "Speichern"}</button>
      </section>
      <section className="card">
        <h2>Lightroom-Dialekt</h2>
        <p className="hint">Aus deinen XMPs gelernt: Prozessversion, Label-Namen, Denoise-Felder, KI-Masken-Typen.</p>
        <pre className="json">{JSON.stringify(dialect, null, 2)}</pre>
        <h2>Lizenzen</h2>
        <table className="list">
          <tbody>
            {licenses.map((l) => (
              <tr key={l.name}><td>{l.commercial_ok ? "✓" : "⚠︎"}</td><td>{l.name}</td><td>{l.license}</td></tr>
            ))}
          </tbody>
        </table>
      </section>
    </div>
  );
}
