import { useEffect, useState } from "react";
import type { AppCtx } from "../App";
import { api } from "../api";
import LogPanel from "../components/LogPanel";
import { More, Segmented } from "../ui";

type S = {
  culling: { reject_rating: number; series_gap_seconds: number };
  keywords: { people_root: string; write_face_regions: boolean };
  denoise: { mode: string; always: boolean; min_amount: number; max_amount: number };
  develop: { auto_straighten: boolean; auto_crop: boolean; write_masks: boolean; ai_masks: boolean; shoot_consistency: number;
    punch: number; bottom_fade: boolean; bottom_fade_strength: number };
  device: string;
  face_backend: string;
  ignored_shirt_words: string[];
};

export default function SettingsView({ ctx }: { ctx: AppCtx }) {
  const [s, setS] = useState<S | null>(null);
  const [licenses, setLicenses] = useState<{ name: string; license: string; commercial_ok: boolean }[]>([]);

  useEffect(() => {
    api.get<S>("/api/settings").then(setS);
    api.get<typeof licenses>("/api/licenses").then(setLicenses);
  }, []);
  if (!s) return null;

  const save = async (next: S) => {
    setS(next);
    await api.put("/api/settings", next);
    ctx.toast("Gespeichert");
  };
  const set = <K extends keyof S>(k: K, v: Partial<S[K]>) => save({ ...s, [k]: { ...(s[k] as object), ...v } } as S);

  return (
    <div className="page settings">
      <h1>Einstellungen</h1>
      <div className="card">
        <div className="setting">
          <div>
            <b>Entrauschen</b>
            <div className="hint">Stärke wird pro Bild aus dem gemessenen Rauschen bestimmt.</div>
          </div>
          <Segmented value={s.denoise.mode} onChange={(v) => set("denoise", { mode: v })}
            options={[["lightroom", "Lightroom rechnet"], ["local", "Imagomat"], ["mark", "Nur markieren"]]} />
        </div>
        <div className="setting">
          <div>
            <b>Kontrast (Weiss/Schwarz)</b>
            <div className="hint">wie am Waveform: oben und unten schlägt ein kleiner Teil leicht an</div>
          </div>
          <Segmented value={s.develop.punch} onChange={(v) => set("develop", { punch: v })}
            options={[[0, "wie gelernt"], [0.6, "etwas"], [1, "knackig"], [1.5, "kräftig"]]} />
        </div>
        <div className="setting">
          <div>
            <b>Dunkler Verlauf von unten</b>
            <div className="hint">dunkler und weicher Rasen/Vordergrund, pro Bild angepasst</div>
          </div>
          <Segmented value={s.develop.bottom_fade ? s.develop.bottom_fade_strength : 0}
            onChange={(v) => set("develop", { bottom_fade: v > 0, bottom_fade_strength: v > 0 ? v : 1 })}
            options={[[0, "aus"], [0.6, "leicht"], [1, "normal"], [1.4, "stark"]]} />
        </div>
        <div className="setting">
          <div><b>Schiefe Bilder begradigen</b></div>
          <input type="checkbox" className="switch" checked={s.develop.auto_straighten} onChange={(e) => set("develop", { auto_straighten: e.target.checked })} />
        </div>
        <div className="setting">
          <div><b>Zuschneiden wie ich</b><div className="hint">übernimmt deinen typischen Bildausschnitt</div></div>
          <input type="checkbox" className="switch" checked={s.develop.auto_crop} onChange={(e) => set("develop", { auto_crop: e.target.checked })} />
        </div>
        <div className="setting">
          <div><b>Masken setzen</b><div className="hint">z. B. Spieler aufhellen, Hintergrund mit Tiefe</div></div>
          <input type="checkbox" className="switch" checked={s.develop.write_masks} onChange={(e) => set("develop", { write_masks: e.target.checked })} />
        </div>
        <div className="setting">
          <div><b>Bilder einer Lichtsituation angleichen</b></div>
          <input type="checkbox" className="switch" checked={s.develop.shoot_consistency > 0} onChange={(e) => set("develop", { shoot_consistency: e.target.checked ? 0.6 : 0 })} />
        </div>
      </div>

      <LogPanel toast={ctx.toast} />

      <More label="Erweitert">
        <div className="card">
          <div className="setting">
            <div><b>Lightroom-KI-Masken verwenden</b><div className="hint">aus: Masken als Pinselstriche aus eigener Erkennung</div></div>
            <input type="checkbox" className="switch" checked={s.develop.ai_masks} onChange={(e) => set("develop", { ai_masks: e.target.checked })} />
          </div>
          <div className="setting">
            <div><b>Aussortierte bekommen 1 Stern</b></div>
            <input type="checkbox" className="switch" checked={s.culling.reject_rating === 1} onChange={(e) => set("culling", { reject_rating: e.target.checked ? 1 : 0 })} />
          </div>
          <div className="setting">
            <div><b>Gesichter in Lightroom markieren</b><div className="hint">schreibt benannte Gesichtsregionen ins XMP</div></div>
            <input type="checkbox" className="switch" checked={s.keywords.write_face_regions} onChange={(e) => set("keywords", { write_face_regions: e.target.checked })} />
          </div>
          <div className="setting">
            <div><b>Stichwort für Personen</b></div>
            <input className="narrow" value={s.keywords.people_root} onChange={(e) => setS({ ...s, keywords: { ...s.keywords, people_root: e.target.value } })}
              onBlur={() => save(s)} />
          </div>
          <div className="setting">
            <div><b>Sponsoren auf Trikots ignorieren</b><div className="hint">diese Wörter sind nie ein Spielername (weitere erkennt Imagomat selbst)</div></div>
            <input className="narrow" value={(s.ignored_shirt_words ?? []).join(", ")}
              onChange={(e) => setS({ ...s, ignored_shirt_words: e.target.value.split(",").map((w) => w.trim()).filter(Boolean) })}
              onBlur={() => save(s)} placeholder="z. B. KELLER, INIT" />
          </div>
          <div className="setting">
            <div><b>Gesichtserkennung</b><div className="hint">InsightFace ist am genauesten, aber nur privat nutzbar</div></div>
            <select className="narrow" value={s.face_backend} onChange={(e) => save({ ...s, face_backend: e.target.value })}>
              <option value="auto">Automatisch</option>
              <option value="insightface">InsightFace</option>
              <option value="yunet">YuNet (kommerziell ok)</option>
            </select>
          </div>
        </div>
        <h3>Lizenzen der KI-Modelle</h3>
        <table className="list">
          <tbody>
            {licenses.map((l) => <tr key={l.name}><td>{l.commercial_ok ? "✓" : "⚠︎"}</td><td>{l.name}</td><td>{l.license}</td></tr>)}
          </tbody>
        </table>
      </More>
    </div>
  );
}
