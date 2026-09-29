import { useEffect, useState } from "react";
import type { AppCtx } from "../App";
import { api, IS_APP, Overview, pickFolder, Shoot } from "../api";
import { SELECTION_HINT, SELECTION_PARAMS, SELECTIONS, Selection } from "../selection";
import { Segmented } from "../ui";

export default function HomeView({ ctx, setDrop }: { ctx: AppCtx; setDrop: (h: ((p: string[]) => void) | null) => void }) {
  const [ov, setOv] = useState<Overview | null>(null);
  const [shoots, setShoots] = useState<Shoot[]>([]);
  const [folder, setFolder] = useState("");
  const [profile, setProfile] = useState<string>("");
  const [sel, setSel] = useState<Selection>("normal");
  const [maxKeep, setMaxKeep] = useState("");
  const [teams, setTeams] = useState<string[]>([]);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    api.get<Overview>("/api/overview").then((o) => {
      setOv(o);
      api.get<{ default_profile: string | null }>("/api/settings").then((s) => setProfile(s.default_profile ?? ""));
    });
  }, []);
  useEffect(() => {
    api.get<Shoot[]>("/api/shoots").then(setShoots);
  }, [ctx.tick]);
  useEffect(() => {
    setDrop((paths) => paths[0] && setFolder(paths[0]));
    return () => setDrop(null);
  }, [setDrop]);

  const start = async () => {
    setBusy(true);
    try {
      const r = await api.post<{ shoot_id: number }>("/api/shoots/import", {
        folder, profile: profile || undefined, teams, ...SELECTION_PARAMS[sel],
        max_keep: maxKeep ? Math.max(1, parseInt(maxKeep, 10)) : undefined,
      });
      ctx.refreshJobs();
      ctx.go({ name: "shoot", id: r.shoot_id });
    } catch (e) {
      ctx.toast(String((e as Error).message), "error");
    } finally {
      setBusy(false);
    }
  };

  const folderName = folder.split("/").filter(Boolean).pop();

  return (
    <div className="page home">
      <h1>Neuer Shoot</h1>
      <div
        className={`dropzone ${folder ? "has" : ""}`}
        onClick={async () => {
          const f = await pickFolder("Ordner oder Speicherkarte wählen");
          if (f) setFolder(f);
        }}
      >
        {folder ? (
          <>
            <div className="dz-big">📁 {folderName}</div>
            <div className="dz-small">{folder}</div>
          </>
        ) : (
          <>
            <div className="dz-big">{IS_APP ? "Ordner hierher ziehen" : "Ordner wählen"}</div>
            <div className="dz-small">oder klicken, um einen Ordner bzw. die Speicherkarte auszuwählen</div>
          </>
        )}
      </div>

      <div className="options">
        <div className="opt">
          <label>Stil</label>
          <select value={profile} onChange={(e) => setProfile(e.target.value)}>
            <option value="">Automatisch (passender Stil je Situation)</option>
            {ov?.profiles.map((p) => <option key={p.name} value={p.name}>{p.name}</option>)}
          </select>
          {ov && ov.profiles.length === 0 && (
            <button className="link" onClick={() => ctx.go({ name: "style" })}>Eigenen Stil aus Lightroom lernen →</button>
          )}
        </div>
        <div className="opt">
          <label>Auswahl</label>
          <Segmented value={sel} options={SELECTIONS} onChange={setSel} />
          <div className="hint">{SELECTION_HINT[sel]}</div>
          <div className="max-keep">
            höchstens
            <input type="number" min={1} placeholder="–" value={maxKeep} onChange={(e) => setMaxKeep(e.target.value)} />
            Bilder <span className="muted">(leer = nur Prozent)</span>
          </div>
        </div>
        {ov && ov.teams.length > 0 && (
          <div className="opt">
            <label>Teams im Bild (für Rückennummern)</label>
            <div className="chips">
              {ov.teams.map((t) => (
                <button key={t} className={teams.includes(t) ? "chip on" : "chip"}
                  onClick={() => setTeams(teams.includes(t) ? teams.filter((x) => x !== t) : [...teams, t])}>
                  {t}
                </button>
              ))}
            </div>
          </div>
        )}
      </div>
      <button className="primary big" disabled={!folder || busy} onClick={start}>Los</button>

      {shoots.length > 0 && (
        <>
          <h2 className="mt">Deine Shoots</h2>
          <div className="shoot-cards">
            {shoots.map((s) => {
              const running = s.job && ["running", "queued"].includes(s.job.status);
              return (
                <button key={s.id} className="shoot-card" onClick={() => ctx.go({ name: "shoot", id: s.id })}>
                  <div className="cover">{s.cover ? <img src={api.img(`/api/images/${s.cover}/preview`)} /> : null}</div>
                  <div className="sc-name">{s.name}</div>
                  <div className="sc-meta">
                    {running ? `wird verarbeitet …` : s.kept != null ? `${s.kept} von ${s.n} behalten` : `${s.n} Bilder`}
                  </div>
                </button>
              );
            })}
          </div>
        </>
      )}
    </div>
  );
}
