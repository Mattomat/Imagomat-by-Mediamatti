import { useEffect, useState } from "react";
import type { AppCtx } from "../App";
import { api, IS_APP, JobInfo, Overview, pickFile, pickFolder, Preset, Profile, Shoot, waitForJob } from "../api";
import { Modal, More, Progress } from "../ui";

function accuracy(p: Profile): string | null {
  const e = p.metrics.mae_model?.Exposure2012;
  if (e === undefined) return null;
  return `Belichtung trifft im Schnitt auf ±${e.toFixed(2).replace(".", ",")} Blenden`;
}

export default function StyleView({ ctx, setDrop }: { ctx: AppCtx; setDrop: (h: ((p: string[]) => void) | null) => void }) {
  const [ov, setOv] = useState<Overview | null>(null);
  const [presets, setPresets] = useState<Preset[]>([]);
  const [def, setDef] = useState<string | null>(null);
  const [source, setSource] = useState<{ catalog?: string; folder?: string }>({});
  const [name, setName] = useState("Mein Stil");
  const [onlyGood, setOnlyGood] = useState(false);
  const [job, setJob] = useState<JobInfo | null>(null);
  const [shoots, setShoots] = useState<Shoot[]>([]);
  const [looks, setLooks] = useState<{ name: string; n: number; base: string }[]>([]);
  const [lookName, setLookName] = useState("Mein Look");
  const [lookBase, setLookBase] = useState("fb_signature");
  const [lookJob, setLookJob] = useState<JobInfo | null>(null);

  const load = () => {
    api.get<Overview>("/api/overview").then((o) => {
      setOv(o);
      setSource((s) => (s.catalog || s.folder ? s : o.catalogs[0] ? { catalog: o.catalogs[0] } : {}));
    });
    api.get<{ default_profile: string | null }>("/api/settings").then((s) => setDef(s.default_profile));
    api.get<{ name: string; n: number; base: string }[]>("/api/looks").then(setLooks).catch(() => setLooks([]));
  };
  useEffect(load, [ctx.tick]);
  useEffect(() => {
    api.get<Preset[]>("/api/presets").then(setPresets);
    api.get<Shoot[]>("/api/shoots").then(setShoots);
  }, []);
  useEffect(() => {
    setDrop((paths) => {
      const p = paths[0];
      if (!p) return;
      setSource(p.endsWith(".lrcat") ? { catalog: p } : { folder: p });
    });
    return () => setDrop(null);
  }, [setDrop]);

  const learnLook = async () => {
    const folder = await pickFolder("Ordner mit deinen fertigen Bildern (JPEG)");
    if (!folder) return;
    try {
      const r = await api.post<{ job_id: number }>("/api/jobs", {
        kind: "learn_look", shoot_id: null, params: { name: lookName.trim() || "Mein Look", folder, base: lookBase },
      });
      ctx.refreshJobs();
      const j = await waitForJob(r.job_id, setLookJob);
      setLookJob(null);
      if (j.status === "done") ctx.toast(j.message ?? "Look gelernt");
      else ctx.toast(j.error?.split("\n")[0] ?? "Look lernen fehlgeschlagen", "error");
      load();
    } catch (e) {
      setLookJob(null);
      ctx.toast((e as Error).message, "error");
    }
  };

  const deleteLook = async (n: string) => {
    await api.del(`/api/looks/${encodeURIComponent(n)}`);
    load();
  };

  const learn = async () => {
    try {
      const r = await api.post<{ job_id: number }>("/api/profiles/train", {
        name: name.trim() || "Mein Stil", catalog: source.catalog ?? null,
        folders: source.folder ? [source.folder] : [], min_rating: onlyGood ? 2 : 0,
      });
      ctx.refreshJobs();
      const j = await waitForJob(r.job_id, setJob);
      setJob(null);
      if (j.status === "done") ctx.toast(j.message ?? "Stil gelernt");
      else ctx.toast(j.error?.split("\n")[0] ?? "Lernen fehlgeschlagen", "error");
      load();
    } catch (e) {
      setJob(null);
      ctx.toast((e as Error).message, "error");
    }
  };

  const [deleting, setDeleting] = useState<string | null>(null);
  const [renaming, setRenaming] = useState<{ old: string; name: string } | null>(null);
  const rename = (old: string) => setRenaming({ old, name: old });
  const doRename = async () => {
    if (!renaming) return;
    const { old } = renaming;
    const neu = renaming.name.trim();
    setRenaming(null);
    if (!neu || neu === old) return;
    try {
      await api.patch(`/api/profiles/${encodeURIComponent(old)}`, { name: neu });
      if (def === old) setDef(neu);
      ctx.toast(`Umbenannt in „${neu}“`);
      load();
    } catch (e) {
      ctx.toast((e as Error).message, "error");
    }
  };

  const setDefault = async (p: string | null) => {
    await api.put("/api/settings", { default_profile: p });
    setDef(p);
    ctx.toast(p ? `„${p}“ ist jetzt dein Standard-Stil` : "Standard: automatisch");
  };

  const feedback = async (profile: string, shootId: number) => {
    const r = await api.post<{ job_id: number }>(`/api/shoots/${shootId}/feedback`, { profile });
    ctx.refreshJobs();
    const j = await waitForJob(r.job_id);
    ctx.toast(j.status === "done" ? j.message ?? "Korrekturen gelernt" : "Keine Korrekturen gefunden", j.status === "done" ? "ok" : "error");
    load();
  };

  const srcLabel = source.catalog ? `Lightroom-Katalog „${source.catalog.split("/").pop()}“`
    : source.folder ? `Ordner „${source.folder.split("/").pop()}“` : "noch nichts gewählt";

  return (
    <div className="page style">
      <h1>Mein Stil</h1>
      <div className="card learn-card">
        <div className="ac-title">Stil aus deinen Bearbeitungen lernen</div>
        <p className="ac-text">
          Imagomat schaut sich an, wie du jedes Bild in Lightroom bearbeitet hast, und bearbeitet neue Shoots genauso,
          individuell pro Bild. Personen, die du in Lightroom benannt hast, werden gleich mit übernommen.
        </p>
        <div className="source-row">
          <div className="source">Quelle: <b>{srcLabel}</b></div>
          <button onClick={async () => { const f = await pickFile(["lrcat"], "Lightroom-Katalog wählen"); if (f) setSource({ catalog: f }); }}>Katalog …</button>
          <button onClick={async () => { const f = await pickFolder("Ordner mit bearbeiteten Bildern (XMP)"); if (f) setSource({ folder: f }); }}>Ordner …</button>
        </div>
        {IS_APP && <div className="hint">Tipp: Katalog oder Ordner einfach hierher ziehen.</div>}
        <More>
          <label>Name des Stils</label>
          <input value={name} onChange={(e) => setName(e.target.value)} />
          <label className="check"><input type="checkbox" checked={onlyGood} onChange={(e) => setOnlyGood(e.target.checked)} /> nur Bilder mit mindestens 2 Sternen verwenden</label>
        </More>
        {job ? (
          <Progress value={job.total ? job.progress / job.total : 0.03} label={job.message ?? "Lerne …"} />
        ) : (
          <button className="primary big" disabled={!source.catalog && !source.folder} onClick={learn}>Stil lernen</button>
        )}
        {job && <div className="hint">Beim ersten Mal dauert das je nach Anzahl Bilder einige Minuten.</div>}
      </div>

      <div className="card learn-card">
        <div className="ac-title">Look aus deinen fertigen Bildern</div>
        <p className="ac-text">
          Lege 10–50 deiner fertig bearbeiteten Bilder (JPEG-Export) in einen Ordner. Imagomat misst daran, wie hell
          Spieler, Himmel, Zuschauer und der Rand unten sind, wie neutral Weiss ist, wie satt jede Farbe ist und wo Weiss
          und Schwarz anschlagen, und bringt dann jedes neue Bild genau dorthin.
        </p>
        <More>
          <label>Name des Looks</label>
          <input value={lookName} onChange={(e) => setLookName(e.target.value)} />
          <label>Grundlage (Masken, Verlauf, Kurve)</label>
          <select value={lookBase} onChange={(e) => setLookBase(e.target.value)}>
            {presets.map((p) => <option key={p.key} value={p.key}>{p.group ? `${p.group}: ` : ""}{p.name}</option>)}
          </select>
        </More>
        {lookJob ? (
          <Progress value={lookJob.total ? lookJob.progress / lookJob.total : 0.03} label={lookJob.message ?? "Messe …"} />
        ) : (
          <button className="primary big" onClick={learnLook}>Ordner wählen und Look lernen</button>
        )}
        {looks.length > 0 && (
          <div className="preset-list mt">
            {looks.map((l) => (
              <div key={l.name} className="preset">
                <b>Look: {l.name}</b>
                <span>aus {l.n} Bildern · im Shoot unter „Stil“ wählbar
                  {" · "}{def === `look:${l.name}` ? "Standard" : <button className="link" onClick={() => setDefault(`look:${l.name}`)}>als Standard</button>}
                  {" · "}<button className="link" style={{ color: "var(--bad)" }} onClick={() => deleteLook(l.name)}>Löschen</button>
                </span>
              </div>
            ))}
          </div>
        )}
      </div>

      {ov && ov.profiles.length > 0 && (
        <>
          <h2 className="mt">Deine Stile</h2>
          <div className="profile-cards">
            {ov.profiles.map((p) => (
              <div key={p.name} className={`card profile-card ${def === p.name ? "default" : ""}`}>
                <div className="pc-head">
                  <div className="ac-title">{p.name}</div>
                  {def === p.name ? <span className="tag">Standard</span>
                    : <button className="link" onClick={() => setDefault(p.name)}>als Standard</button>}
                </div>
                <div className="muted">gelernt aus {p.n} Bildern</div>
                {accuracy(p) && <div className="muted">{accuracy(p)}</div>}
                {p.templates.length > 0 && <div className="muted">Masken: {p.templates.slice(0, 3).map((t) => t.name).join(", ")}</div>}
                <div className="pc-actions">
                  <button className="link" onClick={() => rename(p.name)}>Umbenennen</button>
                  <button className="link" style={{ color: "var(--bad)" }} onClick={() => setDeleting(p.name)}>Löschen</button>
                </div>
                {shoots.length > 0 && (
                  <More label="Aus meinen Korrekturen lernen">
                    <p className="hint">Hast du einen Shoot in Lightroom nachkorrigiert und die Metadaten gespeichert (⌘S)? Dann lernt der Stil daraus.</p>
                    {shoots.slice(0, 5).map((s) => (
                      <button key={s.id} className="small" onClick={() => feedback(p.name, s.id)}>{s.name}</button>
                    ))}
                  </More>
                )}
              </div>
            ))}
          </div>
        </>
      )}

      <h2 className="mt">Fussball-Stile</h2>
      <p className="hint">Zum Ausprobieren: im Shoot unter „Stile vergleichen“ nebeneinander ansehen und für alle übernehmen.
        Belichtung, Weiss/Schwarz, Verlauf und Denoise passen sich weiterhin jedem Bild an.</p>
      <div className="preset-list">
        {presets.filter((p) => p.group === "Fussball").map((p) => (
          <div key={p.key} className="preset"><b>{p.name}</b><span>{p.description}</span></div>
        ))}
      </div>
      <h2 className="mt">Mitgelieferte Stile</h2>
      <p className="hint">Werden automatisch je nach Situation gewählt, solange du keinen eigenen Stil als Standard hast.</p>
      <div className="preset-list">
        {presets.filter((p) => !p.group).map((p) => (
          <div key={p.key} className="preset"><b>{p.name}</b><span>{p.description}</span></div>
        ))}
      </div>
      {def && <button className="link mt" onClick={() => setDefault(null)}>Standard auf „automatisch“ zurücksetzen</button>}
      {renaming && (
        <Modal title="Stil umbenennen" onClose={() => setRenaming(null)}>
          <input autoFocus value={renaming.name} onChange={(e) => setRenaming({ ...renaming, name: e.target.value })}
            onKeyDown={(e) => e.key === "Enter" && doRename()} />
          <div className="modal-actions">
            <button className="ghost" onClick={() => setRenaming(null)}>Abbrechen</button>
            <button className="primary" onClick={doRename}>Umbenennen</button>
          </div>
        </Modal>
      )}
      {deleting && (
        <Modal title={`Stil „${deleting}“ löschen?`} onClose={() => setDeleting(null)}>
          <p>Der gelernte Stil wird entfernt. Bereits bearbeitete Shoots und exportierte XMPs bleiben, wie sie sind.</p>
          <div className="modal-actions">
            <button className="ghost" onClick={() => setDeleting(null)}>Abbrechen</button>
            <button className="danger" onClick={async () => {
              try {
                await api.del(`/api/profiles/${encodeURIComponent(deleting)}`);
                ctx.toast(`Stil „${deleting}“ gelöscht`);
                load();
              } catch (e) {
                ctx.toast((e as Error).message, "error");
              }
              setDeleting(null);
            }}>Löschen</button>
          </div>
        </Modal>
      )}
    </div>
  );
}
