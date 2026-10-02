import { useEffect, useState } from "react";
import type { AppCtx } from "../App";
import { api, IS_APP, Overview, pickFolder, Shoot } from "../api";
import { Modal } from "../ui";

export default function HomeView({ ctx, setDrop }: { ctx: AppCtx; setDrop: (h: ((p: string[]) => void) | null) => void }) {
  const [ov, setOv] = useState<Overview | null>(null);
  const [shoots, setShoots] = useState<Shoot[]>([]);
  const [folder, setFolder] = useState("");
  const [teams, setTeams] = useState<string[]>([]);
  const [busy, setBusy] = useState(false);
  const [deleting, setDeleting] = useState<Shoot | null>(null);
  const [info, setInfo] = useState<{ count: number } | null>(null);

  useEffect(() => { api.get<Overview>("/api/overview").then(setOv).catch(() => undefined); }, []);
  useEffect(() => {
    setInfo(null);
    if (folder) api.get<{ count: number }>(`/api/import/info?folder=${encodeURIComponent(folder)}`).then(setInfo).catch(() => undefined);
  }, [folder]);
  useEffect(() => { api.get<Shoot[]>("/api/shoots").then(setShoots); }, [ctx.tick]);
  useEffect(() => {
    setDrop((paths) => paths[0] && setFolder(paths[0]));
    return () => setDrop(null);
  }, [setDrop]);

  const start = async () => {
    setBusy(true);
    try {
      const r = await api.post<{ shoot_id: number }>("/api/shoots/import", { folder, teams, mode: "people" });
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
      <h1>Personen benennen</h1>
      <p className="muted" style={{ marginTop: -8 }}>Ordner mit Fotos (RAW oder JPG) wählen – Imagomat erkennt die Gesichter, du benennst sie einmal, die Namen gehen nach Lightroom.</p>
      <div className={`dropzone ${folder ? "has" : ""}`}
        onClick={async () => { const f = await pickFolder("Ordner mit Fotos wählen"); if (f) setFolder(f); }}>
        {folder ? (
          <>
            <div className="dz-big">📁 {folderName}</div>
            <div className="dz-small">{folder}{info ? ` · ${info.count} Bilder` : ""}</div>
          </>
        ) : (
          <>
            <div className="dz-big">{IS_APP ? "Ordner hierher ziehen" : "Ordner wählen"}</div>
            <div className="dz-small">oder klicken, um einen Ordner auszuwählen</div>
          </>
        )}
      </div>
      {ov && ov.teams.length > 0 && (
        <div className="opt" style={{ marginTop: 18 }}>
          <label>Team (optional): nur Personen dieses Teams vorschlagen</label>
          <div className="chips">
            {ov.teams.map((t) => (
              <button key={t} className={teams.includes(t) ? "chip on" : "chip"}
                onClick={() => setTeams(teams.includes(t) ? teams.filter((x) => x !== t) : [...teams, t])}>{t}</button>
            ))}
          </div>
        </div>
      )}
      {ov && ov.persons === 0 && (
        <p className="hint">Noch keine Personen bekannt: unter <button className="link" onClick={() => ctx.go({ name: "people" })}>Personen</button> aus
          Lightroom übernehmen oder einen Kader (CSV) laden – oder einfach loslegen und die Gesichter im Shoot benennen.</p>
      )}
      <button className="primary big" disabled={!folder || busy} onClick={start}>Personen erkennen</button>

      {shoots.length > 0 && (
        <>
          <h2 className="mt">Deine Shoots</h2>
          <div className="shoot-cards">
            {shoots.map((s) => {
              const running = s.job && ["running", "queued"].includes(s.job.status);
              return (
                <div key={s.id} className="shoot-card" role="button" tabIndex={0}
                  onClick={() => ctx.go({ name: "shoot", id: s.id })}
                  onKeyDown={(e) => e.key === "Enter" && ctx.go({ name: "shoot", id: s.id })}>
                  <div className="cover">{s.cover ? <img loading="lazy" src={api.img(`/api/images/${s.cover}/thumb`)} /> : null}</div>
                  <button className="sc-del" title="Shoot entfernen" onClick={(e) => { e.stopPropagation(); setDeleting(s); }}>✕</button>
                  <div className="sc-name">{s.name}</div>
                  <div className="sc-meta">
                    {running ? `wird erkannt …` : `${s.n} Bilder`}
                  </div>
                </div>
              );
            })}
          </div>
        </>
      )}
      {deleting && (
        <Modal title={`„${deleting.name}“ entfernen?`} onClose={() => setDeleting(null)}>
          <p>Der Shoot verschwindet aus Imagomat, samt Auswahl und Bearbeitung.</p>
          <p className="hint">Deine Originalbilder bleiben unangetastet. Gesichter, die du benannt hast, behält Imagomat
            für die Personenerkennung.</p>
          <div className="modal-actions">
            <button className="ghost" onClick={() => setDeleting(null)}>Abbrechen</button>
            <button className="danger" onClick={async () => {
              try {
                await api.del(`/api/shoots/${deleting.id}`);
                ctx.toast(`„${deleting.name}“ entfernt`);
                setShoots((xs) => xs.filter((x) => x.id !== deleting.id));
              } catch (e) {
                ctx.toast((e as Error).message, "error");
              }
              setDeleting(null);
            }}>Entfernen</button>
          </div>
        </Modal>
      )}
    </div>
  );
}
