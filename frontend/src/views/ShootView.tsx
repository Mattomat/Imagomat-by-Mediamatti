// Personen benennen: Shoot als Raster/Lupe, "Wer ist das?" für Gesichtsgruppen, Namen speichern für Lightroom.
import { memo, useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { AppCtx } from "../App";
import { api, ImageItem, IS_APP, pickFolder, reveal, Shoot, waitForJob } from "../api";
import PeoplePanel from "../components/PeoplePanel";
import WhoPanel from "../components/WhoPanel";
import { Modal, Progress } from "../ui";

type Tab = "all" | "named" | "unnamed" | "check";
const stored = (k: string, d: string) => { try { return localStorage.getItem(k) ?? d; } catch { return d; } };
const store = (k: string, v: string) => { try { localStorage.setItem(k, v); } catch { /* egal */ } };

export default function ShootView({ ctx, id }: { ctx: AppCtx; id: number }) {
  const [shoot, setShoot] = useState<Shoot | null>(null);
  const [items, setItems] = useState<ImageItem[]>([]);
  const [tab, setTab] = useState<Tab>("all");
  const [person, setPerson] = useState("");
  const [sel, setSel] = useState(0);
  const [loupe, setLoupe] = useState(false);
  const [who, setWho] = useState(false);
  const [saving, setSaving] = useState(false);
  const [cell, setCell] = useState(() => +stored("imagomat.cell", "210"));
  const [teamsAll, setTeamsAll] = useState<string[]>([]);
  const gridRef = useRef<HTMLDivElement>(null);
  const openLoupe = useCallback((idx: number) => { setSel(idx); setLoupe(true); }, []);

  const load = useCallback(() => {
    api.get<Shoot>(`/api/shoots/${id}`).then(setShoot).catch(() => undefined);
    api.get<ImageItem[]>(`/api/shoots/${id}/images`).then(setItems).catch(() => undefined);
  }, [id]);
  useEffect(load, [load, ctx.tick]);
  useEffect(() => { api.get<{ teams: string[] }>("/api/overview").then((o) => setTeamsAll(o.teams)).catch(() => undefined); }, []);

  const liveJob = ctx.jobs.find((j) => j.id === shoot?.job?.id);
  const job = liveJob ?? shoot?.job ?? null;
  const running = !!job && ["running", "queued"].includes(job.status);
  useEffect(() => {
    if (!running) return;
    const t = setInterval(() => api.get<ImageItem[]>(`/api/shoots/${id}/images`).then(setItems).catch(() => undefined), 2500);
    return () => clearInterval(t);
  }, [running, id]);

  const isCheck = (i: ImageItem) => (i.people_check?.length ?? 0) > 0;
  const counts = useMemo(() => ({
    all: items.length,
    named: items.filter((i) => i.people.length > 0).length,
    unnamed: items.filter((i) => (i.faces ?? 0) > 0 && i.people.length === 0).length,
    check: items.filter(isCheck).length,
  }), [items]);
  const people = useMemo(() => {
    const m = new Map<string, number>();
    items.forEach((i) => i.people.forEach((p) => m.set(p, (m.get(p) ?? 0) + 1)));
    return [...m.entries()].sort((a, b) => b[1] - a[1]);
  }, [items]);
  const shown = useMemo(() => items.filter((i) => {
    if (person && !i.people.includes(person)) return false;
    if (tab === "named") return i.people.length > 0;
    if (tab === "unnamed") return (i.faces ?? 0) > 0 && i.people.length === 0;
    if (tab === "check") return isCheck(i);
    return true;
  }), [items, tab, person]);
  const curIdx = Math.min(sel, Math.max(0, shown.length - 1));
  const cur = shown[curIdx];
  const settings = (() => { try { return JSON.parse(shoot?.settings || "{}"); } catch { return {}; } })();
  const shootTeams: string[] = settings.teams ?? [];

  const setTeam = async (t: string) => {
    await api.patch(`/api/shoots/${id}`, { teams: t ? [t] : [] });
    const r = await api.post<{ job_id: number }>(`/api/shoots/${id}/run/people`, {});
    ctx.refreshJobs();
    await waitForJob(r.job_id);
    load();
    ctx.toast(t ? `Personen neu erkannt (nur Team ${t})` : "Personen neu erkannt");
  };

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement;
      if (["INPUT", "SELECT", "TEXTAREA"].includes(t.tagName) || saving || who || !cur) return;
      const k = e.key;
      const cols = gridRef.current ? Math.max(1, Math.floor(gridRef.current.clientWidth / (cell + 8))) : 5;
      if (k === "ArrowRight") setSel((s) => Math.min(shown.length - 1, s + 1));
      else if (k === "ArrowLeft") setSel((s) => Math.max(0, s - 1));
      else if (k === "ArrowDown" && !loupe) setSel((s) => Math.min(shown.length - 1, s + cols));
      else if (k === "ArrowUp" && !loupe) setSel((s) => Math.max(0, s - cols));
      else if (k === "Enter" || k === "e" || k === "E") setLoupe((l) => !l);
      else if (k === "g" || k === "G" || k === "Escape") setLoupe(false);
      else return;
      e.preventDefault();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [cur, shown.length, loupe, saving, who, cell]);
  useEffect(() => { document.getElementById(`t-${cur?.id}`)?.scrollIntoView({ block: "nearest" }); }, [cur?.id]);

  if (!shoot) return <div className="page" />;
  const step = job?.message ?? "Starte …";
  return (
    <div className="lib" style={{ ["--cell" as string]: `${cell}px` }}>
      <header className="lib-top">
        <div className="lib-title">
          <h1>{shoot.name}</h1>
          <span className="muted">{items.length} Bilder · {counts.named} mit Namen</span>
        </div>
        <div className="lib-actions">
          <button className={who ? "on" : ""} onClick={() => setWho((w) => !w)} disabled={running}>Wer ist das?</button>
          <button className="primary" disabled={running || items.length === 0} onClick={() => setSaving(true)}>Namen speichern …</button>
        </div>
      </header>

      {running && (
        <div className="lib-progress">
          <div className="bar"><div style={{ width: `${job && job.total ? Math.round((100 * job.progress) / job.total) : 3}%` }} /></div>
          <span className="muted lib-msg">{step}</span>
        </div>
      )}
      {!running && items.length > 0 && job?.status === "failed" && (
        <div className="lib-progress">
          <span>Erkennung fehlgeschlagen: {job.error?.split("\n")[0]}</span>
          <button className="primary small" onClick={async () => { await api.post(`/api/shoots/${id}/run/pipeline`, { mode: "people" }); ctx.refreshJobs(); load(); }}>Nochmals</button>
        </div>
      )}

      {!who && (
        <div className="lib-filter">
          <div className="segmented">
            {([["all", "Alle"], ["named", "Mit Namen"], ["unnamed", "Ohne Namen"], ...(counts.check ? [["check", "Prüfen"]] : [])] as [Tab, string][]).map(([k, l]) => (
              <button key={k} className={tab === k ? "on" : ""} onClick={() => { setTab(k); setSel(0); }}>{l} <b>{counts[k]}</b></button>
            ))}
          </div>
          {people.length > 0 && (
            <select value={person} onChange={(e) => { setPerson(e.target.value); setSel(0); }}>
              <option value="">Alle Personen ({people.length})</option>
              {people.map(([p, n]) => <option key={p} value={p}>{p} ({n})</option>)}
            </select>
          )}
          <span className="spacer" />
          {teamsAll.length > 0 && (
            <select value={shootTeams[0] ?? ""} disabled={running} title="Nur Personen dieses Teams erkennen"
              onChange={(e) => setTeam(e.target.value)}>
              <option value="">Team: alle</option>
              {teamsAll.map((t) => <option key={t} value={t}>Team: {t}</option>)}
            </select>
          )}
        </div>
      )}

      {who ? <div className="lib-grid lib-who"><WhoPanel ctx={ctx} shootId={id} onChanged={load} /></div>
        : !loupe ? (
          <div className="lib-grid" ref={gridRef}>
            {shown.length === 0 && <div className="empty">{running ? "Bilder werden eingelesen …" : "Keine Bilder in dieser Ansicht."}</div>}
            {shown.map((i, idx) => <Thumb key={i.id} i={i} idx={idx} selected={idx === curIdx} onSelect={setSel} onOpen={openLoupe} />)}
          </div>
        ) : cur && (
          <div className="lib-loupe">
            <div className="stage">
              <StageImage key={cur.id} src={api.img(`/api/images/${cur.id}/preview`)} />
            </div>
            <PeoplePanel ctx={ctx} image={cur} onChanged={load} />
            <div className="loupe-bar">
              <span className="fname">{cur.filename}</span>
              {cur.people.length > 0 && <span className="who">{cur.people.join(", ")}</span>}
              <span className="spacer" />
              <button disabled={curIdx === 0} onClick={() => setSel(curIdx - 1)}>←</button>
              <button disabled={curIdx >= shown.length - 1} onClick={() => setSel(curIdx + 1)}>→</button>
              <button className="ghost" onClick={() => setLoupe(false)}>Raster</button>
            </div>
          </div>
        )}
      <footer className="lib-bottom">
        <button className={!loupe && !who ? "on" : ""} onClick={() => { setWho(false); setLoupe(false); }} title="Raster (G)">▦</button>
        <button className={loupe && !who ? "on" : ""} onClick={() => { setWho(false); if (cur) setLoupe(true); }} title="Lupe (E)">▢</button>
        <span className="muted keyhint">Doppelklick: Bild öffnen und Gesichter benennen · ← → blättern · G Raster</span>
        <span className="spacer" />
        {!loupe && !who && <label className="lib-size">Miniaturen <input type="range" min={130} max={420} value={cell}
          onChange={(e) => { setCell(+e.target.value); store("imagomat.cell", e.target.value); }} /></label>}
      </footer>
      {saving && <TagExportDialog ctx={ctx} shoot={shoot} total={items.length} named={counts.named} onClose={() => setSaving(false)} />}
    </div>
  );
}

/** Namen speichern: RAW-Dateien bekommen die Namen als XMP daneben (Original unverändert), JPGs als Kopie mit Namen. */
function TagExportDialog({ ctx, shoot, total, named, onClose }: { ctx: AppCtx; shoot: Shoot; total: number; named: number; onClose: () => void }) {
  const parent = shoot.folder.replace(/\/[^/]+\/?$/, "");
  const [mode, setMode] = useState<"inplace" | "copy">(stored("imagomat.tagMode", "inplace") as "inplace" | "copy");
  const [root, setRoot] = useState(stored("imagomat.tagRoot", parent));
  const [folderName, setFolderName] = useState(`${shoot.name} – mit Namen`);
  const [onlyPeople, setOnlyPeople] = useState(false);
  const [state, setState] = useState<"form" | "busy" | "done">("form");
  const [progress, setProgress] = useState(0);
  const [msg, setMsg] = useState("");
  const target = folderName.trim() ? `${root.replace(/\/$/, "")}/${folderName.trim()}` : root;

  const go = async () => {
    setState("busy");
    store("imagomat.tagMode", mode);
    try {
      const r = await api.post<{ job_id: number }>(`/api/shoots/${shoot.id}/tag-export`, { target, only_with_people: onlyPeople, inplace: mode === "inplace" });
      ctx.refreshJobs();
      const j = await waitForJob(r.job_id, (x) => setProgress(x.total ? x.progress / x.total : 0));
      if (j.status !== "done") throw new Error(j.error?.split("\n")[0] ?? "Speichern fehlgeschlagen");
      setMsg((j.message ?? "").split(" -> ")[0]);
      setState("done");
    } catch (e) {
      ctx.toast((e as Error).message, "error");
      setState("form");
    }
  };

  return (
    <Modal title="Namen speichern" onClose={onClose}>
      {state === "form" && (
        <>
          <p>{named} von {total} Bildern haben Namen. Gespeichert werden Stichwörter und Gesichtsbereiche – Lightroom zeigt die Personen danach in „Personen“ und in den Stichwörtern.</p>
          <label className="check"><input type="radio" checked={mode === "inplace"} onChange={() => setMode("inplace")} />
            Direkt zu den Bildern (RAW: XMP-Datei daneben, Original bleibt unverändert)</label>
          <label className="check"><input type="radio" checked={mode === "copy"} onChange={() => setMode("copy")} />
            Kopie in einen neuen Ordner (JPGs mit Namen in der Datei)</label>
          {(mode === "copy") && <>
            <label>Ordner (wird gemerkt)</label>
            <div className="row">
              <input value={root} onChange={(e) => { setRoot(e.target.value); store("imagomat.tagRoot", e.target.value); }} />
              <button onClick={async () => { const d = await pickFolder("Zielordner wählen"); if (d) { setRoot(d); store("imagomat.tagRoot", d); } }}>Wählen…</button>
            </div>
            <label>Neuer Ordner</label>
            <input value={folderName} onChange={(e) => setFolderName(e.target.value)} />
          </>}
          {mode === "inplace" && <div className="hint">JPGs werden nie verändert: sie landen mit Namen in „{shoot.name} – mit Namen“ neben dem Shoot-Ordner.</div>}
          <label className="check"><input type="checkbox" checked={onlyPeople} onChange={(e) => setOnlyPeople(e.target.checked)} /> nur Bilder mit erkannten Personen</label>
          <div className="modal-foot">
            <button className="ghost" onClick={onClose}>Abbrechen</button>
            <button className="primary" onClick={go}>Speichern</button>
          </div>
        </>
      )}
      {state === "busy" && <Progress value={progress} label="Namen werden geschrieben …" />}
      {state === "done" && (
        <>
          <p className="success">{msg}</p>
          <ol className="steps">
            {mode === "inplace"
              ? <li>Schon in Lightroom: Bilder markieren → <b>Metadaten → Metadaten aus Datei lesen</b>. Noch nicht: einfach importieren (<b>Hinzufügen</b>).</li>
              : <li>In Lightroom <b>Importieren</b> → Ordner <code>{target.split("/").pop()}</code> → <b>Hinzufügen</b>.</li>}
          </ol>
          <div className="modal-foot">
            <button onClick={() => reveal(mode === "inplace" ? shoot.folder : target)}>Im Finder zeigen</button>
            {IS_APP && mode === "copy" && <button onClick={() => api.post("/api/lightroom/open", { path: target }).catch((e) => ctx.toast(String((e as Error).message), "error"))}>In Lightroom importieren</button>}
            <button className="primary" onClick={onClose}>Schliessen</button>
          </div>
        </>
      )}
    </Modal>
  );
}

const Thumb = memo(function Thumb({ i, idx, selected, onSelect, onOpen }: {
  i: ImageItem; idx: number; selected: boolean; onSelect: (i: number) => void; onOpen: (i: number) => void;
}) {
  return (
    <div id={`t-${i.id}`} className={`cellx ${selected ? "sel" : ""}`}
      onClick={() => onSelect(idx)} onDoubleClick={() => onOpen(idx)}>
      <span className="cell-n">{idx + 1}</span>
      <div className="cell-img"><img loading="lazy" decoding="async" draggable={false}
        src={api.img(`/api/images/${i.id}/thumb`)} /></div>
      <div className="cell-foot">
        {i.people.length > 0 ? <span className="cell-names">{i.people.join(", ")}</span>
          : (i.faces ?? 0) > 0 ? <span className="cell-who">{i.faces} {i.faces === 1 ? "Gesicht" : "Gesichter"} ohne Namen</span>
            : <span className="cell-who">keine Person</span>}
      </div>
    </div>
  );
});

function StageImage({ src, placeholder }: { src: string; placeholder?: string }) {
  // Sofort die Kamera-Vorschau zeigen, die Bearbeitung blendet darüber, sobald sie fertig ist
  const [state, setState] = useState<"load" | "ok" | "err">("load");
  const showPh = placeholder && placeholder !== src && state !== "ok";
  return (
    <>
      {showPh && <img src={placeholder} className="stage-ph" />}
      <img src={src} onLoad={() => setState("ok")} onError={() => setState("err")}
        style={state === "ok" ? undefined : showPh ? { position: "absolute", opacity: 0 } : { opacity: 0.35 }} />
      {state === "err" && <div className="stage-loading">Vorschau nicht möglich (Protokoll in den Einstellungen)</div>}
    </>
  );
}

