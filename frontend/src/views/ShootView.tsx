// Bibliothek wie in Lightroom: Raster (bzw. Lupe) mit Filterleiste; Verarbeitung läuft oben schmal mit.
import { memo, useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { AppCtx } from "../App";
import { api, ImageItem, IS_APP, pickFiles, pickFolder, reveal, Shoot, waitForJob } from "../api";
import PeoplePanel from "../components/PeoplePanel";
import ExportDialog from "../components/ExportDialog";
import Develop from "../develop/Develop";
import { SELECTION_HINT, SELECTION_PARAMS, SELECTIONS, Selection, selectionFromSettings } from "../selection";
import { Modal, Progress, Segmented } from "../ui";

type Tab = "all" | "keep" | "reject" | "check";
type StyleOpt = { key: string; label: string; group?: string; user?: boolean };

const STEPS = ["Kopieren", "Vorschauen", "Aussortieren", "Personen", "Entwickeln"];

function needsCheck(i: ImageItem): boolean {
  return (i.people_check?.length ?? 0) > 0
    || (i.decision === "keep" && i.confidence !== null && i.confidence < 0.35);
}

function stepOf(msg: string | null | undefined): number {
  if (!msg) return 1;
  if (/^Kopier/.test(msg)) return 0;
  const m = msg.match(/^(\d)\/\d/);
  if (m) return +m[1];
  if (/Culling|behalten/.test(msg)) return 2;
  if (/Personen|Gesichter wiedererkannt|Rückennummer/.test(msg)) return 3;
  if (/Entwick|Neutral/.test(msg)) return 4;
  return 1;
}
const loupeSize = () => {
  const px = Math.max(window.innerWidth, window.innerHeight) * Math.min(window.devicePixelRatio || 1, 2);
  return px <= 2100 ? 2048 : px <= 2950 ? 2880 : 3600;
};
const stored = (k: string, d: string) => { try { return localStorage.getItem(k) ?? d; } catch { return d; } };
const store = (k: string, v: string) => { try { localStorage.setItem(k, v); } catch { /* egal */ } };

export default function ShootView({ ctx, id }: { ctx: AppCtx; id: number }) {
  const [shoot, setShoot] = useState<Shoot | null>(null);
  const [items, setItems] = useState<ImageItem[]>([]);
  const [tab, setTab] = useState<Tab | null>(null);
  const [person, setPerson] = useState("");
  const [minStars, setMinStars] = useState(0);
  const [sel, setSel] = useState(0);
  const [loupe, setLoupe] = useState(false);
  const [before, setBefore] = useState(false);
  const [editing, setEditing] = useState(false);
  const [exporting, setExporting] = useState(false);
  const [social, setSocial] = useState(false);
  const [tagging, setTagging] = useState(false);
  const [cell, setCell] = useState(() => +stored("imagomat.cell", "210"));
  const [editV, setEditV] = useState(0);
  const [styles, setStyles] = useState<StyleOpt[]>([]);
  const [teamsAll, setTeamsAll] = useState<string[]>([]);
  const gridRef = useRef<HTMLDivElement>(null);
  const openLoupe = useCallback((idx: number) => { setSel(idx); setLoupe(true); }, []);

  const load = useCallback(() => {
    api.get<Shoot>(`/api/shoots/${id}`).then(setShoot).catch(() => undefined);
    api.get<ImageItem[]>(`/api/shoots/${id}/images`).then(setItems).catch(() => undefined);
  }, [id]);
  useEffect(load, [load, ctx.tick]);
  const loadStyles = useCallback(() => api.get<StyleOpt[]>("/api/styles").then(setStyles).catch(() => undefined), []);
  useEffect(() => { loadStyles(); }, [loadStyles, ctx.tick]);
  useEffect(() => { api.get<{ teams: string[] }>("/api/overview").then((o) => setTeamsAll(o.teams)).catch(() => undefined); }, []);

  const liveJob = ctx.jobs.find((j) => j.id === shoot?.job?.id);
  const job = liveJob ?? shoot?.job ?? null;
  const running = !!job && ["running", "queued"].includes(job.status);
  // während der Verarbeitung (und beim Kopieren) füllt sich die Galerie laufend
  useEffect(() => {
    if (!running) return;
    const t = setInterval(() => api.get<ImageItem[]>(`/api/shoots/${id}/images`).then(setItems).catch(() => undefined), 2500);
    return () => clearInterval(t);
  }, [running, id]);
  // bearbeitete Kacheln erscheinen, sobald sie im Hintergrund vorberechnet sind
  const waiting = !editing && !running && items.some((i) => i.decision === "keep" && i.edited && !i.rendered);
  useEffect(() => {
    if (!waiting) return;
    const t = setInterval(() => api.get<ImageItem[]>(`/api/shoots/${id}/images`).then(setItems).catch(() => undefined), 6000);
    return () => clearInterval(t);
  }, [waiting, id]);

  const counts = useMemo(() => ({
    all: items.length,
    keep: items.filter((i) => i.decision === "keep").length,
    reject: items.filter((i) => i.decision === "reject").length,
    check: items.filter((i) => needsCheck(i)).length,
  }), [items]);
  const curTab: Tab = tab ?? (counts.keep > 0 ? "keep" : "all");
  const people = useMemo(() => [...new Set(items.flatMap((i) => i.people))].sort(), [items]);
  const shown = useMemo(() => items.filter((i) => {
    if (person && !i.people.includes(person)) return false;
    if (minStars && (i.rating ?? 0) < minStars) return false;
    if (curTab === "keep") return i.decision === "keep";
    if (curTab === "reject") return i.decision === "reject";
    if (curTab === "check") return needsCheck(i);
    return true;
  }), [items, curTab, person, minStars]);
  const curIdx = Math.min(sel, Math.max(0, shown.length - 1));
  const cur = shown[curIdx];
  const settings = (() => { try { return JSON.parse(shoot?.settings || "{}"); } catch { return {}; } })();
  const peopleMode = settings.mode === "people";
  const selection = selectionFromSettings(shoot?.settings);
  const shootTeams: string[] = settings.teams ?? [];
  const decided = items.some((i) => i.decision !== null);

  usePrefetch(loupe && !editing ? [1, -1].map((d) => shown[curIdx + d]).filter((x) => x && x.edited)
    .map((x) => api.img(`/api/images/${x!.id}/render?size=${loupeSize()}&v=${editV}`)) : []);

  const patch = async (i: ImageItem, body: Partial<Pick<ImageItem, "decision" | "rating">>) => {
    await api.patch(`/api/images/${i.id}/culling`, body);
    setItems((xs) => xs.map((x) => (x.id === i.id ? { ...x, ...body, manual: true } : x)));
  };
  const runJob = async (path: string, body: unknown, done: string) => {
    try {
      const r = await api.post<{ job_id: number }>(path, body);
      ctx.refreshJobs();
      const j = await waitForJob(r.job_id);
      load();
      if (j.status === "done") ctx.toast(done); else ctx.toast(j.error?.split("\n")[0] ?? "Fehlgeschlagen", "error");
    } catch (e) { ctx.toast((e as Error).message, "error"); }
  };
  const applyPreset = async (key: string) => {
    if (key === "__import") { await importPresets(); return; }
    const label = styles.find((s) => s.key === key)?.label ?? key;
    await api.patch(`/api/shoots/${id}`, { profile: key === "preset:auto" ? "" : key });
    await runJob(`/api/shoots/${id}/run/develop`, { profile: key }, `Vorgabe „${label}“ auf alle angewendet (von Hand bearbeitete Bilder bleiben)`);
    setEditV((v) => v + 1);
  };
  const importPresets = async () => {
    const files = await pickFiles(["xmp", "lrtemplate"], "Lightroom-Vorgaben wählen (.xmp oder .lrtemplate)");
    if (!files.length) return;
    try {
      const r = await api.post<{ imported: string[]; errors: string[] }>("/api/presets/import", { paths: files });
      ctx.toast(`${r.imported.length} Vorgabe(n) importiert${r.errors.length ? `, ${r.errors.length} übersprungen` : ""}`);
      loadStyles();
    } catch (e) { ctx.toast((e as Error).message, "error"); }
  };
  const setTeam = (t: string) => runJob(`/api/shoots/${id}/run/people`, {}, t ? `Personen neu erkannt (Team ${t})` : "Personen neu erkannt")
    .then(() => undefined);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement;
      if (["INPUT", "SELECT", "TEXTAREA"].includes(t.tagName) || exporting || editing || social || !cur) return;
      const k = e.key;
      if (["d", "D", "e", "E"].includes(k) && !peopleMode && !e.metaKey) { setEditing(true); e.preventDefault(); return; }
      const cols = gridRef.current ? Math.max(1, Math.floor(gridRef.current.clientWidth / (cell + 8))) : 5;
      if (k === "ArrowRight") setSel((s) => Math.min(shown.length - 1, s + 1));
      else if (k === "ArrowLeft") setSel((s) => Math.max(0, s - 1));
      else if (k === "ArrowDown" && !loupe) setSel((s) => Math.min(shown.length - 1, s + cols));
      else if (k === "ArrowUp" && !loupe) setSel((s) => Math.max(0, s - cols));
      else if (/^[0-5]$/.test(k)) patch(cur, { rating: +k, ...(+k > 0 ? { decision: "keep" as const } : {}) });
      else if (k === "x" || k === "X") patch(cur, { decision: "reject" });
      else if (k === "p" || k === "P") patch(cur, { decision: "keep" });
      else if (k === "Enter" || k === "e" || k === "E") setLoupe((l) => !l);
      else if (k === "g" || k === "G" || k === "Escape") setLoupe(false);
      else if (k === " " && loupe) setBefore((b) => !b);
      else return;
      e.preventDefault();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [cur, shown.length, loupe, exporting, editing, social, cell, peopleMode]);
  useEffect(() => { document.getElementById(`t-${cur?.id}`)?.scrollIntoView({ block: "nearest" }); }, [cur?.id]);

  if (!shoot) return <div className="page" />;

  if (editing && cur && !peopleMode) {
    return (
      <Develop items={shown} index={curIdx} onIndex={(i) => setSel(i)} toast={ctx.toast}
        onClose={() => { setEditing(false); setEditV((v) => v + 1); load(); }}
        onSynced={async (jobId) => {
          ctx.refreshJobs();
          const j = await waitForJob(jobId);
          ctx.toast(j.status === "done" ? "Auf alle Bilder übertragen" : (j.error?.split("\n")[0] ?? "Übertragen fehlgeschlagen"),
            j.status === "done" ? "ok" : "error");
          load(); setEditV((v) => v + 1);
        }} />
    );
  }

  const groups = [...new Set(styles.map((s) => s.group ?? ""))];
  const step = stepOf(job?.message);
  return (
    <div className="lib" style={{ ["--cell" as string]: `${cell}px` }}>
      <header className="lib-top">
        <div className="lib-title">
          <h1>{shoot.name}</h1>
          <span className="muted">{items.length} Bilder{decided ? ` · ${counts.keep} behalten` : ""}</span>
        </div>
        <div className="lib-modules">
          <button className="on">Bibliothek</button>
          <button disabled={!cur || peopleMode} onClick={() => setEditing(true)} title="Entwickeln (D)">Entwickeln</button>
        </div>
        <div className="lib-actions">
          {!peopleMode && <button disabled={running || counts.keep === 0} onClick={() => setSocial(true)}>Social Media</button>}
          {peopleMode
            ? <button className="primary" disabled={running} onClick={() => setTagging(true)}>Mit Namen exportieren …</button>
            : <button className="primary" disabled={running || items.length === 0} onClick={() => setExporting(true)}>Exportieren …</button>}
        </div>
      </header>

      {running && (
        <div className="lib-progress">
          <div className="lib-steps">
            {(peopleMode ? ["Vorschauen", "Personen"] : STEPS).map((s, i) => <span key={s} className={i < step ? "done" : i === step ? "now" : ""}>{s}</span>)}
          </div>
          <div className="bar"><div style={{ width: `${job && job.total ? Math.round((100 * job.progress) / job.total) : 3}%` }} /></div>
          <span className="muted lib-msg">{job?.message ?? "Starte …"}</span>
        </div>
      )}
      {!running && items.length > 0 && !decided && (
        <div className="lib-progress">
          <span>{job?.status === "failed" ? `Verarbeitung fehlgeschlagen: ${job.error?.split("\n")[0]}` : "Noch nicht ausgewertet."}</span>
          <button className="primary small" onClick={async () => { await api.post(`/api/shoots/${id}/run/pipeline`, {}); ctx.refreshJobs(); load(); }}>Jetzt verarbeiten</button>
        </div>
      )}

      <div className="lib-filter">
        <div className="segmented">
          {([["all", "Alle"], ["keep", "Behalten"], ["reject", "Aussortiert"], ...(counts.check ? [["check", "Prüfen"]] : [])] as [Tab, string][]).map(([k, l]) => (
            <button key={k} className={curTab === k ? "on" : ""} onClick={() => { setTab(k); setSel(0); }}>{l} <b>{counts[k]}</b></button>
          ))}
        </div>
        <select value={minStars} onChange={(e) => { setMinStars(+e.target.value); setSel(0); }} title="Nach Sternen filtern">
          <option value={0}>Alle Sterne</option>
          {[1, 2, 3, 4, 5].map((n) => <option key={n} value={n}>{"★".repeat(n)} und mehr</option>)}
        </select>
        {people.length > 0 && (
          <select value={person} onChange={(e) => { setPerson(e.target.value); setSel(0); }}>
            <option value="">Alle Personen</option>
            {people.map((p) => <option key={p}>{p}</option>)}
          </select>
        )}
        <span className="spacer" />
        {!peopleMode && decided && (
          <select value={selection} disabled={running} title={SELECTION_HINT[selection]}
            onChange={(e) => runJob(`/api/shoots/${id}/run/cull`, SELECTION_PARAMS[e.target.value as Selection],
              `Auswahl: ${SELECTIONS.find(([k]) => k === e.target.value)?.[1]}`).then(() => setSel(0))}>
            {SELECTIONS.map(([k, label]) => <option key={k} value={k}>Auswahl: {label}</option>)}
          </select>
        )}
        {!peopleMode && (
          <select value="" disabled={running} onChange={(e) => e.target.value && applyPreset(e.target.value)} title="Vorgabe auf alle Bilder anwenden">
            <option value="">Vorgabe für alle …</option>
            {groups.map((gname) => (
              <optgroup key={gname} label={gname || "Weitere"}>
                {styles.filter((s) => (s.group ?? "") === gname).map((s) => <option key={s.key} value={s.key}>{s.label}</option>)}
              </optgroup>
            ))}
            <option value="__import">Lightroom-Vorgaben importieren …</option>
          </select>
        )}
        {teamsAll.length > 0 && (
          <select value={shootTeams[0] ?? ""} disabled={running} title="Team: Personen nur aus diesem Team erkennen"
            onChange={async (e) => { await api.patch(`/api/shoots/${id}`, { teams: e.target.value ? [e.target.value] : [] }); setTeam(e.target.value); }}>
            <option value="">Team: automatisch</option>
            {teamsAll.map((t) => <option key={t} value={t}>Team: {t}</option>)}
          </select>
        )}
      </div>

      {!loupe ? (
        <div className="lib-grid" ref={gridRef}>
          {shown.length === 0 && <div className="empty">{running ? "Bilder werden eingelesen …" : "Keine Bilder in dieser Ansicht."}</div>}
          {shown.map((i, idx) => <Thumb key={i.id} i={i} idx={idx} selected={idx === curIdx} onSelect={setSel} onOpen={openLoupe} />)}
        </div>
      ) : cur && (
        <div className="lib-loupe">
          <div className="stage" onClick={() => cur.edited && setBefore((b) => !b)}>
            <StageImage key={`${cur.id}-${before}-${editV}`} placeholder={api.img(`/api/images/${cur.id}/preview`)}
              src={api.img(before || peopleMode || !cur.edited ? `/api/images/${cur.id}/preview`
                : `/api/images/${cur.id}/render?size=${loupeSize()}&v=${editV}-${cur.edit_v ?? 0}`)} />
            <div className="stage-badge">{before || peopleMode || !cur.edited ? "Original" : "Bearbeitet"}</div>
          </div>
          <PeoplePanel ctx={ctx} image={cur} onChanged={load} />
          <div className="loupe-bar">
            <span className="fname">{cur.filename}</span>
            <span className="stars-edit">{[1, 2, 3, 4, 5].map((n) => (
              <button key={n} className={(cur.rating ?? 0) >= n ? "on" : ""} onClick={() => patch(cur, { rating: (cur.rating ?? 0) === n ? 0 : n })}>★</button>))}</span>
            {cur.decision === "reject" && <span className="why">Aussortiert: {cur.reasons.join(", ")}</span>}
            {cur.people.length > 0 && <span className="who">{cur.people.join(", ")}</span>}
            <span className="spacer" />
            <button onClick={() => patch(cur, { decision: "keep" })} className={cur.decision === "keep" ? "on" : ""}>Behalten</button>
            <button onClick={() => patch(cur, { decision: "reject" })} className={cur.decision === "reject" ? "on bad" : ""}>Aussortieren</button>
            {!peopleMode && <button className="primary" onClick={() => setEditing(true)} title="Entwickeln (D)">Entwickeln</button>}
            <button className="ghost" onClick={() => setLoupe(false)}>Raster</button>
          </div>
        </div>
      )}
      <footer className="lib-bottom">
        <button className={!loupe ? "on" : ""} onClick={() => setLoupe(false)} title="Raster (G)">▦</button>
        <button className={loupe ? "on" : ""} onClick={() => cur && setLoupe(true)} title="Lupe (E)">▢</button>
        <span className="muted keyhint">← → blättern · E Lupe · D Entwickeln · 1–5 Sterne · P behalten · X aussortieren · Leertaste Vorher</span>
        <span className="spacer" />
        {!loupe && <label className="lib-size">Miniaturen <input type="range" min={130} max={420} value={cell}
          onChange={(e) => { setCell(+e.target.value); store("imagomat.cell", e.target.value); }} /></label>}
      </footer>
      {exporting && <ExportDialog ctx={ctx} shoot={shoot} kept={counts.keep} total={items.length} onClose={() => setExporting(false)} />}
      {tagging && <TagExportDialog ctx={ctx} shoot={shoot} total={items.length} onClose={() => setTagging(false)} />}
      {social && <SocialDialog ctx={ctx} shoot={shoot} single={null} onClose={() => setSocial(false)} />}
    </div>
  );
}

type SocialFormat = "story" | "post" | "square" | "original";
const SOCIAL_FORMATS: [SocialFormat, string][] = [
  ["story", "Story 9:16"],
  ["post", "Post 4:5"],
  ["square", "Quadrat"],
  ["original", "Original"],
];

/** Bilder direkt für Instagram & Co. speichern (automatisch um die Personen zugeschnitten). */
function SocialDialog({ ctx, shoot, single, onClose }: {
  ctx: AppCtx; shoot: Shoot; single: ImageItem | null; onClose: () => void;
}) {
  const [format, setFormat] = useState<SocialFormat>("story");
  const [selection, setSelection] = useState<"top" | "keep" | "one">(single ? "one" : "top");
  const [state, setState] = useState<"form" | "busy" | "done">("form");
  const [progress, setProgress] = useState(0);
  const [result, setResult] = useState<{ target: string; message: string } | null>(null);

  const go = async () => {
    setState("busy");
    try {
      const r = await api.post<{ job_id: number; target: string }>(`/api/shoots/${shoot.id}/social`, {
        format, selection: selection === "one" ? "keep" : selection, ids: selection === "one" && single ? [single.id] : null,
      });
      ctx.refreshJobs();
      const j = await waitForJob(r.job_id, (x) => setProgress(x.total ? x.progress / x.total : 0));
      if (j.status !== "done") throw new Error(j.error?.split("\n")[0] ?? "Speichern fehlgeschlagen");
      setResult({ target: r.target, message: j.message ?? "" });
      setState("done");
    } catch (e) {
      ctx.toast((e as Error).message, "error");
      setState("form");
    }
  };

  return (
    <Modal title="Für Social Media speichern" onClose={onClose}>
      {state === "form" && (
        <>
          <label>Format</label>
          <Segmented value={format} options={SOCIAL_FORMATS} onChange={setFormat} />
          <div className="hint">
            {format === "original" ? "ganzes Bild, 2048 px lange Seite"
              : "automatisch um die Spieler zugeschnitten, in Instagram-Grösse (1080 px breit)"}
          </div>
          <label>Welche Bilder</label>
          <Segmented value={selection} onChange={setSelection}
            options={[...(single ? [["one", "Nur dieses"] as ["one", string]] : []), ["top", "Top-Bilder"], ["keep", "Alle behaltenen"]]} />
          <p className="hint">Gespeichert in Downloads › Imagomat › {shoot.name}. Mit deinem Stil bearbeitet (Vorschau-Qualität).</p>
          <div className="modal-foot">
            <button className="ghost" onClick={onClose}>Abbrechen</button>
            <button className="primary" onClick={go}>Speichern</button>
          </div>
        </>
      )}
      {state === "busy" && <Progress value={progress} label="Bilder werden gerendert …" />}
      {state === "done" && result && (
        <>
          <p className="success">{result.message.split(" -> ")[0]}</p>
          <div className="modal-foot">
            <button onClick={() => reveal(result.target)}>Im Finder zeigen</button>
            <button className="primary" onClick={onClose}>Schliessen</button>
          </div>
        </>
      )}
    </Modal>
  );
}


/** Nur Personen: Bilder mit eingebetteten Namen in einen neuen Ordner schreiben. */
function TagExportDialog({ ctx, shoot, total, onClose }: { ctx: AppCtx; shoot: Shoot; total: number; onClose: () => void }) {
  const parent = shoot.folder.replace(/\/[^/]+\/?$/, "");
  const stored = (k: string) => { try { return localStorage.getItem(k); } catch { return null; } };
  const store = (k: string, v: string) => { try { localStorage.setItem(k, v); } catch { /* egal */ } };
  const [root, setRoot] = useState(stored("imagomat.tagRoot") ?? parent);
  const [folderName, setFolderName] = useState(`${shoot.name} – mit Namen`);
  const [onlyPeople, setOnlyPeople] = useState(false);
  const [state, setState] = useState<"form" | "busy" | "done">("form");
  const [progress, setProgress] = useState(0);
  const [msg, setMsg] = useState("");
  const target = folderName.trim() ? `${root.replace(/\/$/, "")}/${folderName.trim()}` : root;

  const go = async () => {
    setState("busy");
    try {
      const r = await api.post<{ job_id: number }>(`/api/shoots/${shoot.id}/tag-export`, { target, only_with_people: onlyPeople });
      ctx.refreshJobs();
      const j = await waitForJob(r.job_id, (x) => setProgress(x.total ? x.progress / x.total : 0));
      if (j.status !== "done") throw new Error(j.error?.split("\n")[0] ?? "Export fehlgeschlagen");
      setMsg((j.message ?? "").split(" -> ")[0]);
      setState("done");
    } catch (e) {
      ctx.toast((e as Error).message, "error");
      setState("form");
    }
  };

  return (
    <Modal title="Mit Namen exportieren" onClose={onClose}>
      {state === "form" && (
        <>
          <p>{total} Bilder werden mit den erkannten Namen (Stichwörter + Gesichter) gespeichert. JPGs bekommen die Namen direkt in die Datei, Lightroom liest sie beim Import. Deine Originale bleiben unverändert.</p>
          <label>Ordner (wird gemerkt)</label>
          <div className="row">
            <input value={root} onChange={(e) => { setRoot(e.target.value); store("imagomat.tagRoot", e.target.value); }} />
            <button onClick={async () => { const d = await pickFolder("Zielordner wählen"); if (d) { setRoot(d); store("imagomat.tagRoot", d); } }}>Wählen…</button>
          </div>
          <label>Neuer Ordner</label>
          <input value={folderName} onChange={(e) => setFolderName(e.target.value)} />
          <label className="check"><input type="checkbox" checked={onlyPeople} onChange={(e) => setOnlyPeople(e.target.checked)} /> nur Bilder mit erkannten Personen</label>
          <div className="modal-foot">
            <button className="ghost" onClick={onClose}>Abbrechen</button>
            <button className="primary" onClick={go}>Exportieren</button>
          </div>
        </>
      )}
      {state === "busy" && <Progress value={progress} label="Namen werden geschrieben …" />}
      {state === "done" && (
        <>
          <p className="success">{msg}</p>
          <div className="modal-foot">
            <button onClick={() => reveal(target)}>Im Finder zeigen</button>
            {IS_APP && <button onClick={() => api.post("/api/lightroom/open", { path: target }).catch((e) => ctx.toast(String((e as Error).message), "error"))}>In Lightroom importieren</button>}
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
    <div id={`t-${i.id}`} className={`cellx ${selected ? "sel" : ""} ${i.decision === "reject" ? "rej" : ""}`}
      onClick={() => onSelect(idx)} onDoubleClick={() => onOpen(idx)}>
      <span className="cell-n">{idx + 1}</span>
      {i.hand_edited && <span className="cell-badge" title="Von Hand bearbeitet">✎</span>}
      <div className="cell-img"><img loading="lazy" decoding="async" draggable={false}
        src={api.img(`/api/images/${i.id}/thumb?v=${i.edit_v ?? 0}${i.rendered ? "r" : ""}`)} /></div>
      <div className="cell-foot">
        {i.decision === "reject" ? <span className="cell-flag rej" title={i.reasons.join(", ")}>✕</span>
          : i.decision === "keep" ? <span className="cell-flag keep">⚑</span> : <span />}
        <span className="cell-stars">{(i.rating ?? 0) > 0 ? "★".repeat(i.rating ?? 0) : ""}</span>
        <span className="cell-who">{i.people.join(", ")}</span>
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

/** Bearbeitete Vorschauen der Nachbarbilder schon im Hintergrund berechnen lassen (blättern ohne Warten). */
function usePrefetch(urls: string[]) {
  const key = urls.join("|");
  useEffect(() => {
    const imgs = urls.map((u) => { const i = new Image(); i.decoding = "async"; i.src = u; return i; });
    return () => { imgs.forEach((i) => { i.src = ""; }); };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key]);
}
