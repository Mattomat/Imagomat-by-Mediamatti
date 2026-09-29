import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { AppCtx } from "../App";
import { api, ImageItem, IS_APP, pickFolder, reveal, Shoot, waitForJob } from "../api";
import PeoplePanel from "../components/PeoplePanel";
import { SELECTION_HINT, SELECTION_PARAMS, SELECTIONS, Selection, selectionFromSettings } from "../selection";
import { Modal, More, Progress, Segmented } from "../ui";

type Tab = "keep" | "reject" | "check";

const STEPS = ["Vorschauen", "Aussortieren", "Personen", "Bearbeiten"];

function needsCheck(i: ImageItem): boolean {
  return (i.people_check?.length ?? 0) > 0
    || (i.decision === "keep" && i.confidence !== null && i.confidence < 0.35);
}

function stepOf(msg: string | null | undefined): number {
  if (!msg) return 0;
  const m = msg.match(/^(\d)\/\d/);
  if (m) return +m[1] - 1;
  if (/Culling|behalten/.test(msg)) return 1;
  if (/Personen|Gesichter wiedererkannt|Rückennummer/.test(msg)) return 2;
  if (/Entwick/.test(msg)) return 3;
  return 0;
}

export default function ShootView({ ctx, id }: { ctx: AppCtx; id: number }) {
  const [shoot, setShoot] = useState<Shoot | null>(null);
  const [items, setItems] = useState<ImageItem[]>([]);
  const [tab, setTab] = useState<Tab>("keep");
  const [person, setPerson] = useState("");
  const [sel, setSel] = useState(0);
  const [loupe, setLoupe] = useState(false);
  const [before, setBefore] = useState(false);
  const [exporting, setExporting] = useState(false);
  const [social, setSocial] = useState<"none" | "all" | "one">("none");
  const gridRef = useRef<HTMLDivElement>(null);

  const load = useCallback(() => {
    api.get<Shoot>(`/api/shoots/${id}`).then(setShoot).catch(() => undefined);
    api.get<ImageItem[]>(`/api/shoots/${id}/images`).then(setItems);
  }, [id]);
  useEffect(load, [load, ctx.tick]);

  const liveJob = ctx.jobs.find((j) => j.id === shoot?.job?.id);
  const job = liveJob ?? shoot?.job ?? null;
  const running = !!job && ["running", "queued"].includes(job.status);

  const counts = useMemo(() => ({
    keep: items.filter((i) => i.decision === "keep").length,
    reject: items.filter((i) => i.decision === "reject").length,
    check: items.filter((i) => needsCheck(i)).length,
  }), [items]);
  const people = useMemo(() => [...new Set(items.flatMap((i) => i.people))].sort(), [items]);
  const shown = useMemo(() => items.filter((i) => {
    if (person && !i.people.includes(person)) return false;
    if (tab === "keep") return i.decision === "keep";
    if (tab === "reject") return i.decision === "reject";
    return needsCheck(i);
  }), [items, tab, person]);
  const cur = shown[Math.min(sel, Math.max(0, shown.length - 1))];

  const selection = selectionFromSettings(shoot?.settings);
  const peopleMode = (() => { try { return JSON.parse(shoot?.settings || "{}").mode === "people"; } catch { return false; } })();
  const [tagging, setTagging] = useState(false);
  type StyleInfo = { used: string | null; label: string | null; is_preset: boolean; profiles: string[] };
  const [style, setStyle] = useState<StyleInfo | null>(null);
  useEffect(() => {
    api.get<StyleInfo>(`/api/shoots/${id}/style`).then(setStyle).catch(() => undefined);
  }, [id, ctx.tick]);
  const redevelop = async (profile: string) => {
    const r = await api.post<{ job_id: number }>(`/api/shoots/${id}/run/develop`, { profile });
    ctx.refreshJobs();
    await waitForJob(r.job_id);
    ctx.toast(profile === "preset:auto" ? "Neu bearbeitet (Standard)" : "Neu bearbeitet");
    load();
    api.get<StyleInfo>(`/api/shoots/${id}/style`).then(setStyle);
  };
  const [comparing, setComparing] = useState(false);
  const [teamsAll, setTeamsAll] = useState<string[]>([]);
  useEffect(() => { api.get<{ teams: string[] }>("/api/overview").then((o) => setTeamsAll(o.teams)).catch(() => undefined); }, []);
  const shootTeams: string[] = (() => { try { return JSON.parse(shoot?.settings || "{}").teams ?? []; } catch { return []; } })();
  const setTeam = async (t: string) => {
    await api.patch(`/api/shoots/${id}`, { teams: t ? [t] : [] });
    const r = await api.post<{ job_id: number }>(`/api/shoots/${id}/run/people`, {});
    ctx.refreshJobs();
    await waitForJob(r.job_id);
    load();
    ctx.toast(t ? `Personen neu erkannt (Team ${t})` : "Personen neu erkannt");
  };
  const applyStyle = async (profile: string) => {
    setComparing(false);
    await api.patch(`/api/shoots/${id}`, { profile: profile.startsWith("preset:") ? "" : profile });
    await redevelop(profile);
  };
  const recull = async (s: Selection) => {
    try {
      const r = await api.post<{ job_id: number }>(`/api/shoots/${id}/run/cull`, SELECTION_PARAMS[s]);
      ctx.refreshJobs();
      await waitForJob(r.job_id);
      load();
      setSel(0);
      ctx.toast(`Auswahl: ${SELECTIONS.find(([k]) => k === s)?.[1]}`);
    } catch (e) {
      ctx.toast(String((e as Error).message), "error");
    }
  };

  const patch = async (i: ImageItem, body: Partial<Pick<ImageItem, "decision" | "rating">>) => {
    await api.patch(`/api/images/${i.id}/culling`, body);
    setItems((xs) => xs.map((x) => (x.id === i.id ? { ...x, ...body, manual: true } : x)));
  };

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement;
      if (["INPUT", "SELECT", "TEXTAREA"].includes(t.tagName) || exporting || !cur) return;
      const cols = gridRef.current ? Math.max(1, Math.floor(gridRef.current.clientWidth / 236)) : 5;
      const k = e.key;
      if (k === "ArrowRight") setSel((s) => Math.min(shown.length - 1, s + 1));
      else if (k === "ArrowLeft") setSel((s) => Math.max(0, s - 1));
      else if (k === "ArrowDown" && !loupe) setSel((s) => Math.min(shown.length - 1, s + cols));
      else if (k === "ArrowUp" && !loupe) setSel((s) => Math.max(0, s - cols));
      else if (/^[1-5]$/.test(k)) patch(cur, { rating: +k, decision: "keep" });
      else if (k === "x" || k === "X" || k === "Backspace") patch(cur, { decision: "reject" });
      else if (k === "p" || k === "P") patch(cur, { decision: "keep" });
      else if (k === "Enter") setLoupe((l) => !l);
      else if (k === "Escape") setLoupe(false);
      else if (k === " " && loupe) setBefore((b) => !b);
      else return;
      e.preventDefault();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [cur, shown.length, loupe, exporting]);

  useEffect(() => {
    document.getElementById(`t-${cur?.id}`)?.scrollIntoView({ block: "nearest" });
  }, [cur?.id]);

  if (!shoot) return <div className="page" />;

  if (running && items.every((i) => i.decision === null)) {
    const step = stepOf(job?.message);
    return (
      <div className="page center">
        <div className="processing">
          <h1>{shoot.name}</h1>
          <p className="muted">{shoot.n} Bilder</p>
          <div className="steps-row">
            {(peopleMode ? ["Vorschauen", "Personen"] : STEPS).map((s, i) => <div key={s} className={`step ${i < step ? "done" : i === step ? "now" : ""}`}>{s}</div>)}
          </div>
          <Progress value={job && job.total ? job.progress / job.total : 0.02} label={job?.message ?? "Starte …"} />
          <p className="hint">Du kannst währenddessen weiterarbeiten, z. B. unter „Personen“.</p>
        </div>
      </div>
    );
  }

  if (!running && items.length > 0 && items.every((i) => i.decision === null)) {
    const failed = job?.status === "failed";
    return (
      <div className="page center">
        <div className="processing">
          <h1>{shoot.name}</h1>
          <p className="muted">{shoot.n} Bilder · noch nicht verarbeitet</p>
          {failed && <p className="error">Letzter Versuch fehlgeschlagen: {job?.error?.split("\n")[0]}</p>}
          {job?.status === "cancelled" && <p className="hint">Die Verarbeitung wurde abgebrochen.</p>}
          <button className="primary big" onClick={async () => {
            await api.post(`/api/shoots/${id}/run/pipeline`, {});
            ctx.refreshJobs();
            load();
          }}>Jetzt verarbeiten</button>
          <p className="hint">Bereits analysierte Bilder werden übersprungen.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="shoot">
      <div className="shoot-head">
        <div>
          <h1>{shoot.name}</h1>
          <div className="muted">{counts.keep} von {items.length} behalten{running ? " · wird noch bearbeitet …" : ""}</div>
          {!peopleMode && style?.used && (
            <div className={`style-chip ${style.is_preset ? "warn" : ""}`}>
              {style.is_preset
                ? <>Bearbeitet mit Standard-Preset, nicht mit deinem Stil. {style.profiles.length
                  ? <select value="" onChange={(e) => e.target.value && redevelop(e.target.value)}>
                      <option value="">Mit meinem Stil neu bearbeiten …</option>
                      {style.profiles.map((p) => <option key={p}>{p}</option>)}
                    </select>
                  : <button className="link" onClick={() => ctx.go({ name: "style" })}>Eigenen Stil lernen →</button>}</>
                : <>Stil: <b>{style.label ?? style.used}</b> <select value="" onChange={(e) => e.target.value && redevelop(e.target.value)}>
                    <option value="">ändern …</option>
                    {style.profiles.map((p) => <option key={p}>{p}</option>)}
                  </select></>}
              {" "}<button className="link" disabled={running} onClick={() => setComparing(true)}>Stile vergleichen</button>
            </div>
          )}
          {teamsAll.length > 0 && (
            <div className="muted">
              Team: <select className="team-select" value={shootTeams[0] ?? ""} disabled={running} onChange={(e) => setTeam(e.target.value)}>
                <option value="">automatisch erkennen</option>
                {teamsAll.map((t) => <option key={t}>{t}</option>)}
              </select>
            </div>
          )}
        </div>
        <div className="tabs">
          <button className={tab === "keep" ? "on" : ""} onClick={() => { setTab("keep"); setSel(0); }}>Behalten <b>{counts.keep}</b></button>
          <button className={tab === "reject" ? "on" : ""} onClick={() => { setTab("reject"); setSel(0); }}>Aussortiert <b>{counts.reject}</b></button>
          {counts.check > 0 && (
            <button className={tab === "check" ? "on" : ""} onClick={() => { setTab("check"); setSel(0); }}>Prüfen <b>{counts.check}</b></button>
          )}
        </div>
        {!peopleMode && (
          <select className="person-filter" value={selection} disabled={running} title={SELECTION_HINT[selection]}
            onChange={(e) => recull(e.target.value as Selection)}>
            {SELECTIONS.map(([k, label]) => <option key={k} value={k}>Auswahl: {label}</option>)}
          </select>
        )}
        {people.length > 0 && (
          <select className="person-filter" value={person} onChange={(e) => { setPerson(e.target.value); setSel(0); }}>
            <option value="">Alle Personen</option>
            {people.map((p) => <option key={p}>{p}</option>)}
          </select>
        )}
        {!peopleMode && <button disabled={running || counts.keep === 0} onClick={() => setSocial("all")}>Social Media</button>}
        {peopleMode
          ? <button className="primary" disabled={running} onClick={() => setTagging(true)}>Mit Namen exportieren ▸</button>
          : <button className="primary" disabled={running} onClick={() => setExporting(true)}>Nach Lightroom ▸</button>}
      </div>

      {!loupe ? (
        <div className="grid" ref={gridRef}>
          {shown.length === 0 && <div className="empty">Keine Bilder in dieser Ansicht.</div>}
          {shown.map((i, idx) => (
            <div key={i.id} id={`t-${i.id}`} className={`thumb ${idx === sel ? "sel" : ""}`}
              onClick={() => setSel(idx)} onDoubleClick={() => { setSel(idx); setLoupe(true); }}>
              <img loading="lazy" decoding="async" src={api.img(`/api/images/${i.id}/thumb`)} />
              <div className="thumb-foot">
                {i.decision === "keep"
                  ? <span className="stars">{"★".repeat(i.rating ?? 0)}</span>
                  : <span className="why">{i.reasons[0] ?? "aussortiert"}</span>}
                {i.best && i.decision === "keep" && <span className="tag">Top</span>}
                {i.moment && i.decision === "keep" && <span className="tag moment">{i.moment}</span>}
                {i.people.length > 0 && <span className="who">{i.people.join(", ")}</span>}
              </div>
            </div>
          ))}
        </div>
      ) : cur && (
        <div className="loupe">
          <div className="stage" onClick={() => setBefore((b) => !b)}>
            <img key={`${cur.id}-${before}`}
              src={api.img(before || peopleMode || cur.decision !== "keep" ? `/api/images/${cur.id}/preview` : `/api/images/${cur.id}/render?size=2000`)} />
            <div className="stage-badge">{before || peopleMode || cur.decision !== "keep" ? "Original" : "Bearbeitet (Vorschau)"}</div>
          </div>
          <PeoplePanel ctx={ctx} image={cur} onChanged={load} />
          <div className="loupe-bar">
            <span className="fname">{cur.filename}</span>
            {cur.decision === "reject" && <span className="why">Aussortiert: {cur.reasons.join(", ")}</span>}
            {cur.moment && <span className="tag moment">{cur.moment}</span>}
            {cur.people.length > 0 && <span className="who">{cur.people.join(", ")}</span>}
            <span className="spacer" />
            <button onClick={() => patch(cur, { decision: "keep" })} className={cur.decision === "keep" ? "on" : ""}>Behalten</button>
            <button onClick={() => patch(cur, { decision: "reject" })} className={cur.decision === "reject" ? "on bad" : ""}>Aussortieren</button>
            <button onClick={() => setSocial("one")}>Für Story speichern</button>
            <button className="ghost" onClick={() => setLoupe(false)}>Zurück</button>
          </div>
        </div>
      )}
      <div className="keyhint">
        ← → blättern · Enter gross · Leertaste Vorher/Nachher · 1–5 Sterne · X aussortieren · P behalten
      </div>
      {exporting && <ExportDialog ctx={ctx} shoot={shoot} kept={counts.keep} onClose={() => setExporting(false)} />}
      {tagging && <TagExportDialog ctx={ctx} shoot={shoot} total={items.length} onClose={() => setTagging(false)} />}
      {comparing && <StyleCompare shootId={id} current={style?.used ?? null} onApply={applyStyle} onClose={() => setComparing(false)} />}
      {social !== "none" && (
        <SocialDialog ctx={ctx} shoot={shoot} single={social === "one" ? cur ?? null : null} onClose={() => setSocial("none")} />
      )}
    </div>
  );
}

function ExportDialog({ ctx, shoot, kept, onClose }: { ctx: AppCtx; shoot: Shoot; kept: number; onClose: () => void }) {
  const parent = shoot.folder.replace(/\/[^/]+\/?$/, "");
  const stored = (k: string) => { try { return localStorage.getItem(k); } catch { return null; } };
  const store = (k: string, v: string) => { try { localStorage.setItem(k, v); } catch { /* egal */ } };
  const [root, setRoot] = useState(stored("imagomat.exportRoot") ?? parent);
  const [folderName, setFolderName] = useState(shoot.name);
  const target = folderName.trim() ? `${root.replace(/\/$/, "")}/${folderName.trim()}` : root;
  const setTarget = (v: string) => { setRoot(v); store("imagomat.exportRoot", v); };
  const [jpeg, setJpeg] = useState(false);
  const [mode, setMode] = useState<"hardlink" | "inplace">("hardlink");
  const [withRejected, setWithRejected] = useState(false);
  const [openLr, setOpenLr] = useState(stored("imagomat.openLr") !== "0");
  const [final, setFinal] = useState<string | null>(null);
  const [state, setState] = useState<"form" | "busy" | "done">("form");
  const [progress, setProgress] = useState(0);

  const go = async () => {
    setState("busy");
    try {
      const formats = ["xmp", ...(jpeg ? ["jpeg"] : [])];
      const r = await api.post<{ job_id: number; target: string }>(`/api/shoots/${shoot.id}/export`, {
        target: mode === "inplace" ? shoot.folder : target, formats, copy_mode: mode, include_rejected: withRejected,
      });
      setFinal(r.target);
      ctx.refreshJobs();
      const j = await waitForJob(r.job_id, (x) => setProgress(x.total ? x.progress / x.total : 0));
      if (j.status !== "done") throw new Error(j.error?.split("\n")[0] ?? "Export fehlgeschlagen");
      setState("done");
      if (openLr && IS_APP) openInLightroom(r.target);
    } catch (e) {
      ctx.toast((e as Error).message, "error");
      setState("form");
    }
  };
  const openInLightroom = async (folder: string) => {
    try {
      await api.post("/api/lightroom/open", { path: folder });
    } catch (e) {
      ctx.toast(`Lightroom konnte nicht geöffnet werden: ${(e as Error).message}`, "error");
    }
  };

  const dest = final ?? (mode === "inplace" ? shoot.folder : target);
  return (
    <Modal title="Nach Lightroom" onClose={onClose}>
      {state === "form" && (
        <>
          <p>{kept} bearbeitete Bilder (inkl. Masken, Denoise, Zuschnitt) werden für Lightroom vorbereitet. Deine Originale bleiben unverändert.</p>
          <label>Dein Lightroom-Ordner (wird gemerkt)</label>
          <div className="row">
            <input value={root} onChange={(e) => setTarget(e.target.value)} disabled={mode === "inplace"} />
            <button onClick={async () => { const d = await pickFolder("Ordner für RAW + XMP wählen"); if (d) setTarget(d); }} disabled={mode === "inplace"}>Wählen…</button>
          </div>
          <label>Neuer Ordner</label>
          <input value={folderName} disabled={mode === "inplace"} onChange={(e) => setFolderName(e.target.value)}
            placeholder="leer = direkt in den Ordner oben" />
          <div className="hint">Nur die {kept} ausgewählten Bilder (RAW + XMP) landen in: {mode === "inplace" ? shoot.folder : target}</div>
          {IS_APP && (
            <label className="check"><input type="checkbox" checked={openLr}
              onChange={(e) => { setOpenLr(e.target.checked); store("imagomat.openLr", e.target.checked ? "1" : "0"); }} /> danach direkt in Lightroom importieren (Import-Dialog öffnet sich mit diesem Ordner)</label>
          )}
          <label className="check"><input type="checkbox" checked={jpeg} onChange={(e) => setJpeg(e.target.checked)} /> zusätzlich schnelle Vorschau-JPEGs (z. B. zum Verschicken)</label>
          <More>
            <label className="check"><input type="checkbox" checked={mode === "inplace"} onChange={(e) => setMode(e.target.checked ? "inplace" : "hardlink")} /> nur Einstellungen (XMP) neben die Originale schreiben</label>
            <label className="check"><input type="checkbox" checked={withRejected} onChange={(e) => setWithRejected(e.target.checked)} /> auch aussortierte Bilder mitnehmen (1 Stern)</label>
          </More>
          <div className="modal-foot">
            <button className="ghost" onClick={onClose}>Abbrechen</button>
            <button className="primary" onClick={go}>Exportieren</button>
          </div>
        </>
      )}
      {state === "busy" && <Progress value={progress} label="Exportiere …" />}
      {state === "done" && (
        <>
          <p className="success">Fertig! So geht's in Lightroom weiter:</p>
          <ol className="steps">
            {openLr && IS_APP
              ? <li>Im Lightroom-Import ist <code>{dest.split("/").pop()}</code> schon gewählt → oben <b>Hinzufügen</b> → <b>Importieren</b>.</li>
              : <li><b>Importieren</b> → den Ordner <code>{dest.split("/").pop()}</code> wählen → <b>Hinzufügen</b>.</li>}
            <li>Alle Bilder auswählen (⌘A) → <b>Foto › Entwicklungseinstellungen › KI-Einstellungen aktualisieren</b>. Damit rechnet Lightroom Masken und Entrauschen.</li>
          </ol>
          <div className="modal-foot">
            <button onClick={() => reveal(dest)}>Im Finder zeigen</button>
            {IS_APP && <button onClick={() => openInLightroom(dest)}>In Lightroom öffnen</button>}
            <button className="primary" onClick={onClose}>Schliessen</button>
          </div>
        </>
      )}
    </Modal>
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

function StyleCompare({ shootId, current, onApply, onClose }: {
  shootId: number; current: string | null; onApply: (style: string) => void; onClose: () => void;
}) {
  type Opt = { key: string; label: string; n: number | null; group: string; description?: string };
  const [data, setData] = useState<{ images: number[]; styles: Opt[] } | null>(null);
  useEffect(() => { api.get<typeof data>(`/api/shoots/${shootId}/compare`).then(setData).catch(() => undefined); }, [shootId]);
  return (
    <Modal title="Stile vergleichen" onClose={onClose} wide>
      <p className="hint">Dieselben Bilder mit jedem deiner Stile bearbeitet (Vorschau, noch nichts gespeichert). Wähle den, der dir am besten gefällt: er wird dann für alle Bilder des Shoots übernommen.</p>
      {!data ? <p className="muted">Lade …</p> : (
        <div className="compare">
          {data.styles.map((s, i) => (<div key={s.key}>
            {(i === 0 || data.styles[i - 1].group !== s.group) && <h3 className="compare-group">{s.group}</h3>}
            <div className={`compare-row ${current === s.key ? "on" : ""}`} style={{ ["--n" as string]: data.images.length }}>
              <div className="cr-name">
                {s.label}
                {s.n ? <span className="muted">aus {s.n} Bildern gelernt</span> : null}
                {s.description ? <span className="muted">{s.description}</span> : null}
                {current === s.key ? <span className="muted">aktuell</span> : null}
                <button className="primary small" onClick={() => onApply(s.key)}>Für alle übernehmen</button>
              </div>
              {data.images.map((iid) => (
                <img key={iid} loading="lazy" src={api.img(`/api/images/${iid}/styled?style=${encodeURIComponent(s.key)}&size=900`)} />
              ))}
            </div>
          </div>))}
        </div>
      )}
    </Modal>
  );
}
