import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { AppCtx } from "../App";
import { api, ImageItem, pickFolder, reveal, Shoot, waitForJob } from "../api";
import { SELECTION_HINT, SELECTION_PARAMS, SELECTIONS, Selection, selectionFromSettings } from "../selection";
import { Modal, More, Progress } from "../ui";

type Tab = "keep" | "reject" | "check";

const STEPS = ["Vorschauen", "Aussortieren", "Personen", "Bearbeiten"];

function stepOf(msg: string | null | undefined): number {
  if (!msg) return 0;
  const m = msg.match(/^(\d)\/4/);
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
    check: items.filter((i) => i.decision === "keep" && i.confidence !== null && i.confidence < 0.35).length,
  }), [items]);
  const people = useMemo(() => [...new Set(items.flatMap((i) => i.people))].sort(), [items]);
  const shown = useMemo(() => items.filter((i) => {
    if (person && !i.people.includes(person)) return false;
    if (tab === "keep") return i.decision === "keep";
    if (tab === "reject") return i.decision === "reject";
    return i.decision === "keep" && i.confidence !== null && i.confidence < 0.35;
  }), [items, tab, person]);
  const cur = shown[Math.min(sel, Math.max(0, shown.length - 1))];

  const selection = selectionFromSettings(shoot?.settings);
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
            {STEPS.map((s, i) => <div key={s} className={`step ${i < step ? "done" : i === step ? "now" : ""}`}>{s}</div>)}
          </div>
          <Progress value={job && job.total ? job.progress / job.total : 0.02} label={job?.message ?? "Starte …"} />
          <p className="hint">Du kannst währenddessen weiterarbeiten, z. B. unter „Personen“.</p>
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
        </div>
        <div className="tabs">
          <button className={tab === "keep" ? "on" : ""} onClick={() => { setTab("keep"); setSel(0); }}>Behalten <b>{counts.keep}</b></button>
          <button className={tab === "reject" ? "on" : ""} onClick={() => { setTab("reject"); setSel(0); }}>Aussortiert <b>{counts.reject}</b></button>
          {counts.check > 0 && (
            <button className={tab === "check" ? "on" : ""} onClick={() => { setTab("check"); setSel(0); }}>Prüfen <b>{counts.check}</b></button>
          )}
        </div>
        <select className="person-filter" value={selection} disabled={running} title={SELECTION_HINT[selection]}
          onChange={(e) => recull(e.target.value as Selection)}>
          {SELECTIONS.map(([k, label]) => <option key={k} value={k}>Auswahl: {label}</option>)}
        </select>
        {people.length > 0 && (
          <select className="person-filter" value={person} onChange={(e) => { setPerson(e.target.value); setSel(0); }}>
            <option value="">Alle Personen</option>
            {people.map((p) => <option key={p}>{p}</option>)}
          </select>
        )}
        <button className="primary" disabled={running} onClick={() => setExporting(true)}>Nach Lightroom ▸</button>
      </div>

      {!loupe ? (
        <div className="grid" ref={gridRef}>
          {shown.length === 0 && <div className="empty">Keine Bilder in dieser Ansicht.</div>}
          {shown.map((i, idx) => (
            <div key={i.id} id={`t-${i.id}`} className={`thumb ${idx === sel ? "sel" : ""}`}
              onClick={() => setSel(idx)} onDoubleClick={() => { setSel(idx); setLoupe(true); }}>
              <img loading="lazy" src={api.img(`/api/images/${i.id}/preview`)} />
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
              src={api.img(before || cur.decision !== "keep" ? `/api/images/${cur.id}/preview` : `/api/images/${cur.id}/render?size=2000`)} />
            <div className="stage-badge">{before || cur.decision !== "keep" ? "Original" : "Bearbeitet (Vorschau)"}</div>
          </div>
          <div className="loupe-bar">
            <span className="fname">{cur.filename}</span>
            {cur.decision === "reject" && <span className="why">Aussortiert: {cur.reasons.join(", ")}</span>}
            {cur.moment && <span className="tag moment">{cur.moment}</span>}
            {cur.people.length > 0 && <span className="who">{cur.people.join(", ")}</span>}
            <span className="spacer" />
            <button onClick={() => patch(cur, { decision: "keep" })} className={cur.decision === "keep" ? "on" : ""}>Behalten</button>
            <button onClick={() => patch(cur, { decision: "reject" })} className={cur.decision === "reject" ? "on bad" : ""}>Aussortieren</button>
            <button className="ghost" onClick={() => setLoupe(false)}>Zurück</button>
          </div>
        </div>
      )}
      <div className="keyhint">
        ← → blättern · Enter gross · Leertaste Vorher/Nachher · 1–5 Sterne · X aussortieren · P behalten
      </div>
      {exporting && <ExportDialog ctx={ctx} shoot={shoot} kept={counts.keep} onClose={() => setExporting(false)} />}
    </div>
  );
}

function ExportDialog({ ctx, shoot, kept, onClose }: { ctx: AppCtx; shoot: Shoot; kept: number; onClose: () => void }) {
  const parent = shoot.folder.replace(/\/[^/]+\/?$/, "");
  const [target, setTarget] = useState(`${parent}/${shoot.name} – Imagomat`);
  const [jpeg, setJpeg] = useState(false);
  const [mode, setMode] = useState<"hardlink" | "inplace">("hardlink");
  const [withRejected, setWithRejected] = useState(true);
  const [state, setState] = useState<"form" | "busy" | "done">("form");
  const [progress, setProgress] = useState(0);

  const go = async () => {
    setState("busy");
    try {
      const formats = ["xmp", ...(jpeg ? ["jpeg"] : [])];
      const r = await api.post<{ job_id: number }>(`/api/shoots/${shoot.id}/export`, {
        target: mode === "inplace" ? shoot.folder : target, formats, copy_mode: mode, include_rejected: withRejected,
      });
      ctx.refreshJobs();
      const j = await waitForJob(r.job_id, (x) => setProgress(x.total ? x.progress / x.total : 0));
      if (j.status !== "done") throw new Error(j.error?.split("\n")[0] ?? "Export fehlgeschlagen");
      setState("done");
    } catch (e) {
      ctx.toast((e as Error).message, "error");
      setState("form");
    }
  };

  const dest = mode === "inplace" ? shoot.folder : target;
  return (
    <Modal title="Nach Lightroom" onClose={onClose}>
      {state === "form" && (
        <>
          <p>{kept} bearbeitete Bilder (inkl. Masken, Denoise, Zuschnitt) werden für Lightroom vorbereitet. Deine Originale bleiben unverändert.</p>
          <label>Zielordner</label>
          <div className="row">
            <input value={target} onChange={(e) => setTarget(e.target.value)} disabled={mode === "inplace"} />
            <button onClick={async () => { const d = await pickFolder("Zielordner wählen"); if (d) setTarget(d); }} disabled={mode === "inplace"}>Wählen…</button>
          </div>
          <label className="check"><input type="checkbox" checked={jpeg} onChange={(e) => setJpeg(e.target.checked)} /> zusätzlich schnelle Vorschau-JPEGs (z. B. zum Verschicken)</label>
          <More>
            <label className="check"><input type="checkbox" checked={mode === "inplace"} onChange={(e) => setMode(e.target.checked ? "inplace" : "hardlink")} /> nur Einstellungen (XMP) neben die Originale schreiben</label>
            <label className="check"><input type="checkbox" checked={withRejected} onChange={(e) => setWithRejected(e.target.checked)} /> aussortierte Bilder mitnehmen (1 Stern)</label>
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
            <li><b>Importieren</b> → den Ordner <code>{dest.split("/").pop()}</code> wählen → <b>Hinzufügen</b>.</li>
            <li>Alle Bilder auswählen (⌘A) → <b>Foto › Entwicklungseinstellungen › KI-Einstellungen aktualisieren</b>. Damit rechnet Lightroom Masken und Entrauschen.</li>
          </ol>
          <div className="modal-foot">
            <button onClick={() => reveal(dest)}>Im Finder zeigen</button>
            <button className="primary" onClick={onClose}>Schliessen</button>
          </div>
        </>
      )}
    </Modal>
  );
}
