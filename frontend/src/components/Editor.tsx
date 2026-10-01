import { useCallback, useEffect, useRef, useState } from "react";
import { api, BASE } from "../api";

type Pt = [number, number];
export interface EdMask {
  name: string;
  kind: "subject" | "background" | "sky" | "person" | "gradient" | "radial" | "other";
  local: Record<string, number>;
  amount?: number;
  zero?: Pt;
  full?: Pt;
  box?: [number, number, number, number];
  feather?: number;
  invert?: boolean;
  raw?: unknown;
}
export interface EdModel {
  global: Record<string, number>;
  wb_custom: boolean;
  masks: EdMask[];
}
interface EdInfo {
  model: EdModel;
  manual: boolean;
  has_subject: boolean;
  as_shot: [number | null, number | null];
}

type S = [key: string, label: string, min: number, max: number, step?: number];
const LIGHT: S[] = [["Exposure2012", "Belichtung", -5, 5, 0.05], ["Contrast2012", "Kontrast", -100, 100],
  ["Highlights2012", "Lichter", -100, 100], ["Shadows2012", "Tiefen", -100, 100],
  ["Whites2012", "Weiss", -100, 100], ["Blacks2012", "Schwarz", -100, 100]];
const PRESENCE: S[] = [["Texture", "Struktur", -100, 100], ["Clarity2012", "Klarheit", -100, 100],
  ["Dehaze", "Dunst entfernen", -100, 100], ["Vibrance", "Dynamik", -100, 100], ["Saturation", "Sättigung", -100, 100]];
const DETAIL: S[] = [["Sharpness", "Schärfe", 0, 150], ["LuminanceSmoothing", "Rauschreduzierung", 0, 100],
  ["ColorNoiseReduction", "Farbrauschen", 0, 100], ["PostCropVignetteAmount", "Vignette", -100, 100]];
const LOCAL: S[] = [["Exposure2012", "Belichtung", -4, 4, 0.05], ["Contrast2012", "Kontrast", -100, 100],
  ["Highlights2012", "Lichter", -100, 100], ["Shadows2012", "Tiefen", -100, 100],
  ["Whites2012", "Weiss", -100, 100], ["Blacks2012", "Schwarz", -100, 100], ["Texture", "Struktur", -100, 100],
  ["Clarity2012", "Klarheit", -100, 100], ["Dehaze", "Dunst entfernen", -100, 100],
  ["Saturation", "Sättigung", -100, 100], ["Temperature", "Temperatur", -100, 100], ["Tint", "Tönung", -100, 100],
  ["Sharpness", "Schärfe", -100, 100]];
const HSL: [string, string, string][] = [["Red", "Rot", "#e5484d"], ["Orange", "Orange", "#f76b15"],
  ["Yellow", "Gelb", "#ffc53d"], ["Green", "Grün", "#46a758"], ["Aqua", "Aqua", "#12a594"],
  ["Blue", "Blau", "#0090ff"], ["Purple", "Lila", "#8e4ec6"], ["Magenta", "Magenta", "#d6409f"]];
const HSL_KIND: [string, string][] = [["HueAdjustment", "Farbton"], ["SaturationAdjustment", "Sättigung"],
  ["LuminanceAdjustment", "Luminanz"]];
const KIND_LABEL: Record<string, string> = { subject: "Motiv", background: "Hintergrund", sky: "Himmel",
  person: "Personen", gradient: "Linearer Verlauf", radial: "Radialer Verlauf", other: "Maske" };

function Slider({ s, value, onChange, accent }: { s: S; value: number; onChange: (v: number) => void; accent?: string }) {
  const [key, label, min, max, step] = s;
  const fmt = (step ?? 1) < 1 ? (value > 0 ? "+" : "") + value.toFixed(2) : (value > 0 ? "+" : "") + Math.round(value);
  return (
    <div className="ed-slider" key={key}>
      <div className="ed-sl-head" onDoubleClick={() => onChange(0)} title="Doppelklick: zurücksetzen">
        <span style={accent ? { color: accent } : undefined}>{label}</span><b>{fmt}</b>
      </div>
      <input type="range" min={min} max={max} step={step ?? 1} value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))} onDoubleClick={() => onChange(0)} />
    </div>
  );
}

function Section({ title, children, open = true }: { title: string; children: React.ReactNode; open?: boolean }) {
  const [o, setO] = useState(open);
  return (
    <div className="ed-section">
      <button className="ed-sec-head" onClick={() => setO(!o)}>{o ? "▾" : "▸"} {title}</button>
      {o && <div className="ed-sec-body">{children}</div>}
    </div>
  );
}

export default function Editor({ iid, filename, onClose, onSaved, toast }: {
  iid: number; filename: string; onClose: () => void; onSaved: (jobId?: number) => void;
  toast: (m: string, kind?: "ok" | "error") => void;
}) {
  const [info, setInfo] = useState<EdInfo | null>(null);
  const [model, setModel] = useState<EdModel | null>(null);
  const [src, setSrc] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [sel, setSel] = useState<number | null>(null);
  const [showMask, setShowMask] = useState(false);
  const [hslTab, setHslTab] = useState(1);
  const [dirty, setDirty] = useState(false);
  const imgRef = useRef<HTMLImageElement>(null);
  const reqId = useRef(0);
  const shown = useRef(0);
  const drag = useRef<Pt | null>(null);

  useEffect(() => {
    setInfo(null); setModel(null); setSel(null); setDirty(false);
    api.get<EdInfo>(`/api/images/${iid}/editor`).then((i) => { setInfo(i); setModel(i.model); })
      .catch((e) => toast((e as Error).message, "error"));
  }, [iid, toast]);

  // Live-Vorschau (entprellt, nur die neueste Anfrage zählt)
  // Beim Ziehen eines Reglers zuerst schnell und klein, nach kurzer Pause in voller Qualität
  useEffect(() => {
    if (!model) return;
    const full = Math.min(1800, Math.round(Math.max(window.innerHeight, 800) * (window.devicePixelRatio || 1) * 0.8));
    const quick = run(Math.min(900, full), 40);
    const fine = run(full, 380);
    return () => { clearTimeout(quick); clearTimeout(fine); };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [model, iid, showMask, sel, toast]);

  function run(size: number, delay: number) {
    const id = ++reqId.current;
    return setTimeout(async () => {
      setBusy(true);
      try {
        const r = await fetch(`${BASE}/api/images/${iid}/editor/preview`, {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ model, size, overlay: showMask && sel !== null ? sel : null }),
        });
        if (!r.ok) throw new Error(`Vorschau: Fehler ${r.status}`);
        const url = URL.createObjectURL(await r.blob());
        if (id >= shown.current) {
          shown.current = id;
          setSrc((old) => { if (old) URL.revokeObjectURL(old); return url; });
        } else URL.revokeObjectURL(url);
      } catch (e) {
        if (id === reqId.current) toast((e as Error).message, "error");
      } finally {
        if (id === reqId.current) setBusy(false);
      }
    }, delay);
  }

  const update = useCallback((f: (m: EdModel) => EdModel) => { setModel((m) => (m ? f(m) : m)); setDirty(true); }, []);
  const setG = (k: string, v: number) => update((m) => ({ ...m, global: { ...m.global, [k]: v },
    wb_custom: k === "Temperature" || k === "Tint" ? true : m.wb_custom }));
  const setMask = (i: number, patch: Partial<EdMask>) => update((m) => ({ ...m, masks: m.masks.map((x, j) => (j === i ? { ...x, ...patch } : x)) }));
  const addMask = (kind: EdMask["kind"]) => {
    const n: EdMask = { name: KIND_LABEL[kind], kind, local: {} };
    if (kind === "gradient") { n.zero = [0.5, 0.65]; n.full = [0.5, 1.0]; n.local = { Exposure2012: -0.5 }; }
    if (kind === "radial") { n.box = [0.3, 0.25, 0.7, 0.85]; n.feather = 60; n.invert = true; n.local = { Exposure2012: -0.4 }; }
    if (kind === "subject") n.local = { Exposure2012: 0.3 };
    update((m) => ({ ...m, masks: [...m.masks, n] }));
    setSel(model ? model.masks.length : 0);
  };

  const g = (k: string) => model?.global[k] ?? 0;
  const temp = model?.global.Temperature ?? info?.as_shot[0] ?? 5500;
  const tint = model?.global.Tint ?? info?.as_shot[1] ?? 0;
  const cur = sel !== null && model ? model.masks[sel] : null;

  const pos = (e: React.MouseEvent): Pt | null => {
    const r = imgRef.current?.getBoundingClientRect();
    if (!r || !r.width) return null;
    return [Math.min(1, Math.max(0, (e.clientX - r.left) / r.width)), Math.min(1, Math.max(0, (e.clientY - r.top) / r.height))];
  };
  const onDown = (e: React.MouseEvent) => {
    if (!cur || sel === null || (cur.kind !== "gradient" && cur.kind !== "radial")) return;
    const p = pos(e); if (!p) return;
    e.preventDefault(); drag.current = p;
  };
  const onMove = (e: React.MouseEvent) => {
    if (!drag.current || !cur || sel === null) return;
    const p = pos(e); if (!p) return;
    const a = drag.current;
    if (cur.kind === "gradient") setMask(sel, { zero: a, full: p });
    else setMask(sel, { box: [Math.min(a[0], p[0]), Math.min(a[1], p[1]), Math.max(a[0], p[0]), Math.max(a[1], p[1])] });
  };
  const onUp = () => { drag.current = null; };

  const save = async () => {
    if (!model) return;
    await api.put(`/api/images/${iid}/editor`, { model });
    setDirty(false); toast("Edit gespeichert – geht so nach Lightroom"); onSaved();
  };
  const sync = async () => {
    if (!model) return;
    const name = window.prompt("Name für diesen Edit (wird als Vorlage gespeichert):", `Edit ${filename.replace(/\.[^.]+$/, "")}`);
    if (name === null) return;
    const r = await api.post<{ job_id: number; template: string }>(`/api/images/${iid}/editor/sync`, { model, name });
    setDirty(false);
    toast(`„${r.template}“ wird auf alle Bilder übertragen …`); onSaved(r.job_id);
  };
  const reset = async () => {
    if (!window.confirm("Eigenen Edit verwerfen und wieder automatisch bearbeiten?")) return;
    await api.del(`/api/images/${iid}/editor`);
    const i = await api.get<EdInfo>(`/api/images/${iid}/editor`);
    setInfo(i); setModel(i.model); setSel(null); setDirty(false); onSaved();
  };

  if (!model || !info) return <div className="editor"><div className="ed-stage"><div className="stage-loading">Editor lädt …</div></div></div>;
  return (
    <div className="editor">
      <div className="ed-stage" onMouseMove={onMove} onMouseUp={onUp} onMouseLeave={onUp}>
        <div className="ed-imgwrap" onMouseDown={onDown}
          style={cur && (cur.kind === "gradient" || cur.kind === "radial") ? { cursor: "crosshair" } : undefined}>
          {src && <img ref={imgRef} src={src} draggable={false} />}
          {src && cur && cur.kind === "gradient" && cur.zero && cur.full && (
            <svg className="ed-overlay" viewBox="0 0 1 1" preserveAspectRatio="none">
              <GradientLines zero={cur.zero} full={cur.full} />
            </svg>
          )}
          {src && cur && cur.kind === "radial" && cur.box && (
            <svg className="ed-overlay" viewBox="0 0 1 1" preserveAspectRatio="none">
              <ellipse cx={(cur.box[0] + cur.box[2]) / 2} cy={(cur.box[1] + cur.box[3]) / 2} rx={(cur.box[2] - cur.box[0]) / 2}
                ry={(cur.box[3] - cur.box[1]) / 2} fill="none" stroke="#fff" strokeWidth="0.003" vectorEffect="non-scaling-stroke" />
            </svg>
          )}
        </div>
        <div className="stage-badge">{busy ? "rechnet …" : "Eigener Edit (Vorschau ohne Zuschnitt)"}</div>
        {cur?.kind === "gradient" && <div className="stage-loading">Im Bild ziehen: vom Beginn (keine Wirkung) zum Ende (volle Wirkung)</div>}
        {cur?.kind === "radial" && <div className="stage-loading">Im Bild ein Rechteck ziehen: Bereich der Ellipse</div>}
      </div>
      <div className="ed-panel">
        <div className="ed-title">Bearbeiten <span className="muted">{filename}</span>{info.manual && <span className="tag">eigener Edit</span>}</div>
        <Section title="Weissabgleich">
          <Slider s={["Temperature", "Temperatur (K)", 2000, 12000, 50]} value={temp} onChange={(v) => setG("Temperature", v)} />
          <Slider s={["Tint", "Tönung", -150, 150]} value={tint} onChange={(v) => setG("Tint", v)} />
          {model.wb_custom && <button className="link" onClick={() => update((m) => {
            const gl = { ...m.global }; delete gl.Temperature; delete gl.Tint; return { ...m, global: gl, wb_custom: false };
          })}>wie aufgenommen</button>}
        </Section>
        <Section title="Licht">{LIGHT.map((s) => <Slider key={s[0]} s={s} value={g(s[0])} onChange={(v) => setG(s[0], v)} />)}</Section>
        <Section title="Präsenz">{PRESENCE.map((s) => <Slider key={s[0]} s={s} value={g(s[0])} onChange={(v) => setG(s[0], v)} />)}</Section>
        <Section title="Farbe (HSL)" open={false}>
          <div className="seg">{HSL_KIND.map(([, l], i) => <button key={l} className={hslTab === i ? "on" : ""} onClick={() => setHslTab(i)}>{l}</button>)}</div>
          {HSL.map(([c, l, col]) => {
            const k = `${HSL_KIND[hslTab][0]}${c}`;
            return <Slider key={k} s={[k, l, -100, 100]} value={g(k)} onChange={(v) => setG(k, v)} accent={col} />;
          })}
        </Section>
        <Section title="Details" open={false}>{DETAIL.map((s) => <Slider key={s[0]} s={s} value={g(s[0])} onChange={(v) => setG(s[0], v)} />)}</Section>
        <Section title={`Masken (${model.masks.length})`}>
          <div className="ed-addmask">
            {info.has_subject && <button onClick={() => addMask("subject")}>+ Motiv</button>}
            {info.has_subject && <button onClick={() => addMask("background")}>+ Hintergrund</button>}
            <button onClick={() => addMask("sky")}>+ Himmel</button>
            <button onClick={() => addMask("gradient")}>+ Verlauf</button>
            <button onClick={() => addMask("radial")}>+ Radial</button>
          </div>
          <div className="ed-masklist">
            {model.masks.map((m, i) => (
              <button key={i} className={`ed-mask ${sel === i ? "on" : ""}`} onClick={() => setSel(sel === i ? null : i)}>
                <span>{m.name}</span><span className="muted">{KIND_LABEL[m.kind]}</span>
              </button>
            ))}
          </div>
          {cur && sel !== null && (
            <div className="ed-maskedit">
              <input value={cur.name} onChange={(e) => setMask(sel, { name: e.target.value })} />
              <label className="check"><input type="checkbox" checked={showMask} onChange={(e) => setShowMask(e.target.checked)} /> Maske rot zeigen</label>
              {cur.kind === "radial" && <>
                <label className="check"><input type="checkbox" checked={!!cur.invert} onChange={(e) => setMask(sel, { invert: e.target.checked })} /> aussen wirken (umkehren)</label>
                <Slider s={["feather", "Weiche Kante", 0, 100]} value={cur.feather ?? 60} onChange={(v) => setMask(sel, { feather: v })} />
              </>}
              <Slider s={["amount", "Stärke (%)", 0, 100]} value={Math.round((cur.amount ?? 1) * 100)} onChange={(v) => setMask(sel, { amount: v / 100 })} />
              {LOCAL.map((s) => <Slider key={s[0]} s={s} value={cur.local[s[0]] ?? 0}
                onChange={(v) => setMask(sel, { local: { ...cur.local, [s[0]]: v } })} />)}
              <button className="link" style={{ color: "var(--bad)" }} onClick={() => {
                update((m) => ({ ...m, masks: m.masks.filter((_, j) => j !== sel) })); setSel(null);
              }}>Maske löschen</button>
            </div>
          )}
        </Section>
        <div className="ed-actions">
          <button className="primary" onClick={sync} title="Dieser Edit 1:1 für alle Bilder, pro Bild nur Belichtung, Weissabgleich und Begradigen angepasst">Auf alle Bilder übertragen</button>
          <button onClick={save} disabled={!dirty}>Nur dieses Bild speichern</button>
          <div className="row">
            <button className="ghost" onClick={reset}>Automatisch</button>
            <button className="ghost" onClick={() => { if (!dirty || window.confirm("Änderungen verwerfen?")) onClose(); }}>Schliessen</button>
          </div>
        </div>
      </div>
    </div>
  );
}

function GradientLines({ zero, full }: { zero: Pt; full: Pt }) {
  // Linien senkrecht zur Verlaufsrichtung bei Beginn (0 %) und Ende (100 %)
  const dx = full[0] - zero[0], dy = full[1] - zero[1];
  const len = Math.hypot(dx, dy) || 1e-6;
  const px = -dy / len * 2, py = dx / len * 2;
  const line = (p: Pt, dash?: string) => (
    <line x1={p[0] - px} y1={p[1] - py} x2={p[0] + px} y2={p[1] + py} stroke="#fff" strokeWidth="1.5"
      strokeDasharray={dash} vectorEffect="non-scaling-stroke" />
  );
  return (
    <>
      {line(zero, "6 4")}
      {line(full)}
      <line x1={zero[0]} y1={zero[1]} x2={full[0]} y2={full[1]} stroke="#ffc53d" strokeWidth="1.5" vectorEffect="non-scaling-stroke" />
    </>
  );
}
