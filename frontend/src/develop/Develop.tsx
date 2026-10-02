// Entwickeln-Modul wie in Lightroom: links Vorgaben/Verlauf, Mitte das Bild (in der Grafikkarte entwickelt),
// rechts Histogramm, Werkzeuge und Regler, unten der Filmstreifen. Änderungen werden automatisch gespeichert.
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api, BASE, ImageItem, pickFiles } from "../api";
import { Modal } from "../ui";
import { DevelopGL, MASK_SIDE, MAX_MASKS, maskUniforms, OutOpts } from "./gl";
import {
  buildCurveTable, cropRect, EdMask, EdModel, fitCrop, grayWorld, internalAngle, labelOf, MaskKind,
  normalize, parseSource, Pt, samplePixel, solveWB, Source, toLrCrop, View, imgToView, viewToImg, fmtValue, def,
} from "./model";
import { ColorWheel, CurveEditor, Group, Histogram, Panel, Slider } from "./panels";
import * as Ic from "./icons";
import "./develop.css";

interface EdInfo {
  model: EdModel;
  manual: boolean;
  has_subject: boolean;
  orientation: number;
  as_shot: [number | null, number | null];
  is_raw: boolean;
  exif: { iso?: number; exposure_time?: number; aperture?: number; focal_length?: number; camera?: string; lens?: string };
}
type Tool = "edit" | "crop" | "mask";
type StyleOpt = { key: string; label: string; group?: string; user?: boolean };
type Hist = [Uint32Array, Uint32Array, Uint32Array];
interface HEntry { label: string; model: EdModel }

const clone = (m: EdModel): EdModel => JSON.parse(JSON.stringify(m));
let clipboard: EdModel | null = null;
let previous: EdModel | null = null;
const sources = new Map<number, Source>();       // zuletzt geladene Bilder (schnelles Hin- und Herblättern)
function remember(iid: number, s: Source) {
  sources.delete(iid); sources.set(iid, s);
  while (sources.size > 4) sources.delete(sources.keys().next().value as number);
}
// Arbeitsauflösung = Bildschirmauflösung (Retina bis 3600 px), nicht nur 2000 px
const srcSize = () => Math.min(3600, Math.max(1600, Math.round(Math.max(window.innerWidth - 560, 900) * Math.min(window.devicePixelRatio || 1, 2) * 1.15)));
async function fetchSource(iid: number): Promise<Source> {
  const hit = sources.get(iid);
  if (hit) return hit;
  const r = await fetch(`${BASE}/api/images/${iid}/editor/source?size=${srcSize()}`);
  if (!r.ok) {
    let msg = `Bild konnte nicht geladen werden (${r.status})`;
    try { const j = await r.json(); if (typeof j.detail === "string") msg = j.detail; } catch { /* egal */ }
    throw new Error(msg);
  }
  const s = parseSource(await r.arrayBuffer());
  remember(iid, s);
  return s;
}

const WB_PRESETS: [string, number, number][] = [["Tageslicht", 5500, 10], ["Bewölkt", 6500, 10], ["Schatten", 7500, 10],
  ["Kunstlicht", 2850, 0], ["Leuchtstoff", 3800, 21], ["Blitz", 5500, 0]];
const HSL: [string, string, string][] = [["Red", "Rot", "#e5484d"], ["Orange", "Orange", "#f76b15"], ["Yellow", "Gelb", "#ffc53d"],
  ["Green", "Grün", "#46a758"], ["Aqua", "Aquamarin", "#12a594"], ["Blue", "Blau", "#0090ff"], ["Purple", "Lila", "#8e4ec6"],
  ["Magenta", "Magenta", "#d6409f"]];
const T_TEMP = "linear-gradient(90deg,#3b6fd8,#c9c9c9 50%,#e2bf3c)";
const T_TINT = "linear-gradient(90deg,#46b34b,#c9c9c9 50%,#c74bc0)";
const T_LIGHT = "linear-gradient(90deg,#141414,#d8d8d8)";
const T_SAT = "linear-gradient(90deg,#8a8a8a,#e5484d 33%,#ffc53d 50%,#46a758 66%,#0090ff)";
const KIND_LABEL: Record<MaskKind, string> = { subject: "Motiv", background: "Hintergrund", sky: "Himmel", person: "Personen",
  gradient: "Linearer Verlauf", radial: "Radialer Verlauf", brush: "Pinsel", other: "Maske" };
const KIND_ICON: Record<MaskKind, (p: { size?: number }) => JSX.Element> = { subject: Ic.IcSubject, background: Ic.IcBackground,
  sky: Ic.IcSky, person: Ic.IcPeople, gradient: Ic.IcLinear, radial: Ic.IcRadial, brush: Ic.IcBrush, other: Ic.IcMask };
const LOCAL: [string, string, number, number, number?, string?][] = [["Temperature", "Temp.", -100, 100, 1, T_TEMP],
  ["Tint", "Tönung", -100, 100, 1, T_TINT], ["Exposure2012", "Belichtung", -4, 4, 0.01, T_LIGHT], ["Contrast2012", "Kontrast", -100, 100],
  ["Highlights2012", "Lichter", -100, 100], ["Shadows2012", "Tiefen", -100, 100], ["Whites2012", "Weiss", -100, 100],
  ["Blacks2012", "Schwarz", -100, 100], ["Texture", "Struktur", -100, 100], ["Clarity2012", "Klarheit", -100, 100],
  ["Dehaze", "Dunst entfernen", -100, 100], ["Saturation", "Sättigung", -100, 100, 1, T_SAT], ["Sharpness", "Schärfe", -100, 100]];
const ASPECTS: [string, number | null][] = [["Wie Aufnahme", -1], ["Frei", null], ["1 : 1", 1], ["4 : 5", 4 / 5], ["5 : 4", 5 / 4],
  ["3 : 2", 3 / 2], ["2 : 3", 2 / 3], ["16 : 9", 16 / 9], ["9 : 16", 9 / 16]];
const tempPos = (t: number) => Math.round((500 - 1e6 / Math.max(t, 2000)) / 0.48);
const posTemp = (p: number) => Math.round(1e6 / (500 - p * 0.48) / 10) * 10;
const srgbLin = (v: number) => (v <= 0.04045 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4));

export default function Develop({ items, index, onIndex, onClose, toast, onSynced }: {
  items: ImageItem[]; index: number; onIndex: (i: number) => void; onClose: () => void;
  toast: (m: string, kind?: "ok" | "error") => void; onSynced: (jobId: number) => void;
}) {
  const cur = items[Math.min(index, items.length - 1)];
  const iid = cur?.id;
  const [info, setInfo] = useState<EdInfo | null>(null);
  const [model, setModelState] = useState<EdModel | null>(null);
  const modelRef = useRef<EdModel | null>(null);
  const [hist, setHist] = useState<HEntry[]>([]);
  const [hIdx, setHIdx] = useState(0);
  const [src, setSrc] = useState<Source | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [noGpu, setNoGpu] = useState(false);
  const [fallbackUrl, setFallbackUrl] = useState<string | null>(null);
  const [tool, setTool] = useState<Tool>("edit");
  const [sel, setSel] = useState<number | null>(null);
  const [overlay, setOverlay] = useState(true);
  const [adjusting, setAdjusting] = useState(false);
  const adjT = useRef(0);
  const [drawing, setDrawing] = useState(false);
  const [objBox, setObjBox] = useState<[Pt, Pt] | null>(null);
  const [dnPreview, setDnPreview] = useState<{ url: string; method: string } | null>(null);
  const [dnBusy, setDnBusy] = useState(false);
  const [dnPick, setDnPick] = useState(false);
  const [hover, setHover] = useState<number | null>(null);
  const [pipette, setPipette] = useState(false);
  const [before, setBefore] = useState<"off" | "on" | "split">("off");
  const [zoom, setZoom] = useState(false);
  const [clip, setClip] = useState(false);
  const [histo, setHisto] = useState<Hist | null>(null);
  const [save, setSave] = useState<"saved" | "dirty" | "saving" | "error">("saved");
  const [syncOpen, setSyncOpen] = useState(false);
  const [styles, setStyles] = useState<StyleOpt[]>([]);
  const [hoverStyle, setHoverStyle] = useState<string | null>(null);
  const [curveCh, setCurveCh] = useState<"param" | "main" | "red" | "green" | "blue">("main");
  const [hslTab, setHslTab] = useState<"Hue" | "Saturation" | "Luminance" | "all">("Hue");
  const [brush, setBrush] = useState({ size: 0.04, feather: 50, erase: false });
  const [aspect, setAspect] = useState<number | null>(-1);
  const [cropDraft, setCropDraft] = useState<{ rect: [number, number, number, number]; ang: number } | null>(null);
  const [box, setBox] = useState({ w: 0, h: 0 });
  const [cursor, setCursor] = useState<Pt | null>(null);

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const stageRef = useRef<HTMLDivElement>(null);
  const glRef = useRef<DevelopGL | null>(null);
  const layerKeys = useRef<(string | null)[]>([]);
  const layerReady = useRef<(number | null)[]>([]);
  const maskCache = useRef(new Map<string, Uint8Array>());
  const saveTimer = useRef<number | null>(null);
  const savedJson = useRef("");
  const raf = useRef(0);
  const histTimer = useRef<number | null>(null);
  const loadSeq = useRef(0);
  const [layerTick, setLayerTick] = useState(0);

  const orient = info?.orientation ?? 1;

  // ---------------------------------------------------------------- Modell, Verlauf, Speichern
  const setModel = useCallback((m: EdModel, dirty = true) => {
    modelRef.current = m;
    setModelState(m);
    if (dirty) setSave("dirty");
  }, []);
  const update = useCallback((f: (m: EdModel) => EdModel) => {
    const m = modelRef.current;
    if (!m) return;
    setModel(f(m));
  }, [setModel]);
  const commit = useCallback((label: string) => {
    const m = modelRef.current;
    if (!m) return;
    setHist((h) => {
      const base = h.slice(0, hIdxRef.current + 1);
      const last = base[base.length - 1];
      if (last && JSON.stringify(last.model) === JSON.stringify(m)) return h;
      const next = [...base, { label, model: clone(m) }].slice(-200);
      hIdxRef.current = next.length - 1;
      setHIdx(next.length - 1);
      return next;
    });
  }, []);
  const hIdxRef = useRef(0);
  const histRef = useRef<HEntry[]>([]);
  histRef.current = hist;
  const goHistory = useCallback((i: number) => {
    const h = histRef.current[i];
    if (!h) return;
    hIdxRef.current = i;
    setHIdx(i);
    setModel(clone(h.model));
  }, [setModel]);

  const doSave = useCallback(async (id: number, m: EdModel, keepalive = false) => {
    const body = JSON.stringify({ model: m });
    if (body === savedJson.current) { setSave("saved"); return; }
    setSave("saving");
    try {
      const r = await fetch(`${BASE}/api/images/${id}/editor`, { method: "PUT", headers: { "Content-Type": "application/json" }, body, keepalive });
      if (!r.ok) throw new Error(String(r.status));
      savedJson.current = body;
      setSave((s) => (s === "saving" ? "saved" : s));
    } catch {
      setSave("error");
    }
  }, []);
  // automatisch speichern (kurz nach der letzten Änderung)
  useEffect(() => {
    if (save !== "dirty" || !model || !iid) return;
    if (saveTimer.current) clearTimeout(saveTimer.current);
    saveTimer.current = window.setTimeout(() => doSave(iid, model), 700);
    return () => { if (saveTimer.current) clearTimeout(saveTimer.current); };
  }, [model, save, iid, doSave]);
  const flush = useCallback(async () => {
    if (saveTimer.current) clearTimeout(saveTimer.current);
    const m = modelRef.current;
    if (m && iid && JSON.stringify({ model: m }) !== savedJson.current) await doSave(iid, m);
  }, [iid, doSave]);

  // ---------------------------------------------------------------- Bild laden
  useEffect(() => {
    if (!iid) return;
    const seq = ++loadSeq.current;
    setErr(null); setSel(null); setPipette(false); setCropDraft(null); setHisto(null);
    if (tool === "crop") setTool("edit");
    layerKeys.current = []; layerReady.current = [];
    api.get<EdInfo>(`/api/images/${iid}/editor`).then((i) => {
      if (seq !== loadSeq.current) return;
      const m = normalize(i.model);
      setInfo(i);
      savedJson.current = JSON.stringify({ model: m });
      setModel(m, false);
      setSave("saved");
      const h = [{ label: "Geöffnet", model: clone(m) }];
      setHist(h); hIdxRef.current = 0; setHIdx(0);
      if (!i.is_raw) { setErr("Nur RAW-Dateien lassen sich bearbeiten (JPGs sind schon fertig entwickelt)."); return; }
      fetchSource(iid).then((s) => {
        if (seq !== loadSeq.current) return;
        setSrc(s);
        // nächstes Bild schon vorbereiten
        const nx = items[index + 1];
        if (nx) setTimeout(() => fetchSource(nx.id).catch(() => undefined), 300);
      }).catch((e) => seq === loadSeq.current && setErr((e as Error).message));
    }).catch((e) => seq === loadSeq.current && setErr((e as Error).message));
    return () => {
      // beim Wechsel sofort speichern
      const m = modelRef.current;
      if (m && JSON.stringify({ model: m }) !== savedJson.current) { doSave(iid, m, true); previous = clone(m); }
      else if (m && hist.length > 1) previous = clone(m);
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [iid]);

  const loadStyles = useCallback(() => api.get<StyleOpt[]>("/api/styles").then(setStyles).catch(() => undefined), []);
  useEffect(() => { loadStyles(); }, [loadStyles]);
  const importPresets = async () => {
    const files = await pickFiles(["xmp", "lrtemplate"], "Lightroom-Vorgaben wählen (.xmp oder .lrtemplate)");
    if (!files.length) return;
    try {
      const r = await api.post<{ imported: string[]; errors: string[] }>("/api/presets/import", { paths: files });
      toast(`${r.imported.length} Vorgabe(n) importiert${r.errors.length ? `, ${r.errors.length} übersprungen` : ""}`);
      loadStyles();
    } catch (e) { toast((e as Error).message, "error"); }
  };
  const runDnPreview = async (p: Pt) => {
    if (!iid) return;
    setDnBusy(true);
    try {
      const r = await fetch(`${BASE}/api/images/${iid}/denoise/preview`, { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ x: p[0], y: p[1], amount: modelRef.current?.denoise || 50, size: 300 }) });
      if (!r.ok) throw new Error(`Vorschau: Fehler ${r.status}`);
      const url = URL.createObjectURL(await r.blob());
      setDnPreview((o) => { if (o) URL.revokeObjectURL(o.url); return { url, method: r.headers.get("X-Denoise-Method") ?? "" }; });
    } catch (e) { toast((e as Error).message, "error"); } finally { setDnBusy(false); }
  };

  // Grafikkarte einrichten
  useEffect(() => {
    const c = canvasRef.current;
    if (!c || glRef.current || noGpu) return;
    const g = DevelopGL.create(c);
    if (!g) { setNoGpu(true); return; }
    glRef.current = g;
    return () => { g.dispose(); glRef.current = null; };
  }, [noGpu]);
  useEffect(() => {
    const g = glRef.current;
    if (!g || !src) return;
    g.setSource(src);
    layerKeys.current = []; layerReady.current = [];
    setLayerTick((t) => t + 1);
  }, [src]);

  // ---------------------------------------------------------------- Masken-Ebenen (KI-Masken vom Server, Pinsel lokal)
  const toLayer = useCallback((img: CanvasImageSource, w: number, h: number): Uint8Array => {
    const g = glRef.current!;
    const cv = document.createElement("canvas");
    cv.width = g.mw; cv.height = g.mh;
    const ctx = cv.getContext("2d", { willReadFrequently: true })!;
    ctx.drawImage(img, 0, 0, w, h, 0, 0, g.mw, g.mh);
    const d = ctx.getImageData(0, 0, g.mw, g.mh).data;
    const out = new Uint8Array(g.mw * g.mh);
    for (let i = 0; i < out.length; i++) out[i] = d[i * 4];
    return out;
  }, []);
  const brushLayer = useCallback((m: EdMask): Uint8Array => {
    const g = glRef.current!;
    const cv = document.createElement("canvas");
    cv.width = g.mw; cv.height = g.mh;
    const ctx = cv.getContext("2d", { willReadFrequently: true })!;
    ctx.fillStyle = "#000"; ctx.fillRect(0, 0, g.mw, g.mh);
    const long = Math.max(g.mw, g.mh), f = (m.feather ?? 50) / 100;
    for (const [x, y, r] of m.dabs ?? []) {
      const R = r * long, inner = R * (1 - 0.5 * f), outer = R * (1 + 0.3 * f);
      const grd = ctx.createRadialGradient(x * g.mw, y * g.mh, inner, x * g.mw, y * g.mh, Math.max(outer, inner + 0.5));
      grd.addColorStop(0, "rgba(255,255,255,1)"); grd.addColorStop(1, "rgba(255,255,255,0)");
      ctx.fillStyle = grd;
      ctx.beginPath(); ctx.arc(x * g.mw, y * g.mh, Math.max(outer, 1), 0, Math.PI * 2); ctx.fill();
    }
    const d = ctx.getImageData(0, 0, g.mw, g.mh).data;
    const out = new Uint8Array(g.mw * g.mh);
    for (let i = 0; i < out.length; i++) out[i] = d[i * 4];
    return out;
  }, []);
  useEffect(() => {
    const g = glRef.current;
    if (!g || !src || !model || !iid) return;
    model.masks.slice(0, MAX_MASKS).forEach((m, i) => {
      let key: string | null = null;
      if (m.kind === "brush") key = `brush|${m.feather ?? 50}|${JSON.stringify(m.dabs ?? [])}`;
      else if (m.kind !== "gradient" && m.kind !== "radial") key = `${m.kind}|${m.kind === "other" ? JSON.stringify(m.raw) : ""}`;
      if (layerKeys.current[i] === key) return;
      layerKeys.current[i] = key;
      if (key === null) { layerReady.current[i] = null; return; }
      if (m.kind === "brush") { g.setMaskLayer(i, brushLayer(m)); layerReady.current[i] = i; return; }
      const hit = maskCache.current.get(`${iid}|${key}`);
      if (hit) { g.setMaskLayer(i, hit); layerReady.current[i] = i; return; }
      layerReady.current[i] = null;
      fetch(`${BASE}/api/images/${iid}/editor/mask`, { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ mask: m, aspect: src.w / src.h, size: MASK_SIDE }) })
        .then((r) => (r.ok ? r.blob() : Promise.reject(new Error(String(r.status))))).then(createImageBitmap)
        .then((bmp) => {
          if (!glRef.current || layerKeys.current[i] !== key) return;
          const data = toLayer(bmp, bmp.width, bmp.height);
          maskCache.current.set(`${iid}|${key}`, data);
          glRef.current.setMaskLayer(i, data);
          layerReady.current[i] = i;
          setLayerTick((t) => t + 1);
        }).catch(() => undefined);
    });
    layerKeys.current.length = Math.min(model.masks.length, MAX_MASKS);
  }, [model, src, iid, brushLayer, toLayer, layerTick]);

  // ---------------------------------------------------------------- Ansicht (Zuschnitt, Drehung)
  const W = src?.w ?? 3, H = src?.h ?? 2;
  const view: View = useMemo(() => {
    if (!model) return { rect: [0, 0, 1, 1], ang: 0, W, H };
    if (tool === "crop" && cropDraft) return { rect: [0, 0, 1, 1], ang: cropDraft.ang, W, H };
    if (noGpu) return { rect: [0, 0, 1, 1], ang: 0, W, H };
    return { rect: cropRect(model.crop, orient, W, H), ang: internalAngle(model.crop, orient), W, H };
  }, [model, tool, cropDraft, W, H, orient, noGpu]);
  const outW = Math.max(16, Math.round(W * (view.rect[2] - view.rect[0])));
  const outH = Math.max(16, Math.round(H * (view.rect[3] - view.rect[1])));

  // Bühne: Bild einpassen
  useEffect(() => {
    const el = stageRef.current;
    if (!el) return;
    const fit = () => {
      const pad = 24, ah = el.clientHeight - pad * 2;
      const aw = before === "split" ? (el.clientWidth - pad * 2 - 12) / 2 : el.clientWidth - pad * 2;
      if (zoom) { const dpr = window.devicePixelRatio || 1; setBox({ w: outW / dpr * 1, h: outH / dpr * 1 }); return; }
      const s = Math.min(aw / outW, ah / outH);
      setBox({ w: Math.max(10, outW * s), h: Math.max(10, outH * s) });
    };
    fit();
    const ro = new ResizeObserver(fit);
    ro.observe(el);
    return () => ro.disconnect();
  }, [outW, outH, zoom, before]);

  // ---------------------------------------------------------------- Rendern
  // rote Maske wie in Lightroom: beim Malen/Ziehen immer, sonst wenn eingeschaltet (beim Regler-Ziehen kurz aus)
  const overlayIdx = tool === "mask" ? (hover ?? (drawing || (overlay && !adjusting) ? sel : null)) : null;
  const outOpts = useCallback((): OutOpts => ({ rect: view.rect, ang: view.ang, width: outW, height: outH,
    overlay: overlayIdx ?? -1, clip }), [view, outW, outH, overlayIdx, clip]);
  useEffect(() => {
    const g = glRef.current, c = canvasRef.current;
    if (!g || !src || !model || !c) return;
    cancelAnimationFrame(raf.current);
    raf.current = requestAnimationFrame(() => {
      if (c.width !== outW) c.width = outW;
      if (c.height !== outH) c.height = outH;
      g.setLut(buildCurveTable(model));
      const mu = maskUniforms(model, layerReady.current);
      const o = outOpts();
      g.render(model, mu, o);
      if (histTimer.current) clearTimeout(histTimer.current);
      histTimer.current = window.setTimeout(() => {
        if (!glRef.current || !modelRef.current) return;
        setHisto(glRef.current.histogram(modelRef.current, maskUniforms(modelRef.current, layerReady.current), o));
      }, 120);
    });
  }, [model, src, outOpts, outW, outH, layerTick]);

  // ohne Grafikkarte: Vorschau vom Server (langsamer)
  useEffect(() => {
    if (!noGpu || !model || !iid || !info?.is_raw) return;
    const t = setTimeout(async () => {
      try {
        const r = await fetch(`${BASE}/api/images/${iid}/editor/preview`, { method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ model, size: 1400, overlay: overlayIdx }) });
        if (r.ok) { const u = URL.createObjectURL(await r.blob()); setFallbackUrl((o) => { if (o) URL.revokeObjectURL(o); return u; }); }
      } catch { /* nächster Versuch */ }
    }, 120);
    return () => clearTimeout(t);
  }, [noGpu, model, iid, overlayIdx, info?.is_raw]);

  // ---------------------------------------------------------------- Regler
  const setG = useCallback((k: string, v: number) => update((m) => ({ ...m, global: { ...m.global, [k]: v } })), [update]);
  const commitG = useCallback((k: string, v: number) => commit(`${labelOf(k)} ${fmtValue(k, v)}`), [commit]);
  const setTempPos = useCallback((_k: string, p: number) => update((m) => ({ ...m, wb_custom: true,
    global: { ...m.global, Temperature: posTemp(p), Tint: m.global.Tint ?? src?.as_shot[1] ?? 0 } })), [update, src]);
  const setTint = useCallback((_k: string, v: number) => update((m) => ({ ...m, wb_custom: true,
    global: { ...m.global, Tint: v, Temperature: m.global.Temperature ?? src?.as_shot[0] ?? 5500 } })), [update, src]);
  const commitWB = useCallback(() => {
    const g = modelRef.current?.global;
    commit(`Weissabgleich ${Math.round(g?.Temperature ?? 0)} K / ${Math.round(g?.Tint ?? 0)}`);
  }, [commit]);
  const setWB = (t: number | null, ti: number | null, label: string) => {
    update((m) => {
      const gl = { ...m.global };
      if (t === null) { delete gl.Temperature; delete gl.Tint; return { ...m, global: gl, wb_custom: false }; }
      return { ...m, global: { ...gl, Temperature: t, Tint: ti ?? 0 }, wb_custom: true };
    });
    commit(label);
  };
  const g = (k: string) => model?.global[k] ?? def(k);
  const S = (k: string, min: number, max: number, step = 1, track?: string, label?: string) => (
    <Slider k={k} label={label ?? labelOf(k)} value={g(k)} min={min} max={max} step={step} track={track} onChange={setG} onCommit={commitG} />
  );
  const asShot: [number, number] = [src?.as_shot[0] ?? info?.as_shot[0] ?? 5500, src?.as_shot[1] ?? info?.as_shot[1] ?? 0];
  const temp = model?.wb_custom ? (model.global.Temperature ?? asShot[0]) : asShot[0];
  const tint = model?.wb_custom ? (model.global.Tint ?? asShot[1]) : asShot[1];

  // Automatik (Tonwerte) wie der Auto-Knopf in Lightroom: Helligkeit, Weiss, Schwarz am Histogramm ausrichten
  const autoTone = () => {
    const gpu = glRef.current;
    if (!gpu || !modelRef.current) return;
    let m: EdModel = { ...modelRef.current, global: { ...modelRef.current.global, Exposure2012: 0, Contrast2012: 0, Highlights2012: 0,
      Shadows2012: 0, Whites2012: 0, Blacks2012: 0 } };
    const o = { ...outOpts(), overlay: -1, clip: false };
    const stats = (mm: EdModel) => {
      gpu.render(mm, maskUniforms(mm, layerReady.current), o);
      const [r, gg, b] = gpu.histogram(mm, maskUniforms(mm, layerReady.current), o);
      const tot = gg.reduce((a, x) => a + x, 0);
      const q = (ch: Uint32Array, f: number) => { let acc = 0; for (let i = 0; i < 256; i++) { acc += ch[i]; if (acc >= f * tot) return i; } return 255; };
      const lum = new Uint32Array(256);
      for (let i = 0; i < 256; i++) lum[i] = Math.round(0.2126 * r[i] + 0.7152 * gg[i] + 0.0722 * b[i]);
      return { med: q(lum, 0.5), hi: Math.max(q(r, 0.995), q(gg, 0.995), q(b, 0.995)), lo: Math.min(q(r, 0.005), q(gg, 0.005), q(b, 0.005)) };
    };
    for (let it = 0; it < 4; it++) {
      const s = stats(m);
      const ev = Math.log2(srgbLin(0.44) / Math.max(srgbLin(s.med / 255), 1e-4));
      m = { ...m, global: { ...m.global, Exposure2012: Math.round(Math.max(-4, Math.min(4, (m.global.Exposure2012 ?? 0) + ev * 0.8)) * 100) / 100 } };
    }
    let s = stats(m);
    // Lichter/Tiefen nach Bildinhalt (wie Lightroom: helle Bilder holen Lichter zurück, dunkle Ecken öffnen)
    m.global.Highlights2012 = s.hi >= 254 ? -60 : s.hi >= 245 ? -35 : -15;
    m.global.Shadows2012 = s.lo <= 4 ? 40 : s.lo <= 15 ? 25 : 10;
    m.global.Contrast2012 = 8;
    m.global.Vibrance = 12;
    m.global.Saturation = 2;
    for (let it = 0; it < 6; it++) {
      s = stats(m);
      m.global.Whites2012 = Math.round(Math.max(-60, Math.min(60, (m.global.Whites2012 ?? 0) + (247 - s.hi) * 0.8)));
      m.global.Blacks2012 = Math.round(Math.max(-60, Math.min(40, (m.global.Blacks2012 ?? 0) + (6 - s.lo) * 1.2)));
    }
    setModel({ ...m, global: { ...m.global } });
    commit("Auto");
  };

  // ---------------------------------------------------------------- Masken
  const setMask = useCallback((i: number, patch: Partial<EdMask>) => update((m) => ({ ...m, masks: m.masks.map((x, j) => (j === i ? { ...x, ...patch } : x)) })), [update]);
  const addMask = (kind: MaskKind | "object") => {
    if (kind === "object") {
      const m0 = modelRef.current;
      if (!m0) return;
      setModel({ ...m0, masks: [...m0.masks, { name: "Objekt", kind: "brush", local: { Exposure2012: 0.3 }, dabs: [], feather: 30 }] });
      setSel(m0.masks.length); setOverlay(true); setObjMode(true);
      commit("Maske: Objekt");
      return;
    }
    const n: EdMask = { name: KIND_LABEL[kind], kind, local: {} };
    if (kind === "gradient") { n.zero = [0.5, 0.62]; n.full = [0.5, 0.98]; n.local = { Exposure2012: -0.5 }; }
    if (kind === "radial") { n.box = [0.3, 0.2, 0.7, 0.85]; n.feather = 60; n.invert = false; n.local = { Exposure2012: 0.3 }; }
    if (kind === "brush") { n.dabs = []; n.feather = brush.feather; n.local = { Exposure2012: 0.3 }; }
    if (kind === "subject" || kind === "person") n.local = { Exposure2012: 0.3 };
    if (kind === "background") n.local = { Exposure2012: -0.3 };
    if (kind === "sky") n.local = { Exposure2012: -0.3 };
    const m = modelRef.current;
    if (!m) return;
    if (m.masks.length >= MAX_MASKS) { toast(`Höchstens ${MAX_MASKS} Masken`, "error"); return; }
    setModel({ ...m, masks: [...m.masks, n] });
    setSel(m.masks.length);
    setOverlay(true); setObjMode(false);
    commit(`Maske: ${KIND_LABEL[kind]}`);
  };
  const delMask = (i: number) => {
    update((m) => ({ ...m, masks: m.masks.filter((_, j) => j !== i) }));
    setSel(null);
    commit("Maske gelöscht");
  };
  const curMask = sel !== null && model ? model.masks[sel] : null;
  const setLocal = useCallback((k: string, v: number) => {
    const i = selRef.current;
    if (i === null) return;
    setAdjusting(true);
    clearTimeout(adjT.current);
    adjT.current = window.setTimeout(() => setAdjusting(false), 700);
    update((m) => ({ ...m, masks: m.masks.map((x, j) => (j === i ? { ...x, local: { ...x.local, [k]: v } } : x)) }));
  }, [update]);
  const commitLocal = useCallback((k: string, v: number) => commit(`Maske ${labelOf(k)} ${fmtValue(k, v)}`), [commit]);
  const selRef = useRef<number | null>(null);
  selRef.current = sel;
  const [objMode, setObjMode] = useState(false);
  const pickObject = async (a: Pt, b: Pt) => {
    const i = selRef.current;
    if (i === null || !iid) return;
    const p1 = viewToImg(view, a), p2 = viewToImg(view, b);
    try {
      const r = await api.post<{ dabs: [number, number, number][] }>(`/api/images/${iid}/editor/object`,
        { box: [Math.min(p1[0], p2[0]), Math.min(p1[1], p2[1]), Math.max(p1[0], p2[0]), Math.max(p1[1], p2[1])] });
      if (!r.dabs.length) { toast("Kein Objekt gefunden – grösseren Rahmen ziehen", "error"); return; }
      setMask(i, { dabs: [...(modelRef.current?.masks[i]?.dabs ?? []), ...r.dabs] });
      commit("Objekt ausgewählt");
      setObjMode(false);
    } catch (e) { toast((e as Error).message, "error"); }
  };

  // ---------------------------------------------------------------- Zuschneiden
  const startCrop = () => {
    if (!model) return;
    setCropDraft({ rect: cropRect(model.crop, orient, W, H), ang: internalAngle(model.crop, orient) });
    setTool("crop"); setPipette(false);
  };
  const applyCrop = (rect: [number, number, number, number], ang: number) => {
    const r = fitCrop(rect, ang, W, H);
    setCropDraft({ rect: r, ang });
    update((m) => ({ ...m, crop: toLrCrop(r, ang, orient, W, H) }));
  };
  const setCropAspect = (a: number | null) => {
    setAspect(a);
    if (!cropDraft || a === null) return;
    const ratio = a === -1 ? W / H : a;
    const [x0, y0, x1, y1] = cropDraft.rect;
    const cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
    let w = (x1 - x0) * W, h = (y1 - y0) * H;
    if (w / h > ratio) w = h * ratio; else h = w / ratio;
    const big = fitCrop([cx - 2 * w / W, cy - 2 * h / H, cx + 2 * w / W, cy + 2 * h / H], cropDraft.ang, W, H);
    applyCrop(big, cropDraft.ang);
    commit("Seitenverhältnis");
  };
  const leaveCrop = () => { setTool("edit"); setCropDraft(null); commit("Zuschneiden"); };

  // ---------------------------------------------------------------- Zeiger auf dem Bild
  const drag = useRef<{ mode: string; start: Pt; q0: Pt; mask?: EdMask; rect?: [number, number, number, number]; ang?: number; scroll?: Pt } | null>(null);
  const qOf = (e: React.PointerEvent): Pt => {
    const r = canvasRef.current?.getBoundingClientRect() ?? (e.currentTarget as HTMLElement).getBoundingClientRect();
    return [(e.clientX - r.left) / r.width, (e.clientY - r.top) / r.height];
  };
  const pxDist = (a: Pt, b: Pt) => Math.hypot((a[0] - b[0]) * box.w, (a[1] - b[1]) * box.h);
  const brushDab = (q: Pt, erase: boolean) => {
    const i = selRef.current;
    const m = modelRef.current;
    if (i === null || !m) return;
    const mk = m.masks[i];
    const p = viewToImg(view, q);
    const dabs = mk.dabs ?? [];
    if (erase) {
      const long = Math.max(W, H);
      const keep = dabs.filter(([x, y]) => Math.hypot((x - p[0]) * W, (y - p[1]) * H) > brush.size * long);
      if (keep.length !== dabs.length) setMask(i, { dabs: keep });
      return;
    }
    const last = dabs[dabs.length - 1];
    if (last && Math.hypot((last[0] - p[0]) * W, (last[1] - p[1]) * H) < brush.size * Math.max(W, H) * 0.35) return;
    setMask(i, { dabs: [...dabs, [Math.round(p[0] * 1e5) / 1e5, Math.round(p[1] * 1e5) / 1e5, Math.round(brush.size * 1e4) / 1e4]] });
  };
  const onDown = (e: React.PointerEvent) => {
    if (!model || !src) return;
    const q = qOf(e);
    (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
    if (dnPick) {
      const p = viewToImg(view, q);
      setDnPick(false);
      runDnPreview(p);
      return;
    }
    if (tool === "mask" && objMode && curMask && sel !== null) {
      drag.current = { mode: "obj", start: q, q0: q };
      setObjBox([q, q]); setDrawing(true);
      return;
    }
    if (pipette) {
      const p = viewToImg(view, q);
      if (p[0] < 0 || p[1] < 0 || p[0] > 1 || p[1] > 1) return;
      const [t, ti] = solveWB(src.wb, samplePixel(src, p[0], p[1], 3));
      setWB(t, ti, `Weissabgleich (Pipette) ${t} K`);
      setPipette(false);
      return;
    }
    if (tool === "crop" && cropDraft) {
      const [x0, y0, x1, y1] = cropDraft.rect;
      const corners: [string, Pt][] = [["nw", [x0, y0]], ["ne", [x1, y0]], ["se", [x1, y1]], ["sw", [x0, y1]],
        ["n", [(x0 + x1) / 2, y0]], ["s", [(x0 + x1) / 2, y1]], ["w", [x0, (y0 + y1) / 2]], ["e", [x1, (y0 + y1) / 2]]];
      const hit = corners.find(([, p]) => pxDist(p, q) < 14);
      const inside = q[0] > x0 && q[0] < x1 && q[1] > y0 && q[1] < y1;
      drag.current = { mode: hit ? `crop-${hit[0]}` : inside ? "crop-move" : "crop-rot", start: q, q0: q, rect: cropDraft.rect, ang: cropDraft.ang };
      return;
    }
    if (tool === "mask" && curMask && sel !== null) {
      const k = curMask.kind;
      if (k === "brush") { drag.current = { mode: "brush", start: q, q0: q }; setDrawing(true); brushDab(q, e.altKey || brush.erase); return; }
      if (k === "gradient" && curMask.zero && curMask.full) {
        const z = imgToView(view, curMask.zero), f = imgToView(view, curMask.full), mid: Pt = [(z[0] + f[0]) / 2, (z[1] + f[1]) / 2];
        const mode = pxDist(z, q) < 14 ? "g-zero" : pxDist(f, q) < 14 ? "g-full" : pxDist(mid, q) < 16 ? "g-move" : "g-new";
        drag.current = { mode, start: q, q0: q, mask: { ...curMask } };
        setDrawing(true);
        return;
      }
      if (k === "radial" && curMask.box) {
        const [bx0, by0, bx1, by1] = curMask.box;
        const c = imgToView(view, [(bx0 + bx1) / 2, (by0 + by1) / 2]);
        const ex = imgToView(view, [bx1, (by0 + by1) / 2]), ey = imgToView(view, [(bx0 + bx1) / 2, by0]);
        const exl = imgToView(view, [bx0, (by0 + by1) / 2]), eyb = imgToView(view, [(bx0 + bx1) / 2, by1]);
        const p = viewToImg(view, q);
        const inside = ((p[0] - (bx0 + bx1) / 2) / ((bx1 - bx0) / 2)) ** 2 + ((p[1] - (by0 + by1) / 2) / ((by1 - by0) / 2)) ** 2 < 1;
        const mode = pxDist(ex, q) < 12 || pxDist(exl, q) < 12 ? "r-x" : pxDist(ey, q) < 12 || pxDist(eyb, q) < 12 ? "r-y"
          : pxDist(c, q) < 14 || inside ? "r-move" : "r-new";
        drag.current = { mode, start: q, q0: q, mask: { ...curMask } };
        setDrawing(true);
        return;
      }
    }
    if (zoom) {
      const st = stageRef.current!;
      drag.current = { mode: "pan", start: [e.clientX, e.clientY], q0: q, scroll: [st.scrollLeft, st.scrollTop] };
      return;
    }
    drag.current = { mode: "click", start: [e.clientX, e.clientY], q0: q };
  };
  const onMove = (e: React.PointerEvent) => {
    const q = qOf(e);
    setCursor(q);
    const d = drag.current;
    if (!d || !model) return;
    if (d.mode === "pan") {
      const st = stageRef.current!;
      st.scrollLeft = d.scroll![0] - (e.clientX - d.start[0]);
      st.scrollTop = d.scroll![1] - (e.clientY - d.start[1]);
      return;
    }
    if (d.mode === "brush") { brushDab(q, e.altKey || brush.erase); return; }
    if (d.mode === "obj") { setObjBox([d.start, q]); return; }
    if (d.mode.startsWith("crop") && d.rect) {
      let [x0, y0, x1, y1] = d.rect;
      const dx = q[0] - d.start[0], dy = q[1] - d.start[1];
      if (d.mode === "crop-move") {
        const w = x1 - x0, h = y1 - y0;
        let nx = Math.min(1 - w, Math.max(0, x0 + dx)), ny = Math.min(1 - h, Math.max(0, y0 + dy));
        let r: [number, number, number, number] = [nx, ny, nx + w, ny + h];
        // nicht aus dem gedrehten Bild schieben
        for (let k = 0; k < 20 && !fitCropOk(r, d.ang!); k++) { nx = (nx + x0) / 2; ny = (ny + y0) / 2; r = [nx, ny, nx + w, ny + h]; }
        setCropDraft({ rect: fitCropOk(r, d.ang!) ? r : d.rect, ang: d.ang! });
        update((m) => ({ ...m, crop: toLrCrop(fitCropOk(r, d.ang!) ? r : d.rect!, d.ang!, orient, W, H) }));
        return;
      }
      if (d.mode === "crop-rot") {
        const c: Pt = [(x0 + x1) / 2, (y0 + y1) / 2];
        const a0 = Math.atan2((d.start[1] - c[1]) * H, (d.start[0] - c[0]) * W), a1 = Math.atan2((q[1] - c[1]) * H, (q[0] - c[0]) * W);
        const ang = Math.max(-45, Math.min(45, d.ang! + ((a1 - a0) * 180) / Math.PI));
        applyCrop(d.rect, Math.round(ang * 100) / 100);
        return;
      }
      const m = d.mode.slice(5);
      if (m.includes("w")) x0 = Math.min(x1 - 0.05, x0 + dx);
      if (m.includes("e")) x1 = Math.max(x0 + 0.05, x1 + dx);
      if (m.includes("n")) y0 = Math.min(y1 - 0.05, y0 + dy);
      if (m.includes("s")) y1 = Math.max(y0 + 0.05, y1 + dy);
      if (aspect !== null) {
        const ratio = aspect === -1 ? W / H : aspect;
        const w = (x1 - x0) * W;
        const h = w / ratio;
        if (m.includes("n")) y0 = y1 - h / H; else y1 = y0 + h / H;
      }
      const r: [number, number, number, number] = [Math.max(0, x0), Math.max(0, y0), Math.min(1, x1), Math.min(1, y1)];
      if (fitCropOk(r, d.ang!)) {
        setCropDraft({ rect: r, ang: d.ang! });
        update((mm) => ({ ...mm, crop: toLrCrop(r, d.ang!, orient, W, H) }));
      }
      return;
    }
    if (d.mask && sel !== null) {
      const p = viewToImg(view, q), p0 = viewToImg(view, d.start);
      const ddx = p[0] - p0[0], ddy = p[1] - p0[1];
      const mk = d.mask;
      if (d.mode === "g-zero") setMask(sel, { zero: p });
      else if (d.mode === "g-full") setMask(sel, { full: p });
      else if (d.mode === "g-move") setMask(sel, { zero: [mk.zero![0] + ddx, mk.zero![1] + ddy], full: [mk.full![0] + ddx, mk.full![1] + ddy] });
      else if (d.mode === "g-new") { if (pxDist(d.start, q) > 6) setMask(sel, { zero: p0, full: p }); }
      else if (mk.box) {
        const [bx0, by0, bx1, by1] = mk.box, cx = (bx0 + bx1) / 2, cy = (by0 + by1) / 2;
        if (d.mode === "r-move") setMask(sel, { box: [bx0 + ddx, by0 + ddy, bx1 + ddx, by1 + ddy] });
        else if (d.mode === "r-x") { const rx = Math.max(0.01, Math.abs(p[0] - cx)); setMask(sel, { box: [cx - rx, by0, cx + rx, by1] }); }
        else if (d.mode === "r-y") { const ry = Math.max(0.01, Math.abs(p[1] - cy)); setMask(sel, { box: [bx0, cy - ry, bx1, cy + ry] }); }
        else if (d.mode === "r-new" && pxDist(d.start, q) > 6) {
          const rx = Math.abs(p[0] - p0[0]), ry = Math.abs(p[1] - p0[1]);
          setMask(sel, { box: [p0[0] - rx, p0[1] - ry, p0[0] + rx, p0[1] + ry] });
        }
      }
    }
  };
  const onUp = (e: React.PointerEvent) => {
    const d = drag.current;
    drag.current = null;
    setDrawing(false);
    if (!d) return;
    if (d.mode === "obj") {
      setObjBox(null);
      if (pxDist(d.start, qOf(e)) > 8) pickObject(d.start, qOf(e));
      return;
    }
    if (d.mode === "click" && Math.hypot(e.clientX - d.start[0], e.clientY - d.start[1]) < 4 && tool !== "crop") {
      // Klick: 1:1 an dieser Stelle
      const q = d.q0;
      setZoom((z) => {
        if (!z) setTimeout(() => {
          const st = stageRef.current;
          if (st) { st.scrollLeft = q[0] * st.scrollWidth - st.clientWidth / 2; st.scrollTop = q[1] * st.scrollHeight - st.clientHeight / 2; }
        }, 30);
        return !z;
      });
      return;
    }
    if (d.mode === "pan") {
      if (Math.hypot(e.clientX - d.start[0], e.clientY - d.start[1]) < 4) setZoom(false);
      return;
    }
    if (d.mode.startsWith("crop")) return;
    if (d.mode === "brush") commit("Pinsel");
    else if (d.mask) commit(d.mask.kind === "gradient" ? "Verlauf verschoben" : "Radialfilter verschoben");
  };
  const fitCropOk = (r: [number, number, number, number], ang: number) => {
    const f = fitCrop(r, ang, W, H);
    return Math.abs(f[2] - f[0] - (r[2] - r[0])) < 1e-6;
  };

  // ---------------------------------------------------------------- Tastatur
  const keyRef = useRef<(e: KeyboardEvent) => void>(() => undefined);
  keyRef.current = (e: KeyboardEvent) => {
    const t = e.target as HTMLElement;
    if (["INPUT", "SELECT", "TEXTAREA"].includes(t.tagName) && (t as HTMLInputElement).type !== "range") return;
    if (syncOpen) return;
    const mod = e.metaKey || e.ctrlKey;
    const k = e.key.toLowerCase();
    if (mod && k === "z") { e.preventDefault(); goHistory(e.shiftKey ? Math.min(hist.length - 1, hIdx + 1) : Math.max(0, hIdx - 1)); return; }
    if (mod && e.shiftKey && k === "c") { e.preventDefault(); copySettings(); return; }
    if (mod && e.shiftKey && k === "v") { e.preventDefault(); pasteSettings(); return; }
    if (mod && e.shiftKey && k === "s") { e.preventDefault(); setSyncOpen(true); return; }
    if (mod && k === "u") { e.preventDefault(); autoTone(); return; }
    if (mod) return;
    if (t.tagName === "INPUT" && ["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"].includes(e.key)) return;
    if (e.key === "Escape") {
      if (pipette) setPipette(false);
      else if (tool === "crop") leaveCrop();
      else if (tool === "mask") { setTool("edit"); setSel(null); }
      else onClose();
    } else if (e.key === "ArrowRight") go(index + 1);
    else if (e.key === "ArrowLeft") go(index - 1);
    else if (e.key === "\\") setBefore((b) => (b === "on" ? "off" : "on"));
    else if (k === "y") setBefore((b) => (b === "split" ? "off" : "split"));
    else if (k === "r") (tool === "crop" ? leaveCrop() : startCrop());
    else if (k === "m") { setTool((x) => (x === "mask" ? "edit" : "mask")); setCropDraft(null); }
    else if (k === "w") setPipette((p) => !p);
    else if (k === "j") setClip((c) => !c);
    else if (k === "z") setZoom((z) => !z);
    else if (k === "o" && tool === "mask") setOverlay((o) => !o);
    else if (k === "g") onClose();
    else if ((e.key === "Delete" || e.key === "Backspace") && tool === "mask" && sel !== null) delMask(sel);
    else if (e.key === "[" && curMask?.kind === "brush") setBrush((b) => ({ ...b, size: Math.max(0.005, b.size / 1.2) }));
    else if (e.key === "]" && curMask?.kind === "brush") setBrush((b) => ({ ...b, size: Math.min(0.3, b.size * 1.2) }));
    else return;
    e.preventDefault();
  };
  useEffect(() => {
    const h = (e: KeyboardEvent) => keyRef.current(e);
    window.addEventListener("keydown", h);
    return () => window.removeEventListener("keydown", h);
  }, []);

  const go = async (i: number) => {
    if (i < 0 || i >= items.length) return;
    onIndex(i);
  };
  const copySettings = () => { if (modelRef.current) { clipboard = clone(modelRef.current); toast("Einstellungen kopiert (⇧⌘V fügt ein)"); } };
  const pasteSettings = () => {
    if (!clipboard || !modelRef.current) { toast("Nichts kopiert", "error"); return; }
    setModel({ ...clone(clipboard), crop: modelRef.current.crop });
    commit("Einstellungen eingefügt");
  };
  const usePrevious = () => {
    if (!previous || !modelRef.current) { toast("Noch kein vorheriges Bild bearbeitet", "error"); return; }
    setModel({ ...clone(previous), crop: modelRef.current.crop });
    commit("Vorherige Einstellungen");
  };
  const resetAll = async () => {
    if (!iid) return;
    if (saveTimer.current) clearTimeout(saveTimer.current);
    await api.del(`/api/images/${iid}/editor`);
    const i = await api.get<EdInfo>(`/api/images/${iid}/editor`);
    const m = normalize(i.model);
    savedJson.current = JSON.stringify({ model: m });
    setModel(m, false); setSave("saved"); setSel(null);
    commit("Zurückgesetzt");
  };
  const applyStyle = async (key: string, label: string) => {
    if (!iid) return;
    if (saveTimer.current) clearTimeout(saveTimer.current);
    try {
      await api.post(`/api/images/${iid}/style`, { style: key });
      const i = await api.get<EdInfo>(`/api/images/${iid}/editor`);
      const m = normalize(i.model);
      savedJson.current = JSON.stringify({ model: m });
      setModel(m, false); setSave("saved"); setSel(null);
      commit(`Vorgabe: ${label}`);
    } catch (e) { toast((e as Error).message, "error"); }
  };

  // ---------------------------------------------------------------- Darstellung
  const exif = info?.exif ?? {};
  const exifLine = [exif.iso ? `ISO ${Math.round(exif.iso)}` : "", exif.focal_length ? `${Math.round(exif.focal_length)} mm` : "",
    exif.aperture ? `f/${exif.aperture}` : "", exif.exposure_time ? (exif.exposure_time < 1 ? `1/${Math.round(1 / exif.exposure_time)} s` : `${exif.exposure_time} s`) : ""]
    .filter(Boolean).join("   ");
  const curveHist = useMemo(() => {
    if (!histo) return null;
    const l = new Uint32Array(256);
    for (let i = 0; i < 256; i++) l[i] = Math.round(0.2126 * histo[0][i] + 0.7152 * histo[1][i] + 0.0722 * histo[2][i]);
    return curveCh === "red" ? histo[0] : curveCh === "green" ? histo[1] : curveCh === "blue" ? histo[2] : l;
  }, [histo, curveCh]);
  const paramPreview = useMemo(() => {
    if (!model || curveCh !== "param") return null;
    const t = buildCurveTable({ ...model, curves: {} }, 256);
    return Float32Array.from({ length: 256 }, (_, i) => t[i * 4 + 1]);
  }, [model, curveCh]);

  if (!cur) return null;
  const overlaySvg = model && src && box.w > 0 && (
    <svg className="dv-ov" width={box.w} height={box.h} viewBox={`0 0 ${box.w} ${box.h}`}>
      {tool === "crop" && cropDraft && (() => {
        const [x0, y0, x1, y1] = cropDraft.rect.map((v, i) => v * (i % 2 ? box.h : box.w));
        const w = x1 - x0, h = y1 - y0;
        return (
          <g>
            <path d={`M0,0H${box.w}V${box.h}H0Z M${x0},${y0}V${y1}H${x1}V${y0}Z`} fill="rgba(0,0,0,0.55)" fillRule="evenodd" />
            <rect x={x0} y={y0} width={w} height={h} fill="none" stroke="#fff" strokeWidth="1.2" />
            {[1, 2].map((k) => <g key={k} stroke="rgba(255,255,255,0.45)" strokeWidth="0.8">
              <line x1={x0 + (w * k) / 3} y1={y0} x2={x0 + (w * k) / 3} y2={y1} /><line x1={x0} y1={y0 + (h * k) / 3} x2={x1} y2={y0 + (h * k) / 3} /></g>)}
            {[[x0, y0], [x1, y0], [x1, y1], [x0, y1], [(x0 + x1) / 2, y0], [(x0 + x1) / 2, y1], [x0, (y0 + y1) / 2], [x1, (y0 + y1) / 2]].map(([x, y], i) =>
              <rect key={i} x={x - 5} y={y - 5} width="10" height="10" fill="#fff" stroke="#222" strokeWidth="1" />)}
          </g>
        );
      })()}
      {tool === "mask" && curMask?.kind === "gradient" && curMask.zero && curMask.full && (() => {
        const z = imgToView(view, curMask.zero), f = imgToView(view, curMask.full);
        const Z: Pt = [z[0] * box.w, z[1] * box.h], F: Pt = [f[0] * box.w, f[1] * box.h], M: Pt = [(Z[0] + F[0]) / 2, (Z[1] + F[1]) / 2];
        const dx = F[0] - Z[0], dy = F[1] - Z[1], len = Math.hypot(dx, dy) || 1, L = (box.w + box.h) * 2;
        const px = (-dy / len) * L, py = (dx / len) * L;
        const ln = (p: Pt, dash?: string) => <line x1={p[0] - px} y1={p[1] - py} x2={p[0] + px} y2={p[1] + py} stroke="#fff" strokeWidth="1.3" strokeDasharray={dash} />;
        return <g>{ln(Z, "5 4")}{ln(M)}{ln(F, "5 4")}
          <circle cx={Z[0]} cy={Z[1]} r="6" className="dv-h" /><circle cx={F[0]} cy={F[1]} r="6" className="dv-h" /><circle cx={M[0]} cy={M[1]} r="7" className="dv-h main" /></g>;
      })()}
      {tool === "mask" && curMask?.kind === "radial" && curMask.box && (() => {
        const [bx0, by0, bx1, by1] = curMask.box;
        const pts = Array.from({ length: 73 }, (_, i) => {
          const a = (i / 72) * Math.PI * 2;
          const v = imgToView(view, [(bx0 + bx1) / 2 + Math.cos(a) * (bx1 - bx0) / 2, (by0 + by1) / 2 + Math.sin(a) * (by1 - by0) / 2]);
          return `${v[0] * box.w},${v[1] * box.h}`;
        }).join(" ");
        const hs: Pt[] = [[(bx0 + bx1) / 2, (by0 + by1) / 2], [bx1, (by0 + by1) / 2], [bx0, (by0 + by1) / 2], [(bx0 + bx1) / 2, by0], [(bx0 + bx1) / 2, by1]];
        return <g><polyline points={pts} fill="none" stroke="#fff" strokeWidth="1.3" />
          {hs.map((p, i) => { const v = imgToView(view, p); return <circle key={i} cx={v[0] * box.w} cy={v[1] * box.h} r={i ? 5 : 7} className={`dv-h ${i ? "" : "main"}`} />; })}</g>;
      })()}
      {objBox && (
        <rect x={Math.min(objBox[0][0], objBox[1][0]) * box.w} y={Math.min(objBox[0][1], objBox[1][1]) * box.h}
          width={Math.abs(objBox[1][0] - objBox[0][0]) * box.w} height={Math.abs(objBox[1][1] - objBox[0][1]) * box.h}
          fill="rgba(235,40,60,0.18)" stroke="#fff" strokeDasharray="5 4" strokeWidth="1.3" />
      )}
      {tool === "mask" && curMask?.kind === "brush" && cursor && !objMode && (
        <circle cx={cursor[0] * box.w} cy={cursor[1] * box.h} r={brush.size * Math.max(box.w / (view.rect[2] - view.rect[0]), box.h / (view.rect[3] - view.rect[1]))}
          fill="none" stroke={brush.erase ? "#ff8080" : "#fff"} strokeWidth="1.2" />
      )}
    </svg>
  );

  const stageCursor = pipette || dnPick || (tool === "mask" && objMode) ? "crosshair" : tool === "mask" && curMask && curMask.kind !== "other" && !["subject", "background", "sky", "person"].includes(curMask.kind)
    ? (curMask.kind === "brush" ? "none" : "crosshair") : tool === "crop" ? "move" : zoom ? "grab" : "zoom-in";
  const saveLabel = { saved: "Gespeichert", dirty: "Geändert …", saving: "Speichert …", error: "Speichern fehlgeschlagen" }[save];

  return (
    <div className="dv">
      {/* ---------------------------------------------------------------- links */}
      <aside className="dv-left">
        <div className="dv-nav">
          <img src={hoverStyle ? `${BASE}/api/images/${iid}/styled?style=${encodeURIComponent(hoverStyle)}&size=360` : `${BASE}/api/images/${iid}/thumb`} draggable={false} />
          <div className="dv-nav-l">{hoverStyle ? "Vorschau der Vorgabe" : "Navigator"}</div>
        </div>
        <Panel id="presets" title="Vorgaben" right={<button className="dv-iconbtn sm" onClick={importPresets} title="Lightroom-Vorgaben importieren (.xmp, .lrtemplate)"><Ic.IcPlus size={15} /></button>}>
          <div className="dv-list">
            {[...new Set(styles.map((x) => x.group ?? ""))].map((gname) => (
              <div key={gname} className="dv-pgroup">
                <div className="dv-pgroup-t">{gname || "Weitere"}</div>
                {styles.filter((x) => (x.group ?? "") === gname).map((x) => (
                  <div key={x.key} className="dv-preset" onMouseEnter={() => setHoverStyle(x.key)} onMouseLeave={() => setHoverStyle(null)}>
                    <button onClick={() => applyStyle(x.key, x.label)}><Ic.IcPreset size={14} /> {x.label}</button>
                    {x.user && <button className="dv-pdel" title="Vorgabe löschen" onClick={async () => {
                      await api.del(`/api/presets/user/${encodeURIComponent(x.key.slice(3))}`); loadStyles();
                    }}><Ic.IcTrash size={13} /></button>}
                  </div>
                ))}
              </div>
            ))}
            <button className="dv-import" onClick={importPresets}><Ic.IcPlus size={14} /> Lightroom-Vorgaben importieren …</button>
          </div>
        </Panel>
        <Panel id="history" title="Verlauf" right={<span className="dv-muted">{hist.length - 1}</span>}>
          <div className="dv-list dv-hist-list">
            {[...hist].map((h, i) => ({ h, i })).reverse().map(({ h, i }) => (
              <button key={i} className={i === hIdx ? "on" : i > hIdx ? "future" : ""} onClick={() => goHistory(i)}>{h.label}</button>
            ))}
          </div>
        </Panel>
        <div className="dv-left-foot">
          <button onClick={copySettings} title="Einstellungen kopieren (⇧⌘C)"><Ic.IcCopy size={15} /> Kopieren</button>
          <button onClick={pasteSettings} disabled={!clipboard} title="Einstellungen einfügen (⇧⌘V)"><Ic.IcPaste size={15} /> Einfügen</button>
        </div>
      </aside>

      {/* ---------------------------------------------------------------- Mitte */}
      <main className="dv-center">
        <div className="dv-top">
          <button className="dv-iconbtn" onClick={onClose} title="Zurück zur Bibliothek (G)"><Ic.IcGrid /> Bibliothek</button>
          <div className="dv-title"><b>{cur.filename}</b><span className="dv-muted">{index + 1} / {items.length}</span></div>
          <div className="dv-top-r">
            <span className={`dv-save ${save}`}>{save === "saved" ? <Ic.IcCheck size={14} /> : null}{saveLabel}</span>
            <button className="dv-iconbtn" disabled={hIdx === 0} onClick={() => goHistory(hIdx - 1)} title="Rückgängig (⌘Z)"><Ic.IcUndo /></button>
            <button className="dv-iconbtn" disabled={hIdx >= hist.length - 1} onClick={() => goHistory(hIdx + 1)} title="Wiederholen (⇧⌘Z)"><Ic.IcRedo /></button>
          </div>
        </div>
        <div className={`dv-stage ${zoom ? "zoomed" : ""} ${before === "split" ? "split" : ""}`} ref={stageRef}>
          {before === "split" && <div className="dv-split-l" style={{ width: box.w, height: box.h }}><img src={`${BASE}/api/images/${iid}/preview`} draggable={false} /><span>Vorher</span></div>}
          <div className="dv-canvas-wrap" style={{ width: box.w, height: box.h, cursor: stageCursor }}
            onPointerDown={onDown} onPointerMove={onMove} onPointerUp={onUp} onPointerLeave={() => setCursor(null)}>
            <canvas ref={canvasRef} style={{ width: box.w, height: box.h, display: noGpu ? "none" : undefined }} />
            {noGpu && fallbackUrl && <img src={fallbackUrl} style={{ width: box.w, height: box.h }} draggable={false} />}
            {before === "on" && <img className="dv-before" src={`${BASE}/api/images/${iid}/preview`} style={{ width: box.w, height: box.h }} draggable={false} />}
            {overlaySvg}
          </div>
          {before === "split" && <span className="dv-split-r">Nachher</span>}
          {(!src || !model) && !err && <div className="dv-loading">Lädt RAW …</div>}
          {err && <div className="dv-loading err">{err}</div>}
          {before === "on" && <div className="dv-badge">Vorher (\\)</div>}
          {pipette && <div className="dv-badge">Auf etwas Neutrales (Grau/Weiss) klicken · Esc bricht ab</div>}
          {tool === "crop" && <div className="dv-badge">Ecken ziehen · innen verschieben · aussen ziehen dreht · Esc fertig</div>}
          {tool === "mask" && curMask?.kind === "brush" && !objMode && <div className="dv-badge">Malen · Alt gedrückt löscht · [ ] Pinselgrösse</div>}
          {tool === "mask" && objMode && <div className="dv-badge">Rahmen um das Objekt ziehen – Imagomat findet die Kanten</div>}
          {dnPick && <div className="dv-badge">Auf die Stelle klicken, die du 1:1 sehen willst</div>}
          {noGpu && <div className="dv-badge warn">Grafikkarte nicht nutzbar – langsamere Vorschau</div>}
        </div>
        <div className="dv-toolbar">
          <button className={`dv-iconbtn ${before !== "off" ? "on" : ""}`} onClick={() => setBefore((b) => (b === "on" ? "off" : "on"))} title="Vorher/Nachher (\\)"><Ic.IcCompare /> Vorher/Nachher</button>
          <button className={`dv-iconbtn ${before === "split" ? "on" : ""}`} onClick={() => setBefore((b) => (b === "split" ? "off" : "split"))} title="Nebeneinander (Y)">Y|Y</button>
          <button className={`dv-iconbtn ${zoom ? "on" : ""}`} onClick={() => setZoom((z) => !z)} title="Einpassen / 1:1 (Z)"><Ic.IcZoom /> {zoom ? "1:1" : "Einpassen"}</button>
          <button className={`dv-iconbtn ${clip ? "on" : ""}`} onClick={() => setClip((c) => !c)} title="Beschnittene Lichter/Tiefen zeigen (J)"><Ic.IcClip /> Beschneidung</button>
          <span className="dv-spacer" />
          <span className="dv-muted">{exifLine}</span>
        </div>
        <div className="dv-strip">
          {items.map((it, i) => (
            <button key={it.id} id={`dvs-${it.id}`} className={`dv-thumb ${i === index ? "on" : ""} ${it.decision === "reject" ? "rej" : ""}`}
              onClick={() => go(i)} ref={i === index ? (el) => el?.scrollIntoView({ block: "nearest", inline: "nearest" }) : undefined}>
              <img loading="lazy" src={`${BASE}/api/images/${it.id}/thumb?v=${it.edit_v ?? 0}${it.rendered ? "r" : ""}`} draggable={false} />
              {(it.rating ?? 0) > 0 && <span className="dv-stars">{"★".repeat(it.rating ?? 0)}</span>}
            </button>
          ))}
        </div>
      </main>

      {/* ---------------------------------------------------------------- rechts */}
      <aside className="dv-right">
        <Histogram hist={histo} clip={clip} onClip={() => setClip((c) => !c)} />
        <div className="dv-tools">
          <button className={tool === "edit" ? "on" : ""} onClick={() => { if (tool === "crop") leaveCrop(); setTool("edit"); }} title="Bearbeiten"><Ic.IcSliders size={20} /><span>Bearbeiten</span></button>
          <button className={tool === "crop" ? "on" : ""} onClick={() => (tool === "crop" ? leaveCrop() : startCrop())} title="Zuschneiden und Begradigen (R)"><Ic.IcCrop size={20} /><span>Zuschneiden</span></button>
          <button className={tool === "mask" ? "on" : ""} onClick={() => { if (tool === "crop") leaveCrop(); setTool(tool === "mask" ? "edit" : "mask"); }} title="Masken (M)"><Ic.IcMask size={20} /><span>Masken</span></button>
        </div>
        <div className="dv-scroll">
          {model && tool === "edit" && <>
            <Panel id="basic" title="Grundeinstellungen" right={<button className="dv-auto-btn" onClick={autoTone} title="Automatisch wie in Lightroom (⌘U)">Auto</button>}>
              <div className="dv-row dv-wbrow">
                <button className={`dv-pipette ${pipette ? "on" : ""}`} onClick={() => setPipette((p) => !p)} title="Weissabgleich-Pipette (W): auf etwas Neutrales klicken"><Ic.IcPipette size={18} /></button>
                <span className="dv-lbl">WA:</span>
                <select value={!model.wb_custom ? "asshot" : "custom"} onChange={(e) => {
                  const v = e.target.value;
                  if (v === "asshot") setWB(null, null, "Weissabgleich: Wie Aufnahme");
                  else if (v === "auto" && src) { const [t, ti] = solveWB(src.wb, grayWorld(src)); setWB(t, ti, "Weissabgleich: Automatisch"); }
                  else { const p = WB_PRESETS.find(([n]) => n === v); if (p) setWB(p[1], p[2], `Weissabgleich: ${p[0]}`); }
                }}>
                  <option value="asshot">Wie Aufnahme</option>
                  <option value="auto">Automatisch</option>
                  {WB_PRESETS.map(([n]) => <option key={n} value={n}>{n}</option>)}
                  <option value="custom" disabled>Benutzerdefiniert</option>
                </select>
              </div>
              <Slider k="Temperature" label="Temp." value={tempPos(temp)} min={0} max={1000} step={1} track={T_TEMP}
                fmt={() => `${Math.round(temp)}`} onChange={setTempPos} onCommit={commitWB} dflt={tempPos(asShot[0])} />
              <Slider k="Tint" label="Tönung" value={tint} min={-150} max={150} track={T_TINT} onChange={setTint} onCommit={commitWB} dflt={asShot[1]} />
              <Group title="Ton">
                {S("Exposure2012", -5, 5, 0.01, T_LIGHT)}
                {S("Contrast2012", -100, 100)}
                {S("Highlights2012", -100, 100)}
                {S("Shadows2012", -100, 100)}
                {S("Whites2012", -100, 100)}
                {S("Blacks2012", -100, 100)}
              </Group>
              <Group title="Präsenz">
                {S("Texture", -100, 100)}
                {S("Clarity2012", -100, 100)}
                {S("Dehaze", -100, 100)}
                {S("Vibrance", -100, 100, 1, T_SAT)}
                {S("Saturation", -100, 100, 1, T_SAT)}
              </Group>
            </Panel>
            <Panel id="curve" title="Gradationskurve" open={false}>
              <div className="dv-seg">
                {([["param", "Parametrisch"], ["main", "RGB"], ["red", "R"], ["green", "G"], ["blue", "B"]] as const).map(([k, l]) => (
                  <button key={k} className={`${curveCh === k ? "on" : ""} c-${k}`} onClick={() => setCurveCh(k)}>{l}</button>
                ))}
              </div>
              <CurveEditor pts={curveCh === "param" ? undefined : model.curves[curveCh]} preview={paramPreview} hist={curveHist}
                color={{ param: "#ddd", main: "#eee", red: "#ef5f5a", green: "#52c46b", blue: "#5b8cff" }[curveCh]}
                onChange={(p) => curveCh !== "param" && update((m) => ({ ...m, curves: { ...m.curves, [curveCh]: p } }))}
                onCommit={() => commit(`Gradationskurve ${curveCh === "main" ? "RGB" : curveCh.toUpperCase()}`)} />
              {curveCh === "param" ? <>
                {S("ParametricHighlights", -100, 100)}
                {S("ParametricLights", -100, 100)}
                {S("ParametricDarks", -100, 100)}
                {S("ParametricShadows", -100, 100)}
              </> : <div className="dv-hint">Klicken fügt einen Punkt hinzu · Punkt aus dem Feld ziehen oder doppelklicken löscht ·
                <button className="link" onClick={() => { update((m) => ({ ...m, curves: { ...m.curves, [curveCh]: undefined } })); commit("Kurve zurückgesetzt"); }}>Zurücksetzen</button></div>}
            </Panel>
            <Panel id="hsl" title="Farbmischer (HSL)" open={false}>
              <div className="dv-seg">
                {([["Hue", "Farbton"], ["Saturation", "Sättigung"], ["Luminance", "Luminanz"], ["all", "Alle"]] as const).map(([k, l]) => (
                  <button key={k} className={hslTab === k ? "on" : ""} onClick={() => setHslTab(k)}>{l}</button>
                ))}
              </div>
              {(hslTab === "all" ? (["Hue", "Saturation", "Luminance"] as const) : [hslTab]).map((t) => (
                <Group key={t} title={hslTab === "all" ? { Hue: "Farbton", Saturation: "Sättigung", Luminance: "Luminanz" }[t] : undefined}>
                  {HSL.map(([c, l, col], i) => {
                    const prev = HSL[(i + 7) % 8][2], next = HSL[(i + 1) % 8][2];
                    const track = t === "Hue" ? `linear-gradient(90deg,${prev},${col},${next})` : t === "Saturation" ? `linear-gradient(90deg,#8a8a8a,${col})` : `linear-gradient(90deg,#111,${col},#fff)`;
                    return <Slider key={c} k={`${t}Adjustment${c}`} label={l} value={g(`${t}Adjustment${c}`)} min={-100} max={100} track={track} onChange={setG} onCommit={commitG} />;
                  })}
                </Group>
              ))}
            </Panel>
            <Panel id="grading" title="Color Grading" open={false}>
              <div className="dv-wheels">
                {([["Mitteltöne", "ColorGradeMidtoneHue", "ColorGradeMidtoneSat", "ColorGradeMidtoneLum"],
                  ["Tiefen", "SplitToningShadowHue", "SplitToningShadowSaturation", "ColorGradeShadowLum"],
                  ["Lichter", "SplitToningHighlightHue", "SplitToningHighlightSaturation", "ColorGradeHighlightLum"],
                  ["Global", "ColorGradeGlobalHue", "ColorGradeGlobalSat", "ColorGradeGlobalLum"]] as const).map(([l, hk, sk, lk]) => (
                  <div key={l} className="dv-wheel-box">
                    <div className="dv-gtitle">{l}</div>
                    <ColorWheel hue={g(hk)} sat={g(sk)} onChange={(h, s) => update((m) => ({ ...m, global: { ...m.global, [hk]: h, [sk]: s } }))}
                      onCommit={() => commit(`Color Grading ${l}`)} />
                    <Slider k={lk} label="Lum." value={g(lk)} min={-100} max={100} track={T_LIGHT} onChange={setG} onCommit={commitG} />
                  </div>
                ))}
              </div>
              {S("ColorGradeBlending", 0, 100)}
              {S("SplitToningBalance", -100, 100)}
            </Panel>
            <Panel id="detail" title="Details" open={false}>
              <Group title="Schärfen">
                {S("Sharpness", 0, 150, 1, undefined, "Betrag")}
                {S("SharpenRadius", 0.5, 3, 0.1)}
                {S("SharpenDetail", 0, 100)}
              </Group>
              <Group title="Rauschreduzierung">
                {S("LuminanceSmoothing", 0, 100, 1, undefined, "Luminanz")}
                {S("ColorNoiseReduction", 0, 100, 1, undefined, "Farbe")}
              </Group>
              <Group title="KI-Entrauschen">
                <label className="check"><input type="checkbox" checked={!!model.denoise} onChange={(e) => {
                  update((m) => ({ ...m, denoise: e.target.checked ? (m.denoise || 50) : null })); commit(e.target.checked ? "KI-Entrauschen an" : "KI-Entrauschen aus");
                }} /> Entrauschen (KI, wie Lightroom „Entrauschen“)</label>
                {!!model.denoise && <>
                  <Slider k="denoise" label="Stärke" value={model.denoise} min={1} max={100} dflt={50} fmt={(v) => `${Math.round(v)}`}
                    onChange={(_k, v) => update((m) => ({ ...m, denoise: Math.round(v) }))} onCommit={(_k, v) => commit(`KI-Entrauschen ${Math.round(v)}`)} />
                  <div className="dv-btnrow">
                    <button onClick={() => runDnPreview([0.5, 0.5])} disabled={dnBusy}>{dnBusy ? "rechnet …" : "Vorschau 1:1 (Mitte)"}</button>
                    <button className={dnPick ? "on" : ""} onClick={() => setDnPick((x) => !x)} disabled={dnBusy}>Stelle wählen</button>
                  </div>
                  {dnPreview && <div className="dv-dn"><img src={dnPreview.url} /><div className="dv-dn-l"><span>vorher</span><span>nachher ({dnPreview.method === "nafnet" ? "KI" : "klassisch"})</span></div></div>}
                </>}
                <div className="dv-hint">Beim Export entsteht wie in Lightroom eine „…-Enhanced-NR.dng“ mit deinen Einstellungen. Hohe ISO ist schon vorgeschlagen.</div>
              </Group>
            </Panel>
            <Panel id="effects" title="Effekte" open={false}>
              <Group title="Vignettierung nach Freistellen">
                {S("PostCropVignetteAmount", -100, 100, 1, undefined, "Stärke")}
                {S("PostCropVignetteMidpoint", 0, 100)}
                {S("PostCropVignetteFeather", 0, 100)}
              </Group>
            </Panel>
          </>}

          {model && tool === "crop" && cropDraft && (
            <Panel id="crop" title="Freistellen und Begradigen">
              <div className="dv-row">
                <span className="dv-lbl">Seitenverh.</span>
                <select value={String(aspect)} onChange={(e) => setCropAspect(e.target.value === "null" ? null : parseFloat(e.target.value))}>
                  {ASPECTS.map(([l, v]) => <option key={l} value={String(v)}>{l}</option>)}
                </select>
              </div>
              <Slider k="CropAngle" label="Winkel" value={Math.round(cropDraft.ang * 100) / 100} min={-45} max={45} step={0.05}
                fmt={(v) => `${v.toFixed(2)}°`} dflt={0}
                onChange={(_k, v) => applyCrop(cropDraft.rect, v)} onCommit={() => commit("Begradigen")} />
              <div className="dv-btnrow">
                <button onClick={() => { applyCrop([0, 0, 1, 1], 0); commit("Zuschnitt zurückgesetzt"); }}><Ic.IcReset size={15} /> Zurücksetzen</button>
                <button className="primary" onClick={leaveCrop}><Ic.IcCheck size={15} /> Fertig</button>
              </div>
              <div className="dv-hint">Zuschnitt und Winkel gehen 1:1 nach Lightroom. Beim Übertragen auf alle Bilder richtet jedes Bild sich selbst gerade.</div>
            </Panel>
          )}

          {model && tool === "mask" && <>
            <Panel id="masknew" title="Neue Maske">
              <div className="dv-masknew">
                {info?.has_subject && <button onClick={() => addMask("subject")}><Ic.IcSubject size={20} /><span>Motiv</span></button>}
                <button onClick={() => addMask("sky")}><Ic.IcSky size={20} /><span>Himmel</span></button>
                {info?.has_subject && <button onClick={() => addMask("background")}><Ic.IcBackground size={20} /><span>Hintergrund</span></button>}
                {info?.has_subject && <button onClick={() => addMask("person")}><Ic.IcPeople size={20} /><span>Personen</span></button>}
                <button onClick={() => addMask("object")} title="Rahmen um ein Objekt ziehen: die Kanten werden automatisch gefunden"><Ic.IcWand size={20} /><span>Objekt</span></button>
                <button onClick={() => addMask("brush")}><Ic.IcBrush size={20} /><span>Pinsel</span></button>
                <button onClick={() => addMask("gradient")}><Ic.IcLinear size={20} /><span>Linear</span></button>
                <button onClick={() => addMask("radial")}><Ic.IcRadial size={20} /><span>Radial</span></button>
              </div>
            </Panel>
            <Panel id="masks" title={`Masken (${model.masks.length})`} right={
              <button className={`dv-iconbtn sm ${overlay ? "on" : ""}`} onClick={() => setOverlay((o) => !o)} title="Überlagerung zeigen (O)"><Ic.IcEye size={15} /></button>}>
              <div className="dv-masklist">
                {model.masks.map((m, i) => {
                  const Icon = KIND_ICON[m.kind];
                  return (
                    <div key={i} className={`dv-mask ${sel === i ? "on" : ""}`} onClick={() => setSel(sel === i ? null : i)}
                      onMouseEnter={() => setHover(i)} onMouseLeave={() => setHover(null)}>
                      <Icon size={16} />
                      <input value={m.name} onClick={(e) => e.stopPropagation()} onChange={(e) => setMask(i, { name: e.target.value })}
                        onBlur={() => commit("Maske umbenannt")} />
                      <span className="dv-muted">{KIND_LABEL[m.kind]}</span>
                      <button className="dv-iconbtn sm" onClick={(e) => { e.stopPropagation(); delMask(i); }} title="Löschen"><Ic.IcTrash size={14} /></button>
                    </div>
                  );
                })}
                {model.masks.length === 0 && <div className="dv-muted">Oben eine Maske wählen: Motiv, Himmel, Pinsel, Verläufe …</div>}
              </div>
            </Panel>
            {curMask && sel !== null && (
              <Panel id="maskedit" title={curMask.name}>
                {curMask.kind === "brush" && <Group title="Pinsel">
                  <div className="dv-btnrow">
                    <button className={!brush.erase ? "on" : ""} onClick={() => setBrush((b) => ({ ...b, erase: false }))}><Ic.IcBrush size={15} /> Malen</button>
                    <button className={brush.erase ? "on" : ""} onClick={() => setBrush((b) => ({ ...b, erase: true }))}><Ic.IcErase size={15} /> Löschen</button>
                  </div>
                  <Slider k="bsize" label="Grösse" value={Math.round(brush.size * 1000)} min={5} max={300} dflt={40} fmt={(v) => `${v}`}
                    onChange={(_k, v) => setBrush((b) => ({ ...b, size: v / 1000 }))} onCommit={() => undefined} />
                  <Slider k="bfeather" label="Weiche Kante" value={curMask.feather ?? 50} min={0} max={100} dflt={50} fmt={(v) => `${Math.round(v)}`}
                    onChange={(_k, v) => { setBrush((b) => ({ ...b, feather: v })); setMask(sel, { feather: v }); }} onCommit={() => commit("Pinsel: weiche Kante")} />
                </Group>}
                {curMask.kind === "radial" && <>
                  <Slider k="feather" label="Weiche Kante" value={curMask.feather ?? 60} min={0} max={100} dflt={60} fmt={(v) => `${Math.round(v)}`}
                    onChange={(_k, v) => setMask(sel, { feather: v })} onCommit={() => commit("Radial: weiche Kante")} />
                  <label className="check"><input type="checkbox" checked={!!curMask.invert} onChange={(e) => { setMask(sel, { invert: e.target.checked }); commit("Maske umkehren"); }} /> Umkehren (wirkt aussen)</label>
                </>}
                <Slider k="amount" label="Stärke" value={Math.round((curMask.amount ?? 1) * 100)} min={0} max={100} dflt={100} fmt={(v) => `${Math.round(v)}`}
                  onChange={(_k, v) => setMask(sel, { amount: v / 100 })} onCommit={() => commit("Maske Stärke")} />
                {LOCAL.map(([k, l, mn, mx, st, tr]) => (
                  <Slider key={k} k={k} label={l} value={curMask.local[k] ?? 0} min={mn} max={mx} step={st ?? 1} track={tr} dflt={0}
                    onChange={setLocal} onCommit={commitLocal} />
                ))}
                <div className="dv-btnrow">
                  <button onClick={() => { setMask(sel, { local: {} }); commit("Maske Regler zurückgesetzt"); }}><Ic.IcReset size={15} /> Regler zurück</button>
                </div>
              </Panel>
            )}
          </>}
        </div>
        <div className="dv-right-foot">
          <button className="primary dv-sync" onClick={() => setSyncOpen(true)} disabled={!model} title="Diesen Edit auf alle Bilder des Shoots übertragen (⇧⌘S)">
            <Ic.IcSync size={16} /> Auf alle übertragen …</button>
          <div className="dv-btnrow">
            <button onClick={usePrevious} disabled={!previous} title="Einstellungen des vorher bearbeiteten Bildes">Vorherige</button>
            <button onClick={resetAll} title="Alle Regler auf 0 wie frisch importiert (⌘Z macht es rückgängig)">Zurücksetzen</button>
          </div>
        </div>
      </aside>

      {syncOpen && model && iid && (
        <SyncDialog filename={cur.filename} count={items.length} onClose={() => setSyncOpen(false)}
          onGo={async (name, overwrite) => {
            try {
              await flush();
              const r = await api.post<{ job_id: number; template: string }>(`/api/images/${iid}/editor/sync`, { model: modelRef.current, name, overwrite_manual: overwrite });
              savedJson.current = JSON.stringify({ model: modelRef.current });
              setSave("saved");
              setSyncOpen(false);
              toast(`„${r.template}“ wird auf alle Bilder übertragen …`);
              onSynced(r.job_id);
            } catch (e) { toast((e as Error).message, "error"); }
          }} />
      )}
    </div>
  );
}

function SyncDialog({ filename, count, onClose, onGo }: { filename: string; count: number; onClose: () => void; onGo: (name: string, overwrite: boolean) => void }) {
  const [name, setName] = useState(`Edit ${filename.replace(/\.[^.]+$/, "")}`);
  const [overwrite, setOverwrite] = useState(false);
  const [busy, setBusy] = useState(false);
  return (
    <Modal title="Auf alle Bilder übertragen" onClose={onClose}>
      <p>Dieser Edit gilt 1:1 für alle Bilder des Shoots: Regler, Kurven, Farben und Masken (Motiv, Himmel, Verläufe, Pinsel).
        Pro Bild passt Imagomat nur Belichtung, Weissabgleich und Begradigen an, damit jedes Bild gleich aussieht.</p>
      <label>Name (wird als Vorlage gespeichert, später auch für andere Shoots wählbar)</label>
      <input value={name} autoFocus onChange={(e) => setName(e.target.value)} onKeyDown={(e) => { e.stopPropagation(); if (e.key === "Enter" && name.trim()) { setBusy(true); onGo(name.trim(), overwrite); } }} />
      <label className="check"><input type="checkbox" checked={overwrite} onChange={(e) => setOverwrite(e.target.checked)} /> auch Bilder überschreiben, die ich schon von Hand bearbeitet habe</label>
      <div className="hint">{count} Bilder in dieser Ansicht · das aktuelle Bild bleibt genau so, wie es ist.</div>
      <div className="modal-foot">
        <button className="ghost" onClick={onClose}>Abbrechen</button>
        <button className="primary" disabled={!name.trim() || busy} onClick={() => { setBusy(true); onGo(name.trim(), overwrite); }}>Übertragen</button>
      </div>
    </Modal>
  );
}
