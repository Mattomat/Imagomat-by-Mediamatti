// Bausteine der rechten Leiste: Regler, aufklappbare Bereiche, Gradationskurve, Farbräder, Histogramm.
import { memo, ReactNode, useEffect, useRef, useState } from "react";
import { curveLut, def, fmtValue, Pt } from "./model";

export type OnVal = (k: string, v: number) => void;

export const Slider = memo(function Slider({ k, label, value, min, max, step = 1, onChange, onCommit, track, dflt, fmt, disabled }: {
  k: string; label: string; value: number; min: number; max: number; step?: number; onChange: OnVal; onCommit: OnVal;
  track?: string; dflt?: number; fmt?: (v: number) => string; disabled?: boolean;
}) {
  const [edit, setEdit] = useState<string | null>(null);
  const d = dflt ?? def(k);
  const reset = () => { onChange(k, d); onCommit(k, d); };
  const shown = fmt ? fmt(value) : fmtValue(k, value);
  return (
    <div className={`dv-sl ${disabled ? "off" : ""}`}>
      <span className="dv-sl-l" onDoubleClick={reset} title="Doppelklick: zurücksetzen">{label}</span>
      <input type="range" min={min} max={max} step={step} value={value} disabled={disabled}
        style={track ? { background: track } : undefined} className={track ? "tracked" : ""}
        onChange={(e) => onChange(k, parseFloat(e.target.value))}
        onPointerUp={(e) => onCommit(k, parseFloat((e.target as HTMLInputElement).value))}
        onKeyUp={(e) => onCommit(k, parseFloat((e.target as HTMLInputElement).value))}
        onDoubleClick={reset} />
      {edit === null
        ? <span className="dv-sl-v" onClick={() => !disabled && setEdit(shown.replace("+", ""))}>{shown}</span>
        : <input className="dv-sl-in" autoFocus value={edit} onChange={(e) => setEdit(e.target.value)} onFocus={(e) => e.target.select()}
            onBlur={() => setEdit(null)}
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                const v = parseFloat(edit.replace(",", "."));
                if (!Number.isNaN(v)) { const c = Math.min(max, Math.max(min, v)); onChange(k, c); onCommit(k, c); }
                setEdit(null);
              } else if (e.key === "Escape") setEdit(null);
              e.stopPropagation();
            }} />}
    </div>
  );
});

export function Panel({ id, title, children, right, open: dflt = true }: { id: string; title: string; children: ReactNode; right?: ReactNode; open?: boolean }) {
  const key = `imagomat.panel.${id}`;
  const [open, setOpen] = useState(() => { try { const v = localStorage.getItem(key); return v === null ? dflt : v === "1"; } catch { return dflt; } });
  const toggle = () => setOpen((o) => { try { localStorage.setItem(key, o ? "0" : "1"); } catch { /* egal */ } return !o; });
  return (
    <section className="dv-panel">
      <header onClick={toggle}>
        <span className={`dv-tri ${open ? "open" : ""}`}>▸</span>
        <span className="dv-ptitle">{title}</span>
        <span className="dv-pright" onClick={(e) => e.stopPropagation()}>{right}</span>
      </header>
      {open && <div className="dv-pbody">{children}</div>}
    </section>
  );
}

export function Group({ title, children }: { title?: string; children: ReactNode }) {
  return <div className="dv-group">{title && <div className="dv-gtitle">{title}</div>}{children}</div>;
}

// ------------------------------------------------------------------ Gradationskurve
export function CurveEditor({ pts, color, onChange, onCommit, hist, preview }: {
  pts: Pt[] | undefined; color: string; onChange: (p: Pt[]) => void; onCommit: () => void;
  hist?: Uint32Array | null; preview?: Float32Array | null;
}) {
  const ref = useRef<SVGSVGElement>(null);
  const drag = useRef<number | null>(null);
  const cur: Pt[] = pts && pts.length >= 2 ? pts : [[0, 0], [255, 255]];
  const lut = preview ?? curveLut(cur);
  const path = Array.from(lut, (v, i) => `${i === 0 ? "M" : "L"}${(i / (lut.length - 1)) * 255},${255 - v * 255}`).join(" ");
  const at = (e: { clientX: number; clientY: number }): Pt => {
    const r = ref.current!.getBoundingClientRect();
    return [Math.round(((e.clientX - r.left) / r.width) * 255), Math.round(255 - ((e.clientY - r.top) / r.height) * 255)];
  };
  const down = (e: React.PointerEvent) => {
    if (preview) return;
    const [x, y] = at(e);
    let i = cur.findIndex(([px, py]) => Math.hypot(px - x, py - y) < 12);
    let next = cur;
    if (i < 0) {
      const ny = Math.round(lut[Math.min(lut.length - 1, Math.max(0, Math.round((x / 255) * (lut.length - 1))))] * 255);
      next = [...cur, [x, Math.abs(ny - y) < 24 ? ny : y] as Pt].sort((a, b) => a[0] - b[0]);
      i = next.findIndex((p) => p[0] === x);
      onChange(next);
    }
    drag.current = i;
    ref.current!.setPointerCapture(e.pointerId);
  };
  const move = (e: React.PointerEvent) => {
    const i = drag.current;
    if (i === null) return;
    let [x, y] = at(e);
    const out = y < -30 || y > 285 || x < -30 || x > 285;
    if (out && i > 0 && i < cur.length - 1) {          // aus dem Feld ziehen: Punkt löschen
      onChange(cur.filter((_, j) => j !== i)); drag.current = null; return;
    }
    const lo = i > 0 ? cur[i - 1][0] + 1 : 0, hi = i < cur.length - 1 ? cur[i + 1][0] - 1 : 255;
    x = Math.min(hi, Math.max(lo, x)); y = Math.min(255, Math.max(0, y));
    onChange(cur.map((p, j) => (j === i ? [x, y] : p)));
  };
  const up = () => { if (drag.current !== null) onCommit(); drag.current = null; };
  let histPath = "";
  if (hist) {
    const mx = Math.max(1, ...Array.from(hist).slice(2, 254));
    histPath = "M0,255 " + Array.from(hist, (v, i) => `L${i},${255 - Math.min(1, v / mx) * 200}`).join(" ") + " L255,255 Z";
  }
  return (
    <svg ref={ref} className="dv-curve" viewBox="-4 -4 263 263" onPointerDown={down} onPointerMove={move} onPointerUp={up}
      onDoubleClick={(e) => {
        const [x, y] = at(e);
        const i = cur.findIndex(([px, py]) => Math.hypot(px - x, py - y) < 12);
        if (i > 0 && i < cur.length - 1) { onChange(cur.filter((_, j) => j !== i)); onCommit(); }
      }}>
      <rect x="0" y="0" width="255" height="255" className="dv-curve-bg" />
      {histPath && <path d={histPath} className="dv-curve-hist" />}
      {[64, 128, 191].map((v) => <g key={v}><line x1={v} y1="0" x2={v} y2="255" /><line x1="0" y1={v} x2="255" y2={v} /></g>)}
      <line x1="0" y1="255" x2="255" y2="0" className="dv-curve-diag" />
      <path d={path} fill="none" stroke={color} strokeWidth="2" />
      {!preview && cur.map(([x, y], i) => <circle key={i} cx={x} cy={255 - y} r="4.5" className="dv-curve-pt" />)}
    </svg>
  );
}

// ------------------------------------------------------------------ Farbrad (Color Grading)
export function ColorWheel({ hue, sat, size = 92, onChange, onCommit }: {
  hue: number; sat: number; size?: number; onChange: (h: number, s: number) => void; onCommit: () => void;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const active = useRef(false);
  const set = (e: React.PointerEvent) => {
    const r = ref.current!.getBoundingClientRect();
    const x = (e.clientX - r.left) / r.width * 2 - 1, y = (e.clientY - r.top) / r.height * 2 - 1;
    const s = Math.min(1, Math.hypot(x, y));
    const h = ((Math.atan2(-y, x) * 180) / Math.PI + 360) % 360;
    onChange(Math.round(h), Math.round(s * 100));
  };
  const a = (hue * Math.PI) / 180, rr = sat / 100;
  const stops = Array.from({ length: 13 }, (_, i) => `hsl(${(((90 - i * 30) % 360) + 360) % 360} 90% 55%) ${i * 30}deg`).join(",");
  return (
    <div ref={ref} className="dv-wheel" style={{ width: size, height: size, background: `radial-gradient(circle, #8a8a8a 0%, rgba(138,138,138,0) 72%), conic-gradient(${stops})` }}
      onPointerDown={(e) => { active.current = true; ref.current!.setPointerCapture(e.pointerId); set(e); }}
      onPointerMove={(e) => active.current && set(e)}
      onPointerUp={() => { if (active.current) onCommit(); active.current = false; }}
      onDoubleClick={() => { onChange(0, 0); onCommit(); }}>
      <div className="dv-wheel-dot" style={{ left: `${50 + Math.cos(a) * rr * 50}%`, top: `${50 - Math.sin(a) * rr * 50}%` }} />
    </div>
  );
}

// ------------------------------------------------------------------ Histogramm
export function Histogram({ hist, clip, onClip }: {
  hist: [Uint32Array, Uint32Array, Uint32Array] | null; clip: boolean; onClip: () => void;
}) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const c = ref.current;
    if (!c) return;
    const ctx = c.getContext("2d")!;
    const W = c.width, H = c.height;
    ctx.clearRect(0, 0, W, H);
    if (!hist) return;
    let mx = 1;
    for (const ch of hist) for (let i = 3; i < 253; i++) mx = Math.max(mx, ch[i]);
    ctx.globalCompositeOperation = "lighter";
    const cols = ["rgba(220,60,60,0.75)", "rgba(60,200,80,0.75)", "rgba(70,110,235,0.8)"];
    hist.forEach((ch, ci) => {
      ctx.fillStyle = cols[ci];
      ctx.beginPath();
      ctx.moveTo(0, H);
      for (let i = 0; i < 256; i++) ctx.lineTo((i / 255) * W, H - Math.min(1, Math.sqrt(ch[i] / mx)) * (H - 4));
      ctx.lineTo(W, H);
      ctx.closePath();
      ctx.fill();
    });
    ctx.globalCompositeOperation = "source-over";
  }, [hist]);
  const tot = hist ? hist[1].reduce((a, b) => a + b, 0) : 1;
  const hiClip = hist ? (hist[0][255] + hist[1][255] + hist[2][255]) / (3 * tot) > 0.002 : false;
  const loClip = hist ? (hist[0][0] + hist[1][0] + hist[2][0]) / (3 * tot) > 0.002 : false;
  return (
    <div className="dv-hist">
      <canvas ref={ref} width={560} height={170} />
      <button className={`dv-clip lo ${loClip ? "warn" : ""} ${clip ? "on" : ""}`} onClick={onClip} title="Tiefen-Beschneidung zeigen (J)">◤</button>
      <button className={`dv-clip hi ${hiClip ? "warn" : ""} ${clip ? "on" : ""}`} onClick={onClip} title="Lichter-Beschneidung zeigen (J)">◥</button>
    </div>
  );
}
