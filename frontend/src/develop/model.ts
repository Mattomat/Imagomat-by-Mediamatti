// Editor-Modell (wie im Backend style/editor.py) und die Rechenhilfen, die der GPU-Renderer braucht.

export type Pt = [number, number];
export type MaskKind = "subject" | "background" | "sky" | "person" | "gradient" | "radial" | "brush" | "other";
export interface EdMask {
  name: string;
  kind: MaskKind;
  local: Record<string, number>;
  amount?: number;
  zero?: Pt;
  full?: Pt;
  box?: [number, number, number, number];
  feather?: number;
  invert?: boolean;
  dabs?: [number, number, number][];
  raw?: unknown;
}
export interface Crop {
  HasCrop: boolean;
  CropLeft: number;
  CropTop: number;
  CropRight: number;
  CropBottom: number;
  CropAngle: number;
}
export interface Curves { main?: Pt[]; red?: Pt[]; green?: Pt[]; blue?: Pt[] }
export interface RetouchOp { id: string; mode: "remove" | "heal"; dabs: [number, number, number][]; feather: number; variant: number }
export interface EdModel {
  global: Record<string, number>;
  wb_custom: boolean;
  masks: EdMask[];
  curves: Curves;
  crop: Crop;
  denoise?: number | null;      // KI-Entrauschen (Stärke) oder aus
  retouch?: RetouchOp[];         // Entfernen / Reparieren
  profile?: string;              // "Camera Standard" (wie Original) | "Adobe Color"
}

export const DEFAULTS: Record<string, number> = {
  ColorNoiseReduction: 25, Sharpness: 40, SharpenRadius: 1, SharpenDetail: 25, ColorGradeBlending: 50,
  ParametricShadowSplit: 25, ParametricMidtoneSplit: 50, ParametricHighlightSplit: 75,
  PostCropVignetteMidpoint: 50, PostCropVignetteFeather: 50, LuminanceNoiseReductionDetail: 50,
  ColorNoiseReductionDetail: 50,
};
export const def = (k: string) => DEFAULTS[k] ?? 0;
export const NO_CROP: Crop = { HasCrop: false, CropLeft: 0, CropTop: 0, CropRight: 1, CropBottom: 1, CropAngle: 0 };

export function normalize(m: EdModel): EdModel {
  return { global: m.global ?? {}, wb_custom: !!m.wb_custom, masks: m.masks ?? [], curves: m.curves ?? {},
    crop: { ...NO_CROP, ...(m.crop ?? {}) }, denoise: m.denoise ?? null, retouch: m.retouch ?? [], profile: m.profile || "Camera Standard" };
}

// ------------------------------------------------------------------ Quelle (lineare RAW-Daten vom Server)
export interface WBTable { mireds: number[]; tints: number[]; mult: number[]; base: number[] }
export interface Source {
  w: number; h: number; floor: number[]; m: number[]; gain: number; wb: WBTable; as_shot: [number, number];
  source: string; orientation: number; data: Uint16Array; cam_curve?: number[][] | null;
  cam_ratio?: { w: number; h: number; data: number[] } | null;
}

export function parseSource(buf: ArrayBuffer): Source {
  const n = new DataView(buf).getUint32(0, true);
  const head = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 4, n)));
  return { ...head, data: new Uint16Array(buf, 4 + n, head.w * head.h * 3) };
}

export function half(h: number): number {
  const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 0x1f, f = h & 0x3ff;
  if (e === 0) return s * 6.103515625e-5 * (f / 1024);
  if (e === 31) return f ? NaN : s * Infinity;
  return s * Math.pow(2, e - 15) * (1 + f / 1024);
}

const _f32 = new Float32Array(1), _u32 = new Uint32Array(_f32.buffer);
export function toHalf(v: number): number {
  _f32[0] = v;
  const x = _u32[0], sign = (x >> 16) & 0x8000;
  let e = ((x >> 23) & 0xff) - 127 + 15;
  const m = x & 0x7fffff;
  if (e <= 0) return e < -10 ? sign : sign | ((m | 0x800000) >> (1 - e + 13));
  if (e >= 31) return sign | 0x7c00;
  const h = sign | (e << 10) | (m >> 13);
  return (m & 0x1000) ? h + 1 : h;
}

/** Weissabgleich-Faktoren (Kamera-RGB) für Temperatur/Tönung, bilinear aus der Server-Tabelle. */
export function wbMult(t: WBTable, temp: number, tint: number): [number, number, number] {
  const mi = Math.min(Math.max(1e6 / Math.max(temp, 1), t.mireds[0]), t.mireds[t.mireds.length - 1]);
  const ti = Math.min(Math.max(tint, t.tints[0]), t.tints[t.tints.length - 1]);
  const fm = (mi - t.mireds[0]) / (t.mireds[1] - t.mireds[0]);
  const ft = (ti - t.tints[0]) / (t.tints[1] - t.tints[0]);
  const i0 = Math.min(Math.floor(fm), t.mireds.length - 2), j0 = Math.min(Math.floor(ft), t.tints.length - 2);
  const a = fm - i0, b = ft - j0, nt = t.tints.length;
  const out: [number, number, number] = [0, 0, 0];
  for (let c = 0; c < 3; c++) {
    const v = (i: number, j: number) => t.mult[(i * nt + j) * 3 + c];
    out[c] = (v(i0, j0) * (1 - b) + v(i0, j0 + 1) * b) * (1 - a) + (v(i0 + 1, j0) * (1 - b) + v(i0 + 1, j0 + 1) * b) * a;
  }
  return out;
}

/** Temperatur/Tönung, bei der die Kamerafarbe ``p`` (linear, ohne Weissabgleich) neutral grau wird. */
export function solveWB(t: WBTable, p: [number, number, number]): [number, number] {
  const err = (temp: number, tint: number) => {
    const m = wbMult(t, temp, tint);
    const r = p[0] * m[0], g = Math.max(p[1] * m[1], 1e-9), b = p[2] * m[2];
    return Math.log(Math.max(r, 1e-9) / g) ** 2 + Math.log(Math.max(b, 1e-9) / g) ** 2;
  };
  let best: [number, number] = [5500, 0], be = Infinity;
  for (let mi = 20; mi <= 500; mi += 4)
    for (let ti = -150; ti <= 150; ti += 3) {
      const e = err(1e6 / mi, ti);
      if (e < be) { be = e; best = [1e6 / mi, ti]; }
    }
  // verfeinern
  let [bm, bt] = [1e6 / best[0], best[1]];
  for (let step = 2; step >= 0.25; step /= 2)
    for (let it = 0; it < 6; it++) {
      let moved = false;
      for (const [dm, dt] of [[step, 0], [-step, 0], [0, step], [0, -step]]) {
        const e = err(1e6 / (bm + dm), bt + dt);
        if (e < be) { be = e; bm += dm; bt += dt; moved = true; }
      }
      if (!moved) break;
    }
  return [Math.round(1e6 / bm / 10) * 10, Math.round(bt)];
}

export function samplePixel(src: Source, x: number, y: number, r = 2): [number, number, number] {
  const px = Math.round(x * (src.w - 1)), py = Math.round(y * (src.h - 1));
  const acc = [0, 0, 0];
  let n = 0;
  for (let yy = Math.max(0, py - r); yy <= Math.min(src.h - 1, py + r); yy++)
    for (let xx = Math.max(0, px - r); xx <= Math.min(src.w - 1, px + r); xx++) {
      const i = (yy * src.w + xx) * 3;
      for (let c = 0; c < 3; c++) acc[c] += half(src.data[i + c]);
      n++;
    }
  return [acc[0] / n, acc[1] / n, acc[2] / n];
}

/** Grauwelt-Schätzung (Automatik-Weissabgleich): Mittel der mittelhellen, nicht ausgefressenen Pixel. */
export function grayWorld(src: Source): [number, number, number] {
  const acc = [0, 0, 0];
  let n = 0;
  const step = Math.max(1, Math.floor(Math.sqrt((src.w * src.h) / 40000)));
  for (let y = 0; y < src.h; y += step)
    for (let x = 0; x < src.w; x += step) {
      const i = (y * src.w + x) * 3;
      const r = half(src.data[i]), g = half(src.data[i + 1]), b = half(src.data[i + 2]);
      const mx = Math.max(r, g, b);
      if (mx > 0.85 || mx < 0.01) continue;
      acc[0] += r; acc[1] += g; acc[2] += b; n++;
    }
  return n ? [acc[0] / n, acc[1] / n, acc[2] / n] : [1, 1, 1];
}

// ------------------------------------------------------------------ Gradationskurve (wie pipeline.curve_lut)
export function curveLut(pts: Pt[] | undefined, n = 256): Float32Array {
  const out = new Float32Array(n);
  const sorted = [...(pts ?? [])].sort((a, b) => a[0] - b[0]);
  const xs: number[] = [], ys: number[] = [];
  for (const [x, y] of sorted) {
    if (xs.length && x - xs[xs.length - 1] < 1e-6) { ys[ys.length - 1] = y; continue; }
    xs.push(x); ys.push(y);
  }
  for (let i = 0; i < n; i++) out[i] = i / (n - 1);
  if (xs.length < 2) return out;
  const k = xs.length;
  const d = xs.slice(0, -1).map((_, i) => (ys[i + 1] - ys[i]) / (xs[i + 1] - xs[i]));
  const m = new Array(k).fill(0);
  m[0] = d[0]; m[k - 1] = d[k - 2];
  for (let i = 1; i < k - 1; i++) m[i] = d[i - 1] * d[i] > 0 ? (d[i - 1] + d[i]) / 2 : 0;
  if (k > 2)
    for (let i = 0; i < k - 1; i++) {
      if (Math.abs(d[i]) < 1e-9) { m[i] = 0; m[i + 1] = 0; continue; }
      const a = m[i] / d[i], b = m[i + 1] / d[i], h = a * a + b * b;
      if (h > 9) { const s = 3 / Math.sqrt(h); m[i] = s * a * d[i]; m[i + 1] = s * b * d[i]; }
    }
  for (let i = 0; i < n; i++) {
    const t = (i / (n - 1)) * 255;
    let y: number;
    if (t <= xs[0]) y = ys[0];
    else if (t >= xs[k - 1]) y = ys[k - 1];
    else {
      let j = 0;
      while (j < k - 2 && t >= xs[j + 1]) j++;
      const hs = xs[j + 1] - xs[j], u = (t - xs[j]) / hs;
      if (k === 2) y = ys[j] + (ys[j + 1] - ys[j]) * u;
      else {
        const h00 = 2 * u ** 3 - 3 * u ** 2 + 1, h10 = u ** 3 - 2 * u ** 2 + u, h01 = -2 * u ** 3 + 3 * u ** 2, h11 = u ** 3 - u ** 2;
        y = h00 * ys[j] + h10 * hs * m[j] + h01 * ys[j + 1] + h11 * hs * m[j + 1];
      }
    }
    out[i] = Math.min(255, Math.max(0, y)) / 255;
  }
  return out;
}

function lutAt(lut: Float32Array, x: number): number {
  const f = Math.min(Math.max(x, 0), 1) * (lut.length - 1);
  const i = Math.min(Math.floor(f), lut.length - 2), w = f - i;
  return lut[i] * (1 - w) + lut[i + 1] * w;
}

/** Alle Kurven (Punkt RGB, Kanäle, parametrisch) als eine Tabelle pro Kanal (RGBA-Float, n Einträge). */
export function buildCurveTable(model: EdModel, n = 1024): Float32Array {
  const main = curveLut(model.curves.main), chan = [curveLut(model.curves.red), curveLut(model.curves.green), curveLut(model.curves.blue)];
  const g = model.global;
  const p = ["ParametricShadows", "ParametricDarks", "ParametricLights", "ParametricHighlights"].map((k) => (g[k] ?? 0) / 100);
  const bumps: Pt[] = [[0.125, 0.12], [0.375, 0.14], [0.625, 0.14], [0.875, 0.12]];
  const out = new Float32Array(n * 4);
  for (let i = 0; i < n; i++) {
    const x = i / (n - 1);
    const mx = lutAt(main, x);
    for (let c = 0; c < 3; c++) {
      let v = lutAt(chan[c], mx);
      if (p.some((q) => Math.abs(q) > 1e-3)) {
        let dsum = 0;
        for (let b = 0; b < 4; b++) dsum += p[b] * 0.12 * Math.exp(-((v - bumps[b][0]) ** 2) / (2 * bumps[b][1] ** 2));
        v = Math.min(1, Math.max(0, v + dsum));
      }
      out[i * 4 + c] = v;
    }
    out[i * 4 + 3] = 1;
  }
  return out;
}

// ------------------------------------------------------------------ Geometrie (wie pipeline.crop_rect / geometry.py)
export const CROP_ANGLE_SIGN = -1;

export function displayToSensor(x: number, y: number, o: number): Pt {
  switch (o) {
    case 3: return [1 - x, 1 - y];
    case 6: return [y, 1 - x];
    case 8: return [1 - y, x];
    case 2: return [1 - x, y];
    case 4: return [x, 1 - y];
    case 5: return [y, x];
    case 7: return [1 - y, 1 - x];
    default: return [x, y];
  }
}
export function sensorToDisplay(x: number, y: number, o: number): Pt {
  if (o === 6) return [1 - y, x];
  if (o === 8) return [y, 1 - x];
  if (o === 7) return [1 - y, 1 - x];
  return displayToSensor(x, y, o);
}

const rot = (x: number, y: number, deg: number): Pt => {
  const a = (deg * Math.PI) / 180;
  return [x * Math.cos(a) - y * Math.sin(a), x * Math.sin(a) + y * Math.cos(a)];
};

/** Interner Drehwinkel (Grad, positiv = im Uhrzeigersinn) aus dem Lightroom-Wert. */
export function internalAngle(crop: Crop, o: number): number {
  const a = (crop.CropAngle || 0) * CROP_ANGLE_SIGN;
  return [2, 4, 5, 7].includes(o) ? -a : a;
}

/** Zuschnitt im gedrehten Bild (normiert 0..1): wie pipeline.crop_rect. */
export function cropRect(crop: Crop, o: number, W: number, H: number): [number, number, number, number] {
  if (!crop.HasCrop) return [0, 0, 1, 1];
  const ang = internalAngle(crop, o);
  const a = sensorToDisplay(crop.CropLeft, crop.CropTop, o), b = sensorToDisplay(crop.CropRight, crop.CropBottom, o);
  const pts = [a, b].map(([x, y]) => {
    const [rx, ry] = rot(x * W - W / 2, y * H - H / 2, ang);
    return [rx + W / 2, ry + H / 2] as Pt;
  });
  const x0 = Math.min(pts[0][0], pts[1][0]), x1 = Math.max(pts[0][0], pts[1][0]);
  const y0 = Math.min(pts[0][1], pts[1][1]), y1 = Math.max(pts[0][1], pts[1][1]);
  return [x0 / W, y0 / H, x1 / W, y1 / H];
}

function corners(rect: [number, number, number, number], ang: number, W: number, H: number): Pt[] {
  const [x0, y0, x1, y1] = rect;
  return ([[x0, y0], [x1, y0], [x1, y1], [x0, y1]] as Pt[]).map(([x, y]) => {
    const [px, py] = rot(x * W - W / 2, y * H - H / 2, -ang);
    return [px + W / 2, py + H / 2] as Pt;
  });
}

export function cropFits(rect: [number, number, number, number], ang: number, W: number, H: number): boolean {
  return corners(rect, ang, W, H).every(([x, y]) => x >= -0.5 && x <= W + 0.5 && y >= -0.5 && y <= H + 0.5);
}

/** Zuschnitt so verkleinern (um seine Mitte, Seitenverhältnis bleibt), dass er ganz im gedrehten Bild liegt. */
export function fitCrop(rect: [number, number, number, number], ang: number, W: number, H: number): [number, number, number, number] {
  if (cropFits(rect, ang, W, H)) return rect;
  const cx = (rect[0] + rect[2]) / 2, cy = (rect[1] + rect[3]) / 2, hw = (rect[2] - rect[0]) / 2, hh = (rect[3] - rect[1]) / 2;
  let lo = 0, hi = 1;
  for (let i = 0; i < 30; i++) {
    const s = (lo + hi) / 2;
    if (cropFits([cx - hw * s, cy - hh * s, cx + hw * s, cy + hh * s], ang, W, H)) lo = s; else hi = s;
  }
  // Mitte Richtung Bildmitte ziehen, falls auch so zu klein
  return [cx - hw * lo, cy - hh * lo, cx + hw * lo, cy + hh * lo];
}

/** Zuschnitt (gedrehtes Bild, normiert) + interner Winkel -> Lightroom-Felder (wie geometry.to_lightroom_crop). */
export function toLrCrop(rect: [number, number, number, number], ang: number, o: number, W: number, H: number): Crop {
  const full = rect[0] <= 0.0005 && rect[1] <= 0.0005 && rect[2] >= 0.9995 && rect[3] >= 0.9995;
  let lrAngle = ang * CROP_ANGLE_SIGN;
  if ([2, 4, 5, 7].includes(o)) lrAngle = -lrAngle;
  if (full && Math.abs(ang) < 0.01) return { ...NO_CROP };
  const pts = corners(rect, ang, W, H).map(([x, y]) => displayToSensor(x / W, y / H, o));
  let a = 0;
  for (let i = 1; i < 4; i++) if (pts[i][0] + pts[i][1] < pts[a][0] + pts[a][1]) a = i;
  const b = (a + 2) % 4;
  const cl = (v: number) => Math.min(1, Math.max(0, v));
  return { HasCrop: true, CropLeft: cl(pts[a][0]), CropTop: cl(pts[a][1]), CropRight: cl(pts[b][0]), CropBottom: cl(pts[b][1]),
    CropAngle: Math.round(lrAngle * 1e4) / 1e4 };
}

/** Anzeige-Koordinaten (ungedreht, normiert) -> Ansicht (Zuschnitt, normiert) und zurück. */
export interface View { rect: [number, number, number, number]; ang: number; W: number; H: number }
export function imgToView(v: View, p: Pt): Pt {
  const [rx, ry] = rot(p[0] * v.W - v.W / 2, p[1] * v.H - v.H / 2, v.ang);
  const qx = (rx + v.W / 2) / v.W, qy = (ry + v.H / 2) / v.H;
  return [(qx - v.rect[0]) / (v.rect[2] - v.rect[0]), (qy - v.rect[1]) / (v.rect[3] - v.rect[1])];
}
export function viewToImg(v: View, q: Pt): Pt {
  const qx = (v.rect[0] + q[0] * (v.rect[2] - v.rect[0])) * v.W, qy = (v.rect[1] + q[1] * (v.rect[3] - v.rect[1])) * v.H;
  const [px, py] = rot(qx - v.W / 2, qy - v.H / 2, -v.ang);
  return [(px + v.W / 2) / v.W, (py + v.H / 2) / v.H];
}

// ------------------------------------------------------------------ Beschriftungen
export const LABELS: Record<string, string> = {
  Temperature: "Temp.", Tint: "Tönung", Exposure2012: "Belichtung", Contrast2012: "Kontrast", Highlights2012: "Lichter",
  Shadows2012: "Tiefen", Whites2012: "Weiss", Blacks2012: "Schwarz", Texture: "Struktur", Clarity2012: "Klarheit",
  Dehaze: "Dunst entfernen", Vibrance: "Dynamik", Saturation: "Sättigung", Sharpness: "Schärfen", SharpenRadius: "Radius",
  SharpenDetail: "Details", LuminanceSmoothing: "Luminanz-Rauschen", ColorNoiseReduction: "Farbrauschen",
  PostCropVignetteAmount: "Vignette", PostCropVignetteMidpoint: "Mittelpunkt", PostCropVignetteFeather: "Weiche Kante",
  ParametricShadows: "Tiefen", ParametricDarks: "Dunkle Töne", ParametricLights: "Helle Töne", ParametricHighlights: "Lichter",
  ColorGradeBlending: "Überblenden", SplitToningBalance: "Abgleich", amount: "Stärke",
};
export function labelOf(k: string): string {
  if (LABELS[k]) return LABELS[k];
  const m = k.match(/^(Hue|Saturation|Luminance)Adjustment(\w+)$/);
  if (m) return `${{ Hue: "Farbton", Saturation: "Sättigung", Luminance: "Luminanz" }[m[1]]} ${m[2]}`;
  if (k.startsWith("ColorGrade") || k.startsWith("SplitToning")) return "Color Grading";
  return k;
}
export function fmtValue(k: string, v: number): string {
  if (k === "Temperature") return `${Math.round(v)}`;
  if (k === "Exposure2012") return (v > 0 ? "+" : "") + v.toFixed(2);
  if (k === "SharpenRadius") return v.toFixed(1);
  return (v > 0 && !/Split$|Midpoint|Feather|Detail|Sharpness|Smoothing|NoiseReduction|Blending|amount/.test(k) ? "+" : "") + Math.round(v);
}
