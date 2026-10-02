// WebGL2-Renderer für den Editor: rechnet die Entwicklung direkt in der Grafikkarte (jede Regler-Bewegung < 10 ms).
import { BLUR, DOWN, FINAL, LIN, MAIN, VERT } from "./shaders";
import { EdModel, Source, def, wbMult } from "./model";

export const MAX_MASKS = 16;
export const MASK_SIDE = 1024;

type Prog = { p: WebGLProgram; u: Map<string, WebGLUniformLocation | null> };
interface Target { tex: WebGLTexture; fbo: WebGLFramebuffer; w: number; h: number }

export interface MaskUniform {
  type: 0 | 1 | 2 | 3;              // 1 Bild (Ebene), 2 linearer Verlauf, 3 radial
  layer: number;
  invert: boolean;
  geo: [number, number, number, number];
  feather: number;
  amount: number;
  l0: [number, number, number, number];  // Belichtung (EV), Temp, Tönung, Sättigung
  l1: [number, number, number, number];  // Kontrast(+Klarheit), Lichter, Tiefen, Weiss
  l2: [number, number, number, number];  // Schwarz, Weichzeichnen
}

export interface OutOpts {
  rect: [number, number, number, number];   // Ausschnitt im gedrehten Bild (normiert)
  ang: number;                              // interner Winkel (Grad)
  width: number;
  height: number;
  overlay: number;
  clip: boolean;
}

const HSL = ["Red", "Orange", "Yellow", "Green", "Aqua", "Blue", "Purple", "Magenta"];

function hsvTint(h: number): [number, number, number] {
  const f = (n: number) => { const k = (n + h / 60) % 6; return 1 - Math.max(0, Math.min(k, 4 - k, 1)); };
  const c: [number, number, number] = [f(5), f(3), f(1)];
  const m = (c[0] + c[1] + c[2]) / 3;
  return [c[0] - m, c[1] - m, c[2] - m];
}

export function maskUniforms(model: EdModel, layers: (number | null)[]): MaskUniform[] {
  const out: MaskUniform[] = [];
  model.masks.slice(0, MAX_MASKS).forEach((m, i) => {
    const L = (k: string) => (m.local[k] ?? 0);
    const soft = -Math.min(0, L("Sharpness")) / 100 + 0.5 * -Math.min(0, L("Texture")) / 100 + 0.3 * -Math.min(0, L("Clarity2012")) / 100;
    const u: MaskUniform = {
      type: 0, layer: 0, invert: false, geo: [0, 0, 0, 0], feather: 0.6, amount: m.amount ?? 1,
      l0: [L("Exposure2012"), L("Temperature") / 100, L("Tint") / 100, L("Saturation") / 100],
      l1: [(L("Contrast2012") + 0.5 * L("Clarity2012")) / 100, L("Highlights2012") / 100, L("Shadows2012") / 100, L("Whites2012") / 100],
      l2: [L("Blacks2012") / 100, soft, 0, 0],
    };
    if (m.kind === "gradient" && m.zero && m.full) { u.type = 2; u.geo = [m.zero[0], m.zero[1], m.full[0], m.full[1]]; }
    else if (m.kind === "radial" && m.box) {
      const [x0, y0, x1, y1] = m.box;
      u.type = 3; u.geo = [(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) / 2, (y1 - y0) / 2];
      u.feather = Math.max((m.feather ?? 60) / 100, 0.02); u.invert = !!m.invert;
    } else if (layers[i] !== null && layers[i] !== undefined) { u.type = 1; u.layer = layers[i]!; }
    out.push(u);
  });
  return out;
}

export class DevelopGL {
  private gl: WebGL2RenderingContext;
  private progs: Record<string, Prog> = {};
  private src: WebGLTexture | null = null;
  private t: Record<string, Target> = {};
  private maskTex: WebGLTexture;
  private lut: WebGLTexture;
  private hist: Target | null = null;
  readonly floatOk: boolean;
  w = 0;
  h = 0;
  mw = MASK_SIDE;
  mh = MASK_SIDE;
  private source: Source | null = null;

  static create(canvas: HTMLCanvasElement): DevelopGL | null {
    const gl = canvas.getContext("webgl2", { antialias: false, premultipliedAlpha: false, preserveDrawingBuffer: false });
    if (!gl) return null;
    const ok = !!(gl.getExtension("EXT_color_buffer_float") || gl.getExtension("EXT_color_buffer_half_float"));
    if (!ok) return null;
    try { return new DevelopGL(gl); } catch (e) { console.error(e); return null; }
  }

  private constructor(gl: WebGL2RenderingContext) {
    this.gl = gl;
    this.floatOk = true;
    gl.bindVertexArray(gl.createVertexArray());
    for (const [k, fs] of Object.entries({ BLUR, DOWN, LIN, MAIN, FINAL })) this.progs[k] = this.compile(fs);
    this.maskTex = gl.createTexture()!;
    this.lut = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, this.lut);
    this.params(gl.TEXTURE_2D);
  }

  private compile(fs: string): Prog {
    const gl = this.gl;
    const sh = (type: number, s: string) => {
      const x = gl.createShader(type)!;
      gl.shaderSource(x, s);
      gl.compileShader(x);
      if (!gl.getShaderParameter(x, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(x) ?? "Shader-Fehler");
      return x;
    };
    const p = gl.createProgram()!;
    gl.attachShader(p, sh(gl.VERTEX_SHADER, VERT));
    gl.attachShader(p, sh(gl.FRAGMENT_SHADER, fs));
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p) ?? "Link-Fehler");
    return { p, u: new Map() };
  }

  private params(target: number, filter: number = this.gl.LINEAR) {
    const gl = this.gl;
    gl.texParameteri(target, gl.TEXTURE_MIN_FILTER, filter);
    gl.texParameteri(target, gl.TEXTURE_MAG_FILTER, filter);
    gl.texParameteri(target, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(target, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  }

  private target(w: number, h: number, fmt: "f16" | "u8" = "f16"): Target {
    const gl = this.gl;
    const tex = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, tex);
    if (fmt === "f16") gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA16F, w, h, 0, gl.RGBA, gl.HALF_FLOAT, null);
    else gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
    this.params(gl.TEXTURE_2D);
    const fbo = gl.createFramebuffer()!;
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
    if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) throw new Error("Grafikkarte: Puffer nicht nutzbar");
    return { tex, fbo, w, h };
  }

  private free() {
    const gl = this.gl;
    for (const t of Object.values(this.t)) { gl.deleteTexture(t.tex); gl.deleteFramebuffer(t.fbo); }
    this.t = {};
    if (this.src) gl.deleteTexture(this.src);
    this.src = null;
  }

  setSource(s: Source) {
    const gl = this.gl;
    this.free();
    this.source = s;
    this.w = s.w; this.h = s.h;
    this.src = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, this.src);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGB16F, s.w, s.h, 0, gl.RGB, gl.HALF_FLOAT, s.data);
    this.params(gl.TEXTURE_2D);
    const qw = Math.max(1, Math.ceil(s.w / 4)), qh = Math.max(1, Math.ceil(s.h / 4));
    this.t = {
      sb: this.target(s.w, s.h), tmp: this.target(s.w, s.h), A: this.target(s.w, s.h), D: this.target(s.w, s.h),
      Q: this.target(qw, qh), Qt: this.target(qw, qh), Qb: this.target(qw, qh),
      R: this.target(qw, qh), Rt: this.target(qw, qh), Rb: this.target(qw, qh),
    };
    // Maskenebenen im Seitenverhältnis des Bildes
    const asp = s.w / s.h;
    this.mw = asp >= 1 ? MASK_SIDE : Math.round(MASK_SIDE * asp);
    this.mh = asp >= 1 ? Math.round(MASK_SIDE / asp) : MASK_SIDE;
    gl.deleteTexture(this.maskTex);
    this.maskTex = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D_ARRAY, this.maskTex);
    gl.texStorage3D(gl.TEXTURE_2D_ARRAY, 1, gl.R8, this.mw, this.mh, MAX_MASKS);
    this.params(gl.TEXTURE_2D_ARRAY);
    // einmalig: weichgezeichnete Quelle (Farbrauschen in den Tiefen, wie _calm_shadows)
    const sig = Math.max(1, 2 * (Math.max(s.w, s.h) / 1600));
    this.blur(this.src, this.t.tmp, this.t.sb, [sig, sig, sig, sig]);
  }

  /** Maskenebene setzen (Graustufen, Grösse mw x mh). */
  setMaskLayer(layer: number, data: Uint8Array) {
    const gl = this.gl;
    if (layer < 0 || layer >= MAX_MASKS) return;
    gl.bindTexture(gl.TEXTURE_2D_ARRAY, this.maskTex);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    gl.texSubImage3D(gl.TEXTURE_2D_ARRAY, 0, 0, 0, layer, this.mw, this.mh, 1, gl.RED, gl.UNSIGNED_BYTE, data);
  }

  setLut(data: Float32Array) {
    const gl = this.gl;
    gl.bindTexture(gl.TEXTURE_2D, this.lut);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA16F, data.length / 4, 1, 0, gl.RGBA, gl.FLOAT, data);
  }

  private use(name: string): Prog {
    const pr = this.progs[name];
    this.gl.useProgram(pr.p);
    return pr;
  }
  private loc(pr: Prog, n: string) {
    if (!pr.u.has(n)) pr.u.set(n, this.gl.getUniformLocation(pr.p, n));
    return pr.u.get(n)!;
  }
  private tex(pr: Prog, n: string, unit: number, t: WebGLTexture, target: number = this.gl.TEXTURE_2D) {
    const gl = this.gl;
    gl.activeTexture(gl.TEXTURE0 + unit);
    gl.bindTexture(target, t);
    gl.uniform1i(this.loc(pr, n), unit);
  }
  private draw(t: Target | null, w?: number, h?: number) {
    const gl = this.gl;
    gl.bindFramebuffer(gl.FRAMEBUFFER, t ? t.fbo : null);
    gl.viewport(0, 0, t ? t.w : w!, t ? t.h : h!);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }

  private blur(src: WebGLTexture, tmp: Target, dst: Target, sig: [number, number, number, number]) {
    const gl = this.gl;
    const pr = this.use("BLUR");
    const r = Math.min(48, Math.ceil(3 * Math.max(...sig)));
    gl.uniform4f(this.loc(pr, "uSig"), ...sig);
    gl.uniform1i(this.loc(pr, "uR"), r);
    this.tex(pr, "uTex", 0, src);
    gl.uniform2f(this.loc(pr, "uDir"), 1, 0);
    this.draw(tmp);
    this.tex(pr, "uTex", 0, tmp.tex);
    gl.uniform2f(this.loc(pr, "uDir"), 0, 1);
    this.draw(dst);
  }

  private down(src: Target, dst: Target, mode: number) {
    const gl = this.gl;
    const pr = this.use("DOWN");
    this.tex(pr, "uTex", 0, src.tex);
    gl.uniform2f(this.loc(pr, "uOut"), dst.w, dst.h);
    gl.uniform1i(this.loc(pr, "uMode"), mode);
    this.draw(dst);
  }

  private masks(pr: Prog, mu: MaskUniform[], unit: number) {
    const gl = this.gl;
    const n = Math.min(mu.length, MAX_MASKS);
    gl.uniform1i(this.loc(pr, "uMaskN"), n);
    const mi = new Int32Array(MAX_MASKS * 4), mg = new Float32Array(MAX_MASKS * 4), mf = new Float32Array(MAX_MASKS * 4);
    mu.slice(0, n).forEach((m, i) => {
      mi.set([m.type, m.layer, m.invert ? 1 : 0, 0], i * 4);
      mg.set(m.geo, i * 4);
      mf.set([m.feather, m.amount, 0, 0], i * 4);
    });
    gl.uniform4iv(this.loc(pr, "uMI[0]"), mi);
    gl.uniform4fv(this.loc(pr, "uMG[0]"), mg);
    gl.uniform4fv(this.loc(pr, "uMF[0]"), mf);
    gl.uniform2f(this.loc(pr, "uImg"), this.w, this.h);
    this.tex(pr, "uMasks", unit, this.maskTex, gl.TEXTURE_2D_ARRAY);
    return n;
  }

  /** Vollständig entwickeln bis vor den letzten Schritt (Ergebnis in D, Rb). */
  private develop(model: EdModel, mu: MaskUniform[]) {
    const gl = this.gl, s = this.source!, g = model.global, t = this.t;
    const scale = Math.max(this.w, this.h) / 1600;
    const temp = g.Temperature ?? s.as_shot[0], tint = g.Tint ?? s.as_shot[1];
    const mult = model.wb_custom && g.Temperature !== undefined ? wbMult(s.wb, temp, tint) : s.wb.base;
    // LIN
    let pr = this.use("LIN");
    this.tex(pr, "uSrc", 0, this.src!);
    this.tex(pr, "uSrcBlur", 1, t.sb.tex);
    gl.uniform3f(this.loc(pr, "uFloor"), s.floor[0], s.floor[1], s.floor[2]);
    gl.uniform3f(this.loc(pr, "uMult"), mult[0], mult[1], mult[2]);
    gl.uniformMatrix3fv(this.loc(pr, "uM"), true, new Float32Array(s.m));
    gl.uniform1f(this.loc(pr, "uGain"), s.gain * Math.pow(2, g.Exposure2012 ?? 0));
    this.draw(t.A);
    // Basis für Lichter/Tiefen (12 px) und lokalen Kontrast (20 px), in Viertel-Auflösung
    this.down(t.A, t.Q, 0);
    this.blur(t.Q.tex, t.Qt, t.Qb, [12 * scale / 4, 20 * scale / 4, 1, 1]);
    // MAIN
    pr = this.use("MAIN");
    this.tex(pr, "uA", 0, t.A.tex);
    this.tex(pr, "uQb", 1, t.Qb.tex);
    const n = this.masks(pr, mu, 2);
    const l0 = new Float32Array(MAX_MASKS * 4), l1 = new Float32Array(MAX_MASKS * 4), l2 = new Float32Array(MAX_MASKS * 4);
    mu.slice(0, n).forEach((m, i) => { l0.set(m.l0, i * 4); l1.set(m.l1, i * 4); l2.set(m.l2, i * 4); });
    gl.uniform4fv(this.loc(pr, "uL0[0]"), l0);
    gl.uniform4fv(this.loc(pr, "uL1[0]"), l1);
    gl.uniform4fv(this.loc(pr, "uL2[0]"), l2);
    gl.uniform1f(this.loc(pr, "uHL"), (g.Highlights2012 ?? 0) / 100);
    gl.uniform1f(this.loc(pr, "uSH"), (g.Shadows2012 ?? 0) / 100);
    gl.uniform1f(this.loc(pr, "uContrast"), (g.Contrast2012 ?? 0) / 100);
    gl.uniform1f(this.loc(pr, "uWH"), (g.Whites2012 ?? 0) / 100);
    gl.uniform1f(this.loc(pr, "uBL"), (g.Blacks2012 ?? 0) / 100);
    gl.uniform1f(this.loc(pr, "uScale"), scale);
    this.draw(t.D);
    // Klarheit (25 px) und Dunst (30 px)
    this.down(t.D, t.R, 1);
    this.blur(t.R.tex, t.Rt, t.Rb, [25 * scale / 4, 30 * scale / 4, 1, 1]);
  }

  private final(model: EdModel, mu: MaskUniform[], out: OutOpts, target: Target | null) {
    const gl = this.gl, g = model.global, t = this.t;
    const pr = this.use("FINAL");
    this.tex(pr, "uD", 0, t.D.tex);
    this.tex(pr, "uRb", 1, t.Rb.tex);
    this.tex(pr, "uLut", 2, this.lut);
    this.masks(pr, mu, 3);
    gl.uniform4f(this.loc(pr, "uRect"), ...out.rect);
    gl.uniform1f(this.loc(pr, "uAng"), out.ang);
    gl.uniform2f(this.loc(pr, "uOut"), target ? target.w : out.width, target ? target.h : out.height);
    gl.uniform1f(this.loc(pr, "uScale"), Math.max(this.w, this.h) / 1600);
    gl.uniform1f(this.loc(pr, "uClar"), (g.Clarity2012 ?? 0) / 100);
    gl.uniform1f(this.loc(pr, "uTexA"), (g.Texture ?? 0) / 100);
    gl.uniform1f(this.loc(pr, "uDehaze"), (g.Dehaze ?? 0) / 100);
    gl.uniform1f(this.loc(pr, "uVib"), (g.Vibrance ?? 0) / 100);
    gl.uniform1f(this.loc(pr, "uSat"), (g.Saturation ?? 0) / 100);
    gl.uniform1f(this.loc(pr, "uCNR"), g.ColorNoiseReduction ?? def("ColorNoiseReduction"));
    gl.uniform1fv(this.loc(pr, "uHueA[0]"), HSL.map((c) => g[`HueAdjustment${c}`] ?? 0));
    gl.uniform1fv(this.loc(pr, "uSatA[0]"), HSL.map((c) => g[`SaturationAdjustment${c}`] ?? 0));
    gl.uniform1fv(this.loc(pr, "uLumA[0]"), HSL.map((c) => g[`LuminanceAdjustment${c}`] ?? 0));
    const zones: [string, string, string][] = [["SplitToningShadowHue", "SplitToningShadowSaturation", "ColorGradeShadowLum"],
      ["ColorGradeMidtoneHue", "ColorGradeMidtoneSat", "ColorGradeMidtoneLum"],
      ["SplitToningHighlightHue", "SplitToningHighlightSaturation", "ColorGradeHighlightLum"],
      ["ColorGradeGlobalHue", "ColorGradeGlobalSat", "ColorGradeGlobalLum"]];
    gl.uniform3fv(this.loc(pr, "uCGTint[0]"), zones.flatMap(([h]) => hsvTint(g[h] ?? 0)));
    gl.uniform2fv(this.loc(pr, "uCG[0]"), zones.flatMap(([, s, l]) => [(g[s] ?? 0) / 100, (g[l] ?? 0) / 100]));
    gl.uniform1f(this.loc(pr, "uBal"), (g.SplitToningBalance ?? 0) / 100);
    gl.uniform1f(this.loc(pr, "uBlend"), (g.ColorGradeBlending ?? def("ColorGradeBlending")) / 100);
    gl.uniform3f(this.loc(pr, "uVig"), (g.PostCropVignetteAmount ?? 0) / 100, (g.PostCropVignetteMidpoint ?? 50) / 100,
      Math.max((g.PostCropVignetteFeather ?? 50) / 100, 0.05));
    gl.uniform1i(this.loc(pr, "uOverlay"), target ? -1 : out.overlay);
    gl.uniform1i(this.loc(pr, "uClip"), !target && out.clip ? 1 : 0);
    gl.uniform3f(this.loc(pr, "uBg"), 0.12, 0.12, 0.13);
    // Schärfen wie pipeline.sharpen_params (Radius relativ zur Grösse des entwickelten Bildes)
    const amt = (g.Sharpness ?? def("Sharpness")) / 150 * 1.2;
    const ppx = Math.min(1, Math.max(0.35, Math.max(this.w, this.h) / 6000));
    gl.uniform2f(this.loc(pr, "uSharp"), target ? 0 : amt, Math.max(0.5, (g.SharpenRadius ?? 1) * 0.7 * ppx));
    this.draw(target, out.width, out.height);
  }

  render(model: EdModel, mu: MaskUniform[], out: OutOpts) {
    if (!this.source) return;
    this.develop(model, mu);
    this.final(model, mu, out, null);
  }

  /** Histogramm (R, G, B, je 256 Klassen) des zuletzt entwickelten Bildes, aus einer kleinen Kopie. */
  histogram(model: EdModel, mu: MaskUniform[], out: OutOpts): [Uint32Array, Uint32Array, Uint32Array] {
    const gl = this.gl;
    const hw = 256, hh = Math.max(16, Math.round(256 * (out.height / Math.max(out.width, 1))));
    if (!this.hist || this.hist.w !== hw || this.hist.h !== hh) {
      if (this.hist) { gl.deleteTexture(this.hist.tex); gl.deleteFramebuffer(this.hist.fbo); }
      this.hist = this.target(hw, hh, "u8");
    }
    this.final(model, mu, { ...out, width: hw, height: hh }, this.hist);
    const px = new Uint8Array(hw * hh * 4);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.hist.fbo);
    gl.readPixels(0, 0, hw, hh, gl.RGBA, gl.UNSIGNED_BYTE, px);
    const r = new Uint32Array(256), g = new Uint32Array(256), b = new Uint32Array(256);
    for (let i = 0; i < px.length; i += 4) { r[px[i]]++; g[px[i + 1]]++; b[px[i + 2]]++; }
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    return [r, g, b];
  }

  dispose() {
    this.free();
    this.gl.getExtension("WEBGL_lose_context")?.loseContext();
  }
}
