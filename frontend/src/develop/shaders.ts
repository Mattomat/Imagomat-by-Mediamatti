// GLSL-Fassung von backend/imagomat/render/pipeline.py:render (gleiche Formeln, gleiche Reihenfolge).
// Ablauf pro Bild: Quelle -> (einmalig) weichgezeichnete Quelle; pro Änderung: LIN -> DOWN/BLUR (Basis für
// Lichter/Tiefen, lokalen Kontrast) -> MAIN (Masken, Basiskurve, Weiss/Schwarz) -> DOWN/BLUR (Klarheit, Dunst)
// -> FINAL (Präsenz, HSL, Color Grading, Kurven, Vignette, Drehen/Zuschneiden).

export const VERT = `#version 300 es
void main() {
  vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}`;

const HEAD = `#version 300 es
precision highp float;
precision highp int;
precision highp sampler2DArray;
out vec4 o;
float lum(vec3 c) { return dot(c, vec3(0.2126, 0.7152, 0.0722)); }
float ss(float e0, float e1, float x) { float t = clamp((x - e0) / (e1 - e0), 0.0, 1.0); return t * t * (3.0 - 2.0 * t); }
vec3 srgbEnc(vec3 x) {
  x = clamp(x, 0.0, 1.0);
  return mix(1.055 * pow(x, vec3(1.0 / 2.4)) - 0.055, x * 12.92, vec3(lessThanEqual(x, vec3(0.0031308))));
}
`;

export const BLUR = HEAD + `
uniform sampler2D uTex;
uniform vec2 uDir;
uniform vec4 uSig;
uniform int uR;
void main() {
  vec2 sz = vec2(textureSize(uTex, 0));
  vec2 uv = gl_FragCoord.xy / sz;
  vec4 inv = 1.0 / (2.0 * uSig * uSig);
  vec4 acc = vec4(0.0), ws = vec4(0.0);
  for (int i = -uR; i <= uR; i++) {
    float f = float(i);
    vec4 w = exp(-f * f * inv);
    acc += w * texture(uTex, uv + uDir * f / sz);
    ws += w;
  }
  o = acc / ws;
}`;

export const DOWN = HEAD + `
uniform sampler2D uTex;
uniform vec2 uOut;
uniform int uMode;
vec4 pick(vec4 s) {
  if (uMode == 0) return vec4(s.a, s.a, 0.0, 0.0);
  return vec4(s.a, min(min(s.r, s.g), s.b), 0.0, 0.0);
}
void main() {
  vec2 isz = 1.0 / vec2(textureSize(uTex, 0));
  vec2 uv = gl_FragCoord.xy / uOut;
  vec4 s = pick(texture(uTex, uv + vec2(-1.0, -1.0) * isz)) + pick(texture(uTex, uv + vec2(1.0, -1.0) * isz))
         + pick(texture(uTex, uv + vec2(-1.0, 1.0) * isz)) + pick(texture(uTex, uv + vec2(1.0, 1.0) * isz));
  o = s * 0.25;
}`;

export const LIN = HEAD + `
uniform sampler2D uSrc;
uniform sampler2D uSrcBlur;
uniform vec3 uFloor;
uniform vec3 uMult;
uniform mat3 uM;
uniform float uGain;
void main() {
  vec2 uv = gl_FragCoord.xy / vec2(textureSize(uSrc, 0));
  vec3 s = texture(uSrc, uv).rgb;
  vec3 raw = s + uFloor;
  float clipped = ss(0.90, 0.99, max(max(raw.r, raw.g), raw.b));
  vec3 c = s * uMult;
  if (clipped > 0.0) {
    vec3 cm = min(c, vec3(1.0));
    c = mix(c, vec3(max(max(cm.r, cm.g), cm.b)), clipped);
  }
  c = (uM * c) * uGain;
  vec3 cb = (uM * (texture(uSrcBlur, uv).rgb * uMult)) * uGain;
  float L = lum(c), Lb = lum(cb);
  c = vec3(L) + (cb - vec3(Lb)) * ss(0.006, 0.05, Lb);
  c = max(c, vec3(0.0));
  o = vec4(c, log2(max(lum(c), 1e-6)));
}`;

const MASKS = `
uniform int uMaskN;
uniform ivec4 uMI[16];
uniform vec4 uMG[16];
uniform vec4 uMF[16];
uniform vec2 uImg;
uniform sampler2DArray uMasks;
float maskAt(int i, vec2 p) {
  ivec4 inf = uMI[i];
  float m = 0.0;
  if (inf.x == 1) {
    m = texture(uMasks, vec3(p, float(inf.y))).r;
  } else if (inf.x == 2) {
    vec2 z = uMG[i].xy * uImg, f = uMG[i].zw * uImg, v = f - z;
    m = clamp(dot(p * uImg - z, v) / max(dot(v, v), 1e-6), 0.0, 1.0);
  } else if (inf.x == 3) {
    vec2 c = uMG[i].xy * uImg, r = max(uMG[i].zw * uImg, vec2(1.0));
    float d = length((p * uImg - c) / r);
    float fe = max(uMF[i].x, 0.02);
    m = 1.0 - ss(1.0 - fe, 1.0, d);
  }
  if (inf.z == 1) m = 1.0 - m;
  return clamp(m, 0.0, 1.0) * uMF[i].y;
}
`;

export const MAIN = HEAD + MASKS + `
uniform sampler2D uA;
uniform sampler2D uQb;
uniform float uHL, uSH, uContrast, uWH, uBL, uScale;
uniform vec4 uL0[16];
uniform vec4 uL1[16];
uniform vec4 uL2[16];
vec3 blurA(vec2 uv, vec2 sz) {
  float s = 3.5 * uScale * 1.2;
  vec3 acc = vec3(0.0); float ws = 0.0;
  for (int y = -1; y <= 1; y++) for (int x = -1; x <= 1; x++) {
    float w = exp(-float(x * x + y * y) * 0.72);
    acc += w * texture(uA, uv + vec2(float(x), float(y)) * s / sz).rgb; ws += w;
  }
  return acc / ws;
}
void main() {
  vec2 sz = vec2(textureSize(uA, 0));
  vec2 uv = gl_FragCoord.xy / sz;
  vec4 a = texture(uA, uv);
  vec4 qb = texture(uQb, uv);
  vec3 lin = a.rgb;
  lin *= exp2(uHL * 1.4 * ss(-2.8, 0.5, qb.r) + uSH * 1.6 * (1.0 - ss(-7.5, -3.0, qb.r)));
  for (int i = 0; i < 16; i++) {
    if (i >= uMaskN) break;
    float m = maskAt(i, uv);
    if (m <= 0.0) continue;
    vec4 l0 = uL0[i], l1 = uL1[i], l2 = uL2[i];
    lin *= exp2(l0.x * m);
    if (l0.y != 0.0 || l0.z != 0.0) {
      vec3 g = vec3(1.0 + 0.25 * l0.y, 1.0 - 0.2 * l0.z, 1.0 - 0.25 * l0.y);
      lin *= 1.0 + (g - 1.0) * m;
    }
    if (l0.w != 0.0) { float L = lum(lin); lin = vec3(L) + (lin - vec3(L)) * (1.0 + l0.w * m); }
    if (l1.x != 0.0) lin *= exp2((a.a - qb.g) * l1.x * 0.6 * m);
    if (l1.y != 0.0 || l1.z != 0.0) {
      float lL = log2(max(lum(lin), 1e-6));
      lin *= exp2((l1.y * 1.2 * ss(-2.8, 0.5, lL) + l1.z * 1.4 * (1.0 - ss(-7.5, -3.0, lL))) * m);
    }
    if (l1.w != 0.0 || l2.x != 0.0) {
      float lL = log2(max(lum(lin), 1e-6));
      lin *= exp2((l1.w * 0.9 * ss(-3.0, 0.0, lL) + l2.x * (1.0 - ss(-8.5, -4.5, lL))) * m);
    }
    if (l2.y > 0.0) {
      vec3 b = blurA(uv, sz) * (lum(lin) / max(lum(a.rgb), 1e-6));
      lin = mix(lin, b, min(l2.y * 1.2, 1.0) * m);
    }
  }
  lin = max(lin, vec3(0.0));
  float k = 1.0 + uContrast * 0.6;
  vec3 y = lin * (1.0 + lin / 16.0) / (1.0 + lin) * 1.18;
  vec3 gm = srgbEnc(y) - 0.46;
  vec3 disp = clamp(0.46 + gm * k - (k - 1.0) * 0.35 * gm * gm * gm / 0.25, 0.0, 1.0);
  if (uWH != 0.0 || uBL != 0.0) {
    float Y = clamp(lum(disp), 1e-4, 1.0);
    float Y2 = Y * (1.0 + (uWH > 0.0 ? 0.45 : 0.3) * uWH * pow(Y, 1.5));
    float c2 = 1.0 - clamp(Y2, 0.0, 1.0);
    Y2 += (uBL < 0.0 ? 0.25 : 0.15) * uBL * c2 * c2 * c2;
    disp *= max(Y2, 0.0) / Y;
  }
  o = vec4(disp, lum(disp));
}`;

export const FINAL = HEAD + MASKS + `
uniform sampler2D uD;
uniform sampler2D uRb;
uniform sampler2D uLut;
uniform vec4 uRect;
uniform float uAng;
uniform vec2 uOut;
uniform float uScale;
uniform float uClar, uTexA, uDehaze, uVib, uSat, uCNR;
uniform float uHueA[8];
uniform float uSatA[8];
uniform float uLumA[8];
uniform vec3 uCGTint[4];
uniform vec2 uCG[4];
uniform float uBal, uBlend;
uniform vec3 uVig;
uniform int uOverlay;
uniform int uClip;
uniform vec3 uBg;
uniform vec2 uSharp;
const float HUE_C[8] = float[8](0.0, 30.0, 60.0, 120.0, 180.0, 240.0, 275.0, 315.0);
vec3 rgb2hsv(vec3 c) {
  float v = max(max(c.r, c.g), c.b), mn = min(min(c.r, c.g), c.b), d = v - mn;
  float s = v > 0.0 ? d / v : 0.0, h = 0.0;
  if (d > 0.0) {
    if (v == c.r) h = 60.0 * (c.g - c.b) / d;
    else if (v == c.g) h = 120.0 + 60.0 * (c.b - c.r) / d;
    else h = 240.0 + 60.0 * (c.r - c.g) / d;
    if (h < 0.0) h += 360.0;
  }
  return vec3(h, s, v);
}
vec3 hsv2rgb(vec3 c) {
  vec3 k = mod(vec3(5.0, 3.0, 1.0) + c.x / 60.0, 6.0);
  return c.z - c.z * c.y * clamp(min(k, 4.0 - k), 0.0, 1.0);
}
vec4 blurD(vec2 p) {
  float s = 2.5 * uScale * 1.2;
  vec4 acc = vec4(0.0); float ws = 0.0;
  for (int y = -1; y <= 1; y++) for (int x = -1; x <= 1; x++) {
    float w = exp(-float(x * x + y * y) * 0.72);
    acc += w * texture(uD, p + vec2(float(x), float(y)) * s / uImg); ws += w;
  }
  return acc / ws;
}
float sharpDetail(vec2 p, float Y) {
  if (uSharp.x <= 0.0) return 0.0;
  float acc = 0.0, ws = 0.0;
  float inv = 1.0 / (2.0 * uSharp.y * uSharp.y);
  for (int y = -2; y <= 2; y++) for (int x = -2; x <= 2; x++) {
    float w = exp(-float(x * x + y * y) * inv);
    acc += w * texture(uD, p + vec2(float(x), float(y)) / uImg).a; ws += w;
  }
  return uSharp.x * (Y - acc / ws);
}
void main() {
  vec2 q = vec2(gl_FragCoord.x, uOut.y - gl_FragCoord.y) / uOut;
  vec2 qp = (uRect.xy + q * (uRect.zw - uRect.xy)) * uImg;
  vec2 c0 = uImg * 0.5;
  float an = radians(uAng);
  vec2 d0 = qp - c0;
  vec2 p = (vec2(d0.x * cos(an) + d0.y * sin(an), -d0.x * sin(an) + d0.y * cos(an)) + c0) / uImg;
  if (p.x < 0.0 || p.y < 0.0 || p.x > 1.0 || p.y > 1.0) { o = vec4(uBg, 1.0); return; }
  vec4 dd = texture(uD, p);
  vec3 col = dd.rgb;
  float Y = dd.a;
  float sharp = sharpDetail(p, Y);
  vec4 rb = texture(uRb, p);
  vec4 bs = blurD(p);
  if (abs(uClar) > 1e-3) col += uClar * 0.9 * (Y - rb.r) * clamp(1.0 - abs(Y - 0.5) * 2.0, 0.2, 1.0);
  if (abs(uTexA) > 1e-3) col += uTexA * 0.8 * (Y - bs.a);
  if (abs(uDehaze) > 1e-3) {
    float dk = rb.g;
    col = (col - uDehaze * 0.6 * dk) / (1.0 - uDehaze * 0.6 * clamp(dk, 0.0, 0.9));
    col += (col - vec3(lum(col))) * uDehaze * 0.3;
  }
  col = clamp(col, 0.0, 1.0);
  if (uCNR > 0.0) col += (bs.rgb - vec3(lum(bs.rgb))) - (dd.rgb - vec3(Y));   // Farbrauschen: Farbanteil geglättet
  col = clamp(col, 0.0, 1.0);
  // HSL, Dynamik, Sättigung
  vec3 hsv = rgb2hsv(col);
  float S = hsv.y * (1.0 + uSat) * (1.0 + uVib * (1.0 - hsv.y) * 1.2);
  float dh = 0.0, ds = 1.0, dv = 1.0;
  for (int i = 0; i < 8; i++) {
    float dist = abs(mod(hsv.x - HUE_C[i] + 180.0, 360.0) - 180.0);
    float w = pow(clamp(1.0 - dist / 40.0, 0.0, 1.0), 1.5) * clamp(S * 3.0, 0.0, 1.0);
    dh += w * uHueA[i] * 0.3;
    ds *= 1.0 + w * uSatA[i] / 100.0;
    dv *= 1.0 + w * uLumA[i] / 100.0 * 0.5;
  }
  col = hsv2rgb(vec3(mod(hsv.x + dh, 360.0), clamp(S * ds, 0.0, 1.0), clamp(hsv.z * dv, 0.0, 1.0)));
  // Color Grading
  float Yc = lum(col);
  float wsh = 1.0 - ss(0.1, 0.5 + uBal * 0.3 + uBlend * 0.2, Yc);
  float whi = ss(0.5 + uBal * 0.3 - uBlend * 0.2, 0.9, Yc);
  float wmid = clamp(1.0 - wsh - whi, 0.0, 1.0);
  float wz[4] = float[4](wsh, wmid, whi, 1.0);
  for (int i = 0; i < 4; i++) {
    if (uCG[i].x > 0.0) col += wz[i] * uCG[i].x * 0.35 * uCGTint[i];
    if (uCG[i].y != 0.0) col *= 1.0 + wz[i] * uCG[i].y * 0.4;
  }
  col = clamp(col, 0.0, 1.0);
  // Gradationskurven (Punkt, Kanäle, parametrisch in einer Tabelle)
  float n = float(textureSize(uLut, 0).x);
  vec3 lc = col * (n - 1.0) / n + 0.5 / n;
  col = vec3(texture(uLut, vec2(lc.r, 0.5)).r, texture(uLut, vec2(lc.g, 0.5)).g, texture(uLut, vec2(lc.b, 0.5)).b);
  // Vignette (wie pipeline: im ganzen Bild vor dem Zuschneiden)
  if (abs(uVig.x) > 1e-3) {
    float dv2 = length((p - 0.5) * 2.0) / sqrt(2.0);
    col = clamp(col * (1.0 + uVig.x * 0.8 * ss(uVig.y * 0.9, uVig.y * 0.9 + uVig.z, dv2)), 0.0, 1.0);
  }
  col = clamp(col + sharp, 0.0, 1.0);
  if (uOverlay >= 0) {
    float m = maskAt(uOverlay, p) / max(uMF[uOverlay].y, 1e-3);
    col = mix(col, vec3(0.92, 0.16, 0.24), 0.55 * clamp(m, 0.0, 1.0));
  }
  if (uClip == 1) {
    if (max(max(col.r, col.g), col.b) >= 0.996) col = vec3(1.0, 0.1, 0.1);
    else if (max(max(col.r, col.g), col.b) <= 0.004) col = vec3(0.1, 0.35, 1.0);
  }
  o = vec4(col, 1.0);
}`;
