"""Mitgelieferte, adaptive Presets.

Anders als Lightroom-Presets berechnen diese Presets die Werte **pro Bild**:
- Belichtung aus der gemessenen Szenenhelligkeit (RAW-linear), gewichtet aufs Motiv,
- Lichter/Tiefen/Weiss/Schwarz aus Clipping und Histogramm,
- Weissabgleich je nach Situation: As Shot, neutralisieren (Grauwelt) oder wärmen,
- Denoise-Stärke aus dem gemessenen Rauschen und der Belichtungsanhebung,
- Masken-Rezepte (Motiv anheben, Hintergrund zurücknehmen, Himmel abdunkeln).

Der feste "Look" (Kontrast, HSL, Color Grading, Vignette) ist ein Ausgangspunkt.
Ein eigenes Stilprofil baut auf einem Preset auf und ersetzt dessen Werte, sobald
genug eigene Beispiele vorhanden sind.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..config import DenoiseSettings
from ..io.color import mired, multipliers_to_temp_tint

HSL = ("Red", "Orange", "Yellow", "Green", "Aqua", "Blue", "Purple", "Magenta")


@dataclass
class MaskRecipe:
    kind: str                      # subject | background | sky | person | gradient_bottom | gradient_top | spotlight
    name: str
    local: dict[str, float]        # UI-Einheiten, z. B. {"Exposure2012": 0.3, "Clarity2012": 10}
    condition: str = "always"      # always | has_subject | has_sky | has_face


@dataclass
class Preset:
    key: str
    name: str
    description: str
    target_log: float              # gewünschter linearer Median (log2) nach Belichtung
    subject_weight: float = 0.5    # 0 = ganzes Bild, 1 = nur Motiv
    wb_mode: str = "as_shot"       # as_shot | neutralize | warm
    neutralize: float = 0.0        # Anteil der Grauwelt-Korrektur
    warm_mired: float = 0.0        # negativ = wärmer
    tint_fix: float = 0.0          # Anteil der Tint-Korrektur Richtung 0 (Flutlicht-Grünstich)
    highlight_protect: float = 1.0
    shadow_lift: float = 1.0
    denoise_bias: int = 0
    look: dict[str, float] = field(default_factory=dict)
    curve: list[float] | None = None          # Abweichung an CURVE_X
    masks: list[MaskRecipe] = field(default_factory=list)
    fade: float = 1.0                         # Stärke des Verlaufs unten (mal Einstellung)
    group: str = ""                           # z. B. "Fussball": in der Auswahl gruppiert, nie automatisch


def _hsl(**kw: float) -> dict[str, float]:
    """_hsl(sat_Green=-10, lum_Blue=-5, hue_Orange=3)"""
    out = {}
    for k, v in kw.items():
        kind, color = k.split("_")
        out[{"hue": "HueAdjustment", "sat": "SaturationAdjustment", "lum": "LuminanceAdjustment"}[kind] + color] = v
    return out


def _grade(shadow: tuple[float, float] = (0, 0), mid: tuple[float, float] = (0, 0),
           high: tuple[float, float] = (0, 0)) -> dict[str, float]:
    out = {}
    for (h, s), name in ((shadow, "grade_shadow"), (mid, "grade_mid"), (high, "grade_high")):
        out[f"{name}_a"] = s * math.cos(math.radians(h))
        out[f"{name}_b"] = s * math.sin(math.radians(h))
    return out


SUBJECT_POP = MaskRecipe("subject", "Motiv anheben", {"Exposure2012": 0.4, "Clarity2012": 10, "Texture": 8,
                                                     "Shadows2012": 10}, "has_subject")
# Hintergrund: etwas dunkler, aber mit Kontrast und Tiefe (nicht nur Helligkeit runter -> sonst fade)
BACKGROUND_CALM = MaskRecipe("background", "Hintergrund",
                             {"Exposure2012": -0.25, "Contrast2012": 15, "Dehaze": 10, "Highlights2012": -15,
                              "Saturation": -5}, "has_subject")
# Dunkler Verlauf von unten (Rasen/Vordergrund zurücknehmen, lenkt den Blick auf die Spieler)
BOTTOM_FADE = MaskRecipe("gradient_bottom", "Verlauf unten", {"Exposure2012": -1.1, "Highlights2012": -25,
                                                             "Saturation": -15, "Sharpness": -60,
                                                             "Clarity2012": -30, "Texture": -40}, "always")
SKY_DEEPEN = MaskRecipe("sky", "Himmel", {"Exposure2012": -0.35, "Highlights2012": -20, "Saturation": 8}, "has_sky")

PRESETS: dict[str, Preset] = {p.key: p for p in [
    Preset("sport_day", "Sport Tag", "Fussball/Outdoor bei Tageslicht: knackig, sattes aber natürliches Grün.",
           target_log=-2.6, subject_weight=0.7, wb_mode="as_shot",
           look={"Contrast2012": 18, "Clarity2012": 10, "Texture": 10, "Dehaze": 5, "Vibrance": 22,
                 "Sharpness": 50, "PostCropVignetteAmount": -10,
                 **_hsl(sat_Green=-12, lum_Green=-8, hue_Green=5, sat_Orange=4, lum_Orange=4, sat_Blue=5)},
           curve=[-3, -4, -2, 0, 2, 3, 2], masks=[SUBJECT_POP, BACKGROUND_CALM, BOTTOM_FADE]),
    Preset("sport_floodlight", "Sport Flutlicht", "Stadion bei Nacht: Grünstich weg, Spieler freistellen.",
           target_log=-2.4, subject_weight=0.85, wb_mode="as_shot",
           highlight_protect=1.2, shadow_lift=1.3, denoise_bias=5,
           look={"Contrast2012": 22, "Clarity2012": 12, "Texture": 10, "Dehaze": 8, "Vibrance": 22,
                 "Saturation": 0, "Sharpness": 45, "PostCropVignetteAmount": -15,
                 **_hsl(hue_Green=10, sat_Green=-18, lum_Green=-10, sat_Yellow=-10, lum_Orange=5),
                 **_grade(shadow=(215, 8), high=(40, 6))},
           curve=[-5, -6, -3, 0, 3, 4, 2], masks=[SUBJECT_POP, BACKGROUND_CALM, BOTTOM_FADE]),
    Preset("sport_indoor", "Sport Halle", "Hallensport: Mischlicht neutralisieren, Hallenboden beruhigen.",
           target_log=-2.6, subject_weight=0.8, wb_mode="neutralize", neutralize=0.6, tint_fix=0.4,
           highlight_protect=1.0, shadow_lift=1.0, denoise_bias=5,
           look={"Contrast2012": 15, "Clarity2012": 10, "Texture": 8, "Vibrance": 10, "Sharpness": 45,
                 "PostCropVignetteAmount": -10, **_hsl(sat_Orange=-8, sat_Yellow=-12, lum_Yellow=-5)},
           masks=[SUBJECT_POP, BACKGROUND_CALM]),
    Preset("concert_stage", "Konzert Bühne", "Grosse Bühne mit LED-Licht: Stimmung erhalten, Spots zähmen.",
           target_log=-3.6, subject_weight=0.8, wb_mode="as_shot", highlight_protect=1.4, shadow_lift=0.7,
           denoise_bias=8,
           look={"Contrast2012": 20, "Blacks2012": -15, "Clarity2012": 10, "Dehaze": 10, "Vibrance": 5,
                 "Saturation": -5, "Sharpness": 40, "PostCropVignetteAmount": -20,
                 **_hsl(sat_Red=-10, sat_Magenta=-15, sat_Purple=-12, sat_Blue=-10, lum_Blue=5, lum_Red=5,
                        lum_Orange=6), **_grade(shadow=(230, 10))},
           curve=[-6, -6, -3, 0, 2, 3, 1],
           masks=[MaskRecipe("subject", "Künstler", {"Exposure2012": 0.35, "Texture": 8, "Highlights2012": -15},
                             "has_subject")]),
    Preset("concert_club", "Konzert Club", "Kleiner Club, sehr dunkel: dichte Schwarztöne, dezentes Korn.",
           target_log=-4.0, subject_weight=0.85, wb_mode="as_shot", highlight_protect=1.3, shadow_lift=0.5,
           denoise_bias=10,
           look={"Contrast2012": 25, "Blacks2012": -25, "Clarity2012": 15, "Dehaze": 15, "Saturation": -10,
                 "GrainAmount": 12, "Sharpness": 35, "PostCropVignetteAmount": -25,
                 **_hsl(sat_Magenta=-15, sat_Purple=-15, sat_Blue=-12, lum_Orange=6)},
           curve=[-8, -7, -4, 0, 3, 4, 2],
           masks=[MaskRecipe("subject", "Künstler", {"Exposure2012": 0.4, "Clarity2012": 10}, "has_subject")]),
    Preset("portrait", "Porträt", "Weiche Haut, sanfte Lichter, Fokus aufs Gesicht.",
           target_log=-2.8, subject_weight=0.8, wb_mode="warm", warm_mired=-8, highlight_protect=1.0,
           look={"Contrast2012": 5, "Clarity2012": -5, "Texture": -10, "Vibrance": 10, "Sharpness": 35,
                 "PostCropVignetteAmount": -12, **_hsl(lum_Orange=8, sat_Orange=-5, sat_Red=-5)},
           masks=[MaskRecipe("subject", "Person", {"Exposure2012": 0.2, "Texture": -10}, "has_subject")]),
    Preset("event", "Event", "Innenräume, Mischlicht: neutral, offen, freundlich.",
           target_log=-2.8, subject_weight=0.6, wb_mode="neutralize", neutralize=0.5, shadow_lift=1.2,
           look={"Contrast2012": 10, "Clarity2012": 5, "Vibrance": 10, "Sharpness": 40,
                 "PostCropVignetteAmount": -8, **_hsl(lum_Orange=5, sat_Yellow=-8)},
           masks=[SUBJECT_POP]),
    Preset("wedding", "Hochzeit hell", "Hell und luftig: weiche Kontraste, warme Lichter, dezentes Grün.",
           target_log=-2.5, subject_weight=0.6, wb_mode="warm", warm_mired=-12, highlight_protect=1.1,
           shadow_lift=1.3,
           look={"Contrast2012": -10, "Whites2012": 10, "Blacks2012": 5, "Clarity2012": -5, "Vibrance": 5,
                 "Saturation": -10, "Sharpness": 35,
                 **_hsl(lum_Orange=10, sat_Orange=-5, sat_Green=-20, lum_Green=5, sat_Blue=-10),
                 **_grade(high=(45, 8), shadow=(200, 5))},
           curve=[10, 6, 3, 1, 0, -1, -2], masks=[SUBJECT_POP]),
    Preset("landscape", "Landschaft", "Kräftige Farben, Himmel mit Zeichnung.",
           target_log=-3.0, subject_weight=0.2, wb_mode="as_shot", highlight_protect=1.4, shadow_lift=1.3,
           look={"Contrast2012": 15, "Clarity2012": 20, "Texture": 15, "Dehaze": 10, "Vibrance": 20,
                 "Sharpness": 45, **_hsl(sat_Blue=10, lum_Blue=-10, sat_Green=5, lum_Green=-5)},
           masks=[SKY_DEEPEN]),
]}


# ---------------------------------------------------------------------------
# Fussball-Stile (zum Ausprobieren; werden nie automatisch gewählt)
# Orientierung an gängigen Matchday-Looks: knackig/kommerziell, Flutlicht-moody, Kino (Teal/Orange),
# gritty (Zweikampf), Schwarzweiss. Belichtung, Weiss/Schwarz und Denoise bleiben pro Bild adaptiv.
# ---------------------------------------------------------------------------
GRADIENT_TOP = MaskRecipe("gradient_top", "Verlauf oben", {"Exposure2012": -0.4, "Highlights2012": -30,
                                                           "Dehaze": 10, "Saturation": -10}, "always")
SPOTLIGHT = MaskRecipe("spotlight", "Spot auf Spieler", {"Exposure2012": -0.45, "Contrast2012": 10,
                                                         "Saturation": -10}, "has_subject")

FOOTBALL: list[Preset] = [
    Preset("fb_matchday", "Matchday Punch", "Knackig und sauber wie auf Club-Kanälen: satte, aber natürliche "
           "Farben, Spieler klar vor dem Hintergrund.",
           target_log=-2.6, subject_weight=0.8, wb_mode="as_shot",
           highlight_protect=0.9, shadow_lift=1.0, denoise_bias=3,
           look={"Contrast2012": 28, "Clarity2012": 14, "Texture": 12, "Dehaze": 8, "Vibrance": 20,
                 "Sharpness": 50, "PostCropVignetteAmount": -12,
                 **_hsl(hue_Green=6, sat_Green=-10, lum_Green=-10, sat_Red=8, lum_Red=-4, lum_Orange=4,
                        sat_Yellow=-8)},
           curve=[-2, -3, -2, 0, 3, 5, 3],
           masks=[MaskRecipe("subject", "Spieler", {"Exposure2012": 0.35, "Clarity2012": 15, "Texture": 12,
                                                    "Whites2012": 10, "Shadows2012": 12}, "has_subject"),
                  MaskRecipe("background", "Hintergrund", {"Exposure2012": -0.3, "Contrast2012": 18, "Dehaze": 12,
                                                           "Highlights2012": -20, "Saturation": -8}, "has_subject"),
                  BOTTOM_FADE], group="Fussball"),
    Preset("fb_night", "Flutlicht Moody", "Dunkel und dramatisch unter Flutlicht: dichte Schwarztöne, kühle "
           "Schatten, warme Lichter, Spot auf den Spieler.",
           target_log=-3.0, subject_weight=0.9, wb_mode="as_shot",
           highlight_protect=1.3, shadow_lift=0.5, denoise_bias=6,
           look={"Contrast2012": 32, "Clarity2012": 12, "Texture": 10, "Dehaze": 12, "Vibrance": 8,
                 "Saturation": -8, "Sharpness": 45, "PostCropVignetteAmount": -28,
                 **_hsl(hue_Green=12, sat_Green=-30, lum_Green=-18, sat_Yellow=-20, sat_Blue=-10, lum_Orange=6),
                 **_grade(shadow=(210, 14), high=(38, 10))},
           curve=[-5, -4, -2, 0, 3, 5, 3],
           masks=[MaskRecipe("subject", "Spieler", {"Exposure2012": 0.45, "Clarity2012": 12, "Texture": 10,
                                                    "Shadows2012": 15}, "has_subject"),
                  SPOTLIGHT, GRADIENT_TOP, BOTTOM_FADE], fade=1.25, group="Fussball"),
    Preset("fb_cinematic", "Cinematic Teal/Orange", "Kino-Look: Hauttöne warm, Schatten und Rasen Richtung "
           "Petrol, weiche Lichter.",
           target_log=-2.8, subject_weight=0.8, wb_mode="as_shot", highlight_protect=1.2, shadow_lift=0.8,
           look={"Contrast2012": 22, "Clarity2012": 8, "Dehaze": 6, "Vibrance": 12, "Saturation": -6,
                 "Sharpness": 40, "PostCropVignetteAmount": -18,
                 **_hsl(hue_Green=30, sat_Green=-25, lum_Green=-15, hue_Aqua=-10, sat_Aqua=10, hue_Blue=-12,
                        sat_Orange=6, lum_Orange=8, hue_Yellow=-10, sat_Yellow=-20),
                 **_grade(shadow=(195, 18), mid=(200, 4), high=(40, 14))},
           curve=[4, -2, -2, 0, 3, 4, 0],
           masks=[MaskRecipe("subject", "Spieler", {"Exposure2012": 0.3, "Clarity2012": 8, "Texture": 6},
                             "has_subject"),
                  MaskRecipe("background", "Hintergrund", {"Exposure2012": -0.35, "Contrast2012": 12, "Dehaze": 8,
                                                           "Saturation": -12}, "has_subject"),
                  BOTTOM_FADE], group="Fussball"),
    Preset("fb_gritty", "Gritty Zweikampf", "Rau und intensiv: viel Struktur auf Spielern (Schweiss, Dreck), "
           "entsättigt, harte Kontraste, feines Korn.",
           target_log=-2.9, subject_weight=0.85, wb_mode="as_shot",
           highlight_protect=1.1, shadow_lift=0.7, denoise_bias=-5,
           look={"Contrast2012": 28, "Clarity2012": 25, "Texture": 22, "Dehaze": 15, "Vibrance": -5,
                 "Saturation": -22, "Sharpness": 55, "GrainAmount": 14, "PostCropVignetteAmount": -22,
                 **_hsl(sat_Green=-35, lum_Green=-15, sat_Orange=-5, lum_Orange=-4, sat_Red=5)},
           curve=[-4, -4, -2, 0, 4, 6, 3],
           masks=[MaskRecipe("subject", "Spieler", {"Exposure2012": 0.3, "Clarity2012": 25, "Texture": 25,
                                                    "Contrast2012": 10}, "has_subject"),
                  MaskRecipe("background", "Hintergrund", {"Exposure2012": -0.45, "Contrast2012": 10,
                                                           "Saturation": -15, "Clarity2012": -10}, "has_subject"),
                  BOTTOM_FADE], fade=1.1, group="Fussball"),
    Preset("fb_fcw_red", "Rot knallt", "Vereinsfarbe im Fokus: Rot kräftig und dicht, übrige Farben "
           "zurückgenommen, dunkler Rand.",
           target_log=-2.8, subject_weight=0.85, wb_mode="as_shot",
           highlight_protect=1.1, shadow_lift=0.8,
           look={"Contrast2012": 28, "Clarity2012": 12, "Texture": 10, "Dehaze": 10, "Vibrance": 5,
                 "Saturation": -5, "Sharpness": 50, "PostCropVignetteAmount": -22,
                 **_hsl(sat_Red=18, lum_Red=-8, hue_Red=3, sat_Orange=-5, sat_Green=-35, lum_Green=-15,
                        sat_Blue=-25, sat_Yellow=-30, sat_Aqua=-25, sat_Purple=-30, sat_Magenta=-20),
                 **_grade(shadow=(220, 8))},
           curve=[-3, -4, -2, 0, 3, 5, 3],
           masks=[MaskRecipe("subject", "Spieler", {"Exposure2012": 0.35, "Clarity2012": 12, "Texture": 10},
                             "has_subject"),
                  MaskRecipe("background", "Hintergrund", {"Exposure2012": -0.4, "Contrast2012": 15, "Dehaze": 10,
                                                           "Saturation": -20}, "has_subject"),
                  BOTTOM_FADE], group="Fussball"),
    Preset("fb_bw", "Schwarzweiss Drama", "Monochrom mit tiefem Schwarz: Emotionen, Jubel, Porträts nach "
           "dem Spiel.",
           target_log=-2.8, subject_weight=0.9, wb_mode="as_shot", highlight_protect=1.1, shadow_lift=0.6,
           look={"Contrast2012": 28, "Clarity2012": 18, "Texture": 15, "Dehaze": 10, "Saturation": -100,
                 "Sharpness": 50, "GrainAmount": 18, "PostCropVignetteAmount": -25,
                 **_hsl(lum_Red=-10, lum_Orange=10, lum_Green=-25, lum_Blue=-15)},
           curve=[-4, -4, -2, 0, 5, 7, 4],
           masks=[MaskRecipe("subject", "Spieler", {"Exposure2012": 0.35, "Clarity2012": 15, "Texture": 12},
                             "has_subject"),
                  SPOTLIGHT, BOTTOM_FADE], fade=1.2, group="Fussball"),
]
# Nach den Referenzbildern des Nutzers (Nachtspiele FC Winterthur): tiefschwarzer, leicht navyblauer Himmel,
# helle knackige Spieler mit geschützten Weisstönen, sattes dunkles Rasengrün, sehr dunkler weicher Verlauf
# unten, ruhiger dunkler Hintergrund, kein Rauschen.
SIGNATURE_NIGHT = Preset(
    "fb_signature", "Flutlicht Signature", "Nach deinen Referenzbildern: schwarzer Himmel, helle Spieler mit "
    "neutralem Weiss, leuchtendes Rot und Rasengrün, Zuschauer mit Zeichnung, weicher dunkler Verlauf unten.",
    target_log=-2.2, subject_weight=0.8, wb_mode="as_shot", highlight_protect=1.3, shadow_lift=1.4, denoise_bias=10,
    look={"Contrast2012": 14, "Highlights2012": -25, "Shadows2012": 18, "Whites2012": 10, "Blacks2012": -4,
          "Clarity2012": 8, "Texture": 12, "Dehaze": 4, "Vibrance": 20, "Sharpness": 50,
          "PostCropVignetteAmount": -15,
          **_hsl(hue_Green=6, sat_Green=10, lum_Green=-6, hue_Yellow=-4, sat_Yellow=-8, lum_Orange=5,
                 sat_Orange=3, sat_Red=8, lum_Red=2, sat_Blue=-8, lum_Blue=-15, sat_Aqua=-10, sat_Magenta=-25,
                 sat_Purple=-25),
          **_grade(shadow=(225, 5), high=(45, 5))},
    curve=[-2, -2, -1, 0, 3, 4, 2],
    masks=[MaskRecipe("subject", "Spieler", {"Exposure2012": 0.3, "Clarity2012": 12, "Texture": 12,
                                             "Whites2012": 8, "Shadows2012": 10}, "has_subject"),
           MaskRecipe("background", "Hintergrund", {"Exposure2012": -0.2, "Contrast2012": 8, "Dehaze": 5,
                                                    "Saturation": -5}, "has_subject"),
           MaskRecipe("sky", "Himmel", {"Exposure2012": -0.6, "Saturation": -15}, "always"),
           MaskRecipe("gradient_top", "Verlauf oben", {"Exposure2012": -0.35, "Saturation": -10}, "always"),
           BOTTOM_FADE], fade=1.3, group="Fussball")
FOOTBALL.insert(0, SIGNATURE_NIGHT)
PRESETS.update({p.key: p for p in FOOTBALL})


# Weissabgleich bei Sport/Flutlicht: wie aufgenommen (Kamera-Automatik unter LED-Flutlicht ist gut).
# Eine Grauwelt-Neutralisierung kippt Rasenbilder ins Violette, und eigene Temperaturwerte weichen von
# Lightroom ab. Wärme kommt bei den Fussball-Stilen über das Color Grading.


def choose_preset(scene: dict[str, float] | None) -> str:
    if not scene:
        return "sport_floodlight"
    return max(scene.items(), key=lambda kv: kv[1])[0] if scene else "sport_floodlight"


def _neutralize(a: dict[str, Any]) -> tuple[float, float] | None:
    """Grauwelt-Weissabgleich aus den linearen RAW-Mittelwerten -> (Temperatur, Tint)."""
    m = a.get("xyz_to_cam")
    wb = a.get("camera_wb")
    rgb = a.get("lin_rgb_mean")
    if not (m and wb and rgb) or min(rgb) <= 0:
        return None
    mat = np.asarray(m, dtype=float).reshape(3, 3)
    mult = np.asarray(wb[:3], dtype=float) * np.array([rgb[1] / rgb[0], 1.0, rgb[1] / rgb[2]])
    try:
        return multipliers_to_temp_tint(mat, mult)
    except (np.linalg.LinAlgError, ValueError, ZeroDivisionError):
        return None


def measured_log(a: dict[str, Any], subject_weight: float) -> float | None:
    lm = a.get("lin_log_median")
    if lm is None:
        return None
    med = max(float(a.get("median") or 0.4), 1e-3)
    subj = a.get("face_luma") or a.get("subject_luma")
    if subj:
        delta = 2.2 * math.log2(max(float(subj), 1e-3) / med)   # Vorschau ist gamma-kodiert
        return float(lm) + subject_weight * float(np.clip(delta, -3, 3))
    return float(lm)


def denoise_amount(a: dict[str, Any], exposure: float, ds: DenoiseSettings, bias: int = 0) -> int | None:
    sigma = a.get("noise_sigma_mid")
    if sigma is None:
        iso = float(a.get("iso") or 100)
        base = ds.default_amount + 8 * math.log2(max(iso, 100) / 3200)
    else:
        eff = float(sigma) * 2 ** max(exposure, 0.0)
        base = 45 + 12 * math.log2(max(eff, 1e-5) / 0.004)
    amount = int(round(np.clip(base + bias, ds.min_amount, ds.max_amount)))
    if not ds.always and amount <= ds.min_amount:
        return None
    return amount


def apply(preset: Preset, a: dict[str, Any], ds: DenoiseSettings | None = None) -> dict[str, float]:
    """Preset auf ein Bild anwenden -> Zielwerte (Schlüssel wie style.targets.TARGETS)."""
    from .targets import CURVE_X, TARGETS

    ds = ds or DenoiseSettings()
    t: dict[str, float] = {k: 0.0 for k in TARGETS}
    for k in ("Sharpness", "SharpenRadius", "SharpenDetail", "ColorNoiseReduction", "PostCropVignetteMidpoint",
              "PostCropVignetteFeather", "ColorGradeBlending"):
        t[k] = {"Sharpness": 40, "SharpenRadius": 1.0, "SharpenDetail": 25, "ColorNoiseReduction": 25,
                "PostCropVignetteMidpoint": 50, "PostCropVignetteFeather": 50, "ColorGradeBlending": 50}[k]
    t["crop_area"] = 1.0
    t.update(preset.look)
    m = measured_log(a, preset.subject_weight)
    if m is not None:
        exp = float(np.clip(preset.target_log - m, -2.0, 3.0))
    else:
        med = max(float(a.get("median") or 0.45), 0.02)
        exp = float(np.clip(2.2 * math.log2(0.45 / med) * 0.8, -2.0, 2.5))
    t["Exposure2012"] = round(exp, 2)
    p99 = (a.get("lin_log_p99") if a.get("lin_log_p99") is not None else -1.5) + exp
    p05 = (a.get("lin_log_p05") if a.get("lin_log_p05") is not None else -7.0) + exp
    p01 = (a.get("lin_log_p01") if a.get("lin_log_p01") is not None else -9.0) + exp
    clip = float(a.get("raw_clip") or 0.0)
    hl = (20 + 400 * clip + max(0.0, p99 + 1.0) * 25) * preset.highlight_protect
    t["Highlights2012"] = -float(np.clip(hl, 0, 100)) + preset.look.get("Highlights2012", 0)
    sh = max(0.0, (-p05 - 6.5) * 10) * preset.shadow_lift
    t["Shadows2012"] = float(np.clip(sh, 0, 60)) + preset.look.get("Shadows2012", 0)
    t["Whites2012"] = float(np.clip((-0.8 - p99) * 12, -30, 30)) + preset.look.get("Whites2012", 0)
    t["Blacks2012"] = float(np.clip(-5 - max(0.0, p01 + 8.0) * 6, -40, 0)) + preset.look.get("Blacks2012", 0)
    # Weissabgleich
    as_temp, as_tint = a.get("as_shot_temp"), a.get("as_shot_tint")
    if as_temp and preset.wb_mode != "as_shot":
        dm, dt = preset.warm_mired, 0.0
        if preset.wb_mode == "neutralize":
            n = _neutralize(a)
            if n is not None:
                dm += preset.neutralize * float(np.clip(mired(n[0]) - mired(as_temp), -120, 120))
                dt += preset.neutralize * float(np.clip(n[1] - (as_tint or 0), -40, 40))
        if preset.tint_fix and as_tint is not None:
            dt += -preset.tint_fix * float(np.clip(as_tint + dt, -30, 30)) * 0.5
        t["wb_dmired"], t["wb_dtint"], t["wb_custom"] = dm, dt, 1.0
    if preset.curve:
        for x, v in zip(CURVE_X, preset.curve):
            t[f"curve_{x}"] = v
    amt = denoise_amount(a, exp, ds, preset.denoise_bias)
    t["denoise"] = float(amt or 0)
    return t
