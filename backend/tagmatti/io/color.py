"""Weissabgleich: Temperatur/Tint <-> Chromatizität <-> Kamera-Multiplikatoren.

Das Verfahren folgt dem Adobe DNG SDK (dng_temperature, Robertson-Isothermen,
Tint-Skala -3000), damit Werte mit Lightroom vergleichbar sind. Absolute Kelvinwerte können
trotzdem um einige 100 K von Lightroom abweichen, weil Lightroom seine eigenen
Kameraprofile (DCP) nutzt. Das Stilmodell lernt Weissabgleich deshalb relativ zum As-Shot-Wert.
"""

from __future__ import annotations

import math

import numpy as np

_TEMP_TABLE = np.array([
    (0, 0.18006, 0.26352, -0.24341), (10, 0.18066, 0.26589, -0.25479), (20, 0.18133, 0.26846, -0.26876),
    (30, 0.18208, 0.27119, -0.28539), (40, 0.18293, 0.27407, -0.30470), (50, 0.18388, 0.27709, -0.32675),
    (60, 0.18494, 0.28021, -0.35156), (70, 0.18611, 0.28342, -0.37915), (80, 0.18740, 0.28668, -0.40955),
    (90, 0.18880, 0.28997, -0.44278), (100, 0.19032, 0.29326, -0.47888), (125, 0.19462, 0.30141, -0.58204),
    (150, 0.19962, 0.30921, -0.70471), (175, 0.20525, 0.31647, -0.84901), (200, 0.21142, 0.32312, -1.0182),
    (225, 0.21807, 0.32909, -1.2168), (250, 0.22511, 0.33439, -1.4512), (275, 0.23247, 0.33904, -1.7298),
    (300, 0.24010, 0.34308, -2.0637), (325, 0.24702, 0.34655, -2.4681), (350, 0.25591, 0.34951, -2.9641),
    (375, 0.26400, 0.35200, -3.5814), (400, 0.27218, 0.35407, -4.3633), (425, 0.28039, 0.35577, -5.3762),
    (450, 0.28863, 0.35714, -6.7262), (475, 0.29685, 0.35823, -8.5955), (500, 0.30505, 0.35907, -11.324),
    (525, 0.31320, 0.35968, -15.628), (550, 0.32129, 0.36011, -23.325), (575, 0.32931, 0.36038, -40.770),
    (600, 0.33724, 0.36051, -116.45),
])
TINT_SCALE = -3000.0

# Lineares sRGB (D65) <-> XYZ
SRGB_TO_XYZ = np.array([[0.4124564, 0.3575761, 0.1804375],
                        [0.2126729, 0.7151522, 0.0721750],
                        [0.0193339, 0.1191920, 0.9503041]])
XYZ_TO_SRGB = np.linalg.inv(SRGB_TO_XYZ)


def temp_tint_to_xy(temperature: float, tint: float) -> tuple[float, float]:
    r = 1.0e6 / max(temperature, 1.0)
    offset = tint * (1.0 / TINT_SCALE)
    t = _TEMP_TABLE
    for i in range(30):
        if r < t[i + 1, 0] or i == 29:
            f = (t[i + 1, 0] - r) / (t[i + 1, 0] - t[i, 0])
            u = t[i, 1] * f + t[i + 1, 1] * (1 - f)
            v = t[i, 2] * f + t[i + 1, 2] * (1 - f)
            uu1, vv1 = 1.0, t[i, 3]
            uu2, vv2 = 1.0, t[i + 1, 3]
            l1, l2 = math.hypot(uu1, vv1), math.hypot(uu2, vv2)
            uu1, vv1, uu2, vv2 = uu1 / l1, vv1 / l1, uu2 / l2, vv2 / l2
            uu3, vv3 = uu1 * f + uu2 * (1 - f), vv1 * f + vv2 * (1 - f)
            l3 = math.hypot(uu3, vv3)
            u += uu3 / l3 * offset
            v += vv3 / l3 * offset
            d = u - 4.0 * v + 2.0
            return 1.5 * u / d, v / d
    raise AssertionError("unreachable")


def xy_to_temp_tint(x: float, y: float) -> tuple[float, float]:
    d = 1.5 - x + 6.0 * y
    u, v = 2.0 * x / d, 3.0 * y / d
    t = _TEMP_TABLE
    last_dt = last_du = last_dv = 0.0
    for i in range(1, 31):
        du, dv = 1.0, t[i, 3]
        ln = math.hypot(du, dv)
        du, dv = du / ln, dv / ln
        uu, vv = u - t[i, 1], v - t[i, 2]
        dt = -uu * dv + vv * du
        if dt <= 0 or i == 30:
            dt = -min(dt, 0.0)
            f = 0.0 if i == 1 else dt / (last_dt + dt)
            temperature = 1.0e6 / (t[i - 1, 0] * f + t[i, 0] * (1 - f))
            uu = u - (t[i - 1, 1] * f + t[i, 1] * (1 - f))
            vv = v - (t[i - 1, 2] * f + t[i, 2] * (1 - f))
            du, dv = du * (1 - f) + last_du * f, dv * (1 - f) + last_dv * f
            ln = math.hypot(du, dv)
            du, dv = du / ln, dv / ln
            return temperature, (uu * du + vv * dv) * TINT_SCALE
        last_dt, last_du, last_dv = dt, du, dv
    raise AssertionError("unreachable")


def xy_to_xyz(x: float, y: float) -> np.ndarray:
    return np.array([x / y, 1.0, (1.0 - x - y) / y])


def xyz_to_xy(xyz: np.ndarray) -> tuple[float, float]:
    s = float(np.sum(xyz))
    return float(xyz[0] / s), float(xyz[1] / s)


def camera_neutral(xyz_to_cam: np.ndarray, temperature: float, tint: float) -> np.ndarray:
    """Kamera-Neutralwert (wie DNG AsShotNeutral), Grün = 1."""
    n = xyz_to_cam @ xy_to_xyz(*temp_tint_to_xy(temperature, tint))
    return n / n[1]


def wb_multipliers(xyz_to_cam: np.ndarray, temperature: float, tint: float) -> np.ndarray:
    n = camera_neutral(xyz_to_cam, temperature, tint)
    return 1.0 / np.maximum(n, 1e-6)


def neutral_to_temp_tint(xyz_to_cam: np.ndarray, neutral: np.ndarray) -> tuple[float, float]:
    xyz = np.linalg.solve(xyz_to_cam, np.asarray(neutral, dtype=float))
    return xy_to_temp_tint(*xyz_to_xy(xyz))


def multipliers_to_temp_tint(xyz_to_cam: np.ndarray, mult: np.ndarray) -> tuple[float, float]:
    m = np.asarray(mult[:3], dtype=float)
    return neutral_to_temp_tint(xyz_to_cam, 1.0 / np.maximum(m, 1e-6))


def mired(temperature: float) -> float:
    return 1.0e6 / max(temperature, 1.0)


def from_mired(m: float) -> float:
    return 1.0e6 / max(m, 1.0)
