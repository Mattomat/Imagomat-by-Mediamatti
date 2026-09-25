"""Analyse eines Shoots: Import, Vorschauen, Metriken, Gesichter, Embeddings.

Stufe 1 (parallel, CPU): Vorschau extrahieren, Belichtung/Farbe/Schärfe, RAW-Statistik,
Rauschen, Neigung.
Stufe 2 (gebatcht, GPU/NPU): Gesichter, Embeddings, Segmentierung, OCR.

Jeder Schritt wird pro Bild in ``steps`` vermerkt, sodass ein abgebrochener Job fortsetzt.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import cv2
import imagehash
import numpy as np
from PIL import Image

from .config import IMAGE_EXTENSIONS, cache_dir, load_settings
from .db import Database, dumps, f32_to_blob
from .io import exif as exif_io
from .io import raw as raw_io
from .jobs import JobContext, job
from .vision import quality
from .vision.embeddings import get_embedder, heuristic_scene
from .vision.faces import detect_faces
from .vision.geometry import estimate_tilt
from .vision.segmentation import body_boxes, segment

log = logging.getLogger(__name__)
STEP_PREVIEW, STEP_METRICS, STEP_FACES, STEP_EMBED = "preview", "metrics", "faces", "embed"
PREVIEW_SIDE = 2048


def scan_folder(folder: Path) -> list[Path]:
    files = [p for p in sorted(folder.rglob("*")) if p.suffix.lower() in IMAGE_EXTENSIONS and p.is_file()
             and not p.name.startswith("._")]
    # RAW+JPEG-Paare: nur die RAW verarbeiten
    raws = {p.with_suffix("").as_posix().lower() for p in files if raw_io.is_raw(p)}
    return [p for p in files if raw_io.is_raw(p) or p.with_suffix("").as_posix().lower() not in raws]


def preview_path(image_id: int) -> Path:
    d = cache_dir() / "previews" / f"{image_id // 1000:04d}"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{image_id}.jpg"


def load_cached_preview(row: Any) -> np.ndarray:
    p = Path(row["preview_path"]) if row["preview_path"] else None
    if p and p.exists():
        return cv2.cvtColor(cv2.imread(str(p), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    img, _ = raw_io.load_preview(row["path"], PREVIEW_SIDE)
    return img


# ---------------------------------------------------------------------------
# Import
# ---------------------------------------------------------------------------

def import_folder(db: Database, folder: Path, name: str | None = None, profile: str | None = None,
                  ctx: JobContext | None = None) -> int:
    folder = Path(folder).expanduser().resolve()
    files = scan_folder(folder)
    shoot_id = db.upsert_shoot(name or folder.name, str(folder), profile)
    if ctx:
        ctx.set_total(len(files))
        ctx.progress(0, f"EXIF lesen ({len(files)} Dateien)")
    meta = exif_io.read_exif(files)
    for i, p in enumerate(files):
        m = meta.get(str(p), {})
        db.upsert_image(shoot_id, {
            "path": str(p), "filename": p.name, "capture_time": m.get("capture_time"), "camera": m.get("camera"),
            "lens": m.get("lens"), "iso": m.get("iso"), "exposure_time": m.get("exposure_time"),
            "aperture": m.get("aperture"), "focal_length": m.get("focal_length"),
            "orientation": m.get("orientation") or 1, "exif": m.get("raw", {}),
        })
        if ctx and i % 50 == 0:
            ctx.progress(i, "Import")
    return shoot_id


# ---------------------------------------------------------------------------
# Stufe 1
# ---------------------------------------------------------------------------

def compute_metrics(path: Path) -> tuple[np.ndarray, dict[str, Any], int, int | None, int | None, str]:
    """Stufe-1-Metriken für eine Datei (ohne Datenbank; auch fürs Stil-Training genutzt)."""
    img, orientation = raw_io.load_preview(path, PREVIEW_SIDE)
    data: dict[str, Any] = {"preview_w": img.shape[1], "preview_h": img.shape[0]}
    g = quality.gray(img)
    data.update(quality.exposure_stats(img))
    data.update(quality.color_stats(img))
    data["sharpness"] = quality.sharpness(g)
    data["edge_sharpness"] = quality.edge_sharpness(g)
    data.update({f"motion_{k}": v for k, v in quality.motion_blur(g).items()})
    tilt = estimate_tilt(img, load_settings().develop.max_straighten_deg)
    data.update({"tilt_angle": tilt.angle, "tilt_conf": tilt.confidence, "tilt_source": tilt.source})
    width = height = None
    if raw_io.is_raw(path):
        try:
            lin, info = raw_io.read_linear(path, max_side=1024)
            data.update(quality.linear_stats(lin, info.camera_wb))
            data.update({"as_shot_temp": info.as_shot_temp, "as_shot_tint": info.as_shot_tint,
                         "camera_wb": info.camera_wb.tolist(), "xyz_to_cam": info.xyz_to_cam.ravel().tolist()})
            orientation = info.orientation
            width, height = info.width, info.height
            data.update({f"noise_{k}": v for k, v in raw_io.estimate_noise(path).items()})
        except Exception as e:  # noqa: BLE001 - defekte RAWs sollen den Shoot nicht stoppen
            log.warning("RAW-Analyse fehlgeschlagen für %s: %s", path.name, e)
            data["raw_error"] = str(e)
    data["orientation"] = orientation
    ph = str(imagehash.phash(Image.fromarray(img).resize((256, int(256 * img.shape[0] / img.shape[1])))))
    return img, data, orientation, width, height, ph


def _stage1(db: Database, row: Any) -> None:
    image_id, path = row["id"], Path(row["path"])
    img, data, orientation, width, height, ph = compute_metrics(path)
    pp = preview_path(image_id)
    cv2.imwrite(str(pp), cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
    db.update_analysis(image_id, data, phash=ph)
    with db.tx() as c:
        c.execute("UPDATE images SET preview_path=?, orientation=?, width=COALESCE(?, width),"
                  " height=COALESCE(?, height) WHERE id=?",
                  (str(pp), orientation, width, height, image_id))
    db.mark_step(image_id, STEP_METRICS)


# ---------------------------------------------------------------------------
# Stufe 2
# ---------------------------------------------------------------------------

def compute_content(img: np.ndarray):
    """Gesichter + Motiv/Himmel für ein Vorschaubild (ohne Datenbank)."""
    s = load_settings()
    faces = detect_faces(img, ear_threshold=s.culling.eyes_closed_threshold)
    main = max(faces, key=lambda f: f.height * f.score, default=None)
    data: dict[str, Any] = {
        "face_count": len(faces),
        "max_face_height": max((f.height for f in faces), default=0.0),
        "main_face": main.to_dict() if main else None,
        "eyes_closed_any": any(f.eyes_open is not None and f.eyes_open < 0.5 and f.height > 0.05 for f in faces),
        "faces_cut": any(f.bbox[0] <= 0.002 or f.bbox[2] >= 0.998 or f.bbox[1] <= 0.002 for f in faces),
    }
    if main is not None:
        crop = quality.crop_region(quality.gray(img), main.bbox)
        data["face_luma"] = float(crop.mean()) if crop.size else None
    boxes = [f.bbox for f in faces]
    seg = segment(img, boxes)
    data.update(seg.stats(img))
    data["subject_bbox"] = list(seg.subject_bbox) if seg.subject_bbox else None
    if seg.subject_bbox:
        sub = quality.crop_region(quality.gray(img), seg.subject_bbox)
        data["subject_sharpness"] = quality.edge_sharpness(sub)
        # Angeschnitten = Motiv berührt linken, rechten oder oberen Rand (unten ist bei Halbtotalen normal)
        x0, y0, x1, _ = seg.subject_bbox
        data["subject_cut"] = bool(x0 < 0.005 or x1 > 0.995 or y0 < 0.005)
    return faces, data, seg


def _stage2_faces(db: Database, row: Any, img: np.ndarray) -> list[tuple[float, ...]]:
    faces, data, seg = compute_content(img)
    with db.tx() as c:
        c.execute("DELETE FROM faces WHERE image_id=? AND person_id IS NULL", (row["id"],))
        for f in faces:
            c.execute(
                "INSERT INTO faces(image_id, bbox, landmarks, det_score, embedding, eyes_open, sharpness, yaw)"
                " VALUES(?,?,?,?,?,?,?,?)",
                (row["id"], dumps(list(f.bbox)), dumps(f.kps), f.score, f32_to_blob(f.embedding), f.eyes_open,
                 f.eye_sharpness, f.yaw))
    np.save(_mask_file(row["id"]), np.stack([seg.subject, seg.sky]).astype(np.float16))
    db.update_analysis(row["id"], data)
    db.mark_step(row["id"], STEP_FACES)
    return body_boxes([f.bbox for f in faces], img.shape[1] / img.shape[0])


def _mask_file(image_id: int) -> Path:
    d = cache_dir() / "masks" / f"{image_id // 1000:04d}"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{image_id}.npy"


def load_masks(image_id: int) -> tuple[np.ndarray, np.ndarray] | None:
    p = _mask_file(image_id)
    if not p.exists():
        return None
    m = np.load(p).astype(np.float32)
    return m[0], m[1]


def _stage2_numbers(db: Database, row: Any, img: np.ndarray, bodies: list[tuple[float, ...]]) -> None:
    from .vision import ocr

    if not ocr.available():
        return
    hits = ocr.find_numbers(img)
    with db.tx() as c:
        c.execute("DELETE FROM numbers WHERE image_id=? AND person_id IS NULL", (row["id"],))
        for h in hits:
            c.execute("INSERT INTO numbers(image_id, text, confidence, bbox) VALUES(?,?,?,?)",
                      (row["id"], h.text, h.confidence, dumps(list(h.bbox))))
    db.update_analysis(row["id"], {"numbers": [h.text for h in hits]})


def _stage2_embed(db: Database, rows: list[Any], imgs: list[np.ndarray]) -> None:
    emb = get_embedder()
    vecs = emb.embed(imgs)
    scenes = emb.scenes_from_embeddings(vecs) if hasattr(emb, "scenes_from_embeddings") else None
    aest = emb.aesthetic(vecs)
    for i, row in enumerate(rows):
        data: dict[str, Any] = {"embedder": emb.name}
        a = db.get_analysis(row["id"])
        if scenes is not None:
            data["scene"] = scenes[i]
        else:
            data["scene"] = heuristic_scene({**a, "iso": row["iso"]})
        if aest is not None:
            data["aesthetic"] = float(aest[i])
        db.update_analysis(row["id"], data, embedding=vecs[i])
        db.mark_step(row["id"], STEP_EMBED)


# ---------------------------------------------------------------------------
# Job
# ---------------------------------------------------------------------------

@job("analyze")
def analyze_shoot(ctx: JobContext, shoot_id: int, force: bool = False) -> None:
    db = ctx.db
    rows = db.images(shoot_id)
    total = len(rows) * 2
    ctx.set_total(total)
    done = 0
    todo1 = [r for r in rows if force or not db.step_done(r["id"], STEP_METRICS)]
    done += len(rows) - len(todo1)
    ctx.progress(done, "Vorschauen und Metriken")
    workers = load_settings().workers
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_stage1, db, r) for r in todo1]
        for f in futures:
            ctx.check()
            try:
                f.result()
            except Exception as e:  # noqa: BLE001
                log.warning("Analyse fehlgeschlagen: %s", e)
            done += 1
            ctx.progress(done, "Vorschauen und Metriken")
    rows = db.images(shoot_id)
    todo2 = [r for r in rows if force or not (db.step_done(r["id"], STEP_FACES) and db.step_done(r["id"], STEP_EMBED))]
    done += len(rows) - len(todo2)
    batch_rows: list[Any] = []
    batch_imgs: list[np.ndarray] = []
    for r in todo2:
        ctx.check()
        img = load_cached_preview(r)
        if force or not db.step_done(r["id"], STEP_FACES):
            bodies = _stage2_faces(db, r, img)
            _stage2_numbers(db, r, img, bodies)
        if force or not db.step_done(r["id"], STEP_EMBED):
            small = cv2.resize(img, (448, int(448 * img.shape[0] / img.shape[1])), interpolation=cv2.INTER_AREA)
            batch_rows.append(r)
            batch_imgs.append(small)
            if len(batch_rows) >= 32:
                _stage2_embed(db, batch_rows, batch_imgs)
                batch_rows, batch_imgs = [], []
        done += 1
        ctx.progress(done, "Gesichter, Motiv, Embeddings")
    if batch_rows:
        _stage2_embed(db, batch_rows, batch_imgs)
