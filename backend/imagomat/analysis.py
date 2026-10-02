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
STEP_ACTION = "action"
STEP_OCR = "ocr"
OCR_VERSION = 2        # 2: Rückennummern + Trikotnamen
METRICS_VERSION = 2    # 2: Apples RAW-Engine/DNG Converter für unbekannte Kameras, Vorschau auf RAW-Skala
ACTION_VERSION = 2     # erhöht, wenn sich die Moment-Erkennung ändert -> wird neu berechnet
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
    add_files(db, shoot_id, files, ctx)
    return shoot_id


def add_files(db: Database, shoot_id: int, files: list[Path], ctx: JobContext | None = None) -> None:
    """Dateien in den Shoot aufnehmen (EXIF lesen); schon bekannte werden aktualisiert."""
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
            lin, info = raw_io.read_linear_any(path, max_side=1024, reference=img)
            data.update(quality.linear_stats(lin, info.camera_wb))
            data.update({"as_shot_temp": info.as_shot_temp, "as_shot_tint": info.as_shot_tint,
                         "camera_wb": info.camera_wb.tolist(), "xyz_to_cam": info.xyz_to_cam.ravel().tolist()})
            orientation = info.orientation
            width, height = info.width, info.height
            source = info.extra.get("source", "libraw")
            data["raw_source"] = source
            if source == "libraw":
                data.update({f"noise_{k}": v for k, v in raw_io.estimate_noise(path).items()})
            else:
                data.update(iso_noise(path))
                _note_unsupported(path, RuntimeError(f"gelesen über {source}"))
        except Exception as e:  # noqa: BLE001 - defekte RAWs sollen den Shoot nicht stoppen
            data["raw_error"] = str(e)
            data.update(preview_linear_stats(img, path))
            _note_unsupported(path, e)
    data["orientation"] = orientation
    ph = str(imagehash.phash(Image.fromarray(img).resize((256, int(256 * img.shape[0] / img.shape[1])))))
    return img, data, orientation, width, height, ph


_UNSUPPORTED_SEEN: set[str] = set()
PREVIEW_TO_RAW_EV = raw_io.PREVIEW_TO_RAW_EV


def _note_unsupported(path: Path, err: Exception) -> None:
    """Einmal pro Kameramodell protokollieren, statt für jedes Bild (sonst ist das Protokoll voll)."""
    from .io.tiffmeta import read_tiff_exif

    ex = read_tiff_exif(path)
    model = f"{ex.get('Make', '')} {ex.get('Model', '')}".strip() or path.suffix.upper()
    if model in _UNSUPPORTED_SEEN:
        return
    _UNSUPPORTED_SEEN.add(model)
    how = str(err).replace("gelesen über ", "") if str(err).startswith("gelesen über") else None
    if how == "coreimage":
        log.warning("LibRaw kennt die Kamera %s noch nicht. Imagomat liest sie über Apples RAW-Engine "
                    "(Weissabgleich gemessen, Rauschen aus ISO geschätzt).", model)
    elif how == "dng_converter":
        log.warning("LibRaw kennt die Kamera %s noch nicht. Imagomat liest sie über den Adobe DNG Converter.", model)
    else:
        log.warning("Die Kamera %s kann weder LibRaw noch Apples RAW-Engine lesen (%s, z. B. %s). Imagomat nutzt "
                    "die eingebettete Vorschau; Weissabgleich und Rauschen werden geschätzt. Tipp: Adobe DNG "
                    "Converter (gratis) installieren.", model, err, path.name)


def preview_linear_stats(img: np.ndarray, path: Path) -> dict[str, Any]:
    """Ersatz-Statistik, wenn LibRaw die RAW nicht lesen kann: linearisierte Vorschau + Rauschen aus ISO."""
    x = cv2.resize(img, (512, int(512 * img.shape[0] / img.shape[1])), interpolation=cv2.INTER_AREA)
    x = x.astype(np.float32) / 255.0
    lin = np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)
    out: dict[str, Any] = dict(quality.linear_stats(lin))
    # Auf die Skala echter RAW-Daten bringen: dort liegt Mittelgrau ca. 3.2 EV unter dem Weisspunkt,
    # in der (sRGB-)Vorschau bei 0.18 (-2.5 EV). Ohne Angleich würden solche Bilder zu dunkel entwickelt.
    for k in list(out):
        if k.startswith("lin_log_"):
            out[k] = float(out[k]) - PREVIEW_TO_RAW_EV
    out["raw_source"] = "preview"
    out.update(iso_noise(path))
    return out


def iso_noise(path: Path) -> dict[str, Any]:
    """Rauschen aus der ISO schätzen (wenn keine Bayer-Daten vorliegen)."""
    from .io.tiffmeta import read_tiff_exif

    iso = read_tiff_exif(path).get("ISO")
    if not (isinstance(iso, (int, float)) and iso > 0):
        return {}
    # Vollformat-Näherung: SNR bei 18 % Grau ~100 bei ISO 100, fällt mit Wurzel(ISO)
    sigma = 0.0018 * (float(iso) / 100.0) ** 0.5
    return {"noise_sigma_mid": sigma, "noise_sigma_shadow": sigma * 0.75, "noise_snr_mid": 0.18 / sigma,
            "noise_a": sigma ** 2 / 0.18, "noise_b": 1e-8, "noise_source": "iso"}


def _stage1(db: Database, row: Any) -> None:
    image_id, path = row["id"], Path(row["path"])
    img, data, orientation, width, height, ph = compute_metrics(path)
    data.setdefault("raw_error", None)           # alten Lesefehler überschreiben, falls es jetzt klappt
    pp = preview_path(image_id)
    cv2.imwrite(str(pp), cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
    db.update_analysis(image_id, data, phash=ph)
    with db.tx() as c:
        c.execute("UPDATE images SET preview_path=?, orientation=?, width=COALESCE(?, width),"
                  " height=COALESCE(?, height) WHERE id=?",
                  (str(pp), orientation, width, height, image_id))
    db.mark_step(image_id, STEP_METRICS, METRICS_VERSION)


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
    hits = ocr.read_shirts(img)
    with db.tx() as c:
        c.execute("DELETE FROM numbers WHERE image_id=? AND text NOT LIKE '#%'", (row["id"],))
        for h in hits:
            c.execute("INSERT INTO numbers(image_id, text, confidence, bbox) VALUES(?,?,?,?)",
                      (row["id"], h.text, h.confidence, dumps(list(h.bbox))))
    db.update_analysis(row["id"], {"numbers": [h.text for h in hits if not h.text.startswith("@")],
                                   "shirt_names": [h.text[1:] for h in hits if h.text.startswith("@")]})
    db.mark_step(row["id"], STEP_OCR, OCR_VERSION)


def ensure_ocr(ctx: JobContext, shoot_id: int) -> None:
    """Trikot-Texte für Bilder nachlesen, die mit einer älteren Version analysiert wurden."""
    from .vision import ocr

    if not ocr.available():
        return
    db = ctx.db
    rows = [r for r in db.images(shoot_id) if db.step_done(r["id"], STEP_METRICS, METRICS_VERSION)
            and not db.step_done(r["id"], STEP_OCR, OCR_VERSION)]
    for i, r in enumerate(rows):
        ctx.check()
        try:
            _stage2_numbers(db, r, load_cached_preview(r), [])
        except Exception as e:  # noqa: BLE001
            log.warning("Trikot-Texte fehlgeschlagen für %s: %s", r["filename"], e)
            db.mark_step(r["id"], STEP_OCR, OCR_VERSION)
        if i % 20 == 0:
            ctx.progress(message=f"Rückennummern lesen {i + 1}/{len(rows)}")


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


def _stage3_action(db: Database, rows: list[Any], imgs: list[np.ndarray]) -> None:
    """Action-Momente: CLIP aus dem gespeicherten Embedding + Pose (falls GPU) + Ersatz."""
    from .vision import action

    emb = get_embedder()
    pose = action.get_pose_detector()
    analyses = [db.get_analysis(r["id"]) for r in rows]
    clips: list[tuple[float, str | None] | None] = [None] * len(rows)
    if emb.name == "clip":
        stored = db.embeddings([r["id"] for r, a in zip(rows, analyses) if a.get("embedder") == "clip"])
        vecs = [stored.get(r["id"]) for r in rows]
        idx = [i for i, v in enumerate(vecs) if v is not None]
        if idx:
            res = action.clip_action(emb, np.stack([vecs[i] for i in idx]))
            for i, c in zip(idx, res):
                clips[i] = c
    poses: list[Any] = [None] * len(rows)
    if pose is not None:
        try:
            poses = pose.analyze(imgs)
        except Exception as e:  # noqa: BLE001
            log.warning("Pose-Erkennung fehlgeschlagen: %s", e)
    for r, a, c, p in zip(rows, analyses, clips, poses):
        db.update_analysis(r["id"], action.combine(a, c, p))
        db.mark_step(r["id"], STEP_ACTION, ACTION_VERSION)


def ensure_action(ctx: JobContext, shoot_id: int) -> None:
    """Action-Momente für Bilder nachrechnen, die noch eine ältere Version haben (schnell)."""
    db = ctx.db
    rows = [r for r in db.images(shoot_id)
            if db.step_done(r["id"], STEP_METRICS, METRICS_VERSION) and not db.step_done(r["id"], STEP_ACTION, ACTION_VERSION)]
    for i in range(0, len(rows), 8):
        ctx.check()
        chunk = rows[i:i + 8]
        try:
            _stage3_action(db, chunk, [load_cached_preview(r) for r in chunk])
        except Exception as e:  # noqa: BLE001
            log.warning("Action-Momente fehlgeschlagen: %s", e)
            for r in chunk:
                db.mark_step(r["id"], STEP_ACTION, ACTION_VERSION)
        ctx.progress(message=f"Momente {min(i + 8, len(rows))}/{len(rows)}")


# ---------------------------------------------------------------------------
# Job
# ---------------------------------------------------------------------------

def refresh_raw_metrics(ctx: JobContext, shoot_id: int) -> int:
    """Bilder, die früher nur über die eingebettete Vorschau gemessen wurden (Kamera damals unbekannt, z. B.
    Sony A7 V), neu messen, sobald der RAW-Leser sie kennt. Danach stimmen Belichtung und Vorschau."""
    import rawpy

    db = ctx.db
    rows = [r for r in db.images(shoot_id) if raw_io.is_raw(Path(r["path"]))
            and db.get_analysis(r["id"]).get("raw_source") == "preview"]
    if not rows:
        return 0
    try:
        with rawpy.imread(rows[0]["path"]):
            pass
    except Exception:  # noqa: BLE001 - Kamera weiterhin unbekannt: nichts zu tun
        return 0
    n = 0
    for i, r in enumerate(rows):
        ctx.check()
        try:
            _stage1(db, r)
            n += 1
        except Exception as e:  # noqa: BLE001
            log.warning("Neu messen fehlgeschlagen für %s: %s", r["filename"], e)
        for p in [cache_dir() / "linear" / f"{r['id']}.npz", *(cache_dir() / "linear").glob(f"{r['id']}_*.npz"),
                  *(cache_dir() / "renders").glob(f"{r['id']}_*.jpg")]:
            p.unlink(missing_ok=True)
        if i % 10 == 0:
            ctx.progress(message=f"RAW-Daten neu lesen {i + 1}/{len(rows)}")
    log.info("%d Bilder mit echten RAW-Daten neu gemessen (vorher nur Vorschau)", n)
    return n


@job("analyze")
def analyze_shoot(ctx: JobContext, shoot_id: int, force: bool = False, light: bool = False) -> None:
    """light=True (nur Personen): ohne Bild-KI (CLIP) und Action-Momente, deutlich schneller."""
    db = ctx.db
    if not light and not force:
        refresh_raw_metrics(ctx, shoot_id)
    rows = db.images(shoot_id)
    total = len(rows) * 3
    ctx.set_total(total)
    done = 0
    todo1 = [r for r in rows if force or not db.step_done(r["id"], STEP_METRICS, METRICS_VERSION)]
    failed = 0
    done += len(rows) - len(todo1)
    ctx.progress(done, "Vorschauen und Metriken")
    workers = load_settings().workers
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [(r, ex.submit(_stage1, db, r)) for r in todo1]
        for r, f in futures:
            ctx.check()
            try:
                f.result()
            except Exception as e:  # noqa: BLE001 - eine defekte Datei soll den Shoot nicht stoppen
                log.warning("Analyse fehlgeschlagen für %s: %s", r["filename"], e, exc_info=True)
                db.update_analysis(r["id"], {"read_error": str(e)[:300]})
                failed += 1
            done += 1
            ctx.progress(done, "Vorschauen und Metriken")
    rows = db.images(shoot_id)
    todo2 = [r for r in rows if force or not (db.step_done(r["id"], STEP_FACES)
                                              and (light or db.step_done(r["id"], STEP_EMBED)))]
    done += len(rows) - len(todo2)
    batch_rows: list[Any] = []
    batch_imgs: list[np.ndarray] = []
    readable = {r["id"] for r in rows if db.step_done(r["id"], STEP_METRICS, METRICS_VERSION)}
    for r in todo2:
        ctx.check()
        if r["id"] not in readable:
            done += 1
            continue
        img = None
        try:
            img = load_cached_preview(r)
            if force or not db.step_done(r["id"], STEP_FACES):
                bodies = _stage2_faces(db, r, img)
                _stage2_numbers(db, r, img, bodies)
        except Exception as e:  # noqa: BLE001
            log.warning("Gesichter/Motiv fehlgeschlagen für %s: %s", r["filename"], e, exc_info=True)
            db.update_analysis(r["id"], {"content_error": str(e)[:300]})
            db.mark_step(r["id"], STEP_FACES)
        if img is not None and not light and (force or not db.step_done(r["id"], STEP_EMBED)):
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
    # Stufe 3: Action-Momente (braucht Embeddings und Gesichter)
    rows = db.images(shoot_id)
    todo3 = [] if light else [r for r in rows if r["id"] in readable
                              and (force or not db.step_done(r["id"], STEP_ACTION, ACTION_VERSION))]
    done += len(rows) - len(todo3)
    for i in range(0, len(todo3), 8):
        ctx.check()
        chunk = todo3[i:i + 8]
        try:
            _stage3_action(db, chunk, [load_cached_preview(r) for r in chunk])
        except Exception as e:  # noqa: BLE001 - Action-Momente sind optional
            log.warning("Action-Momente fehlgeschlagen: %s", e, exc_info=True)
            for r in chunk:
                db.mark_step(r["id"], STEP_ACTION, ACTION_VERSION)
        done += len(chunk)
        ctx.progress(done, "Action-Momente")
    if failed:
        ctx.progress(done, f"{failed} von {len(rows)} Dateien konnten nicht gelesen werden")
        if failed == len(rows):
            first = db.get_analysis(rows[0]["id"]).get("read_error", "")
            raise RuntimeError(f"Keine der {len(rows)} Dateien konnte gelesen werden. Erster Fehler: {first}")
