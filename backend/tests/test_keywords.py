"""Eigene Stichwörter: Bibliothek, Vorschläge, Export, Lightroom-Abgleich."""

from __future__ import annotations

import shutil
import sqlite3
from pathlib import Path

import numpy as np

from imagomat import keywords as K
from imagomat.analysis import import_folder
from imagomat.db import Database
from imagomat.jobs import JobContext

from .synth import write_shoot


def _shoot(tmp_path: Path, n: int = 4):
    write_shoot(tmp_path / "s", n=n)
    db = Database(tmp_path / "a.db")
    sid = import_folder(db, tmp_path / "s")
    ids = [r["id"] for r in db.query("SELECT id FROM images WHERE shoot_id=? ORDER BY filename", (sid,))]
    return db, sid, ids


def test_library_theme_and_states(tmp_path: Path):
    db, sid, ids = _shoot(tmp_path, 3)
    kid = K.get_or_create(db, "  Fahne   Löwe ", "flag with a lion")
    assert K.get_or_create(db, "fahne löwe") == kid                     # gleich, egal welche Schreibweise
    assert "Fans" in K.add_theme(db, "Fussball")
    assert {k["name"] for k in K.library(db)} >= {"Fahne Löwe", "Fans", "Jubel"}
    K.set_state(db, ids[:2], kid, "manual")
    K.set_state(db, [ids[2]], kid, "rejected")
    assert K.image_keywords(db, ids[0]) == ["Fahne Löwe"] and K.image_keywords(db, ids[2]) == []
    row = next(k for k in K.shoot_keywords(db, sid) if k["id"] == kid)
    assert row["assigned"] == 2
    K.set_state(db, [ids[0]], kid, None)
    assert K.image_keywords(db, ids[0]) == []


def test_score_images_text_and_examples():
    rng = np.random.default_rng(0)
    img = rng.normal(size=(40, 16))
    img /= np.linalg.norm(img, axis=1, keepdims=True)
    target = img[3] + img[7]
    target /= np.linalg.norm(target)
    s = K.score_images(img, None, np.stack([img[3]]))
    assert s[3] > 0.9 and (s > 0).sum() <= 3                          # nur wirklich ähnliche Bilder
    text = target[None, :] * 0.3 + rng.normal(size=(1, 16)) * 0.01
    text /= np.linalg.norm(text)
    s2 = K.score_images(img * 0.3, text, None)                        # CLIP-typisch kleine Werte
    assert s2.argmax() in (3, 7)


def test_suggest_job_with_fake_clip(tmp_path: Path, monkeypatch):
    db, sid, ids = _shoot(tmp_path, 4)
    vecs = {i: np.eye(8, dtype=np.float32)[n % 8] for n, i in enumerate(ids)}
    vecs[ids[3]] = (np.eye(8)[0] * 0.95 + np.eye(8)[1] * 0.31).astype(np.float32)   # ähnlich wie Bild 0
    vecs[ids[3]] /= np.linalg.norm(vecs[ids[3]])
    import imagomat.vision.tags as T

    monkeypatch.setattr(T, "clip_embeddings", lambda db_, ids_, progress=None: (object(), vecs))
    kid = K.get_or_create(db, "Maskottchen")
    K.set_state(db, [ids[0]], kid, "manual")                          # Beispielbild
    jid = db.create_job("kw_suggest", sid, {})
    res = K.kw_suggest(JobContext(db, jid), sid, kid)
    sug = [s["image_id"] for s in K.suggestions(db, sid, kid)]
    assert res["suggested"] == len(sug) and ids[3] in sug and ids[0] not in sug
    K.set_state(db, [ids[3]], kid, "confirmed")
    assert K.image_keywords(db, ids[3]) == ["Maskottchen"]


def test_keywords_in_xmp_export(tmp_path: Path):
    from imagomat.export.tagging import tag_export
    from imagomat.lightroom.xmp import read_xmp

    db, sid, ids = _shoot(tmp_path, 2)
    dng = Path(db.one("SELECT path FROM images WHERE id=?", (ids[0],))["path"])
    arw = dng.with_suffix(".ARW")
    shutil.move(dng, arw)
    with db.tx() as c:
        c.execute("UPDATE images SET path=? WHERE id=?", (str(arw), ids[0]))
    K.set_state(db, [ids[0]], K.get_or_create(db, "Choreo"), "manual")
    jid = db.create_job("tag_export", sid, {})
    tag_export(JobContext(db, jid), sid, str(tmp_path / "out"), inplace=True)
    assert "Choreo" in read_xmp(arw.with_suffix(".xmp")).keywords


def test_keywords_lightroom_pull_and_push(tmp_path: Path):
    from imagomat.lightroom.catalog_writer import CatalogPhoto, write_catalog
    from imagomat.lightroom.sync import pull, push

    from .lrcat_fixture import make_template

    write_shoot(tmp_path / "s", n=2)
    files = sorted((tmp_path / "s").glob("*.dng"))
    cat = tmp_path / "c.lrcat"
    write_catalog(make_template(tmp_path / "t.lrcat"), cat, tmp_path / "s",
                  [CatalogPhoto(path=f, width=900, height=600) for f in files])
    c = sqlite3.connect(cat)
    img = c.execute("SELECT i.id_local FROM Adobe_images i JOIN AgLibraryFile f ON i.rootFile=f.id_local "
                    "WHERE f.baseName=?", (files[0].stem,)).fetchone()[0]
    c.execute("INSERT INTO AgLibraryKeyword (id_local, id_global, name, parent) VALUES (950, 'KW', 'Pyro', 20)")
    c.execute("INSERT INTO AgLibraryKeywordImage (image, tag) VALUES (?, 950)", (img,))
    c.commit()
    c.close()
    db = Database(tmp_path / "a.db")
    sid = import_folder(db, tmp_path / "s")
    r = pull(db, sid, cat)
    iid0 = db.one("SELECT id FROM images WHERE path=?", (str(files[0]),))["id"]
    iid1 = db.one("SELECT id FROM images WHERE path=?", (str(files[1]),))["id"]
    assert r["keywords"] == 1 and K.image_keywords(db, iid0) == ["Pyro"]
    K.set_state(db, [iid1], K.get_or_create(db, "Maskottchen"), "manual")
    res = push(db, sid, cat)
    assert res["keywords"] >= 1
    c = sqlite3.connect(cat)
    names = {r[0] for r in c.execute(
        "SELECT k.name FROM AgLibraryKeywordImage ki JOIN AgLibraryKeyword k ON k.id_local=ki.tag")}
    assert {"Pyro", "Maskottchen"} <= names
    assert c.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
