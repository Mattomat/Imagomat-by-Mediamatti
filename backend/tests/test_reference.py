from pathlib import Path

import numpy as np

from imagomat.lightroom.xmp import FaceRegion, XmpDoc
from imagomat.people.reference import assign, name_from_filename, names_from_doc
from imagomat.vision.faces import Face


def test_names_from_keywords_regions_and_filename():
    doc = XmpDoc(keywords=["Personen|FC Winterthur|Max Muster", "Imagomat|Culling|Behalten", "Stadion"],
                 regions=[FaceRegion("Luca Beispiel", 0.5, 0.3, 0.1, 0.15)])
    pairs, regions = names_from_doc(doc, known=set())
    names = dict(pairs)
    assert names["Max Muster"] == "FC Winterthur" and "Luca Beispiel" in names
    assert "Stadion" not in names and "Behalten" not in names
    assert regions[0][0] == "Luca Beispiel" and abs(regions[0][1][0] - 0.45) < 1e-9
    # flache Stichwörter (z. B. aus einem JPG-Export) mit Namensform
    pairs, _ = names_from_doc(XmpDoc(keywords=["Nora Keller", "Fussball Match"]), known=set())
    assert [n for n, _ in pairs] == ["Nora Keller"]
    assert name_from_filename(Path("Max_Muster_03.jpg")) == "Max Muster"
    assert name_from_filename(Path("DSC01234.jpg")) is None


def _face(box, h=None):
    return Face(bbox=box, score=0.9, embedding=np.ones(4, np.float32) / 2)


def test_assign_by_region_and_single_face():
    faces = [_face((0.1, 0.1, 0.2, 0.25)), _face((0.45, 0.22, 0.55, 0.38))]
    regions = [("Luca Beispiel", (0.45, 0.225, 0.55, 0.375))]
    empty = np.zeros((0, 4)), np.zeros(0, int)
    out = assign(faces, ["Luca Beispiel"], regions, *empty, {}, 0.4)
    assert out == {1: "Luca Beispiel"}
    # ein Name, ein klar grösstes Gesicht
    big = [_face((0.3, 0.1, 0.6, 0.6)), _face((0.8, 0.1, 0.85, 0.16))]
    assert assign(big, ["Max Muster"], [], *empty, {}, 0.4) == {0: "Max Muster"}
    # ein Name, zwei gleich grosse Gesichter -> unklar
    same = [_face((0.1, 0.1, 0.3, 0.4)), _face((0.5, 0.1, 0.7, 0.4))]
    assert assign(same, ["Max Muster"], [], *empty, {}, 0.4) == {}


def test_named_faces_from_catalog(tmp_path: Path):
    import sqlite3

    from imagomat.lightroom.catalog import CatalogReader
    from imagomat.lightroom.catalog_writer import CatalogPhoto, write_catalog

    from .lrcat_fixture import make_template

    shoot = tmp_path / "shoot"
    shoot.mkdir()
    f = shoot / "A.ARW"
    f.write_bytes(b"x")
    cat = tmp_path / "c.lrcat"
    write_catalog(make_template(tmp_path / "t.lrcat"), cat, shoot, [CatalogPhoto(path=f, width=6000, height=4000)])
    c = sqlite3.connect(cat)
    img = c.execute("SELECT id_local FROM Adobe_images").fetchone()[0]
    c.execute("INSERT INTO AgLibraryKeyword (id_local, id_global, name, parent, keywordType) VALUES (900, 'K', 'Max Muster', 20, 'person')")
    c.execute("INSERT INTO AgLibraryFace (id_local, id_global, image, tl_x, tl_y, br_x, br_y) VALUES (901, 'F1', ?, 0.4, 0.2, 0.5, 0.35)", (img,))
    c.execute("INSERT INTO AgLibraryFace (id_local, id_global, image, tl_x, tl_y, br_x, br_y) VALUES (902, 'F2', ?, 0.1, 0.1, 0.2, 0.2)", (img,))
    c.execute("INSERT INTO AgLibraryKeywordFace (face, tag, userPick) VALUES (901, 900, 1)")
    c.execute("INSERT INTO AgLibraryKeywordFace (face, tag, userPick, userReject) VALUES (902, 900, 0, 1)")
    c.commit()
    c.close()
    with CatalogReader(cat) as r:
        faces = r.named_faces()
    assert faces == {f: [("Max Muster", (0.4, 0.2, 0.5, 0.35))]}
