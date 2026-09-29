"""Personen bearbeiten, zusammenführen und Kader-CSV (Format 'FCW Herren: Nr. 22')."""

from pathlib import Path

from imagomat.db import Database
from imagomat.people import roster
from imagomat.people.registry import norm_name, update_person, upsert_person

CSV = '''"Name";"Fotoanzahl";"Kaderzuordnung (Namensabgleich)";"Mögliche Namensvarianten (unbestätigt)"
"Aldin Turkes";"14";"FCW Herren: Nr. 22";""
"Giuliano Foro";"11";"FCW Herren: Nr. 70 | FCW U21: Nr. 16";""
"Theo Golliard";"49";"";"Théo Golliard"
"Alex Sirupkurve";"7";"";""
'''


def test_kader_csv_updates_existing(tmp_path: Path):
    db = Database(tmp_path / "p.db")
    a = upsert_person(db, "Aldin Turkes")
    t = upsert_person(db, "Théo Golliard", "FC Winterthur")
    roster.save(db, roster.parse_csv(CSV, "Personenuebersicht"))
    rows = {r["name"]: dict(r) for r in db.query("SELECT * FROM persons")}
    assert len(rows) == 4                                   # keine Duplikate
    assert rows["Aldin Turkes"]["id"] == a and rows["Aldin Turkes"]["number"] == "22"
    assert rows["Aldin Turkes"]["team"] == "FCW Herren"
    assert rows["Aldin Turkes"]["keyword"] == "Aldin Turkes"
    assert rows["Giuliano Foro"]["number"] == "70"
    assert rows["Théo Golliard"]["id"] == t                 # über Namensvariante gefunden
    assert rows["Alex Sirupkurve"]["team"] is None          # Dateiname wird nicht zum Team


def test_rename_merges(tmp_path: Path):
    db = Database(tmp_path / "m.db")
    a = upsert_person(db, "Noah Makaja")
    b = upsert_person(db, "Noah Makaya", "FCW U21", "20")
    img = db.upsert_shoot("S", str(tmp_path), None)
    iid = db.upsert_image(img, {"path": str(tmp_path / "x.arw"), "filename": "x.arw"})
    with db.tx() as c:
        c.execute("INSERT INTO faces(image_id, bbox, person_id, assigned_by) VALUES(?,?,?,?)",
                  (iid, "[0,0,1,1]", a, "confirmed"))
    new = update_person(db, a, name="Noah Makaya")
    assert new == b
    assert db.one("SELECT COUNT(*) FROM persons")[0] == 1
    assert db.one("SELECT person_id FROM faces")[0] == b
    p = db.one("SELECT * FROM persons WHERE id=?", (b,))
    assert p["team"] == "FCW U21" and p["number"] == "20"
    assert update_person(db, b, number="") == b and db.one("SELECT number FROM persons")[0] is None
    assert norm_name("Jamie-Lee von Allmen") == norm_name("jamie lee von allmen")
