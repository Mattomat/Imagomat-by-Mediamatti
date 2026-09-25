from pathlib import Path

from imagomat.lightroom.catalog import CatalogReader
from imagomat.lightroom.catalog_writer import CatalogPhoto, write_catalog
from imagomat.lightroom.dialect import learn_dialect
from imagomat.lightroom.develop import crs_to_lua, lua_to_crs, strip_digests
from imagomat.lightroom.lua import dump_lua, parse_lua
from imagomat.lightroom.xmp import FaceRegion, XmpDoc, parse_xmp, serialize, write_sidecar

from .lrcat_fixture import make_template

MASKS = [{
    "What": "Correction", "CorrectionAmount": 1.0, "CorrectionActive": True, "CorrectionName": "Motiv",
    "LocalExposure2012": 0.1,
    "CorrectionMasks": [
        {"What": "Mask/Image", "MaskSubType": "1", "MaskActive": True, "MaskName": "Motiv 1", "MaskDigest": "ABC"},
        {"What": "Mask/CircularGradient", "Top": 0.1, "Left": 0.2, "Bottom": 0.5, "Right": 0.6, "Feather": 50,
         "Roundness": 0, "Midpoint": 50, "Angle": 0, "Flipped": True, "MaskBlendMode": 1},
    ],
}]


def test_xmp_roundtrip_with_masks_and_regions():
    doc = XmpDoc(rating=3, label="Grün", keywords=["Personen|FCW|Max Muster"],
                 crs={"ProcessVersion": "11.0", "Exposure2012": -0.5, "Tint": 7, "HasSettings": True,
                      "ToneCurvePV2012": ["0, 0", "255, 255"], "MaskGroupBasedCorrections": MASKS},
                 regions=[FaceRegion("Max Muster", 0.4, 0.3, 0.1, 0.12)], region_dims=(7008, 4672))
    back = parse_xmp(serialize(doc))
    assert back.rating == 3 and back.label == "Grün"
    assert back.keywords == ["Personen|FCW|Max Muster"]
    assert back.crs["Exposure2012"] == "-0.50"
    assert back.crs["Tint"] == "+7"
    masks = back.crs["MaskGroupBasedCorrections"][0]["CorrectionMasks"]
    assert masks[0]["What"] == "Mask/Image" and masks[1]["Flipped"] == "true"
    assert back.regions[0].name == "Max Muster" and back.region_dims == (7008, 4672)


def test_merge_keeps_foreign_properties(tmp_path: Path):
    raw = tmp_path / "A.ARW"
    raw.write_bytes(b"raw")
    existing = (b'<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
                b'<rdf:Description rdf:about="" xmlns:photoshop="http://ns.adobe.com/photoshop/1.0/" '
                b'xmlns:crs="http://ns.adobe.com/camera-raw-settings/1.0/" photoshop:City="Winterthur" '
                b'crs:Exposure2012="+1.00"/></rdf:RDF></x:xmpmeta>')
    raw.with_suffix(".xmp").write_bytes(existing)
    write_sidecar(raw, XmpDoc(rating=5, crs={"Exposure2012": 0.2}))
    back = parse_xmp(raw.with_suffix(".xmp").read_bytes())
    assert back.other["photoshop:City"] == "Winterthur"
    assert back.crs["Exposure2012"] == "+0.20"
    assert back.rating == 5
    assert raw.read_bytes() == b"raw"  # RAW unverändert


def test_lua_crs_conversion():
    lua = parse_lua('s = { Exposure2012 = 0.35, ToneCurvePV2012 = { 0, 0, 128, 140, 255, 255, }, '
                    'AutoLateralCA = 1, LensProfileEnable = 1, Look = { Name = "Adobe Color", Amount = 1, }, }')
    crs = lua_to_crs(lua)
    assert crs["ToneCurvePV2012"] == ["0, 0", "128, 140", "255, 255"]
    back = crs_to_lua(crs)
    assert back["ToneCurvePV2012"] == [0, 0, 128, 140, 255, 255]
    assert parse_lua(dump_lua(back))["Exposure2012"] == 0.35


def test_strip_digests():
    s = strip_digests({"MaskGroupBasedCorrections": MASKS})
    assert "MaskDigest" not in s["MaskGroupBasedCorrections"][0]["CorrectionMasks"][0]


def test_catalog_write_and_read(tmp_path: Path):
    template = make_template(tmp_path / "template.lrcat")
    shoot = tmp_path / "shoot"
    (shoot / "sub").mkdir(parents=True)
    photos = []
    for i in range(3):
        f = shoot / "sub" / f"DSC0000{i}.ARW"
        f.write_bytes(b"x")
        photos.append(CatalogPhoto(
            path=f, width=7008, height=4672, orientation=6 if i == 1 else 1, capture_time=1_700_000_000 + i,
            rating=4, pick=1 if i else -1, color_label="Grün",
            keywords=["Personen|FC Winterthur|Max Muster", "Imagomat|Denoise"],
            develop={"ProcessVersion": "11.0", "Exposure2012": "+0.40", "WhiteBalance": "Custom",
                     "Temperature": "4300", "MaskGroupBasedCorrections": MASKS},
            collections=["Imagomat|Testshoot|Behalten"], stack=7, stack_position=i, iso=6400,
            aperture=2.8, shutter=1 / 1000, focal_length=200))
    out = tmp_path / "out" / "Imagomat.lrcat"
    res = write_catalog(template, out, shoot, photos)
    assert res["warnings"] == []
    with CatalogReader(out) as r:
        imgs = list(r.images())
        assert len(imgs) == 3
        im = imgs[0]
        assert im.path == shoot / "sub" / "DSC00000.ARW"
        assert im.develop["Exposure2012"] == 0.4
        assert im.develop["MaskGroupBasedCorrections"][0]["CorrectionMasks"][0]["What"] == "Mask/Image"
        assert "Personen|FC Winterthur|Max Muster" in im.keywords
        assert im.pick == -1 and imgs[1].pick == 1
        assert abs(im.aperture - 2.8) < 0.05 and abs(im.shutter - 0.001) < 1e-6
        assert r.color_labels() == {"Grün": 3}
    # Schlüsselwort-Hierarchie wird nicht doppelt angelegt
    import sqlite3
    c = sqlite3.connect(out)
    assert c.execute("SELECT COUNT(*) FROM AgLibraryKeyword WHERE name='Personen'").fetchone()[0] == 1
    assert c.execute("SELECT COUNT(*) FROM AgLibraryFolderStackImage").fetchone()[0] == 3
    counter = int(c.execute("SELECT value FROM Adobe_variablesTable WHERE name='Adobe_entityIDCounter'").fetchone()[0])
    assert counter > c.execute("SELECT MAX(id_local) FROM Adobe_images").fetchone()[0]


def test_dialect_learning():
    crs = [{"ProcessVersion": "15.4", "Version": "18.0", "EnhanceDenoiseLumaAmount": str(a),
            "EnhanceDenoiseVersion": "2",
            "MaskGroupBasedCorrections": [{"CorrectionMasks": [
                {"What": "Mask/Image", "MaskName": "Himmel 1", "MaskSubType": "7", "MaskDigest": "X"}]}]}
           for a in (40, 55, 60)]
    d = learn_dialect(crs, labels=["Grün", "Rot"])
    assert d.process_version == "15.4" and d.crs_version == "18.0"
    assert d.denoise_amount_key == "EnhanceDenoiseLumaAmount"
    assert d.denoise(50) == {"EnhanceDenoiseVersion": "2", "EnhanceDenoiseLumaAmount": "50"}
    assert d.ai_masks["sky"]["MaskSubType"] == "7" and "MaskDigest" not in d.ai_masks["sky"]
    assert d.label("green") == "Grün"
