"""Vorlage: eigene Lightroom-Bearbeitung (XMP) 1:1 übernehmen, pro Bild nur leicht anpassen."""

from pathlib import Path

XMP = """<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="" xmlns:tiff="http://ns.adobe.com/tiff/1.0/" xmlns:exif="http://ns.adobe.com/exif/1.0/"
 xmlns:crs="http://ns.adobe.com/camera-raw-settings/1.0/"
 tiff:Orientation="8" exif:ExposureTime="1/800" exif:FNumber="28/10"
 crs:ProcessVersion="15.4" crs:WhiteBalance="Custom" crs:Temperature="{temp}" crs:Tint="+40"
 crs:Exposure2012="{exp}" crs:Contrast2012="+5" crs:Highlights2012="-70" crs:Shadows2012="+20"
 crs:Whites2012="-6" crs:Blacks2012="-12" crs:Vibrance="+15" crs:SaturationAdjustmentGreen="-33"
 crs:LuminanceAdjustmentGreen="-46" crs:Sharpness="40" crs:CameraProfile="Adobe Standard"
 crs:CropTop="0.1" crs:CropLeft="0.1" crs:CropBottom="0.9" crs:CropRight="0.9" crs:CropAngle="-3.7"
 crs:HasCrop="True">
 <exif:ISOSpeedRatings><rdf:Seq><rdf:li>4000</rdf:li></rdf:Seq></exif:ISOSpeedRatings>
 <crs:MaskGroupBasedCorrections><rdf:Seq>
  <rdf:li><rdf:Description crs:What="Correction" crs:CorrectionAmount="1" crs:CorrectionName="Maske 1"
    crs:LocalExposure2012="{lexp}" crs:LocalSharpness="-1" crs:LocalClarity2012="-1" crs:LocalTexture="-1">
   <crs:CorrectionMasks><rdf:Seq><rdf:li crs:What="Mask/Gradient" crs:MaskName="Linearer Verlauf 1"
     crs:MaskSyncID="AA" crs:ZeroX="0.3" crs:ZeroY="0.4" crs:FullX="-0.05" crs:FullY="0.4"/></rdf:Seq>
   </crs:CorrectionMasks></rdf:Description></rdf:li>
  <rdf:li><rdf:Description crs:What="Correction" crs:CorrectionAmount="1" crs:CorrectionName="Maske 2"
    crs:LocalWhites2012="0.5" crs:LocalShadows2012="0.11">
   <crs:CorrectionMasks><rdf:Seq><rdf:li crs:What="Mask/Image" crs:MaskName="Motiv 1" crs:MaskSubType="1"
     crs:MaskVersion="1" crs:InputDigest="ABC" crs:MaskDigest="DEF" crs:ReferencePoint="0.4 0.6"
     crs:Origin="835,1080" crs:FullMaskSize="2880,1921"/></rdf:Seq></crs:CorrectionMasks>
  </rdf:Description></rdf:li>
 </rdf:Seq></crs:MaskGroupBasedCorrections>
</rdf:Description></rdf:RDF></x:xmpmeta>"""


def _write(folder: Path, name: str, temp: int = 3600, exp: str = "+1.03", lexp: str = "-0.5") -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    p = folder / f"{name}.xmp"
    p.write_text(XMP.format(temp=temp, exp=exp, lexp=lexp), "utf-8")
    return p


def test_template_keeps_edit_and_masks(tmp_path: Path):
    from imagomat.lightroom.params import to_number
    from imagomat.style.template import learn_template, list_templates, masks_for
    from imagomat.vision.geometry import sensor_to_display

    _write(tmp_path / "x", "A", 3600, "+1.03", "-0.51")
    _write(tmp_path / "x", "B", 3550, "+0.98", "-0.34")
    t = learn_template("Flutlicht", [tmp_path / "x"])
    assert [x["name"] for x in list_templates()] == ["Flutlicht"]
    crs = t["crs"]
    assert t["n"] == 2 and t["ref"]["iso"] == 4000 and abs(t["ref"]["exposure_time"] - 1 / 800) < 1e-9
    assert crs["Exposure2012"] == 1.0 and crs["Temperature"] == 3575
    assert to_number(crs["SaturationAdjustmentGreen"]) == -33 and to_number(crs["LuminanceAdjustmentGreen"]) == -46
    assert "CropTop" not in crs and "HasCrop" not in crs                   # Zuschnitt gehört zum Bild
    m1, m2 = crs["MaskGroupBasedCorrections"]
    assert abs(float(m1["LocalExposure2012"]) - (-0.425)) < 1e-6        # Median beider Vorlagen
    # gleiche Ausrichtung: Geometrie 1:1; KI-Maske ohne bildbezogene Felder (Lightroom rechnet sie neu)
    same = masks_for(t, 8)
    g = same[0]["CorrectionMasks"][0]
    assert (float(g["ZeroX"]), float(g["FullX"])) == (0.3, -0.05)
    ai = same[1]["CorrectionMasks"][0]
    assert ai["MaskSubType"] == "1" and not {"InputDigest", "MaskDigest", "ReferencePoint", "Origin"} & set(ai)
    assert float(same[1]["LocalWhites2012"]) == 0.5
    # Querformat: Verlauf bleibt in der Anzeige unten (hochkant 8 -> quer 1)
    land = masks_for(t, 1)[0]["CorrectionMasks"][0]
    zero = sensor_to_display(float(land["ZeroX"]), float(land["ZeroY"]), 1)
    full = sensor_to_display(float(land["FullX"]), float(land["FullY"]), 1)
    assert full[1] > zero[1] and full[1] > 1.0 and abs(zero[0] - full[0]) < 0.01


def test_template_adjusts_only_light_and_white_balance(tmp_path: Path):
    from imagomat.style.template import adjust, learn_template

    _write(tmp_path / "x", "A")
    t = learn_template("T", [tmp_path / "x"])
    exif = {"iso": 4000, "exposure_time": 1 / 800, "aperture": 2.8}
    a = {"lin_log_median": -6.8, "lin_log_p75": -5.8, "as_shot_temp": 3800, "as_shot_tint": 8}
    ref = {"level": -6.3, "as_shot_temp": 3800, "as_shot_tint": 8}
    vals, _ = adjust(t, a, exif, ref)
    assert vals["Exposure2012"] == 1.03 and vals["Temperature"] == 3600 and vals["Tint"] == 40   # gleich: 1:1
    # eine Blende kürzer belichtet: +1 EV; Szene viel dunkler: höchstens +0.35
    vals, notes = adjust(t, {**a, "lin_log_median": -9.8, "lin_log_p75": -8.8}, {**exif, "exposure_time": 1 / 1600},
                         ref)
    assert abs(vals["Exposure2012"] - (1.03 + 1.0 + 0.35)) < 0.011 and notes
    # Kamera hat wärmeres Licht gemessen: Weissabgleich zieht mit (begrenzt)
    vals, _ = adjust(t, {**a, "as_shot_temp": 3300, "as_shot_tint": 14}, exif, ref)
    assert vals["Temperature"] < 3600 and vals["Tint"] == 46


def test_develop_shoot_with_template(tmp_path: Path):
    import json

    from imagomat.analysis import import_folder
    from imagomat.db import Database
    from imagomat.jobs import JobManager
    from imagomat.style import compare
    import imagomat.pipeline  # noqa: F401 - registriert alle Aufträge
    from imagomat.culling import engine as _cull  # noqa: F401

    from .synth import write_shoot

    _write(tmp_path / "xmp", "A")
    write_shoot(tmp_path / "s", n=3)
    db = Database(tmp_path / "a.db")
    sid = import_folder(db, tmp_path / "s")
    jm = JobManager(db)
    j = db.create_job("learn_template", None, {"name": "Meine", "paths": [str(tmp_path / "xmp" / "A.xmp")]})
    assert jm.run_sync(j)["status"] == "done"
    assert "tpl:Meine" in [s["key"] for s in compare.styles()]
    for k in ("analyze", "cull"):
        assert jm.run_sync(db.create_job(k, sid, {}))["status"] == "done"
    assert jm.run_sync(db.create_job("develop", sid, {"profile": "tpl:Meine", "only_keep": False}))["status"] == "done"
    rows = db.query("SELECT profile, params, masks FROM edits")
    assert rows and all(r["profile"] == "tpl:Meine" for r in rows)
    for r in rows:
        p, masks = json.loads(r["params"]), json.loads(r["masks"])
        assert str(p["LuminanceAdjustmentGreen"]) == "-46" and str(p["Highlights2012"]) == "-70"
        assert [m["CorrectionName"] for m in masks] == ["Maske 1", "Maske 2"]
        assert float(masks[1]["LocalWhites2012"]) == 0.5
    iid = db.images(sid)[0]["id"]
    crs = compare.develop_preview(db, sid, iid, "tpl:Meine")
    assert str(crs["Vibrance"]) == "15"
