"""XMP-Sidecars lesen und schreiben (Lightroom-/Camera-Raw-kompatibel).

Unterstützt:
- Bewertung (xmp:Rating), Farblabel (xmp:Label)
- Stichwörter: dc:subject (flach) und lr:hierarchicalSubject ("Personen|Vorname Nachname")
- Entwicklungseinstellungen im Namespace crs, inklusive verschachtelter Strukturen
  (MaskGroupBasedCorrections, CorrectionMasks, CorrectionRangeMask, Look, Tonkurven)
- Gesichtsregionen nach MWG (mwg-rs:Regions)

Beim Schreiben in ein bestehendes Sidecar bleiben alle Eigenschaften erhalten, die
Imagomat nicht verwaltet. Die RAW-Datei selbst wird nie angefasst.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lxml import etree

from .params import fmt_local, fmt_value

NS = {
    "x": "adobe:ns:meta/",
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "xmp": "http://ns.adobe.com/xap/1.0/",
    "dc": "http://purl.org/dc/elements/1.1/",
    "lr": "http://ns.adobe.com/lightroom/1.0/",
    "crs": "http://ns.adobe.com/camera-raw-settings/1.0/",
    "mwg-rs": "http://www.metadataworkinggroup.com/schemas/regions/",
    "stDim": "http://ns.adobe.com/xap/1.0/sType/Dimensions#",
    "stArea": "http://ns.adobe.com/xmp/sType/Area#",
    "tiff": "http://ns.adobe.com/tiff/1.0/",
    "exif": "http://ns.adobe.com/exif/1.0/",
    "photoshop": "http://ns.adobe.com/photoshop/1.0/",
    "xmpMM": "http://ns.adobe.com/xap/1.0/mm/",
    "Iptc4xmpExt": "http://iptc.org/std/Iptc4xmpExt/2008-02-29/",
}
_PREFIX = {v: k for k, v in NS.items()}
RDF = "{%s}" % NS["rdf"]
CRS = "{%s}" % NS["crs"]

# Diese Eigenschaften verwaltet Imagomat; sie werden beim Mergen ersetzt.
_MANAGED = {
    (NS["xmp"], "Rating"), (NS["xmp"], "Label"), (NS["dc"], "subject"), (NS["lr"], "hierarchicalSubject"),
    (NS["mwg-rs"], "Regions"), (NS["xmp"], "MetadataDate"), (NS["Iptc4xmpExt"], "PersonInImage"),
}


@dataclass
class FaceRegion:
    name: str
    x: float  # Mittelpunkt, normiert (MWG)
    y: float
    w: float
    h: float
    type: str = "Face"


@dataclass
class XmpDoc:
    rating: int | None = None
    label: str | None = None
    keywords: list[str] = field(default_factory=list)   # hierarchisch mit "|"
    crs: dict[str, Any] = field(default_factory=dict)
    regions: list[FaceRegion] = field(default_factory=list)
    people: list[str] = field(default_factory=list)       # IPTC "Person im Bild" (Bilddatenbanken lesen das)
    region_dims: tuple[int, int] | None = None
    other: dict[str, Any] = field(default_factory=dict)  # gelesene, nicht verwaltete Eigenschaften

    @property
    def has_develop(self) -> bool:
        return bool(self.crs)


# ---------------------------------------------------------------------------
# Lesen
# ---------------------------------------------------------------------------

def _qname(tag: str) -> tuple[str, str]:
    if tag.startswith("{"):
        ns, local = tag[1:].split("}", 1)
        return ns, local
    return "", tag


def _key(ns: str, local: str) -> str:
    return f"{_PREFIX.get(ns, ns)}:{local}"


def _parse_struct(el: etree._Element) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for attr, val in el.attrib.items():
        ns, local = _qname(attr)
        if ns in (NS["rdf"], "http://www.w3.org/XML/1998/namespace"):
            continue
        out[local if ns == NS["crs"] else _key(ns, local)] = val
    for child in el:
        if not isinstance(child.tag, str):
            continue
        if child.tag == RDF + "Description":
            out.update(_parse_struct(child))
            continue
        ns, local = _qname(child.tag)
        out[local if ns == NS["crs"] else _key(ns, local)] = _parse_value(child)
    return out


def _parse_li(li: etree._Element) -> Any:
    has_attrs = any(_qname(a)[0] not in (NS["rdf"], "http://www.w3.org/XML/1998/namespace") for a in li.attrib)
    children = [c for c in li if isinstance(c.tag, str)]
    if has_attrs or children or li.get(RDF + "parseType") == "Resource":
        return _parse_struct(li)
    return (li.text or "").strip()


def _parse_value(el: etree._Element) -> Any:
    for container in ("Seq", "Bag", "Alt"):
        c = el.find(RDF + container)
        if c is not None:
            items = [_parse_li(li) for li in c.findall(RDF + "li")]
            if container == "Alt":
                return items[0] if items else ""
            return items
    children = [c for c in el if isinstance(c.tag, str)]
    has_attrs = any(_qname(a)[0] != NS["rdf"] for a in el.attrib)
    if children or has_attrs or el.get(RDF + "parseType") == "Resource":
        return _parse_struct(el)
    return (el.text or "").strip()


def parse_xmp(data: bytes | str) -> XmpDoc:
    if isinstance(data, str):
        data = data.encode("utf-8")
    # XMP-Pakete können von <?xpacket ...?> umgeben sein; lxml kommt damit klar.
    parser = etree.XMLParser(remove_blank_text=True, recover=True, huge_tree=True)
    root = etree.fromstring(data.strip(), parser)
    doc = XmpDoc()
    for desc in root.iter(RDF + "Description"):
        # Nur Top-Level-Beschreibungen (direkt unter rdf:RDF)
        parent = desc.getparent()
        if parent is None or parent.tag != RDF + "RDF":
            continue
        props = _parse_struct(desc)
        for k, v in props.items():
            if ":" not in k:  # crs
                doc.crs[k] = v
            elif k == "xmp:Rating":
                try:
                    doc.rating = int(float(v))
                except (TypeError, ValueError):
                    pass
            elif k == "xmp:Label":
                doc.label = v or None
            elif k == "lr:hierarchicalSubject":
                doc.keywords = [str(x) for x in (v if isinstance(v, list) else [v])]
            elif k == "dc:subject":
                doc.other["dc:subject"] = v if isinstance(v, list) else [v]
            elif k == "mwg-rs:Regions" and isinstance(v, dict):
                _read_regions(doc, v)
            else:
                doc.other[k] = v
    if not doc.keywords and "dc:subject" in doc.other:
        doc.keywords = [str(x) for x in doc.other["dc:subject"]]
    return doc


def _read_regions(doc: XmpDoc, v: dict[str, Any]) -> None:
    dims = v.get("mwg-rs:AppliedToDimensions")
    if isinstance(dims, dict):
        try:
            doc.region_dims = (int(float(dims.get("stDim:w", 0))), int(float(dims.get("stDim:h", 0))))
        except ValueError:
            pass
    for r in v.get("mwg-rs:RegionList", []) or []:
        if not isinstance(r, dict):
            continue
        area = r.get("mwg-rs:Area", {}) or {}
        try:
            doc.regions.append(FaceRegion(
                name=r.get("mwg-rs:Name", ""),
                x=float(area.get("stArea:x", 0)), y=float(area.get("stArea:y", 0)),
                w=float(area.get("stArea:w", 0)), h=float(area.get("stArea:h", 0)),
                type=r.get("mwg-rs:Type", "Face"),
            ))
        except ValueError:
            continue


def read_xmp(path: str | Path) -> XmpDoc:
    return parse_xmp(Path(path).read_bytes())


# ---------------------------------------------------------------------------
# Schreiben
# ---------------------------------------------------------------------------

def _q(prefix: str, local: str) -> str:
    return "{%s}%s" % (NS[prefix], local)


def _is_scalar(v: Any) -> bool:
    return not isinstance(v, (dict, list, tuple))


def _set_struct(el: etree._Element, d: dict[str, Any], top_level: bool) -> None:
    """Schreibt ein Dict als Attribute (Skalare) und Kindelemente (Strukturen/Listen)."""
    for k, v in d.items():
        if v is None:
            continue
        qn = _crs_or_prefixed(k)
        if _is_scalar(v):
            el.set(qn, fmt_value(k, v) if top_level else fmt_local(v))
    for k, v in d.items():
        if v is None or _is_scalar(v):
            continue
        child = etree.SubElement(el, _crs_or_prefixed(k))
        _write_complex(child, v)


def _crs_or_prefixed(k: str) -> str:
    if ":" in k:
        p, local = k.split(":", 1)
        return _q(p, local)
    return CRS + k


def _write_complex(el: etree._Element, v: Any) -> None:
    if isinstance(v, (list, tuple)):
        seq = etree.SubElement(el, RDF + "Seq")
        for item in v:
            li = etree.SubElement(seq, RDF + "li")
            if isinstance(item, dict):
                if all(_is_scalar(x) for x in item.values()):
                    _set_struct(li, item, top_level=False)
                else:
                    desc = etree.SubElement(li, RDF + "Description")
                    _set_struct(desc, item, top_level=False)
            else:
                li.text = fmt_local(item) if isinstance(item, (int, float)) else str(item)
    elif isinstance(v, dict):
        if all(_is_scalar(x) for x in v.values()):
            _set_struct(el, v, top_level=False)
        else:
            desc = etree.SubElement(el, RDF + "Description")
            _set_struct(desc, v, top_level=False)


def _bag(parent: etree._Element, tag: str, items: list[str], container: str = "Bag") -> None:
    el = etree.SubElement(parent, tag)
    bag = etree.SubElement(el, RDF + container)
    for it in items:
        li = etree.SubElement(bag, RDF + "li")
        li.text = it


def flat_keywords(keywords: list[str], include_parents: bool = True) -> list[str]:
    out: list[str] = []
    for kw in keywords:
        parts = [p for p in kw.split("|") if p]
        chosen = parts if include_parents else parts[-1:]
        for p in chosen:
            if p not in out:
                out.append(p)
    return out


def _write_managed(desc: etree._Element, doc: XmpDoc, include_parent_keywords: bool, replace_develop: bool) -> None:
    now = _dt.datetime.now().astimezone().isoformat(timespec="seconds")
    desc.set(_q("xmp", "MetadataDate"), now)
    if doc.rating is not None:
        desc.set(_q("xmp", "Rating"), str(int(doc.rating)))
    if doc.label:
        desc.set(_q("xmp", "Label"), doc.label)
    if replace_develop and doc.crs:
        _set_struct(desc, doc.crs, top_level=True)
    if doc.keywords:
        _bag(desc, _q("dc", "subject"), flat_keywords(doc.keywords, include_parent_keywords))
        _bag(desc, _q("lr", "hierarchicalSubject"), doc.keywords)
    if doc.people:
        _bag(desc, _q("Iptc4xmpExt", "PersonInImage"), list(dict.fromkeys(doc.people)))
    if doc.regions and doc.region_dims:
        _write_regions(desc, doc)


def _write_regions(desc: etree._Element, doc: XmpDoc) -> None:
    regions = etree.SubElement(desc, _q("mwg-rs", "Regions"), {RDF + "parseType": "Resource"})
    w, h = doc.region_dims or (0, 0)
    etree.SubElement(regions, _q("mwg-rs", "AppliedToDimensions"), {
        _q("stDim", "w"): str(w), _q("stDim", "h"): str(h), _q("stDim", "unit"): "pixel"})
    rl = etree.SubElement(regions, _q("mwg-rs", "RegionList"))
    bag = etree.SubElement(rl, RDF + "Bag")
    for r in doc.regions:
        li = etree.SubElement(bag, RDF + "li")
        d = etree.SubElement(li, RDF + "Description", {
            _q("mwg-rs", "Name"): r.name, _q("mwg-rs", "Type"): r.type})
        etree.SubElement(d, _q("mwg-rs", "Area"), {
            _q("stArea", "x"): f"{r.x:.6f}", _q("stArea", "y"): f"{r.y:.6f}",
            _q("stArea", "w"): f"{r.w:.6f}", _q("stArea", "h"): f"{r.h:.6f}",
            _q("stArea", "unit"): "normalized"})


def _new_tree() -> tuple[etree._Element, etree._Element]:
    root = etree.Element(_q("x", "xmpmeta"), nsmap={"x": NS["x"]})
    root.set(_q("x", "xmptk"), "Imagomat XMP Core 1.0")
    rdf = etree.SubElement(root, RDF + "RDF", nsmap={"rdf": NS["rdf"]})
    desc = etree.SubElement(rdf, RDF + "Description", nsmap={k: v for k, v in NS.items() if k not in ("x", "rdf")})
    desc.set(RDF + "about", "")
    desc.set(_q("xmp", "CreatorTool"), "Imagomat")
    return root, desc


def _strip_managed(desc: etree._Element, replace_develop: bool, doc: XmpDoc) -> None:
    for attr in list(desc.attrib):
        ns, local = _qname(attr)
        if (ns, local) in _MANAGED or (replace_develop and ns == NS["crs"]):
            if (ns, local) == (NS["xmp"], "Rating") and doc.rating is None:
                continue
            if (ns, local) == (NS["xmp"], "Label") and doc.label is None:
                continue
            del desc.attrib[attr]
    for child in list(desc):
        if not isinstance(child.tag, str):
            continue
        ns, local = _qname(child.tag)
        if (ns, local) in {(NS["dc"], "subject"), (NS["lr"], "hierarchicalSubject")} and not doc.keywords:
            continue
        if (ns, local) == (NS["mwg-rs"], "Regions") and not doc.regions:
            continue
        if (ns, local) in _MANAGED or (replace_develop and ns == NS["crs"]):
            desc.remove(child)


def serialize(doc: XmpDoc, existing: bytes | None = None, *, include_parent_keywords: bool = True,
              replace_develop: bool = True) -> bytes:
    if existing:
        parser = etree.XMLParser(remove_blank_text=True, recover=True, huge_tree=True)
        root = etree.fromstring(existing.strip(), parser)
        descs = [d for d in root.iter(RDF + "Description") if d.getparent() is not None
                 and d.getparent().tag == RDF + "RDF"]
        if not descs:
            root, desc = _new_tree()
        else:
            for d in descs:
                _strip_managed(d, replace_develop and bool(doc.crs), doc)
            desc = descs[0]
    else:
        root, desc = _new_tree()
    _write_managed(desc, doc, include_parent_keywords, replace_develop)
    etree.cleanup_namespaces(root, top_nsmap={k: v for k, v in NS.items() if k not in ("x",)})
    body = etree.tostring(root, pretty_print=True, encoding="UTF-8", xml_declaration=False)
    return b'<?xpacket begin="\xef\xbb\xbf" id="W5M0MpCehiHzreSzNTczkc9d"?>\n' + body + b'<?xpacket end="w"?>\n'


def sidecar_path(image_path: str | Path) -> Path:
    """Lightroom-Konvention: IMG_0001.ARW -> IMG_0001.xmp"""
    p = Path(image_path)
    return p.with_suffix(".xmp")


def write_sidecar(image_path: str | Path, doc: XmpDoc, *, merge: bool = True, target: Path | None = None,
                  include_parent_keywords: bool = True) -> Path:
    out = target or sidecar_path(image_path)
    existing = out.read_bytes() if merge and out.exists() else None
    out.write_bytes(serialize(doc, existing, include_parent_keywords=include_parent_keywords))
    return out
