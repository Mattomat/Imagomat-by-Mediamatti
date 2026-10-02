"""Kader importieren: CSV oder Webseite (z. B. FC Winterthur, FVRZ).

CSV-Format (Trennzeichen ; oder ,), Kopfzeile optional:
    Name;Nummer;Team;Position
    Max Muster;7;FC Winterthur 1. Mannschaft;Stürmer

Web-Import: Die Seiten der Vereine ändern ihr HTML regelmässig. Der Parser sucht deshalb
generisch nach "Nummer + Name"-Paaren in Kader-Karten bzw. Tabellenzeilen. Das Ergebnis
bitte in der UI prüfen, bevor es gespeichert wird. Die URLs stehen in ``DEFAULT_SOURCES``
und lassen sich ändern.
"""

from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass
from pathlib import Path

from ..db import Database
from .registry import upsert_person

# Vorgeschlagene Kader-Seiten (keine: jeder startet mit eigenen Teams)
DEFAULT_SOURCES: dict[str, str] = {}

_NAME = r"[A-ZÄÖÜÀ-Ý][\w'’\-\.]+(?:\s+(?:de|da|van|von|dos|del|di|le|la)?\s*[A-ZÄÖÜÀ-Ý][\w'’\-\.]+){1,3}"
_POSITIONS = re.compile(
    r"\b(Torhüter(?:in)?|Torwart|Goalie|Verteidiger(?:in)?|Abwehr|Mittelfeld(?:spieler(?:in)?)?|Stürmer(?:in)?|"
    r"Angriff|Goalkeeper|Defender|Midfielder|Forward|Trainer(?:in)?|Captain|Captain|Kapitän(?:in)?)\b", re.I)
_PAIR = re.compile(rf"(?:^|\s)(\d{{1,2}})\s*[\.\-–|:]?\s*({_NAME})")
_PAIR_REV = re.compile(rf"({_NAME})\s*[\(#,\-–|:]?\s*#?(\d{{1,2}})\)?(?:\s|$)")


@dataclass
class RosterEntry:
    name: str
    number: str | None
    team: str | None
    position: str | None = None
    aliases: list[str] | None = None


def parse_csv(text: str, default_team: str | None = None) -> list[RosterEntry]:
    sample = text[:2000]
    delim = ";" if sample.count(";") >= sample.count(",") else ","
    rows = list(csv.reader(io.StringIO(text), delimiter=delim))
    if not rows:
        return []
    header = [h.strip().lower() for h in rows[0]]
    kader_col = next((i for i, h in enumerate(header) if h.startswith("kader")), None)
    alias_col = next((i for i, h in enumerate(header) if "variante" in h or h in ("alias", "aliases")), None)
    has_header = any(h in ("name", "nummer", "number", "team", "position", "vorname", "nachname") for h in header)
    idx = {k: i for i, k in enumerate(header)} if has_header else {"name": 0, "nummer": 1, "team": 2, "position": 3}
    out = []
    for r in rows[1:] if has_header else rows:
        if not r or not any(x.strip() for x in r):
            continue
        get = lambda *keys: next((r[idx[k]].strip() for k in keys if k in idx and idx[k] < len(r)), "")  # noqa: E731
        name = get("name")
        if not name and ("vorname" in idx or "nachname" in idx):
            name = f"{get('vorname')} {get('nachname')}".strip()
        if not name:
            continue
        aliases = [a.strip() for a in r[alias_col].split("|")] if alias_col is not None and alias_col < len(r) else []
        aliases = [a for a in aliases if a]
        if kader_col is not None:
            # z. B. "FCW Herren: Nr. 22" oder "FCW Herren: Nr. 70 | FCW U21: Nr. 16" (erste = Hauptteam)
            kader = r[kader_col].strip() if kader_col < len(r) else ""
            m = re.match(r"\s*([^:|]+?)\s*:\s*(?:Nr\.?|#)?\s*(\d+)", kader)
            if m:
                out.append(RosterEntry(name, m.group(2), m.group(1), None, aliases))
            else:
                out.append(RosterEntry(name, None, None, None, aliases))   # ohne Kader: Team nicht raten
            continue
        num = re.sub(r"\D", "", get("nummer", "number", "nr", "#")) or None
        out.append(RosterEntry(name, num, get("team") or default_team, get("position") or None, aliases))
    return out


def parse_html(html: str, team: str) -> list[RosterEntry]:
    """Generischer Parser für Kaderseiten."""
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        for t in soup(["script", "style", "noscript"]):
            t.decompose()
        blocks = [b.get_text(" ", strip=True) for b in soup.find_all(["li", "tr", "article", "a", "div"])
                  if 3 < len(b.get_text(" ", strip=True)) < 120]
    except ImportError:
        text = re.sub(r"(?is)<(script|style).*?</\1>", " ", html)
        text = re.sub(r"(?i)</?(li|tr|div|article|p|br|h\d)[^>]*>", "\n", text)
        text = re.sub(r"<[^>]+>", " ", text)
        blocks = [re.sub(r"\s+", " ", b).strip() for b in text.split("\n")]
        blocks = [b for b in blocks if 3 < len(b) < 120]
    found: dict[str, RosterEntry] = {}
    for b in blocks:
        b = _POSITIONS.sub(" ", b)
        for m in _PAIR.finditer(b):
            num, name = m.group(1), m.group(2).strip()
            found.setdefault(name, RosterEntry(name, num, team))
        for m in _PAIR_REV.finditer(b):
            name, num = m.group(1).strip(), m.group(2)
            found.setdefault(name, RosterEntry(name, num, team))
    return list(found.values())


def fetch(url: str, team: str) -> list[RosterEntry]:
    import urllib.request

    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 Tagmatti roster import"})
    with urllib.request.urlopen(req, timeout=20) as r:
        html = r.read().decode("utf-8", errors="ignore")
    return parse_html(html, team)


def save(db: Database, entries: list[RosterEntry]) -> int:
    """Bestehende Personen (gleicher Name, auch mit Akzent-/Schreibvarianten) werden ergänzt,
    statt doppelt angelegt. Team/Nummer aus dem Kader überschreiben leere Werte."""
    from .registry import find_person, update_person

    for e in entries:
        pid = find_person(db, e.name, e.aliases)
        if pid is None:
            upsert_person(db, e.name, e.team, e.number)
            continue
        kw: dict = {}
        if e.number:
            kw["number"] = e.number
        if e.team:
            kw["team"] = e.team
        if kw:
            update_person(db, pid, **kw)
    return len(entries)


def import_csv_file(db: Database, path: Path, default_team: str | None = None) -> int:
    return save(db, parse_csv(Path(path).read_text("utf-8-sig"), default_team))
