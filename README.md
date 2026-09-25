# Imagomat by Mediamatti

Lokale Desktop-App (macOS, Apple Silicon) für Sport- und Konzertfotografie:
**Culling → Personen → individuelle Entwicklung im eigenen Stil → Lightroom Classic.**
Alles läuft lokal, RAW-Originale werden nie verändert, alle Bearbeitungen sind XMP/Katalog.

## Was die App macht

| Modul | Funktion |
|---|---|
| Import | Ordner/Speicherkarte, EXIF inkl. Sub-Sekunden (Serien), eingebettete JPEG-Vorschauen fürs Tempo |
| AI-Culling | Schärfe (Augen/Motiv), Bewegungsunschärfe, Augen zu, abgewandt, angeschnitten, Über-/Unterbelichtung, Serien + Duplikate (bestes pro Serie), Ästhetik; Strenge „behalte ca. X %“; Sterne + Farblabels; jederzeit manuell übersteuerbar; kalibrierbar auf deine eigene Auswahl |
| Personen | Gesichter clustern, Namen vergeben, Wiedererkennung über Shoots; **Rückennummern** (Apple Vision OCR) + Kader (CSV oder Vereinsseite, z. B. FC Winterthur 1. Mannschaft / Frauen / U21); Stichwörter `Personen|Team|Name` + MWG-Gesichtsregionen |
| Stil-Lernen | Aus deinem Lightroom-Katalog **oder jedem Ordner mit RAW/JPEG + XMP** (jede Person kann ihre eigenen XMPs laden); Lightroom-Presets als fester Look; lernt laufend aus Korrekturen |
| Presets | 9 mitgelieferte, **adaptive** Presets: Sport Tag, Sport Flutlicht, Sport Halle, Konzert Bühne, Konzert Club, Porträt, Event, Hochzeit hell, Landschaft; Situation wird automatisch erkannt |
| Entwicklung | pro Bild: WB, Belichtung, Tonwerte, Kurve, HSL, Color Grading, Details, Vignette, Profil, Objektivkorrektur, **Begradigen + Zuschneiden**, **Masken** (Lightroom-KI-Masken, Radial, Verlauf, Pinsel), Shoot-Konsistenz |
| Denoise | Stärke pro Bild aus gemessenem Rauschen; Lightroom rechnet (Wert im XMP) **oder** lokal zur linearen DNG **oder** nur markieren |
| Review | Raster mit Filtern (Sterne, Personen, Grund, unsicher), Lupe, Vorher/Nachher, Tastatur (Pfeile, 0–5, X, P, Enter, `\`, Y) |
| Export | A: RAW + XMP · B: Lightroom-Katalog (experimentell) · C: Galerie-JPEGs · Log (CSV/JSON) · Lightroom-Plugin |

## Schnellstart (Mac)

```bash
git clone https://github.com/Mattomat/Imagomat-by-Mediamatti.git
cd Imagomat-by-Mediamatti
scripts/setup_mac.sh                 # Homebrew-Pakete, Python-Umgebung, Tests, Frontend
cd backend && source .venv/bin/activate
imagomat serve                       # -> http://127.0.0.1:8765
```

Kommandozeile:

```bash
imagomat train "Sport Nacht" --catalog ~/Pictures/Lightroom/Lightroom\ Catalog.lrcat --min-rating 2
imagomat roster "FC Winterthur 1. Mannschaft" --csv kader.csv
imagomat run /Volumes/Untitled/DCIM --profile "Sport Nacht" --keep 0.2 \
    --teams "FC Winterthur 1. Mannschaft" --export ~/Export/FCW-GCZ --formats xmp jpeg
imagomat feedback 3 --profile "Sport Nacht"     # nach Korrekturen in Lightroom
```

Danach in Lightroom: Exportordner importieren (oder Plugin *Imagomat-Export importieren*),
alle auswählen › **Foto › Entwicklungseinstellungen › KI-Einstellungen aktualisieren**.

## Stand und ehrliche Einschränkungen

**Getestet (automatisiert, 19 Tests):** XMP-/Katalog-Roundtrip, Import → Analyse → Culling auf
synthetischen RAWs, Personen/Rückennummern-Logik, Stil-Training aus XMP-Ordner mit Vorhersage
(Belichtungsfehler < 0,35 EV im Test), Feedback-Loop (lernt nur aus tatsächlich geänderten Bildern), lokales Denoise, kompletter Export, API. Oberfläche
im Browser geprüft.

**Nicht getestet, weil hier weder Mac, Lightroom noch KI-Modelle verfügbar waren:**
- Verhalten in Lightroom: Denoise- und KI-Masken-Deklaration (H1/H2), Crop-Vorzeichen bei
  Hochformat, Pinselradius, Katalog-Export → **Checkliste: [docs/lightroom-roundtrip.md](docs/lightroom-roundtrip.md)**
- KI-Modelle (InsightFace, CLIP, BiRefNet, MediaPipe, NAFNet, Apple Vision): Code vorhanden,
  in den Tests liefen die klassischen Fallbacks
- Tauri-Build und das Lightroom-Plugin (Lua): Code vorhanden, auf dem Mac zu bauen bzw. zu laden
- Kader-Import von fcw.ch: generischer Parser, da die Seite von hier aus nicht erreichbar war
- Die Vorschau ist eine Annäherung an Lightroom, nicht identisch

**Lizenzen:** InsightFace-Modelle sind nur nicht-kommerziell nutzbar (Standard ist „auto“ = InsightFace
für beste Qualität). Vor einem Verkauf in den Einstellungen auf YuNet/SFace umstellen.
Übersicht: `imagomat licenses`.

## Dokumentation

- [Machbarkeitsbericht (Phase 0)](docs/phase0/Machbarkeitsbericht.md)
- [Architektur](docs/architektur.md)
- [Lightroom-Roundtrip-Checkliste](docs/lightroom-roundtrip.md)
