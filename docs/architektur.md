# Architektur

```
┌──────────────── Tauri 2 (macOS-App) ────────────────┐
│  React/TypeScript-Oberfläche (frontend/src)         │
│  Import · Culling & Review · Personen · Profile ·   │
│  Export · Einstellungen                             │
└──────────────┬──────────────────────────────────────┘
               │ HTTP + WebSocket (nur 127.0.0.1:8765)
┌──────────────▼──────────── Python-Backend (backend/tagmatti) ─────────────────────────┐
│ server/app.py  FastAPI            jobs.py  persistente, abbrechbare Job-Queue         │
│ db.py          SQLite (Bilder, Analysen, Personen, Edits, Jobs)                       │
│                                                                                       │
│ io/       raw.py (rawpy/LibRaw: Vorschau, lineare Daten, Rauschen), exif.py,          │
│           color.py (Kelvin/Tint wie DNG SDK), dng.py (DNG-Writer)                     │
│ vision/   quality (Schärfe, Bewegung, Belichtung), faces (InsightFace/YuNet/Haar,     │
│           Augen offen), embeddings (CLIP + Szenen + Ästhetik), segmentation           │
│           (BiRefNet/klassisch), ocr (Apple Vision/EasyOCR), geometry (Begradigen,     │
│           Zuschnitt), models (Download, Gerät, Lizenzen)                              │
│ analysis.py  Import + Analyse-Job                                                     │
│ culling/  engine (Scores, Serien, Duplikate, Strenge, Sterne/Labels), calibrate       │
│ people/   registry (Personen über Shoots), clustering (HDBSCAN, Rückennummern), roster│
│ style/    features, targets, model (kNN + LightGBM + Preset-Prior), presets (9        │
│           adaptive Situationen), masks (KI/Radial/Verlauf/Pinsel, Vorlagen), develop  │
│           (Konsistenz, Crop, Denoise, Masken), sources (Katalog/XMP/Presets), jobs    │
│ denoise/  local.py (NAFNet/klassisch -> lineare DNG)                                  │
│ render/   pipeline.py (Vorschau-Annäherung)                                           │
│ lightroom/ xmp.py, lua.py, catalog.py (lesen), catalog_writer.py (experimentell),     │
│           params.py, develop.py, dialect.py (gelernte Lightroom-Details)              │
│ export/   exporter.py (RAW+XMP, Katalog, JPEG, Log, Manifest)                         │
└───────────────────────────────────────────────────────────────────────────────────────┘
               │ tagmatti.json + XMP
┌──────────────▼───────────── Lightroom Classic ──────────────┐
│ Import (liest XMP) · Tagmatti.lrplugin (Picks, Sammlungen,  │
│ Auswahl für „KI-Einstellungen aktualisieren“)                │
└─────────────────────────────────────────────────────────────┘
```

## Datenfluss eines Shoots

1. **Import**: Dateien + EXIF (ExifTool, sonst exifread) → `images`.
2. **Analyse** (Job `analyze`): eingebettete Vorschau (ARW: 1616 px), Metriken, lineare
   RAW-Statistik (2×2-Binning, kein Demosaicing), Rauschschätzung, Neigung → Gesichter,
   Motiv/Himmel, Rückennummern, Embeddings, Szenen, Ästhetik. Wiederaufnehmbar pro Bild.
3. **Culling** (`cull`): Shoot-normierte Scores, harte Gründe, Serien (Zeit + Embedding +
   pHash), Auswahl nach Strenge, Sterne/Labels. Manuelle Entscheidungen bleiben.
4. **Personen** (`people`): Wiedererkennen → Rückennummern → Clustern.
5. **Entwicklung** (`develop`): Stilmodell oder Preset → Shoot-Konsistenz → crs-Einstellungen
   inkl. Begradigen/Zuschnitt, Masken und Denoise.
6. **Export** (`export`): Kopien + XMP (oder DNG), optional Katalog/JPEG, Log, Manifest.
7. **Feedback** (`feedback`): Korrigierte XMPs zurücklesen → Profil nachtrainieren.

## Stilmodell (Kurzfassung)

- Merkmale: ~95 Zahlen (RAW-Helligkeitsverteilung, As-Shot-WB, EXIF, Farbe, Gesichter,
  Motiv/Himmel, Szenen) + PCA des CLIP-Embeddings.
- Ziele: ~85 Werte (WB relativ zum As-Shot, Grundregler, Kurve, HSL, Color Grading als
  Vektoren, Details, Effekte, Kalibrierung, Denoise, Zuschnitt). Belichtung wird als
  Ausgabehelligkeit gelernt.
- kNN (Top-10) + LightGBM je Ziel (nur wenn es kNN in der Kreuzvalidierung schlägt).
- Prior: adaptives Preset, Gewicht 12/(n+12).
- Masken: Vorlagen = häufige Maskenkombinationen aus deinen Daten; Nutzung per kNN +
  logistischer Regression, Werte per kNN, Geometrie relativ zum Motiv.
- Unsicherheit → Stichwort `Tagmatti|Prüfen` + gelbes Label.

## Tests

`cd backend && TAGMATTI_OFFLINE=1 python -m pytest` – läuft ohne KI-Modelle (klassische
Fallbacks) auf synthetischen Bayer-DNGs, die LibRaw wie echte RAWs liest:
XMP/Lua/Katalog-Roundtrip, Import→Analyse→Culling, Personen/Rückennummern, Stil-Training aus
einem XMP-Ordner mit Vorhersage auf neuem Shoot, lokales Denoise, kompletter Export, API.
