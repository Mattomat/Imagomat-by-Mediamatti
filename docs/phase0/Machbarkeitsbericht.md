# Phase 0: Machbarkeitsbericht Tagmatti

Stand: 25.09.2026 (Referenz: Lightroom Classic 15.5, August 2026)
Status: **umgesetzt.** Auf deinen Wunsch wurden alle Punkte gebaut, auch die riskanten
(Denoise-Deklaration, KI-Masken, Katalog-Export). Offene Hypothesen werden mit
[docs/lightroom-roundtrip.md](../lightroom-roundtrip.md) auf deinem Mac geprüft.

---

## Kurzfassung

| Thema | Ergebnis | Empfehlung |
|---|---|---|
| a) Denoise automatisieren | Seit LrC 14.4 ist AI Denoise eine **normale, nicht-destruktive Entwicklungseinstellung** (keine DNG mehr). Sie lässt sich per Vorgabe (Preset) setzen und steht damit auch im XMP. Die Pixeldaten liegen seit LrC 15 in einer `.acr`-Sidecar-Datei bzw. in `.lrcat-data`. Eine API, die Denoise von aussen *berechnet*, gibt es nicht: weder im LrC-SDK noch in der Firefly/Lightroom-API. | Die App schreibt den **Denoise-Wert pro Bild ins XMP**. Nach dem Import wählst du in Lightroom alle Bilder aus und rufst einmal *Foto > Entwicklungseinstellungen > KI-Einstellungen aktualisieren* auf. Fallback: Stichwort + Farblabel + Smart-Sammlung + Denoise-Preset. **Muss im ersten Spike mit echten XMPs verifiziert werden** (Hypothese H1). |
| b) Masken im XMP | Das Format `crs:MaskGroupBasedCorrections` (seit LR 11) ist gut dokumentiert (ExifTool-Strukturen). Verläufe, Radialfilter, Pinsel und Luminanz-/Farbbereich sind **reine Parameter**, die Lightroom exakt so berechnet, wie wir sie schreiben. Bei KI-Masken (Motiv, Himmel, Personen, Hintergrund, Objekte) steht nur die *Absicht* im XMP. Die Maskenbitmap liegt in `.acr`/Katalog und wird von Lightroom neu berechnet. | **Hybrid:** Lightroom-native KI-Masken als Deklaration schreiben (Lightroom berechnet sie in seiner eigenen Qualität). Verläufe, Radial- und Bereichsmasken deterministisch schreiben. Eigene Segmentierung nur als Fallback über Pinsel-Tupfer (`Mask/Paint`). |
| c) Katalog erzeugen | Technisch möglich (SQLite), aber das Schema ist undokumentiert, jede LrC-Hauptversion migriert es, und seit 14.4 gibt es zusätzlich `.lrcat-data`. Das Risiko, deinen Arbeitskatalog zu beschädigen, ist real. | **Kein `.lrcat` schreiben.** Stattdessen Variante A (RAW + XMP) plus optional ein kleines **LrC-Plugin (offizielles Lua-SDK)** für das, was XMP nicht kann: Pick-Flags, Sammlungen pro Person/Culling-Grund, Import in den bestehenden Katalog. |
| d) Stil aus Katalog lesen | Gut machbar: `Adobe_imageDevelopSettings.text` (Lua-Tabelle mit allen Reglern inkl. Masken), dazu EXIF, Bewertungen, Picks, Farblabels und Entwicklungsverlauf aus weiteren Tabellen. Gelesen wird nur eine **Kopie** des Katalogs. | Katalog-Reader als erstes Stück von M3. Der Verlauf (History) liefert zusätzlich „Startwert → Endwert“ als Lernsignal. |
| e) Open Source | Fast alles Nötige gibt es mit kommerziell nutzbarer Lizenz. **Zwei Fallen:** `pyiqa` (PolyForm *Noncommercial*) und die **InsightFace-Modelle** (nur nicht-kommerzielle Forschung, kommerziell nur per Lizenzanfrage). | Lizenzsaubere Alternativen: siehe Tabelle unten. Gesichtserkennung als austauschbares Modul, standardmässig kommerziell unbedenklich. |
| Tempo (1000 Bilder) | Culling auf den eingebetteten JPEG-Vorschauen: ca. 1 bis 2 Minuten. Parameter-Vorhersage: Sekunden. Halbes RAW-Decoding der behaltenen Bilder für Features und Vorschau: ca. 1 bis 3 Minuten. | Ziel „wenige Minuten“ ist realistisch (ohne Denoise, das Lightroom rechnet). |

**Gesamturteil: machbar.** Die kritischen Punkte (Denoise, KI-Masken) lassen sich über
den Mechanismus lösen, den Lightroom selbst für Presets und Synchronisation nutzt: Wir
schreiben die *Einstellungen*, Lightroom rechnet die *Pixel*. Genau diese Annahme
(H1/H2 unten) müssen wir als Erstes mit echten, von dir erzeugten XMPs absichern.

**Ehrlicher Hinweis zur Konkurrenz:** LrC 15.x hat inzwischen selbst *Assisted Culling*
(Augen-Fokus, Augen offen, Motivschärfe), Auto-Stapeln und seit 15.4 eine
Duplikaterkennung und ein Gesichter-Panel. Beim Culling konkurrieren wir also mit
Adobe. Der Mehrwert von Tagmatti liegt in der **Stil-Entwicklung pro Bild**, der
**Personen-Zuordnung über Shoots hinweg** und darin, dass alles in **einem Durchlauf**
passiert.

---

## a) Lightroom AI Denoise

### Was sich geändert hat
- **Bis LrC 14.3:** *Verbessern > Entrauschen* erzeugte eine neue DNG. Von aussen liess
  sich nur markieren, nicht auslösen.
- **LrC 14.4 (Juni 2025):** Denoise, Rohdetails und Super Resolution sind jetzt
  **nicht-destruktive Regler im Details-Bedienfeld**. Es entsteht keine DNG mehr, und
  die Stärke lässt sich nachträglich ändern, ohne neu zu rechnen. Denoise kann in einer
  **Entwicklungsvorgabe** stecken und per Ad-hoc-Entwicklung auf viele Bilder angewendet
  werden. Die Ergebnisdaten landen in `<Katalog>.lrcat-data`. Anfangs wurden auch die
  XMPs riesig (bis ca. 13 MB).
- **LrC 15.0 (Oktober 2025):** Grosse Pixeldaten (KI-Masken, Denoise, Super Resolution,
  Ablenkungen entfernen) gehen in eine separate **`.acr`-Sidecar-Datei**, das XMP bleibt
  klein.
- **XMP-Felder** (laut ExifTool, genaue Semantik für die neue Variante noch zu prüfen):
  `crs:EnhanceDenoiseAlreadyApplied`, `crs:EnhanceDenoiseVersion`,
  `crs:EnhanceDenoiseLumaAmount`, analog `EnhanceDetails*` und `EnhanceSuperResolution*`.

### Möglichkeiten, Denoise von aussen anzustossen

| Weg | Ergebnis |
|---|---|
| LrC-SDK (Lua, `LrDevelopController`) | Kein dokumentierter Aufruf für Denoise oder „KI-Einstellungen aktualisieren“. Regler setzen geht, KI-Berechnung auslösen nach heutigem Stand nicht. |
| Adobe Firefly Services / Lightroom API | Nur Cloud (verletzt „alles lokal“). Endpunkte: Auto Tone, Auto Straighten, Presets, XMP anwenden, Edit. **Kein Denoise.** Scheidet aus. |
| ACR/Photoshop-Scripting | Ginge über Photoshop plus ACR, ist aber ein anderer Workflow, langsam und verlässt Lightroom. Nicht empfohlen. |
| **XMP-Parameter + „KI-Einstellungen aktualisieren“** | **Bevorzugt.** Denoise ist eine normale Einstellung. Fehlt die `.acr`-Datei, markiert Lightroom das Bild mit „KI-Einstellungen müssen neu berechnet werden“, und *Foto > Entwicklungseinstellungen > KI-Einstellungen aktualisieren* arbeitet auf einer Mehrfachauswahl im Raster. Das ist in Foren für KI-Masken aus XMPs belegt. Für Denoise ist es plausibel, aber **noch nicht verifiziert**. |
| Fallback: Markieren | Stichwort `Tagmatti|Denoise` + Farblabel (konfigurierbar) + Smart-Sammlung. In Lightroom dann: Sammlung öffnen, alles auswählen, Denoise-Preset per Ad-hoc-Entwicklung anwenden. Ein Klickpfad, funktioniert sicher. |

### Automatische Entscheidung, welche Bilder Denoise brauchen (M5)
- Merkmale: ISO, Belichtungszeit, Kamera (Sensor-Rauschprofil), **gemessenes Rauschen**
  direkt aus den RAW-Bayer-Daten (robuste Standardabweichung eines Hochpass-Filters in
  flachen, dunklen Regionen; dazu das Schwarzwert-Rauschen aus maskierten Pixeln, falls
  vorhanden) sowie die vorhergesagte Belichtungsanhebung. Wird ein Bild um +1,5 EV
  angehoben, wird sein Rauschen um diesen Faktor sichtbarer.
- Die Denoise-*Stärke* lernt das Stilmodell aus deinem Katalog, sofern du Denoise
  bereits nutzt. Sonst gibt es eine Kurve Rauschen → Stärke mit Standard 50.

### Lokales Open-Source-Denoising: ehrlicher Vergleich

| Ansatz | Lizenz | Einschätzung gegenüber Adobe Denoise |
|---|---|---|
| NAFNet, Restormer, SCUNet | MIT / MIT / Apache-2.0 | Arbeiten auf **fertig entwickelten RGB-Bildern**, trainiert grösstenteils auf Smartphone-Rauschen (SIDD) oder synthetischem Rauschen. Bei Kamera-High-ISO oft wachsartig, Detailverlust in feinen Texturen (Haare, Rasen, Stoff). Ausserdem ist das Ergebnis eine TIFF oder lineare DNG, also **kein RAW-Workflow mehr**. Deutlich unter Adobe. |
| RAW-basierte Forschungsmodelle (SID/ELD-Familie u. a.) | gemischt | Konzeptionell richtig (Rauschen vor dem Demosaicing), aber pro Sensor kalibrierungsbedürftig und ohne produktionsreife, gepflegte Modelle. Hoher Aufwand, unsichere Qualität. |
| darktable / RawTherapee (profiliertes Denoise, Wavelets) | GPL-3.0 | Solide klassische Verfahren, bei hohen ISO klar hinter KI-Verfahren. |
| DxO PureRAW (kommerziell, lokal) | proprietär | Einzige Alternative auf Adobe-Niveau. Liefert lineare DNG. Nur interessant, falls du Lightroom-Denoise nicht willst. |

**Empfehlung:** Denoise bleibt bei Lightroom (Qualität, RAW-Workflow bleibt erhalten).
Lokales Open-Source-Denoising setzen wir höchstens für Variante C ein (schnelle
JPEG-Galerien), dort reicht SCUNet oder NAFNet.

---

## b) Masken im XMP

### Aufbau (gesichert über die ExifTool-Strukturdefinitionen, seit LR 11)

```
crs:MaskGroupBasedCorrections  (rdf:Seq)
  └─ rdf:li  crs:What="Correction"
       crs:CorrectionAmount, crs:CorrectionActive, crs:CorrectionName, crs:CorrectionSyncID
       crs:LocalExposure2012, LocalContrast2012, LocalHighlights2012, LocalShadows2012,
       LocalWhites2012, LocalBlacks2012, LocalTemperature, LocalTint, LocalSaturation,
       LocalClarity2012, LocalTexture, LocalDehaze, LocalHue, LocalToningHue/-Saturation,
       LocalLuminanceNoise, LocalMoire, LocalDefringe, LocalSharpness, (Kurven, Grain …)
       crs:CorrectionMasks (rdf:Seq)          ← Maskenkomponenten, kombinierbar
         └─ rdf:li crs:What="Mask/…"
              crs:MaskActive, MaskName, MaskBlendMode (add/subtract/intersect),
              MaskInverted, MaskSyncID, MaskVersion, MaskSubType,
              ReferencePoint, InputDigest, MaskDigest, WholeImageArea, Origin
              + typspezifische Felder (s. u.)
              crs:CorrectionRangeMask { Type, LumRange, LumMin/Max/Feather,
                                        ColorAmount, SampleType, AreaModels, Invert … }
```

### Maskentypen: was sich extern zuverlässig erzeugen lässt

| Maskentyp | XMP | Parameter | Extern erzeugbar? |
|---|---|---|---|
| Linearer Verlauf | `Mask/Gradient` | `ZeroX/ZeroY/FullX/FullY` (normiert 0 bis 1) | **Ja, exakt.** Lightroom rechnet aus den Parametern. |
| Radialfilter | `Mask/CircularGradient` | `Top/Left/Bottom/Right/Angle/Midpoint/Roundness/Feather/Flipped` | **Ja, exakt.** Ideal für „Gesicht aufhellen“ oder Vignette. |
| Pinsel | `Mask/Paint` | `Dabs` (Liste `d x y`), `Radius`, `Flow`, `CenterWeight`, `MaskValue` | **Ja.** Eine eigene Segmentierungsmaske lässt sich in Tupfer umrechnen (Kreisüberdeckung, einige hundert bis tausend Dabs). Voll editierbar, braucht keine Neuberechnung. Nachteil: grössere XMPs, weiche Kanten nur angenähert. |
| Luminanz-/Farb-/Tiefenbereich | `Mask/RangeMask` bzw. `CorrectionRangeMask` | `LumRange`, `ColorAmount`, `AreaModels` (Farbproben), `SampleType` | **Ja.** Rein parametrisch, wird von Lightroom neu ausgewertet. |
| KI: Motiv, Himmel, Hintergrund, Personen (inkl. Teile wie Gesichtshaut, Haare, Augen), Objekte, Landschaft | `Mask/Image` + `MaskSubType` (+ ggf. Unterkategorie, `ReferencePoint`) | Bitmap **nicht** im XMP, sondern in `.acr` (ab LrC 15) bzw. im Katalog. Im XMP stehen nur `MaskDigest`/`InputDigest`. | **Als Deklaration ja**, Lightroom berechnet die Maske dann selbst (mit „KI-Einstellungen aktualisieren“, auch im Stapel). Bei LrC 15 berichten Nutzer, dass das nach Sync nicht mehr automatisch passiert, der Menübefehl im Raster aber funktioniert. Die genauen `MaskSubType`-Werte und die Adressierung einzelner Personen **müssen wir aus Referenz-XMPs ablesen** (H2). |

### Bewertung
- **Vorteil der Deklaration:** Wir müssen Lightrooms KI-Masken nicht nachbauen und
  bekommen Adobes Maskenqualität, und du kannst die Masken in Lightroom normal
  weiterbearbeiten, einschliesslich der neuen Feather/Edge-Regler aus 15.5.
- **Risiko:** Adobe ändert die KI-Maskenversionen (`MaskVersion`) gelegentlich. Wir
  schreiben deshalb genau die Werte, die deine installierte LrC-Version selbst schreibt
  (aus den Referenz-XMPs übernommen), und prüfen sie bei jedem LrC-Update per
  Roundtrip-Test erneut.
- **Eigene Segmentierung** (BiRefNet, SAM 2) brauchen wir trotzdem: für die
  **Vorhersage** (Motivhelligkeit, Himmelsanteil usw.), für die **Vorschau** in unserer
  App und als **Pinsel-Fallback**, wo Lightroom keinen passenden KI-Typ hat.

### Vorgehen für den Reverse-Engineering-Spike (benötigt dich, ca. 30 Minuten in LrC)
Lightroom kann ich in dieser Umgebung nicht ausführen. Ich brauche von dir **eine RAW
deiner Hauptkamera** und davon virtuelle Kopien mit je genau einer Änderung. Danach
jeweils *Metadaten in Datei speichern* (Strg/Cmd+S) und mir die `.xmp`- und
`.acr`-Dateien schicken:
1. Nur Grundregler (Belichtung +1, Temperatur 5000 K, Tonkurve, HSL, Color Grading)
2. Linearer Verlauf, Radialfilter, Pinsel (je einzeln)
3. Luminanzbereich und Farbbereich
4. KI: Motiv, Himmel, Hintergrund, Personen (alle), Personen (nur eine Person,
   nur Gesichtshaut), Objekt
5. Kombination: Motiv minus Pinsel, invertierter Radialfilter
6. Denoise 50, Denoise 70; Rohdetails; Super Resolution
7. Sterne, Farblabel, Stichwort `Personen|Test Person`, benanntes Gesicht (Personen-Tag)

Daraus erstelle ich XMP-Vorlagen und automatische Roundtrip-Tests. Der Gegentest:
Unsere erzeugten XMPs importierst du in einen **Testkatalog**, dann *KI-Einstellungen
aktualisieren* und *Metadaten speichern*, und das Tool vergleicht das zurückgeschriebene
XMP mit dem, was wir geschrieben haben.

### Weitere XMP-Fallstricke
- **Pick-Flag** (Auswahl/Abgelehnt) steht nicht im XMP, wie von dir vermutet. Deshalb
  Sterne und Farblabels, optional Picks per LrC-Plugin.
- **Farblabels werden als Text gespeichert** (`xmp:Label`). Lightroom ordnet sie nur zu,
  wenn der Text zum aktiven Farbmarkierungssatz passt, bei deutschem Lightroom also
  eventuell „Rot“ statt „Red“. Sonst erscheint das Label weiss. Wir lesen die
  tatsächlich verwendeten Label-Texte aus deinem Katalog (`Adobe_images.colorLabels`)
  und machen sie konfigurierbar.
- **Sidecar-Namen:** `IMG_0001.CR3` → `IMG_0001.xmp`. Bei RAW+JPEG-Paaren mit gleichem
  Namen teilen sich beide das XMP. Das behandeln wir explizit.
- Nach dem Import eines Ordners, den Lightroom schon kennt, braucht es *Metadaten aus
  Dateien lesen*. Bei Neuimport liest Lightroom die XMPs automatisch.
- `crs:ProcessVersion` und `crs:Version` setzen wir auf die Werte deiner LrC-Version,
  sonst schaltet Lightroom auf eine alte Prozessversion zurück.

---

## c) Lightroom-Katalog erzeugen (.lrcat) oder RAW + XMP?

| Kriterium | Neues `.lrcat` aus Template | RAW + XMP-Ordner | LrC-Plugin (Lua-SDK) |
|---|---|---|---|
| Offizielle Schnittstelle | nein (Schema undokumentiert) | ja (XMP ist Adobe-Standard) | ja |
| Risiko bei LrC-Updates | **hoch**: Katalog-Upgrades, neue Tabellen, `.lrcat-data` (seit 14.4), ID-Generatoren, Vorschau-Caches | gering, Formatänderungen sind additiv | gering |
| Masken, Denoise | ja, aber KI-Daten gehören in `.lrcat-data`, das Format ist unbekannt | ja (Deklaration + Neuberechnung) | nur über XMP |
| Picks, Sammlungen, Stapel | ja | nein | Picks und Sammlungen ja |
| Integration in *deinen* Katalog | nur über „Aus Katalog importieren“ | normaler Import | direkt |

**Empfehlung:** Variante A (RAW + XMP) als Standard. Statt Variante B (fertiges
`.lrcat`) bauen wir **B′: ein kleines Lightroom-Classic-Plugin**. Es importiert den
Ordner in deinen aktuellen Katalog, setzt Picks und Ablehnungen anhand unserer
Stichwörter, legt Sammlungen an („Tagmatti / <Shoot> / Personen / …“, „… / Aussortiert
nach Grund“) und zeigt, welche Bilder noch „KI-Einstellungen aktualisieren“ brauchen.
Das ist dieselbe Architektur, die kommerzielle Anbieter nutzen, und sie übersteht
LrC-Updates. **Den Katalog lesen (Stil-Lernen) ist dagegen unkritisch**, weil wir nur
eine Kopie lesen.

---

## d) Stil-Lernen: Entwicklungseinstellungen auslesen

### Aus dem `.lrcat` (SQLite, nur Kopie, niemals das Original öffnen)
| Tabelle | Inhalt |
|---|---|
| `Adobe_images` | Bild-ID, `rating`, `pick`, `colorLabels`, Aufnahmezeit, Ausrichtung, Verweis auf die Datei |
| `AgLibraryFile`, `AgLibraryFolder`, `AgLibraryRootFolder` | Pfad zur RAW-Datei |
| `Adobe_imageDevelopSettings` | Spalte `text`: **alle Entwicklungsparameter als Lua-Tabelle** (`s = { Exposure2012 = 0.35, … MaskGroupBasedCorrections = { … } }`), dazu `processVersion`, `hasDevelopAdjustments` u. a. |
| `Adobe_libraryImageDevelopHistoryStep` | Entwicklungsverlauf (Name, Zeit, Parametersatz). Damit sehen wir Preset beim Import vs. Endstand. |
| `AgHarvestedExifMetadata` | ISO, Blende, Zeit, Brennweite, Kamera- und Objektiv-Referenzen |
| `AgLibraryKeyword*`, `AgLibraryFace*` | Stichwörter, Gesichter und Personen (Startwerte für die Personendatenbank) |

Wir bauen einen kleinen **Lua-Tabellen-Parser** in Python (das Format ist regulär:
Strings, Zahlen, Booleans, verschachtelte Tabellen). Alternativ wandeln wir die Werte
über dieselbe Abbildung in XMP-Namen um, sodass Katalog und XMP denselben Datenpfad
nutzen.

### Aus XMPs (Feedback-Loop, M3/M4)
ExifTool liest `crs:*` komplett, auch die verschachtelten Maskenstrukturen. Wenn du ein
von uns vorbereitetes Bild in Lightroom korrigierst und die Metadaten speicherst, lesen
wir das XMP zurück. Die Differenz „unsere Vorhersage → dein Endwert“ ist genau das
Trainingssignal.

### Datenhygiene (entscheidend für die Qualität)
- Nur Bilder mit echter Bearbeitung verwenden (Verlauf enthält mehr als Import und
  Preset). Optional nur Bilder ≥ X Sterne, gepickte oder exportierte.
- Nur Prozessversion 2012 oder neuer. Kameraprofil als eigenes Merkmal bzw. Zielwert,
  da „+0,5 Belichtung“ unter Adobe Color etwas anderes bedeutet als unter Camera
  Standard.
- Weissabgleich **relativ zum As-Shot-Wert** lernen (Mired-Verschiebung und
  Tint-Differenz), nicht als absoluten Kelvinwert.

### Vorgeschlagene Modellarchitektur (Details in M3)
1. **Merkmale pro Bild:** globales Embedding (SigLIP 2 oder DINOv2, beide Apache-2.0),
   Statistik aus den **linearen RAW-Daten** (log-Luminanz-Histogramm, Perzentile,
   Clipping-Anteile, As-Shot-WB-Multiplikatoren), EXIF (ISO, Zeit, Blende, Brennweite,
   Objektiv, Blitz, Tageszeit), Szenen-Tags (Zero-Shot: innen/aussen, Nacht, Halle,
   Bühne, Schnee …), Gesichts- und Motivstatistik (Anzahl, Grösse, Helligkeit der
   Gesichter, Himmelsanteil).
2. **Vorhersage:** Kombination aus
   - **kNN-Retrieval** über dein Archiv („was hast du bei ähnlichen Bildern gemacht“).
     Robust schon mit wenigen hundert Beispielen pro Stilprofil und gut erklärbar.
   - **Gradient Boosting** (LightGBM oder CatBoost) pro Zielgrösse, mit den
     kNN-Mittelwerten als Zusatzmerkmal. Mit einigen tausend Bildern schlägt das
     erfahrungsgemäss ein End-to-End-Netz und trainiert in Sekunden.
   - Tonkurven und HSL werden per PCA auf wenige Komponenten reduziert, das Kamera- und
     Kreativprofil als Klassifikation.
   - Masken: Klassifikation „welche Masken-Vorlage nutzt du in dieser Situation“
     (z. B. Motiv aufhellen, Himmel abdunkeln), dazu Regression ihrer Werte.
3. **Shoot-Konsistenz:** Bilder nach Lichtsituation clustern (Zeit, As-Shot-WB,
   Embedding) und Weissabgleich und Belichtung innerhalb eines Clusters robust glätten.
4. **Unsicherheit:** Quantil- bzw. Ensemble-Streuung. Unsichere Bilder markiert die
   Review-Ansicht zuerst.
5. **Feedback-Loop:** Korrigierst du einige Bilder eines Shoots, lernt das Modell sofort
   eine Shoot-Verschiebung dazu (Bias pro Cluster) und beim nächsten Training
   dauerhaft.

---

## e) Open-Source-Projekte und Lizenzen

„Kommerziell ok“ bedeutet: nutzbar für deine geschäftliche Fotografie. GPL/AGPL
betreffen nur die **Weitergabe** der App. Solange die App privat bleibt, darf man
GPL-Code nutzen. Soll sie je verteilt werden, rufen wir GPL-Werkzeuge nur als externes
Programm auf (z. B. `darktable-cli`) und kopieren keinen Code.

| Projekt | Lizenz (geprüft) | Nutzung bei uns |
|---|---|---|
| **rawpy / LibRaw** | MIT / LGPL-2.1 oder CDDL | RAW-Decoding, eingebettete Vorschauen (`extract_thumb`). Kern von Import und Vorschau. |
| **ExifTool / PyExifTool** | Perl (Artistic/GPL) / BSD bzw. GPL | Lesen und Schreiben von EXIF, XMP, MWG-Regions. Als dauerhaft laufender Prozess (`-stay_open`) für Tempo. |
| darktable | GPL-3.0 | Referenz: `src/develop/lightroom.c` (Lightroom-XMP-Import), Pipeline-Ideen. Kein Code-Import. |
| RawTherapee | GPL-3.0 | Referenz für Demosaicing, DCP-Profil-Handhabung. |
| digiKam | GPL-2.0 | Ideen: Gesichts-Workflow, Duplikatsuche. Nutzt OpenCV-DNN (YuNet und SFace). |
| Immich | AGPL-3.0 | Ideen: inkrementelles Gesichts-Clustering (Mindestgrösse, Distanzschwelle, später Zuordnung neuer Gesichter). Kein Code. Nutzt InsightFace-Modelle (Lizenzproblem, s. u.). |
| PhotoPrism | AGPL-3.0 | Ideen: Gesichts-Cluster-Workflow in der UI. |
| **pyiqa (IQA-PyTorch)** | ⚠️ **PolyForm Noncommercial 1.0** | **Nicht als Abhängigkeit.** Wir nutzen die Original-Implementierungen einzelner Metriken mit eigener Lizenz, z. B. den LAION Improved Aesthetic Predictor (Apache-2.0) auf CLIP. Ausserdem trainieren wir einen eigenen Culling-Kopf auf deinen Picks. |
| **SAM 2** | Apache-2.0 | Interaktive und promptbasierte Segmentierung (Klick auf Motiv, Pinsel-Fallback). |
| SAM 3 | „SAM License“ (lizenzfrei, kommerziell erlaubt, mit Nutzungsbeschränkungen z. B. Militär) | Text-Prompt-Segmentierung („sky“, „person“). Optional. |
| **BiRefNet** | MIT | Hochwertige Motiv-Freistellung (Features, Vorschau, Pinsel-Fallback). |
| **InsightFace** | Code MIT, ⚠️ **Modelle (z. B. buffalo_l, SCRFD, ArcFace) nur nicht-kommerzielle Forschung**, kommerziell nur per Lizenzanfrage | Nur als optionales Modul, falls du eine Lizenz hast bzw. rein privat nutzt. |
| face_recognition / dlib | MIT / Boost (Modell frei) | Kommerziell unkritisch, aber schwächer bei Profilen, Unschärfe und kleinen Gesichtern (Sport!). |
| **OpenCV Zoo: YuNet + SFace** | MIT / Apache-2.0 | **Standard-Vorschlag** für Erkennung und Embeddings: kommerziell ok, schnell, bewährt (digiKam). |
| EdgeFace, AdaFace | Code BSD-3 / MIT, Gewichte auf Datensätzen mit Forschungslizenzen trainiert | Genauer als SFace, rechtlich grau. Optional nach deiner Entscheidung. |
| **MediaPipe Face Landmarker** | Apache-2.0 | 478 Landmarken + Blendshapes (`eyeBlinkLeft/Right`) → **geschlossene Augen**, Blickrichtung, Kopfpose. |
| **DINOv2** | Apache-2.0 | Bild-Embeddings (Serien und Duplikate, Stilmerkmale). |
| DINOv3 | „DINOv3 License“ (lizenzfrei, kommerziell erlaubt, mit Einschränkungen) | Optional, stärker als DINOv2. |
| **CLIP / OpenCLIP / SigLIP 2** | MIT / MIT / Apache-2.0 | Szenen-Tags per Zero-Shot, Ästhetik-Kopf, Stilmerkmale. |
| **HDBSCAN** | BSD-3 | Gesichts-Clustering. |
| imagehash (pHash) | BSD-2 | Schneller Duplikat-Vorfilter. |
| NAFNet, Restormer, SCUNet | MIT / MIT / Apache-2.0 | Nur für Variante C (Galerie-JPEGs), s. a). |
| PaddleOCR | Apache-2.0 | **Rückennummern** erkennen (Sport). Zusammen mit Kader-CSV oft zuverlässiger als das Gesicht, wenn Spieler abgewandt sind. |

Hinweis: Bei vielen Modellen sind die *Trainingsdaten* nur für Forschung lizenziert,
auch wenn der Code MIT ist. Wo das relevant ist, steht es oben. Ich markiere es auch in
der App (Modell-Manifest mit Lizenzfeld).

---

## Geschwindigkeit: Abschätzung für 1000 RAWs

| Schritt | Pro Bild | 1000 Bilder |
|---|---|---|
| Eingebettetes JPEG extrahieren + dekodieren | 10 bis 30 ms (CPU, parallel) | ca. 10 s |
| Gesichter + Landmarken + Schärfe + Belichtung | 10 bis 20 ms (GPU/NPU) | ca. 20 s |
| Embeddings (DINOv2 bzw. SigLIP, gebatcht) | 5 bis 15 ms | ca. 15 s |
| Serien, Duplikate, Clustering, Culling-Entscheidung | vernachlässigbar | < 5 s |
| Halbes RAW-Decoding der Behaltenen (z. B. 300) für Features und Vorschau | 0,3 bis 1 s (CPU, 8 bis 10 Kerne) | ca. 30 bis 60 s |
| Segmentierung (BiRefNet) der Behaltenen | 30 bis 100 ms | ca. 15 bis 30 s |
| Parameter-Vorhersage + XMP schreiben | < 5 ms + ExifTool im Stapel | ca. 10 s |
| **Summe** | | **ca. 2 bis 3 Minuten** |

Einschränkung: Sony ARW bettet oft nur ca. 1616 × 1080 px ein. Das genügt für
Belichtung, Augen offen und Komposition, für die **Augen-Schärfe** ist es knapp. Für
Grenzfälle dekodieren wir den Gesichtsausschnitt aus dem RAW nach. Canon CR3 und Nikon
NEF haben volle Vorschauen.

---

## Vorschau-Rendering (Modul 7): klare Erwartung
Adobes Rendering (Prozessversion, Tone-Mapping der Highlights/Shadows-Regler,
Profil-Looktables) ist proprietär. **Unsere Vorschau ist eine Annäherung** und wird in
der UI dauerhaft so gekennzeichnet. Plan: rawpy (linear) → Adobe-DCP-Profil (liegt bei
dir lokal mit Lightroom installiert, wir lesen es nur) → eigene Umsetzung der
Grundregler. Optional **kalibrieren wir die Vorschau**: Du exportierst 100 bis 200
Bilder aus Lightroom, und wir lernen eine Korrektur (3D-LUT oder kleines Netz) von
unserer Vorschau auf den Lightroom-Look. Das macht die Vorschau deutlich näher, aber
nie identisch.

---

## Technik-Vorschlag (Vorschau auf Phase 2)
- **Frontend:** Tauri 2 + React/TypeScript. Deutlich kleiner und sparsamer als Electron.
  Das Python-Backend läuft als Sidecar-Prozess.
- **Backend:** Python 3.12, FastAPI (lokal, nur 127.0.0.1) mit WebSocket für
  Fortschritt, PyTorch (MPS bzw. CUDA), ONNX Runtime (CoreML- bzw. CUDA/TensorRT-EP) für
  die Gesichts- und Segmentierungsmodelle.
- **Daten:** SQLite (WAL) für Bilder, Jobs, Personen und Profile. Embeddings als
  Float16-Blobs (sqlite-vec optional).
- **Jobs:** Eigene persistente Queue in SQLite, idempotente Schritte pro Bild, dadurch
  abbrechbar und wiederaufnehmbar.
- **Modelle:** Download beim ersten Start mit Prüfsumme und Lizenz-Manifest.

## Offene Hypothesen (werden als Erstes geprüft)
- **H1:** Ein XMP mit Denoise-Parametern, aber ohne `.acr`, wird von LrC 15.5 beim
  Import erkannt, und „KI-Einstellungen aktualisieren“ im Stapel berechnet Denoise.
- **H2:** Ein XMP mit `Mask/Image` (Motiv, Himmel, Personen), aber ohne `.acr` wird
  ebenso neu berechnet, und die Korrekturwerte bleiben erhalten.
- **H3:** LrC liest MWG-Gesichtsregionen aus dem XMP beim Import als benannte Personen
  ein. Falls nicht, bleiben Stichwörter als sichere Variante.

Wenn H1 oder H2 scheitern, greifen die beschriebenen Fallbacks (Markieren + Preset bzw.
Pinsel-Tupfer und Bereichsmasken). Das Projekt bleibt machbar, aber mit einem
Handgriff mehr in Lightroom.

---

## Quellen
- ExifTool XMP-Tagdefinitionen (crs-Strukturen, Enhance*-Felder): https://github.com/exiftool/exiftool/blob/master/lib/Image/ExifTool/XMP.pm
- darktable Lightroom-Import: https://github.com/darktable-org/darktable/blob/master/src/develop/lightroom.c
- LrC 14.4 Denoise ohne DNG: https://blog.thomasfitzgeraldphotography.com/blog/2025/6/lightroom-144-released-big-changes-to-raw-details-denoise-and-super-resolution
- Denoise ohne DNG (Tim Grey): https://asktimgrey.com/2025/10/20/denoise-without-dng/
- `.acr`-Sidecars: https://helpx.adobe.com/lightroom-classic/help/create-xmp-acr-files.html und https://asktimgrey.com/2026/02/02/new-acr-files-appearing/
- XMP-Grösse und Speicherort von Denoise-Daten: https://community.adobe.com/questions-675/question-how-much-disk-space-is-consumed-by-ai-denoising-an-image-in-lightroom-classic-14-4-982496
- KI-Einstellungen aktualisieren / KI-Masken nach Sync (LrC 15): https://community.adobe.com/questions-675/ai-masks-not-updating-after-sync-and-sometimes-resetting-to-a-blank-mask-in-lightroom-classic-v15-984011
- Lightroom API (Firefly Services): https://developer.adobe.com/firefly-services/docs/lightroom/api/lightroom_autoTone/ und https://developer.adobe.com/firefly-services/docs/photoshop/guides/photoshop-v2-beta/v1-to-v2/v2-api-catalog
- LrC 15.0 Assisted Culling: https://petapixel.com/2025/11/03/lightrooms-new-features-ai-culling-auto-dust-removal-color-variance-slider-and-more/
- LrC 15.4 Gesichter-Panel und Duplikate: https://community.adobe.com/announcements-673/lightroom-classic-v15-4-is-live-cull-people-shots-faster-auto-detect-duplicates-sync-keywords-everywhere-1627070
- LrC 15.5: https://community.adobe.com/announcements-673/lightroom-classic-v15-5-is-here-better-masking-controls-render-to-dng-and-more-1635193
- Lizenzen: jeweils die `LICENSE`- bzw. README-Datei der GitHub-Repositories (am 25.09.2026 abgerufen), u. a. chaofengc/IQA-PyTorch, deepinsight/insightface, facebookresearch/sam2, sam3, dinov2, dinov3, ZhengPeng7/BiRefNet, opencv/opencv_zoo, google-ai-edge/mediapipe
