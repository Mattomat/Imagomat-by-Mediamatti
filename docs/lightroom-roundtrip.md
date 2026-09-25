# Lightroom-Roundtrip: Checkliste zum Verifizieren

Einige Details des Lightroom-Formats sind nicht öffentlich dokumentiert. Imagomat enthält
dafür Annahmen, die **auf deinem Mac mit Lightroom Classic 15.5.1** geprüft werden müssen.
Alles läuft in einem **Testkatalog**, dein Arbeitskatalog bleibt unberührt.

Dauer: ca. 30 bis 45 Minuten. Schick mir danach die XMP-/`.acr`-Dateien aus Schritt 1 und
die Ergebnisse (OK / nicht OK + Screenshot) der Schritte 2 bis 7.

## 0. Vorbereitung

1. Lightroom: *Datei › Neuer Katalog* → `Imagomat-Test.lrcat`. Lightroom danach **schliessen**
   und eine Kopie dieses leeren Katalogs als Vorlage aufheben (`Imagomat-Vorlage.lrcat`,
   wird für den Katalog-Export gebraucht).
2. Einen kleinen Testordner mit 10 bis 20 ARW-Dateien eines Matches oder Konzerts anlegen.

## 1. Referenz-XMPs (lehrt Imagomat deinen „Lightroom-Dialekt“)

Eine RAW importieren, fünf virtuelle Kopien anlegen, pro Kopie genau eins davon:

| Kopie | Einstellung |
|---|---|
| A | Belichtung +1, Temperatur 4300 K, Tint +10, Tonkurve (Punktkurve), HSL Grün −20, Color Grading Schatten blau |
| B | Maske *Motiv auswählen* (Belichtung +0,3), Maske *Himmel* (−0,3), Maske *Personen › alle* und *Personen › Gesichtshaut* |
| C | Radialfilter, linearer Verlauf, Pinsel (ein paar Striche) |
| D | Denoise 50 (Details-Bedienfeld), bei einer weiteren Kopie Denoise 70 |
| E | Zuschnitt 4:5, gedreht um +3°, Bild im Hochformat |

Dann *Metadaten › Metadaten in Datei speichern*. Da es virtuelle Kopien sind, pro Einstellung
eine eigene Master-RAW verwenden oder nacheinander speichern und die `.xmp` (plus eventuell
`.acr`) jeweils wegkopieren.

Danach in Imagomat: `imagomat dialect <Ordner mit den XMPs>` bzw. Einstellungen ›
Lightroom-Dialekt. Imagomat übernimmt Prozessversion, Denoise-Felder und KI-Masken-Typen
direkt aus diesen Dateien.

## 2. H1: Denoise aus dem XMP

1. Testordner mit Imagomat verarbeiten, Export **A (RAW + XMP)**, Denoise-Modus „Lightroom rechnet“.
2. Exportordner in den Testkatalog importieren.
3. Alle Bilder auswählen › *Foto › Entwicklungseinstellungen › KI-Einstellungen aktualisieren*.
4. Prüfen: Steht im Details-Bedienfeld Denoise mit dem vorhergesagten Wert, und ist das Bild entrauscht?

Falls nein: Einstellungen › Denoise › „Nur markieren“ (Stichwort + lila Label, Denoise-Preset
in Lightroom) oder „Lokal entrauschen“.

## 3. H2: KI-Masken aus dem XMP

Gleicher Import wie oben: Maskenbedienfeld öffnen. Sind die Masken *Motiv anheben* /
*Hintergrund beruhigen* (bzw. deine gelernten Masken) vorhanden und nach „KI-Einstellungen
aktualisieren“ korrekt berechnet? Werte (z. B. Belichtung +0,25) übernommen?

Falls nein: Einstellungen › „Lightroom-KI-Masken“ ausschalten. Imagomat schreibt die Masken
dann als Pinselstriche aus der eigenen Segmentierung.

## 4. H3/H4: Zuschnitt und Drehung

Ein Bild mit schiefem Horizont / schiefen Torpfosten, dazu eines im **Hochformat**:
- Stimmt die Drehrichtung (Bild wird gerade, nicht doppelt schief)? → `CROP_ANGLE_SIGN` in
  `backend/imagomat/vision/geometry.py`.
- Liegt der Zuschnitt beim Hochformat-Bild richtig (Motiv drin, nicht um 90° verschoben)?

## 5. H5: Pinselmasken

Mit ausgeschalteten KI-Masken exportieren: Deckt der Pinsel das Motiv ab, oder sind die Tupfer
zu gross/klein? (Annahme: Radius relativ zur langen Bildseite, `PAINT_RADIUS_BASIS` in
`backend/imagomat/style/masks.py`.)

## 6. H6: Personen

- Stichwörter `Personen|FC Winterthur|Vorname Nachname` vorhanden?
- Erscheinen benannte Gesichter im Personen-Modus von Lightroom (MWG-Regionen)? Falls nicht,
  sind die Stichwörter trotzdem vollständig.
- Farblabels in der richtigen Farbe (nicht weiss)? Sonst zeigt Einstellungen › Lightroom-Dialekt,
  welche Label-Texte gelernt wurden.

## 7. Katalog-Export (experimentell)

Export **B** mit der Vorlage aus Schritt 0. Den erzeugten `.lrcat` in Lightroom öffnen:
- Öffnet er ohne Fehlermeldung / ohne „Katalog reparieren“?
- Sind Bilder, Picks, Sterne, Labels, Stichwörter, Sammlungen und Stapel da?
- Sind die Entwicklungseinstellungen sichtbar?

Falls Lightroom den Katalog ablehnt: Variante A + Plugin verwenden (gleiches Ergebnis, robuster).

## 8. Plugin

*Datei › Zusatzmodul-Manager › Hinzufügen* → `lightroom-plugin/Imagomat.lrplugin`. Dann
*Bibliothek › Zusatzmoduloptionen › Imagomat-Export importieren und übernehmen …*: Werden Picks
gesetzt, Sammlungen angelegt und am Ende die Bilder für „KI-Einstellungen aktualisieren“ ausgewählt?
