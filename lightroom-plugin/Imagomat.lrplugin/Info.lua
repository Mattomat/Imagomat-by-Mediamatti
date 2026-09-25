--[[
Imagomat – Lightroom-Classic-Plugin

Übernimmt einen Imagomat-Export in den aktuell geöffneten Katalog:
- importiert die Dateien (optional),
- setzt Pick-Flags (behalten / abgelehnt), die im XMP nicht transportiert werden,
- legt Sammlungen an (Behalten, Aussortiert nach Grund, Personen, Denoise, Prüfen),
- wählt alle Bilder aus, deren KI-Masken bzw. Denoise noch berechnet werden müssen.

Installation: Datei › Zusatzmodul-Manager › Hinzufügen › diesen Ordner wählen.
]]

return {
  LrSdkVersion = 12.0,
  LrSdkMinimumVersion = 10.0,
  LrToolkitIdentifier = "ch.mediamatti.imagomat",
  LrPluginName = "Imagomat",
  LrPluginInfoUrl = "https://github.com/Mattomat/Imagomat-by-Mediamatti",
  LrLibraryMenuItems = {
    { title = "Imagomat-Export importieren und übernehmen …", file = "ImportExport.lua" },
    { title = "Imagomat-Manifest auf vorhandene Bilder anwenden …", file = "ApplyManifest.lua" },
    { title = "Bilder für „KI-Einstellungen aktualisieren“ auswählen", file = "SelectNeedsAI.lua" },
  },
  VERSION = { major = 0, minor = 1, revision = 0 },
}
