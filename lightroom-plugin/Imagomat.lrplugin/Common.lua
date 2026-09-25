--[[ Gemeinsame Funktionen: Manifest lesen, Picks setzen, Sammlungen anlegen. ]]

local LrApplication = import "LrApplication"
local LrDialogs = import "LrDialogs"
local LrFileUtils = import "LrFileUtils"
local LrPathUtils = import "LrPathUtils"
local LrProgressScope = import "LrProgressScope"

local JSON = require "JSON"

local Common = {}

function Common.chooseExportFolder()
  local r = LrDialogs.runOpenPanel {
    title = "Imagomat-Exportordner wählen (enthält imagomat.json)",
    canChooseFiles = false, canChooseDirectories = true, allowsMultipleSelection = false,
  }
  return r and r[1] or nil
end

function Common.readManifest(folder)
  local path = LrPathUtils.child(folder, "imagomat.json")
  if not LrFileUtils.exists(path) then
    LrDialogs.message("Imagomat", "Keine imagomat.json in " .. folder, "critical")
    return nil
  end
  local f = io.open(path, "r")
  local text = f:read("*a")
  f:close()
  return JSON.decode(text)
end

-- "Imagomat|Shoot|Personen|Name" -> verschachtelte Sammlungssätze + Sammlung
local function ensureCollection(catalog, path, cache)
  if cache[path] then return cache[path] end
  local parts = {}
  for p in string.gmatch(path, "[^|]+") do parts[#parts + 1] = p end
  local parent = nil
  for i = 1, #parts - 1 do
    parent = catalog:createCollectionSet(parts[i], parent, true)
  end
  local coll = catalog:createCollection(parts[#parts], parent, true)
  cache[path] = coll
  return coll
end

function Common.apply(manifest, folder, photosByFile)
  local catalog = LrApplication.activeCatalog()
  local progress = LrProgressScope { title = "Imagomat: Picks und Sammlungen" }
  local cache, byCollection, needsAI = {}, {}, {}
  local n = #manifest.photos
  catalog:withWriteAccessDo("Imagomat: Picks und Sammlungen", function()
    for i, entry in ipairs(manifest.photos) do
      progress:setPortionComplete(i, n)
      local photo = photosByFile[entry.file]
      if photo == nil then
        photo = catalog:findPhotoByPath(LrPathUtils.child(folder, entry.file))
      end
      if photo then
        photo:setRawMetadata("pickStatus", entry.pick or 0)
        for _, c in ipairs(entry.collections or {}) do
          byCollection[c] = byCollection[c] or {}
          table.insert(byCollection[c], photo)
        end
        if entry.needs_ai_update then table.insert(needsAI, photo) end
      end
    end
    for path, photos in pairs(byCollection) do
      ensureCollection(catalog, path, cache):addPhotos(photos)
    end
  end, { timeout = 60 })
  progress:done()
  return needsAI
end

function Common.selectForAIUpdate(photos)
  if #photos == 0 then return end
  local catalog = LrApplication.activeCatalog()
  catalog:setSelectedPhotos(photos[1], photos)
  LrDialogs.message("Imagomat",
    #photos .. " Bilder sind ausgewählt.\n\nBitte jetzt im Raster: Foto › Entwicklungseinstellungen › " ..
    "KI-Einstellungen aktualisieren.\nDamit berechnet Lightroom KI-Masken und Denoise.", "info")
end

return Common
