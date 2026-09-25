--[[ Imagomat-Export in den aktiven Katalog importieren und Picks/Sammlungen übernehmen. ]]

local LrApplication = import "LrApplication"
local LrDialogs = import "LrDialogs"
local LrFunctionContext = import "LrFunctionContext"
local LrPathUtils = import "LrPathUtils"
local LrTasks = import "LrTasks"

local Common = require "Common"

LrTasks.startAsyncTask(function()
  LrFunctionContext.callWithContext("Imagomat Import", function()
    local folder = Common.chooseExportFolder()
    if not folder then return end
    local manifest = Common.readManifest(folder)
    if not manifest then return end
    local catalog = LrApplication.activeCatalog()
    local photosByFile = {}
    catalog:withWriteAccessDo("Imagomat: Import", function()
      for _, entry in ipairs(manifest.photos) do
        local path = LrPathUtils.child(folder, entry.file)
        local existing = catalog:findPhotoByPath(path)
        if existing then
          photosByFile[entry.file] = existing
        else
          -- addPhoto liest die XMP-Sidecars (Entwicklung, Sterne, Label, Stichwörter) mit ein
          local ok, photo = pcall(function() return catalog:addPhoto(path) end)
          if ok and photo then photosByFile[entry.file] = photo end
        end
      end
    end, { timeout = 300 })
    local needsAI = Common.apply(manifest, folder, photosByFile)
    LrDialogs.message("Imagomat", "Import fertig: " .. #manifest.photos .. " Bilder.", "info")
    Common.selectForAIUpdate(needsAI)
  end)
end)
