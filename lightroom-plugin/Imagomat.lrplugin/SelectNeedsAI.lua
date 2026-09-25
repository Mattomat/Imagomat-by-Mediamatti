--[[ Alle Bilder eines Imagomat-Exports auswählen, die KI-Masken oder Denoise enthalten. ]]

local LrFunctionContext = import "LrFunctionContext"
local LrApplication = import "LrApplication"
local LrPathUtils = import "LrPathUtils"
local LrTasks = import "LrTasks"

local Common = require "Common"

LrTasks.startAsyncTask(function()
  LrFunctionContext.callWithContext("Imagomat KI-Auswahl", function()
    local folder = Common.chooseExportFolder()
    if not folder then return end
    local manifest = Common.readManifest(folder)
    if not manifest then return end
    local catalog = LrApplication.activeCatalog()
    local photos = {}
    for _, entry in ipairs(manifest.photos) do
      if entry.needs_ai_update and entry.pick == 1 then
        local p = catalog:findPhotoByPath(LrPathUtils.child(folder, entry.file))
        if p then photos[#photos + 1] = p end
      end
    end
    Common.selectForAIUpdate(photos)
  end)
end)
