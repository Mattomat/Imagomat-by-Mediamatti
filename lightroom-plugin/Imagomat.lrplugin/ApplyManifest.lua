--[[ Manifest auf bereits importierte Bilder anwenden (z. B. nach normalem Lightroom-Import). ]]

local LrFunctionContext = import "LrFunctionContext"
local LrTasks = import "LrTasks"

local Common = require "Common"

LrTasks.startAsyncTask(function()
  LrFunctionContext.callWithContext("Imagomat Manifest", function()
    local folder = Common.chooseExportFolder()
    if not folder then return end
    local manifest = Common.readManifest(folder)
    if not manifest then return end
    local needsAI = Common.apply(manifest, folder, {})
    Common.selectForAIUpdate(needsAI)
  end)
end)
