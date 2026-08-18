-- Darktable AI Retoucher: stable localhost bridge (Darktable 5.6+).
-- Install by requiring this file from ~/.config/darktable/luarc.

local dt = require "darktable"

local API = "http://127.0.0.1:8765"
local last_plan = nil
local last_session = nil

local function quote(value)
  return "'" .. tostring(value):gsub("'", "'\\''") .. "'"
end

local function selected_image()
  local images = dt.gui.action_images
  if images == nil or #images ~= 1 then
    dt.print_error("AI Retoucher: select exactly one image")
    return nil
  end
  return images[1]
end

local mode = dt.new_widget("combobox") {
  label = "Mode",
  "Technical", "Portrait", "Creative",
  selected = 1,
}
local intent = dt.new_widget("entry") { placeholder = "Natural correction or desired mood" }
local strength = dt.new_widget("slider") { label = "Strength", min = 0, max = 1, step = 0.05, value = 0.5 }
local naturalness = dt.new_widget("slider") { label = "Naturalness", min = 0, max = 1, step = 0.05, value = 0.8 }
local protect_skin = dt.new_widget("check_button") { label = "Protect skin", value = true }
local protect_highlights = dt.new_widget("check_button") { label = "Protect highlights", value = true }
local preserve_shadows = dt.new_widget("check_button") { label = "Preserve deep shadows", value = true }
local preserve_colours = dt.new_widget("check_button") { label = "Preserve scene colours", value = true }
local summary = dt.new_widget("label") { label = "No plan analysed" }

local function mode_value()
  return ({ "technical", "portrait", "creative" })[mode.selected]
end

local analyse = dt.new_widget("button") {
  label = "Analyse exported preview",
  clicked_callback = function()
    local image = selected_image()
    if image == nil then return end
    -- Darktable Lua does not expose a stable current-pipeline preview export across all
    -- supported builds. Export a JPEG from Darkroom first; this bridge accepts that path.
    local preview = dt.preferences.read("ai_retoucher", "preview_path", "string")
    if preview == nil or preview == "" then
      dt.print_error("AI Retoucher: set ai_retoucher/preview_path in preferences")
      return
    end
    local context = string.format(
      '{"mode":%q,"intent":%q,"strength":%.3f,"naturalness":%.3f,"protect_skin":%s,"protect_highlights":%s,"preserve_deep_shadows":%s,"preserve_scene_colours":%s,"exif":{},"current_state":""}',
      mode_value(), intent.text, strength.value, naturalness.value,
      tostring(protect_skin.value), tostring(protect_highlights.value),
      tostring(preserve_shadows.value), tostring(preserve_colours.value)
    )
    local output = os.tmpname()
    local command = "curl --fail --silent --show-error -o " .. quote(output)
      .. " -F image=@" .. quote(preview)
      .. " -F context=" .. quote(context)
      .. " " .. quote(API .. "/v1/analyse")
    if os.execute(command) then
      local file = io.open(output, "r")
      if file ~= nil then last_plan = file:read("*a"); file:close() end
      summary.label = last_plan or "Analysis returned no plan"
      dt.print("AI Retoucher: plan ready for review")
    else
      dt.print_error("AI Retoucher: analysis failed; Darktable remains unchanged")
    end
    os.remove(output)
  end,
}

local apply = dt.new_widget("button") {
  label = "Apply plan",
  clicked_callback = function()
    local image = selected_image()
    if image == nil or last_plan == nil then
      dt.print_error("AI Retoucher: analyse and review a plan first")
      return
    end
    local payload_path = os.tmpname()
    local response_path = os.tmpname()
    local payload = io.open(payload_path, "w")
    payload:write('{"image_id":' .. string.format("%q", tostring(image.id))
      .. ',"image_path":' .. string.format("%q", image.path .. "/" .. image.filename)
      .. ',"edit_plan":' .. last_plan .. '}')
    payload:close()
    local command = "curl --fail --silent --show-error -o " .. quote(response_path)
      .. " -H 'Content-Type: application/json' --data-binary @" .. quote(payload_path)
      .. " " .. quote(API .. "/v1/apply")
    if os.execute(command) then
      local response = io.open(response_path, "r")
      local body = response and response:read("*a") or ""
      if response then response:close() end
      last_session = body:match('"session_id"%s*:%s*"([%w]+)"')
      dt.print("AI Retoucher: deterministic operation manifest created")
    else
      dt.print_error("AI Retoucher: apply failed; no partial edit was committed")
    end
    os.remove(payload_path); os.remove(response_path)
  end,
}

local revert = dt.new_widget("button") {
  label = "Revert AI Retouch",
  clicked_callback = function()
    if last_session == nil then dt.print_error("AI Retoucher: no active session"); return end
    local command = "curl --fail --silent -H 'Content-Type: application/json' --data "
      .. quote('{"session_id":"' .. last_session .. '"}') .. " " .. quote(API .. "/v1/revert")
    if os.execute(command) then
      last_session = nil
      dt.print("AI Retoucher: session reverted")
    else
      dt.print_error("AI Retoucher: revert failed")
    end
  end,
}

dt.register_lib(
  "ai_retoucher", "AI Retoucher", true, false,
  { [dt.gui.views.darkroom] = { "DT_UI_CONTAINER_PANEL_RIGHT_CENTER", 100 } },
  dt.new_widget("box") {
    orientation = "vertical",
    mode, intent, strength, naturalness, protect_skin, protect_highlights,
    preserve_shadows, preserve_colours, analyse, summary, apply, revert,
  },
  nil, nil
)
