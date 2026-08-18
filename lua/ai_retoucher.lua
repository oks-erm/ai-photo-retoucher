-- Darktable AI Retoucher: stable localhost bridge (Darktable 5.6+).
-- Install by requiring this file from ~/.config/darktable/luarc.

local dt = require "darktable"

local API = "http://127.0.0.1:8765"
local last_plan = nil
local last_session = nil
local applied_state = nil
local NAN = 0 / 0

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

local function object(json, key)
  return json:match('"' .. key .. '"%s*:%s*(%b{})')
end

local function number(json, key)
  if json == nil then return nil end
  local value = json:match('"' .. key .. '"%s*:%s*(-?[%d%.]+)')
  return tonumber(value)
end

local function in_darkroom()
  return dt.gui.current_view() == dt.gui.views.darkroom
end

local ACTIONS = {
  -- EditPlan values are semantic deltas. scale converts them to the units used
  -- by Darktable's corresponding slider. Paths are stable shortcut action paths.
  { section = "global", key = "exposure_ev", path = "iop/exposure/exposure", scale = 1.0 },
  { section = "global", key = "contrast", path = "iop/colorbalancergb/global contrast", scale = 100.0 },
  { section = "global", key = "saturation", path = "iop/colorbalancergb/global saturation", scale = 100.0 },
  { section = "global", key = "vibrance", path = "iop/colorbalancergb/global vibrance", scale = 100.0 },
}

local function read_action(path)
  local ok, status = pcall(dt.gui.action, path, 0, "value", "set", NAN)
  if not ok then return nil, tostring(status) end
  local value = tonumber(status)
  if value == nil then return nil, "non-numeric action status: " .. tostring(status) end
  return value, nil
end

local function set_action(path, value)
  local ok, status = pcall(dt.gui.action, path, 0, "value", "set", value)
  if not ok then return false, tostring(status) end
  return true, tostring(status)
end

local function enable_module(path)
  local module = path:match("^(iop/[^/]+)")
  if module == nil then return false end
  local ok = pcall(dt.gui.action, module, 0, "enable", "on", 1.0)
  return ok
end

local function apply_plan_to_darktable(plan)
  if not in_darkroom() then
    return nil, "open the selected image in Darkroom before applying"
  end
  local snapshot = {}
  local applied = 0
  local failures = {}
  for _, action in ipairs(ACTIONS) do
    local section = object(plan, action.section)
    local delta = number(section, action.key)
    if delta ~= nil and math.abs(delta) > 0.000001 then
      enable_module(action.path)
      local previous, read_error = read_action(action.path)
      if previous == nil then
        table.insert(failures, action.key .. ": " .. read_error)
      else
        local target = previous + delta * action.scale * strength.value
        local ok, set_error = set_action(action.path, target)
        if ok then
          table.insert(snapshot, { path = action.path, value = previous })
          applied = applied + 1
        else
          table.insert(failures, action.key .. ": " .. set_error)
        end
      end
    end
  end
  if applied == 0 then
    return nil, "no supported non-zero adjustments could be applied; " .. table.concat(failures, "; ")
  end
  return snapshot, table.concat(failures, "; ")
end

local function restore_darktable(snapshot)
  if not in_darkroom() then return false, "open the edited image in Darkroom first" end
  local failures = {}
  for index = #snapshot, 1, -1 do
    local item = snapshot[index]
    local ok, err = set_action(item.path, item.value)
    if not ok then table.insert(failures, item.path .. ": " .. err) end
  end
  return #failures == 0, table.concat(failures, "; ")
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
      local snapshot, apply_message = apply_plan_to_darktable(last_plan)
      if snapshot == nil then
        dt.print_error("AI Retoucher: plan stored, but Darktable apply failed: " .. apply_message)
      else
        applied_state = snapshot
        if apply_message ~= "" then
          dt.print("AI Retoucher: applied with skipped controls: " .. apply_message)
        else
          dt.print("AI Retoucher: edit applied to the current Darktable history")
        end
      end
    else
      dt.print_error("AI Retoucher: apply failed; no partial edit was committed")
    end
    os.remove(payload_path); os.remove(response_path)
  end,
}

local revert = dt.new_widget("button") {
  label = "Revert AI Retouch",
  clicked_callback = function()
    if last_session == nil or applied_state == nil then
      dt.print_error("AI Retoucher: no active applied session"); return
    end
    local restored, restore_error = restore_darktable(applied_state)
    if not restored then
      dt.print_error("AI Retoucher: could not restore every Darktable value: " .. restore_error)
      return
    end
    local command = "curl --fail --silent -H 'Content-Type: application/json' --data "
      .. quote('{"session_id":"' .. last_session .. '"}') .. " " .. quote(API .. "/v1/revert")
    if os.execute(command) then
      last_session = nil
      applied_state = nil
      dt.print("AI Retoucher: Darktable values and backend session reverted")
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
