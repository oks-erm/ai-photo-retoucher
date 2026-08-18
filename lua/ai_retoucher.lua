-- Darktable AI Retoucher: one-click macOS/Linux bridge for Darktable 5.6+.
local dt = require "darktable"
local API, NAN = "http://127.0.0.1:8765", 0 / 0
local last_plan, last_session, original_preview, applied_state

local function quote(v) return "'" .. tostring(v):gsub("'", "'\\''") .. "'" end
local function json_quote(v)
  local s = tostring(v):gsub("\\", "\\\\"):gsub('"', '\\"')
  s = s:gsub("\n", "\\n"):gsub("\r", "\\r"):gsub("\t", "\\t")
  return '"' .. s .. '"'
end
local function remove_file(p) if p then pcall(os.remove, p) end end
local function read_file(p)
  local f = io.open(p, "r"); if not f then return nil end
  local body = f:read("*a"); f:close(); return body
end
local function selected_image()
  local images = dt.gui.action_images
  if not images or #images ~= 1 then dt.print_error("AI Retoucher: open exactly one image"); return nil end
  return images[1]
end
local function in_darkroom() return dt.gui.current_view() == dt.gui.views.darkroom end
local function object(json, key) return json and json:match('"' .. key .. '"%s*:%s*(%b{})') end
local function number(json, key)
  return json and tonumber(json:match('"' .. key .. '"%s*:%s*(-?[%d%.]+)')) or nil
end
local function boolean(json, key)
  if not json then return nil end
  if json:match('"' .. key .. '"%s*:%s*true') then return true end
  if json:match('"' .. key .. '"%s*:%s*false') then return false end
  return nil
end
local function text_value(json, key)
  if not json then return nil end
  local value = json:match('"' .. key .. '"%s*:%s*"(.-)"')
  if value then return value:gsub('\\n', ' '):gsub('\\"', '"'):gsub('\\\\', '\\') end
  return nil
end
local function api_available()
  return os.execute("curl --connect-timeout 1 --max-time 2 --fail --silent "
    .. quote(API .. "/health") .. " >/dev/null 2>&1") == true
end
local function darktable_cli()
  local configured = dt.preferences.read("ai_retoucher", "darktable_cli", "string")
  if configured and configured ~= "" then return configured end
  local os_name = tostring(dt.configuration.running_os):lower()
  if os_name == "macos" or os_name == "osx" or os_name == "darwin" then
    return "/Applications/darktable.app/Contents/MacOS/darktable-cli"
  end
  return "darktable-cli"
end
local function render_preview(image)
  local source, output = image.path .. "/" .. image.filename, os.tmpname() .. ".jpg"
  local config_dir = os.tmpname() .. "-darktable-config"
  if os.execute("mkdir -p " .. quote(config_dir)) ~= true then
    return nil, "could not create isolated preview configuration"
  end
  local command = quote(darktable_cli()) .. " " .. quote(source) .. " " .. quote(output)
    .. " --width 2048 --height 2048 --hq false --core"
    .. " --configdir " .. quote(config_dir) .. " --library :memory:"
    .. " >/tmp/darktable-ai-retoucher-export.log 2>&1"
  local ok = os.execute(command) == true
  os.execute("rm -rf " .. quote(config_dir))
  if not ok then remove_file(output); return nil,
    "preview render failed; see /tmp/darktable-ai-retoucher-export.log" end
  return output
end
local function analyse_api(preview, context)
  local output = os.tmpname()
  local command = "curl --fail --silent --show-error -o " .. quote(output)
    .. " -F image=@" .. quote(preview) .. " -F context=" .. quote(context)
    .. " " .. quote(API .. "/v1/analyse")
  if os.execute(command) ~= true then remove_file(output); return nil end
  local body = read_file(output); remove_file(output); return body
end
local function apply_api(image, plan)
  local payload_path, response_path = os.tmpname(), os.tmpname()
  local payload = io.open(payload_path, "w"); if not payload then return nil end
  payload:write('{"image_id":' .. json_quote(tostring(image.id))
    .. ',"image_path":' .. json_quote(image.path .. "/" .. image.filename)
    .. ',"edit_plan":' .. plan .. '}'); payload:close()
  local command = "curl --fail --silent --show-error -o " .. quote(response_path)
    .. " -H 'Content-Type: application/json' --data-binary @" .. quote(payload_path)
    .. " " .. quote(API .. "/v1/apply")
  local ok = os.execute(command) == true; remove_file(payload_path)
  if not ok then remove_file(response_path); return nil end
  local body = read_file(response_path); remove_file(response_path)
  return body and body:match('"session_id"%s*:%s*"([%w]+)"')
end
local function critique_api(original, edited, plan)
  local output = os.tmpname()
  local command = "curl --fail --silent --show-error -o " .. quote(output)
    .. " -F original=@" .. quote(original) .. " -F edited=@" .. quote(edited)
    .. " -F previous_plan=" .. quote(plan) .. " " .. quote(API .. "/v1/critique")
  if os.execute(command) ~= true then remove_file(output); return nil end
  local body = read_file(output); remove_file(output); return body
end
local function revert_api(id)
  return os.execute("curl --fail --silent -H 'Content-Type: application/json' --data "
    .. quote('{"session_id":' .. json_quote(id) .. '}') .. " "
    .. quote(API .. "/v1/revert") .. " >/dev/null") == true
end

local mode = dt.new_widget("combobox") { label = "Mode", "Technical", "Portrait", "Creative", selected = 1 }
local intent = dt.new_widget("entry") { placeholder = "Natural correction or desired mood" }
local strength = dt.new_widget("slider") {
  label = "Strength", soft_min = 0, soft_max = 1, hard_min = 0, hard_max = 1,
  step = 0.05, digits = 2, value = 0.5,
}
local naturalness = dt.new_widget("slider") {
  label = "Naturalness", soft_min = 0, soft_max = 1, hard_min = 0, hard_max = 1,
  step = 0.05, digits = 2, value = 0.8,
}
local protect_skin = dt.new_widget("check_button") { label = "Protect skin", value = true }
local protect_highlights = dt.new_widget("check_button") { label = "Protect highlights", value = true }
local preserve_shadows = dt.new_widget("check_button") { label = "Preserve deep shadows", value = true }
local preserve_colours = dt.new_widget("check_button") { label = "Preserve scene colours", value = true }
local review_first = dt.new_widget("check_button") { label = "Review before applying", value = false }
local summary = dt.new_widget("label") { label = "Ready" }
local function mode_value() return ({ "technical", "portrait", "creative" })[mode.selected] end
local function context()
  return '{"mode":' .. json_quote(mode_value()) .. ',"intent":'
    .. json_quote(intent.text ~= "" and intent.text or "Natural professional correction")
    .. string.format(',"strength":%.3f,"naturalness":%.3f', strength.value, naturalness.value)
    .. ',"protect_skin":' .. tostring(protect_skin.value)
    .. ',"protect_highlights":' .. tostring(protect_highlights.value)
    .. ',"preserve_deep_shadows":' .. tostring(preserve_shadows.value)
    .. ',"preserve_scene_colours":' .. tostring(preserve_colours.value)
    .. ',"exif":{},"current_state":""}'
end

local ACTIONS = {
  { section="global", key="exposure_ev", path="iop/exposure/exposure", scale=1 },
  { section="global", key="contrast", path="iop/colorbalancergb/contrast", scale=1 },
  { section="global", key="saturation", path="iop/colorbalancergb/global saturation", scale=1 },
  { section="global", key="vibrance", path="iop/colorbalancergb/global vibrance", scale=1 },
}
local function read_action(path)
  local ok, status = pcall(dt.gui.action, path, 0, "value", "set", NAN)
  local value = ok and tonumber(status) or nil
  return value, value and nil or tostring(status)
end
local function set_action(path, value)
  local ok, status = pcall(dt.gui.action, path, 0, "value", "set", value)
  return ok, tostring(status)
end
local function enable(path)
  local module = path:match("^(iop/[^/]+)")
  if module then pcall(dt.gui.action, module, 0, "enable", "on", 1) end
end
local function apply_values(values, snapshot)
  local applied, failures = 0, {}
  for _, action in ipairs(ACTIONS) do
    local delta = number(object(values, action.section), action.key)
    if delta and math.abs(delta) > 0.000001 then
      enable(action.path)
      local previous, err = read_action(action.path)
      if not previous then table.insert(failures, action.key .. ": " .. err) else
        if snapshot then table.insert(snapshot, { path=action.path, value=previous }) end
        -- Strength is already part of the model request, so EditPlan values are final
        -- deltas. Scaling them again here makes low/medium-strength edits invisible.
        local ok, set_err = set_action(action.path, previous + delta * action.scale)
        if ok then applied = applied + 1 else table.insert(failures, action.key .. ": " .. set_err) end
      end
    end
  end
  return applied, table.concat(failures, "; ")
end
local function restore(snapshot)
  local failures = {}
  for i = #snapshot, 1, -1 do
    local ok, err = set_action(snapshot[i].path, snapshot[i].value)
    if not ok then table.insert(failures, err) end
  end
  return #failures == 0, table.concat(failures, "; ")
end
local function analyse_current()
  if not in_darkroom() then return nil, nil, "open one image in Darkroom" end
  if not api_available() then return nil, nil, "backend is not running" end
  local image = selected_image(); if not image then return nil, nil, "no image" end
  summary.label = "Rendering preview…"
  local preview, err = render_preview(image); if not preview then return nil, nil, err end
  summary.label = "Analysing…"
  local plan = analyse_api(preview, context())
  if not plan then remove_file(preview); return nil, nil, "analysis failed" end
  return image, preview, plan
end
local function apply_reviewed(image, preview, plan)
  local session = apply_api(image, plan); if not session then return false, "session failed" end
  local snapshot = {}; local count, message = apply_values(plan, snapshot)
  if count == 0 then revert_api(session); return false, message ~= "" and message or "no supported changes" end
  remove_file(original_preview); original_preview, last_plan, last_session, applied_state = preview, plan, session, snapshot
  return true, message
end

local analyse_apply = dt.new_widget("button") {
  label = "Analyse & Apply",
  clicked_callback = function()
    local image, preview, plan = analyse_current()
    if not image then summary.label = "Error: " .. plan; dt.print_error(summary.label); return end
    last_plan = plan
    if review_first.value then remove_file(original_preview); original_preview = preview
      local description = text_value(plan, "summary") or "Plan ready"
      summary.label = description .. " — press Apply reviewed plan"; return end
    local ok, message = apply_reviewed(image, preview, plan)
    summary.label = ok and (message ~= "" and "Applied; some controls skipped" or "Applied") or "Apply failed: " .. message
    if not ok then dt.print_error(summary.label) end
  end,
}
local apply_reviewed_button = dt.new_widget("button") {
  label = "Apply reviewed plan",
  clicked_callback = function()
    if not last_plan or not original_preview then dt.print_error("No reviewed plan"); return end
    local image = selected_image(); if not image then return end
    local preview = original_preview; original_preview = nil
    local ok, message = apply_reviewed(image, preview, last_plan)
    summary.label = ok and "Applied" or "Apply failed: " .. message
  end,
}
local refine = dt.new_widget("button") {
  label = "Refine once",
  clicked_callback = function()
    if not original_preview or not last_plan or not applied_state then dt.print_error("Apply first"); return end
    local image = selected_image(); if not image then return end
    summary.label = "Rendering edited preview…"
    local edited, err = render_preview(image); if not edited then summary.label = err; return end
    summary.label = "Critiquing…"; local delta = critique_api(original_preview, edited, last_plan); remove_file(edited)
    if not delta then summary.label = "Critique failed"; return end
    if boolean(delta, "accepted") then summary.label = "Critic accepted the edit"; return end
    local exposure_delta = number(delta, "global_exposure_delta") or 0
    local synthetic = '{"global":{"exposure_ev":' .. tostring(exposure_delta) .. '}}'
    local count, message = apply_values(synthetic, nil)
    summary.label = count > 0 and "Refinement applied" or message
  end,
}
local revert = dt.new_widget("button") {
  label = "Revert AI Retouch",
  clicked_callback = function()
    if not applied_state or not last_session then dt.print_error("No active retouch"); return end
    if not in_darkroom() then dt.print_error("Open edited image in Darkroom"); return end
    local ok, err = restore(applied_state); if not ok then dt.print_error("Revert incomplete: " .. err); return end
    revert_api(last_session); remove_file(original_preview)
    last_plan, last_session, original_preview, applied_state = nil, nil, nil, nil
    summary.label = "Reverted"
  end,
}
dt.register_event("ai-retoucher-exit", "exit", function() remove_file(original_preview) end)
dt.register_lib("ai_retoucher", "AI Retoucher", true, false,
  { [dt.gui.views.darkroom] = { "DT_UI_CONTAINER_PANEL_RIGHT_CENTER", 100 } },
  dt.new_widget("box") { orientation="vertical", mode, intent, strength, naturalness,
    protect_skin, protect_highlights, preserve_shadows, preserve_colours, review_first,
    analyse_apply, apply_reviewed_button, refine, revert, summary }, nil, nil)
