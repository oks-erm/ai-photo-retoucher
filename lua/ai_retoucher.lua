-- Darktable AI Retoucher: intent -> validated plan -> local 16-bit masked render.
local dt = require "darktable"
local API = "http://127.0.0.1:8765"
local reviewed_plan, reviewed_preview

local function quote(value) return "'" .. tostring(value):gsub("'", "'\\''") .. "'" end
local function json_quote(value)
  local text = tostring(value):gsub("\\", "\\\\"):gsub('"', '\\"')
  text = text:gsub("\n", "\\n"):gsub("\r", "\\r"):gsub("\t", "\\t")
  return '"' .. text .. '"'
end
local function read_file(path)
  local file = io.open(path, "rb"); if not file then return nil end
  local body = file:read("*a"); file:close(); return body
end
local function remove_file(path) if path then pcall(os.remove, path) end end
local function selected_image()
  local images = dt.gui.action_images
  if not images or #images ~= 1 then
    dt.print_error("AI Retoucher: open exactly one image in Darkroom"); return nil
  end
  return images[1]
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
local function api_available()
  return os.execute("curl --connect-timeout 1 --max-time 2 --fail --silent "
    .. quote(API .. "/health") .. " >/dev/null 2>&1") == true
end
local function render(image, extension, export_options, core_options)
  local source, output = image.path .. "/" .. image.filename, os.tmpname() .. extension
  local config_dir = os.tmpname() .. "-darktable-config"
  if os.execute("mkdir -p " .. quote(config_dir)) ~= true then
    return nil, "could not create isolated Darktable configuration"
  end
  local command = quote(darktable_cli()) .. " " .. quote(source) .. " " .. quote(output)
    .. " " .. export_options .. " --core --configdir " .. quote(config_dir)
    .. " --library :memory: " .. (core_options or "")
    .. " >/tmp/darktable-ai-retoucher-export.log 2>&1"
  local ok = os.execute(command) == true
  os.execute("rm -rf " .. quote(config_dir))
  if not ok then
    remove_file(output)
    return nil, "Darktable export failed; see /tmp/darktable-ai-retoucher-export.log"
  end
  return output
end
local function render_preview(image)
  return render(image, ".jpg", "--width 2048 --height 2048 --hq false --out-ext jpg")
end
local function render_working_tiff(image)
  return render(image, ".tif",
    "--width 0 --height 0 --hq true --out-ext tif",
    "--conf plugins/imageio/format/tiff/bpp=16")
end
local function request_file(command)
  local response = os.tmpname()
  local ok = os.execute(command .. " -o " .. quote(response)) == true
  if not ok then remove_file(response); return nil end
  local body = read_file(response); remove_file(response); return body
end
local function analyse_api(preview, context)
  return request_file("curl --fail --silent --show-error -F image=@" .. quote(preview)
    .. " -F context=" .. quote(context) .. " " .. quote(API .. "/v1/analyse"))
end
local function latest_plan_api()
  return request_file("curl --fail --silent --show-error " .. quote(API .. "/v1/plans/latest"))
end
local function output_path(image, style_name)
  local stem = image.filename:gsub("%.[^%.]+$", "")
  local style_slug = style_name:gsub("_", "-")
  return image.path .. "/" .. stem .. "-ai-" .. style_slug .. "-"
    .. os.date("%Y%m%d-%H%M%S") .. ".tif"
end
local function retouch_api(image, input, output, plan, style_name)
  local payload_path = os.tmpname()
  local payload = io.open(payload_path, "w"); if not payload then return nil, "request failed" end
  payload:write('{"image_id":' .. json_quote(tostring(image.id))
    .. ',"input_path":' .. json_quote(input) .. ',"output_path":' .. json_quote(output)
    .. ',"edit_plan":' .. plan .. ',"export_masks":true,"style":'
    .. json_quote(style_name) .. ',"auto_refine":true,"refinement_passes":6}')
  payload:close()
  local body = request_file("curl --fail --silent --show-error -H 'Content-Type: application/json'"
    .. " --data-binary @" .. quote(payload_path) .. " " .. quote(API .. "/v1/retouch"))
  remove_file(payload_path)
  if not body then return nil, "local retouch failed; check backend log" end
  local returned = body:match('"output_path"%s*:%s*"(.-)"')
  if not returned then return nil, "backend returned no output" end
  return returned:gsub("\\/", "/"), nil
end

local mode = dt.new_widget("combobox") {
  label = "Mode", "Technical", "Portrait", "Creative", selected = 2,
}
local style = dt.new_widget("combobox") {
  label = "Style",
  "Golden hour cinematic",
  "Malick — luminous natural",
  "Coppola — nostalgic dream",
  "Pre-Raphaelite forest",
  "Fairytale twilight",
  "Custom intent only",
  selected = 1,
}
local intent = dt.new_widget("entry") {
  placeholder = "e.g. luminous golden-hour portrait, natural skin",
}
local strength = dt.new_widget("slider") {
  label = "Strength", soft_min = 0, soft_max = 1, hard_min = 0, hard_max = 1,
  step = 0.05, digits = 2, value = 0.65,
}
local naturalness = dt.new_widget("slider") {
  label = "Naturalness", soft_min = 0, soft_max = 1, hard_min = 0, hard_max = 1,
  step = 0.05, digits = 2, value = 0.85,
}
local protect_skin = dt.new_widget("check_button") {
  label = "Protect identity and skin texture", value = true,
}
local protect_highlights = dt.new_widget("check_button") {
  label = "Protect white clothing/highlights", value = true,
}
local preserve_shadows = dt.new_widget("check_button") {
  label = "Preserve deep-shadow detail", value = true,
}
local review_first = dt.new_widget("check_button") {
  label = "Review plan before rendering", value = false,
}
local status = dt.new_widget("label") { label = "Ready — original RAW will not be changed" }
local styles = {
  "golden_cinematic",
  "malick_luminous",
  "coppola_nostalgic",
  "preraphaelite_enchanted",
  "fairytale_twilight",
  "custom",
}
local function context()
  local modes = { "technical", "portrait", "creative" }
  local requested = intent.text ~= "" and intent.text or
    "Apply the selected preset faithfully with natural skin and professional restraint"
  return '{"mode":' .. json_quote(modes[mode.selected])
    .. ',"style":' .. json_quote(styles[style.selected]) .. ',"intent":' .. json_quote(requested)
    .. string.format(',"strength":%.3f,"naturalness":%.3f', strength.value, naturalness.value)
    .. ',"protect_skin":' .. tostring(protect_skin.value)
    .. ',"protect_highlights":' .. tostring(protect_highlights.value)
    .. ',"preserve_deep_shadows":' .. tostring(preserve_shadows.value)
    .. ',"preserve_scene_colours":true,"exif":{},"current_state":""}'
end
local function import_output(path)
  local ok, imported = pcall(dt.database.import, path)
  if not ok or not imported then return false end
  return true
end
local function render_plan(image, plan)
  status.label = "Exporting current edit as 16-bit TIFF…"
  local working, export_error = render_working_tiff(image)
  if not working then return false, export_error end
  local style_name = styles[style.selected]
  local output = output_path(image, style_name)
  status.label = "Building masks and retouching locally…"
  local rendered, retouch_error = retouch_api(image, working, output, plan, style_name)
  remove_file(working)
  if not rendered then return false, retouch_error end
  status.label = "Importing editable TIFF into Darktable…"
  if not import_output(rendered) then
    return false, "TIFF created at " .. rendered .. " but Darktable could not import it"
  end
  reviewed_plan = nil; remove_file(reviewed_preview); reviewed_preview = nil
  return true, "Done — editable TIFF and reusable masks saved beside the RAW"
end

local analyse_retouch = dt.new_widget("button") {
  label = "Analyse & Retouch",
  clicked_callback = function()
    if dt.gui.current_view() ~= dt.gui.views.darkroom then status.label = "Open one image in Darkroom"; return end
    if not api_available() then status.label = "Backend is not running"; return end
    local image = selected_image(); if not image then return end
    status.label = "Rendering private analysis preview…"
    local preview, export_error = render_preview(image)
    if not preview then status.label = export_error; return end
    status.label = "Turning your intent into a plan…"
    local plan = analyse_api(preview, context())
    if not plan then remove_file(preview); status.label = "Analysis failed; check backend log"; return end
    if review_first.value then
      remove_file(reviewed_preview); reviewed_preview, reviewed_plan = preview, plan
      status.label = (plan:match('"summary"%s*:%s*"(.-)"') or "Plan ready")
        .. " — press Render reviewed plan"
      return
    end
    remove_file(preview)
    local ok, message = render_plan(image, plan); status.label = message
    if not ok then dt.print_error("AI Retoucher: " .. message) end
  end,
}
local render_reviewed = dt.new_widget("button") {
  label = "Render reviewed plan",
  clicked_callback = function()
    if not reviewed_plan then status.label = "No reviewed plan is waiting"; return end
    local image = selected_image(); if not image then return end
    local ok, message = render_plan(image, reviewed_plan); status.label = message
    if not ok then dt.print_error("AI Retoucher: " .. message) end
  end,
}
local render_saved = dt.new_widget("button") {
  label = "Retouch with saved plan (free)",
  clicked_callback = function()
    if not api_available() then status.label = "Backend is not running"; return end
    local image = selected_image(); if not image then return end
    status.label = "Loading original model plan — no OpenAI call…"
    local plan = latest_plan_api(); if not plan then status.label = "No saved plan found"; return end
    local ok, message = render_plan(image, plan); status.label = message
    if not ok then dt.print_error("AI Retoucher: " .. message) end
  end,
}

dt.register_event("ai-retoucher-exit", "exit", function() remove_file(reviewed_preview) end)
dt.register_lib("ai_retoucher", "AI Retoucher", true, false,
  { [dt.gui.views.darkroom] = { "DT_UI_CONTAINER_PANEL_RIGHT_CENTER", 100 } },
  dt.new_widget("box") {
    orientation = "vertical", mode, style, intent, strength, naturalness,
    protect_skin, protect_highlights, preserve_shadows, review_first,
    analyse_retouch, render_reviewed, render_saved, status,
  }, nil, nil)
