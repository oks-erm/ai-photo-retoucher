#!/bin/bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
UV_BIN="$(command -v uv || true)"
DARKTABLE_CLI="/Applications/darktable.app/Contents/MacOS/darktable-cli"
CONFIG_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/darktable"
LUA_DIR="$CONFIG_DIR/lua"
LUARC="$CONFIG_DIR/luarc"
AGENT_DIR="$HOME/Library/LaunchAgents"
AGENT_FILE="$AGENT_DIR/com.oksanaerm.darktable-ai-retoucher.plist"
LOG_DIR="$HOME/Library/Logs/darktable-ai-retoucher"

if [[ -z "$UV_BIN" ]]; then
  echo "uv is required. Install it from https://docs.astral.sh/uv/ and run this again." >&2
  exit 1
fi
if [[ ! -x "$DARKTABLE_CLI" ]]; then
  echo "Darktable was not found in /Applications. Install or move darktable.app there." >&2
  exit 1
fi
if [[ ! -f "$PROJECT_DIR/.env" ]]; then
  cp "$PROJECT_DIR/.env.example" "$PROJECT_DIR/.env"
  echo "Created $PROJECT_DIR/.env. Add your OPENAI_API_KEY, then run this installer again." >&2
  exit 1
fi
if ! grep -Eq '^OPENAI_API_KEY=.+$' "$PROJECT_DIR/.env"; then
  echo "Add OPENAI_API_KEY to $PROJECT_DIR/.env, then run this installer again." >&2
  exit 1
fi

cd "$PROJECT_DIR"
uv sync
mkdir -p "$LUA_DIR" "$AGENT_DIR" "$LOG_DIR"
cp "$PROJECT_DIR/lua/ai_retoucher.lua" "$LUA_DIR/ai_retoucher.lua"
touch "$LUARC"
if ! grep -Fq 'require "ai_retoucher"' "$LUARC"; then
  printf '\nrequire "ai_retoucher"\n' >> "$LUARC"
fi

cat > "$AGENT_FILE" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>com.oksanaerm.darktable-ai-retoucher</string>
  <key>ProgramArguments</key>
  <array><string>$UV_BIN</string><string>run</string><string>ai-retoucher</string></array>
  <key>WorkingDirectory</key><string>$PROJECT_DIR</string>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>StandardOutPath</key><string>$LOG_DIR/backend.log</string>
  <key>StandardErrorPath</key><string>$LOG_DIR/backend-error.log</string>
</dict>
</plist>
PLIST

launchctl bootout "gui/$(id -u)" "$AGENT_FILE" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$AGENT_FILE"
launchctl kickstart -k "gui/$(id -u)/com.oksanaerm.darktable-ai-retoucher"

echo "Installed. Restart Darktable, open one photo in Darkroom, and click Analyse & Apply."
