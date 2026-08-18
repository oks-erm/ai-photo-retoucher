#!/bin/bash
set -euo pipefail

CONFIG_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/darktable"
LUARC="$CONFIG_DIR/luarc"
PLUGIN_FILE="$CONFIG_DIR/lua/ai_retoucher.lua"
AGENT_FILE="$HOME/Library/LaunchAgents/com.oksanaerm.darktable-ai-retoucher.plist"

launchctl bootout "gui/$(id -u)" "$AGENT_FILE" 2>/dev/null || true
rm -f "$AGENT_FILE" "$PLUGIN_FILE"
if [[ -f "$LUARC" ]]; then
  sed -i '' '/^[[:space:]]*require "ai_retoucher"[[:space:]]*$/d' "$LUARC"
fi
echo "AI Retoucher was removed. Restart Darktable to unload its panel."
