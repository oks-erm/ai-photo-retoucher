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
uv sync --python 3.12
echo "Preparing the local high-detail portrait-mask model (one-time 214 MB download)..."
uv run --python 3.12 python -c 'from rembg import new_session; new_session("birefnet-general-lite")'
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

EXPECTED_VERSION="2026.08.21.3"
READY=false
for _ in {1..20}; do
  if curl --connect-timeout 1 --max-time 2 --fail --silent \
    http://127.0.0.1:8765/health | grep -Fq "$EXPECTED_VERSION"; then
    READY=true
    break
  fi
  sleep 0.25
done
if [[ "$READY" != true ]]; then
  echo "Backend did not start renderer $EXPECTED_VERSION. Check $LOG_DIR/backend-error.log" >&2
  exit 1
fi

echo "Installed renderer $EXPECTED_VERSION. Restart Darktable, open one photo in Darkroom, and click Analyse & Retouch."
