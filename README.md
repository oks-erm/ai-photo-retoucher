# Darktable AI Retoucher

A one-click, vision-guided retouching panel for Darktable 5.6+. It renders a temporary reduced preview locally, asks an OpenAI vision model for a strictly validated edit plan, and applies allow-listed controls to the open Darkroom image. RAW files never leave the Mac.

## macOS: install once

Requirements: macOS, Darktable in `/Applications`, [uv](https://docs.astral.sh/uv/), and an OpenAI API key. No separate architecture or local AI model is required; Apple Silicon and Intel Macs use the same setup.

```bash
git clone https://github.com/oks-erm/ai-photo-retoucher.git
cd ai-photo-retoucher
cp .env.example .env
# Open .env and set OPENAI_API_KEY
bash scripts/install_macos.sh
```

Restart Darktable after the installer finishes. The installer copies the Lua panel and registers a macOS LaunchAgent, so the local API starts automatically at login and restarts if it exits. You do not need to run `uv run ai-retoucher` manually afterward.

## Retouch a photo

1. Open exactly one photo in **Darkroom**.
2. In **AI Retoucher** on the right, choose Technical, Portrait, or Creative and optionally enter your intent.
3. Click **Analyse & Apply**.
4. Optionally click **Refine once** for a model critique, or **Revert AI Retouch** to restore the captured values.

Enable **Review before applying** if you want to see the model's summary before pressing **Apply reviewed plan**. Preview rendering and temporary-file cleanup are automatic.

## What is implemented

- Loopback-only FastAPI service with `/v1/analyse`, `/v1/critique`, `/v1/apply`, `/v1/revert`, `/health`, `/docs`, and a useful `/` status response.
- OpenAI Responses API with Pydantic Structured Outputs, bounded values, objective preview statistics, and `store=False`.
- Automatic 2048 px preview rendering through the `darktable-cli` bundled with the macOS app; only that JPEG preview is sent to OpenAI.
- Direct Darktable application of exposure, global contrast, saturation, and vibrance through the documented shortcut-action API.
- Optional one-pass visual critique and exposure refinement.
- Per-control snapshots, session records, XMP snapshot restoration, and one-click revert.
- macOS installer/uninstaller and automatic backend startup.
- Deterministic mappings for additional white balance, tone, foliage, denoise, sharpen, subject, and background operations, with mask-confidence safety gates.

Complex controls whose Darktable 5.6 action units have not been calibrated—white-balance chromatic adaptation, tone-equalizer bands, masks, foliage hue ranges, denoise, and diffuse/sharpen—remain validated in the operation manifest but are skipped by the Lua executor rather than guessed. Face/skin masking and pixel-level healing are not performed.

## Service and troubleshooting

The backend listens only on `http://127.0.0.1:8765`. Check it with:

```bash
curl http://127.0.0.1:8765/health
open http://127.0.0.1:8765/docs
```

Logs are in `~/Library/Logs/darktable-ai-retoucher/`. Automatic preview errors are written to `/tmp/darktable-ai-retoucher-export.log`.

To run the backend in a terminal instead, unload the LaunchAgent with `bash scripts/uninstall_macos.sh`, reinstall the Lua file if desired, then run `uv run ai-retoucher` from this project. The warning about an unrelated active `VIRTUAL_ENV` is harmless; `deactivate` first if you want to remove it.

## Privacy and safety

- RAW and XMP files stay local; only a resized JPEG preview is uploaded for analysis.
- The model cannot emit Lua or executable module settings.
- Invalid plans, unsafe ranges, unsupported media, oversized previews, low-confidence masks, and unsupported Darktable controls fail closed.
- Darktable remains non-destructive: edits live in its history/XMP, and the panel snapshots values before changing them.

## Development

```bash
uv sync --extra dev
uv run pytest
uv run ruff check .
```

## Uninstall on macOS

```bash
bash scripts/uninstall_macos.sh
```

## License

MIT
