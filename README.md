# Darktable AI Retoucher

A vision-guided, non-destructive editing service for Darktable 5.6+. A reduced preview is analysed by an OpenAI vision model, the response is constrained to a Pydantic `EditPlan`, and deterministic code maps semantic photographic controls to an allow-listed Darktable operation manifest. The model never emits Lua or executable module settings.

## What is implemented

- Loopback-only FastAPI service with `/v1/analyse`, `/v1/critique`, `/v1/apply`, `/v1/revert`, and `/health`.
- Async OpenAI Responses API integration with Pydantic Structured Outputs and image inputs.
- Strict, bounded `EditPlan` and `DeltaPlan` contracts. Unknown fields and unsafe ranges fail before application.
- Objective preview statistics (luminance percentiles and clipping fractions), 2048 px maximum analysis preview, and no RAW upload.
- Deterministic mappings for exposure, white balance, tone, colour, foliage, denoise, sharpen, subject, and background operations.
- Mask-confidence gate: all local subject/background edits are skipped below the configured threshold.
- Transactional session records and XMP snapshot restoration for revert.
- Thin Darktable Lua panel that applies supported controls to the current Darkroom image.
- Unit tests for schema safety, mapping, prompts, and session storage.

The Lua bridge applies global exposure, contrast, saturation, and vibrance directly through Darktable's documented shortcut-action API. Every value is snapshotted before application so **Revert AI Retouch** restores it. More complex controls (white-balance chromatic adaptation, tone-equalizer bands, masks, foliage hue ranges, denoise and diffuse/sharpen parameters) remain in the validated operation manifest until their Darktable 5.6 action paths and units are calibrated. Unsupported controls are reported rather than approximated unsafely.

## Requirements

- Python 3.12+
- Darktable 5.6+ with Lua enabled
- `curl` for the Lua bridge
- An OpenAI API key for analysis and critique

## Install and run

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
cp .env.example .env
# Add OPENAI_API_KEY to .env
ai-retoucher
```

The service listens only on `127.0.0.1:8765`. Open `http://127.0.0.1:8765/docs` for the local API schema.

## Darktable integration

1. Copy `lua/ai_retoucher.lua` into `~/.config/darktable/lua/`.
2. Add `require "ai_retoucher"` to `~/.config/darktable/luarc`.
3. Start the Python service before Darktable.
4. Export the current Darkroom rendering as a JPEG or PNG, and set the Darktable preference `ai_retoucher/preview_path` to that path.
5. Open exactly one image in **Darkroom**, choose a mode and protections, analyse, review the plan, then apply. Applying from Lighttable is rejected because processing-control actions are only safe in Darkroom.

The preview export step is explicit in this first release because the Lua API does not expose one stable current-pipeline preview export method across the supported builds. The final image remains the original RAW plus Darktable/XMP history; the service does not rewrite it.

## API examples

Analyse:

```bash
curl -F image=@preview.jpg \
  -F 'context={"mode":"portrait","intent":"warm natural outdoor portrait","strength":0.6,"naturalness":0.9}' \
  http://127.0.0.1:8765/v1/analyse
```

Apply accepts an `image_id`, optional image/XMP paths, the validated `edit_plan`, and optional `mask_confidence`. Revert accepts the returned `session_id`.

## Privacy and failure behavior

- Only resized JPEG previews are sent to OpenAI; RAW files and XMP files remain local.
- API responses are requested with `store=False`.
- The service stores operation/session JSON locally with owner-only permissions and no pixels unless a future debug workflow explicitly adds them.
- Missing API credentials disable analysis but leave health, deterministic mapping/application, revert, and Darktable itself usable.
- Invalid plans, oversized uploads, unsupported media, and low-confidence masks fail closed.

## Development

```bash
pytest
ruff check .
```

The next implementation step is calibrating the remaining Darktable-5.6 action paths and implementing subject-mask import. Face/skin masking and pixel-level healing remain intentionally out of scope.

## License

MIT
