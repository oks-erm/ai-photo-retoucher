# Darktable AI Retoucher

Describe the finish you want in Darktable and get a full-resolution, masked, 16-bit TIFF retouch beside the original RAW. OpenAI turns a private 2048 px preview and your intent into a bounded edit plan; all final pixel processing and masks run locally on the Mac.

The result is a new TIFF imported into Darktable. The RAW and its existing history remain untouched, and the TIFF can be developed further with normal Darktable modules. Six full-resolution 16-bit masks are saved beside it for subject, background, skin, face, foliage, and sky.

## Install on macOS

Requirements: Darktable 5.6+ in `/Applications`, [uv](https://docs.astral.sh/uv/), about 500 MB of free space for dependencies/model, and an OpenAI API key.

```bash
git clone https://github.com/oks-erm/ai-photo-retoucher.git
cd ai-photo-retoucher
git switch agent/implement-darktable-ai-retoucher
cp .env.example .env
# Edit .env and set OPENAI_API_KEY
bash scripts/install_macos.sh
```

The installer uses Python 3.12, downloads the Apache-2.0 local portrait model once (176 MB), installs the Lua panel, and starts the loopback backend automatically. Restart Darktable after installation.

## Retouch one photo

1. Open exactly one photo in **Darkroom**.
2. Open **AI Retoucher** in the right panel and choose Portrait, Technical, or Creative.
3. Describe the desired result, for example:

   `Luminous golden-hour maternity portrait. Lift the woman and face, keep the white dress detailed, deepen and warm the forest, mute harsh greens, natural skin and fine texture.`

4. Set Strength around `0.65` and Naturalness around `0.85`.
5. Click **Analyse & Retouch**.

The first local render after installation can be slower while ONNX initializes. The new file is named like `RVAZ4030-ai-retouched-20260818-174500.tif` and is imported into Darktable automatically. Its mask directory has the same stem plus `-masks`.

Enable **Review plan before rendering** if you want to read the plan summary first. **Retouch with saved plan (free)** reuses the most recently rendered plan on the current photo without calling OpenAI—useful for testing or applying a consistent look.

## What is actually applied

- 16-bit global exposure, S-curve contrast, black-depth shaping, highlight compression/softness, and shadow lift
- white-balance temperature/tint, saturation, vibrance, and separate highlight/shadow warmth
- local ONNX subject matte and inverse-background adjustment
- masked green/yellow foliage lightness and chroma
- confidence-gated skin tone, warmth, chroma, and texture-preserving bilateral smoothing
- edge-preserving denoise and bounded unsharp detail
- highlight and deep-shadow protection
- reusable full-resolution 16-bit masks, with explicit applied/skipped reporting in the session JSON

The local model never sends pixels anywhere. OpenAI receives only the temporary 2048 px JPEG used to interpret intent. `store=False` is set on the Responses API request.

## Why the result is a TIFF

Darktable's stable Lua API cannot create arbitrary processing-module instances and drawn/parametric masks reliably. Earlier versions tried to drive a handful of shortcut actions and could only apply a fraction of a plan. This version exports Darktable's current RAW development at full resolution, performs the complete masked retouch locally, and imports a 16-bit TIFF derivative. This preserves the original and gives you a high-bit-depth image you can continue editing, while avoiding unsupported XMP/database manipulation.

## Service and logs

```bash
curl http://127.0.0.1:8765/health
open http://127.0.0.1:8765/docs
```

Backend logs are in `~/Library/Logs/darktable-ai-retoucher/`. Darktable export failures are logged to `/tmp/darktable-ai-retoucher-export.log`. Session plans and capability reports are stored under `~/.cache/darktable-ai-retoucher/sessions/` by default.

To run the service manually, stop the LaunchAgent with `bash scripts/uninstall_macos.sh`, then run `uv run --python 3.12 ai-retoucher` from the repository.

## Development

```bash
uv sync --python 3.12 --extra dev
uv run --python 3.12 pytest
uv run --python 3.12 ruff check .
```

## Uninstall

```bash
bash scripts/uninstall_macos.sh
```

Generated TIFFs and their mask folders are user files and are intentionally not deleted by the uninstaller.

## License

MIT. The optional local portrait mask uses the Apache-2.0 U2Net human-segmentation model downloaded by `rembg`.
