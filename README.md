# Darktable AI Retoucher

Describe the finish you want in Darktable and get a full-resolution, masked, 16-bit TIFF retouch beside the original RAW. OpenAI turns a private 2048 px preview and your intent into a bounded edit plan with exactly one API request. Up to six deterministic local measurement/render passes then tune that same plan toward the selected style—without further API use. All final pixel processing and masks run locally on the Mac.

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
2. Open **AI Retoucher** in the right panel, choose Portrait, Technical, or Creative, and select a photographic preset.
3. The intent box is optional. Use it only for a concrete exception or priority, for example:

   `Prioritise the woman's face and keep the white dress detailed.`

4. Set Strength around `0.65` and Naturalness around `0.85`.
5. Click **Analyse & Retouch**.

The first local render after installation can be slower while ONNX initializes. The new file is rendered at the source dimensions, named like `RVAZ4030-ai-malick-luminous-20260818-174500.tif`, and imported into Darktable automatically. Its mask directory has the same stem plus `-masks`.

## Presets

| Preset | Recognizable result |
|---|---|
| **Golden hour cinematic** | The approved reference look: luminous warm subject, darker amber sky, deep quiet foliage and strong separation. |
| **Malick — luminous natural** | Peach-honey skin, pale-gold highlights, olive/moss foliage, cool green-grey shadows, lifted blacks and restrained directional haze. |
| **Coppola — nostalgic dream** | Creamy highlights, lifted faded blacks, muted yellow-green foliage, dusty-pink/lavender tonality, fine grain, bloom and small halation. |
| **Pre-Raphaelite forest** | Warm ivory skin, emerald/blue-green foliage, preserved reds, cyan deep shadows, amber highlights and selective painterly darkness. |
| **Fairytale twilight** | A cool blue-hour environment with warm subject light, deeper teal-green background, pale-yellow highlights and directional glow. |
| **Custom intent only** | No preset signature or local style convergence; the model follows the written intent alone. |

Preset names describe broad photographic and cinematic colour vocabularies. They do not copy a specific film frame or painting, and the pipeline never changes identity, geometry, pose or location.

**Golden cinematic** is a reusable style signature measured from the supplied example, not a hardcoded recipe for that one frame. It locally aims for a brighter, warmer subject; protected white clothing and skin texture; a substantially darker background and sky; and deeper, quieter foliage. Each pass measures the current image, adjusts bounded plan values, renders again, and keeps only an improvement. Choose **Custom intent only** to apply the model's plan without this style convergence.

Enable **Review plan before rendering** if you want to read the plan summary first. After the first analysis, change the preset and click **Retouch with saved plan (free)** to create as many comparisons as you want without calling OpenAI again. The button always reloads the untouched original model plan, so styles never stack or contaminate each other.

## What is actually applied

- 16-bit global exposure, S-curve contrast, black-depth shaping, highlight compression/softness, and shadow lift
- white-balance temperature/tint, saturation, vibrance, and separate highlight/shadow warmth
- local ONNX subject matte and inverse-background adjustment
- soft sky/background luminance mask with bounded darkening, warmth, and saturation
- masked green/yellow foliage lightness and chroma
- confidence-gated skin tone, warmth, chroma, and texture-preserving bilateral smoothing
- deterministic bloom, halation, fine grain, background softness, directional amber haze and selective edge darkening
- edge-preserving denoise and bounded unsharp detail
- highlight and deep-shadow protection
- reusable full-resolution 16-bit masks, with explicit applied/skipped reporting in the session JSON

The local models and refinement loop never send pixels anywhere. OpenAI receives only the temporary 2048 px JPEG used for the single intent-to-plan request. `store=False` is set on the Responses API request.

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
