# Asciify

Convert images and videos into ASCII art. Each pixel is mapped to a character representing its brightness, rendered in full color or monochrome — with optional edge-detection overlays for a sketch-like look.

Before Asciify| After Asciify
:-:|:-:
![Pre-Asciify](https://github.com/TheNebulo/Asciify/blob/main/photo_pre.jpg?raw=true) | ![Post-Asciify](https://github.com/TheNebulo/Asciify/blob/main/photo_post.png?raw=true) 

## Features

- **Image conversion** — Convert any image to an ASCII art render
- **Video conversion** — Convert videos frame-by-frame with audio preserved
- **Full color or monochrome** — Characters are drawn in the original pixel colors, or grayscale
- **Contour/edge overlay** — Replaces brightness characters with directional edge characters (`| / - \`) using Laplacian + Canny edge detection and Sobel gradients
- **Multiprocessing** — Frame conversion parallelized across all CPU cores
- **Chunked processing** — Streams video in chunks to keep memory usage bounded
- **Low-res audio mode** — Optionally downsamples audio to reduce output file size

> [!NOTE]
> The script currently does not support outputting pure text, only rendered photos and images, although this a trivial modification that will be done in future.

## Installation

Asciify uses a few dependencies. All of these can be installed via pip from the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

Any other dependencies should come standard with Python 3.6+, and the `luton.ttf` font can be found in this repository.

### Convert an Image

```python
from ascii_converter import ascii_photo, AsciiConfig

ascii_photo("input.jpg", "output.png")

# With custom settings
cfg = AsciiConfig(scale_factor=0.2, monochrome=True)
ascii_photo("input.jpg", "output.png", cfg=cfg, progress_bar=True)
```

To asciify a video, use `ascii_video()` like so:

### Convert a Video

```python
from ascii_converter import ascii_video, AsciiConfig

ascii_video("input.mp4", "output.mp4")

# With contour overlay and full-res audio
cfg = AsciiConfig(overlay_contours=True, low_res_audio=False)
ascii_video("input.mp4", "output.mp4", cfg=cfg, progress_bar=True, chunk_size=128) # Chunk size is how many frames are processed in parallel in memory
```

> [!WARNING]
> Do not that if `ascii_video()` is used at all (in any script), the script that is initially run must contain the following lines.
> ```python
> from multiprocessing import freeze_support
>
> if __name__ == "__main__":
>  freeze_support()
>  # Continue the code execution here ( i.e. call main() ).
>```
>
>This is for Windows, Linux, and likely MacOS machines, and is used to prevent subprocesses freezing from new creations.

## Configuration

Most options are set via `AsciiConfig`:

| Parameter | Type | Default | Description |
|---|---|---|---|
| `scale_factor` | `float` | `0.15` | Controls output resolution — fraction of the source width converted to ASCII columns |
| `char_width` | `int` | `7` | Pixel width of each character cell (tune to match your font) |
| `char_height` | `int` | `9` | Pixel height of each character cell |
| `color_brightness` | `float` | `1.0` | Multiplier applied to RGB channels when coloring characters |
| `pixel_brightness` | `float` | `2.15` | Multiplier applied to luminance before character selection — increase to use denser characters |
| `monochrome` | `bool` | `False` | Render characters in grayscale instead of original colors |
| `overlay_contours` | `bool` | `False` | Replace edge pixels with directional characters for a sketch/outline effect |
| `contour_min_threshold` | `int` | `0` | Minimum threshold for Laplacian edge mask |
| `contour_max_threshold` | `int` | `255` | Maximum threshold for Laplacian edge mask |
| `low_res_audio` | `bool` | `True` | Downsample audio to 8kHz then upsample to 16kHz to reduce file size |
| `num_workers` | `int` | `cpu_count()` | Number of parallel worker processes for frame conversion |

To configure the characters used during the ASCII conversion (although not recommended), change the following 3 constants in `asciify.py`:
- `CHARS`: Characters used in mapping pixel luminance, one string, darkest first
- `CONTOUR_CHARS`: Characters used in mapping contours, one string, starting at 0 degrees rotating clockwise
- `FONT`: Font used for the displayed characters, FreeTypeFont object from Pillow library

## How It Works

**Character mapping** — A 256-entry lookup table maps each pixel's luminance value to one of 70 characters, ranging from dense (`$@B%8&WM#*...`) to empty (space). Brighter pixels get sparser characters.

**Color** — In color mode, each character is drawn using the original RGB value of the pixel it represents. In monochrome mode, the luminance value is used for all three channels.

**Contour overlay** — When enabled, edge detection runs on the grayscale frame. Pixels identified as edges are assigned one of four directional characters (`|`, `/`, `-`, `\`) based on the gradient angle from Sobel, giving outlines a hand-drawn appearance.

**Video pipeline** — Frames are read in chunks, converted in parallel using a `multiprocessing.Pool`, and written sequentially via `FFMPEG_VideoWriter`. Audio is extracted, optionally resampled, and muxed into the output.

## Tips

- **Lower `scale_factor`** for faster processing and a blockier look; **higher** for more detail and larger output files.
- **Increase `pixel_brightness`** if your output looks mostly empty/white characters; decrease it if everything looks like `@` or `$`.
- **`overlay_contours`** works best on footage with clear edges — faces, objects, and architectural subjects tend to look great.
- For fine-tuning contour detection, adjust `contour_min_threshold` and `contour_max_threshold` to control how aggressively Laplacian edges are included.
- Set `low_res_audio=False` if audio quality matters; the default trades fidelity for smaller file sizes.
- Set `chunk_size` in `ascii_video` to a value that balances concurrency and memory usage.

## License
This project uses the [MIT License](https://choosealicense.com/licenses/mit/).
