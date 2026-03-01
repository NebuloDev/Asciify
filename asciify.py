import os, shutil, tempfile
from dataclasses import dataclass, field
from multiprocessing import Pool, cpu_count

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from moviepy import VideoFileClip, AudioFileClip
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CHARS = "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\\|()1{}[]?-_+~<>i!lI;:,\"^`'. "[::-1]
CONTOUR_CHARS = "|/-\\"
FONT = ImageFont.truetype("lucon.ttf", 10)


@dataclass
class AsciiConfig:
    scale_factor: float = 0.15
    char_width: int = 7
    char_height: int = 9
    color_brightness: float = 1.0
    pixel_brightness: float = 2.15
    monochrome: bool = False
    overlay_contours: bool = False
    contour_min_threshold: int = 0
    contour_max_threshold: int = 255
    low_res_audio: bool = True
    num_workers: int = field(default_factory=cpu_count)


# ---------------------------------------------------------------------------
# Core frame conversion
# ---------------------------------------------------------------------------

def _build_char_lut() -> np.ndarray:
    """Precompute a 256-entry lookup table mapping brightness -> char index."""
    n = len(CHARS)
    lut = np.floor(np.arange(256) * (n / 256)).astype(np.int32)
    lut = np.clip(lut, 0, n - 1)
    return lut


_CHAR_LUT = _build_char_lut()


def asciify_frame(frame: np.ndarray, cfg: AsciiConfig) -> np.ndarray:
    """Convert a single BGR frame to an ASCII-art RGB frame."""
    # Normalise input
    if frame.ndim == 3 and frame.shape[2] == 4:
        frame = frame[:, :, :3]

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Downscale
    src_h, src_w = frame_rgb.shape[:2]
    new_w = max(1, int(cfg.scale_factor * src_w))
    new_h = max(1, int(cfg.scale_factor * src_h * (cfg.char_width / cfg.char_height)))
    small = cv2.resize(frame_rgb, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    # Grayscale + brightness
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)  # correct channel order

    # Weighted luminance (same coefficients as before but vectorised)
    r, g, b = small[:, :, 0], small[:, :, 1], small[:, :, 2]
    lum = (0.299 * cfg.color_brightness * r
           + 0.587 * cfg.color_brightness * g
           + 0.114 * cfg.color_brightness * b)
    lum = np.clip(lum * cfg.pixel_brightness, 0, 255).astype(np.uint8)

    # Map each pixel to a character via LUT
    char_indices = _CHAR_LUT[lum]  # shape (new_h, new_w)

    # Contour / edge detection
    contour_map = None
    if cfg.overlay_contours:
        depth = cv2.Laplacian(gray, cv2.CV_64F)
        depth = cv2.convertScaleAbs(depth)
        _, depth_mask = cv2.threshold(
            depth,
            cfg.contour_min_threshold,
            cfg.contour_max_threshold,
            cv2.THRESH_BINARY,
        )
        edges = cv2.Canny(gray, 50, 70)

        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        angles = cv2.phase(grad_x, grad_y, angleInDegrees=True) % 180

        contour_mask = (edges > 0) & (depth_mask > 0)
        contour_map = (angles, contour_mask)

    # Render to image
    out_w = cfg.char_width * new_w
    out_h = cfg.char_height * new_h
    output = Image.new("RGB", (out_w, out_h), color=(0, 0, 0))
    draw = ImageDraw.Draw(output)

    colors = small if not cfg.monochrome else np.stack([lum, lum, lum], axis=-1)

    for i in range(new_h):
        for j in range(new_w):
            if contour_map is not None:
                angles_map, cmask = contour_map
                if cmask[i, j]:
                    a = angles_map[i, j]
                    ci = 0 if a < 45 else 1 if a < 90 else 2 if a < 135 else 3
                    char = CONTOUR_CHARS[ci]
                else:
                    char = CHARS[char_indices[i, j]]
            else:
                char = CHARS[char_indices[i, j]]

            rgb = tuple(int(c) for c in colors[i, j])
            draw.text((j * cfg.char_width, i * cfg.char_height), char, font=FONT, fill=rgb)

    return np.array(output)


# Multiprocessing shim: pool.imap requires a top-level picklable callable
def _convert_frame_worker(args):
    frame, cfg = args
    return asciify_frame(frame, cfg)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ascii_photo(
    in_path: str,
    out_path: str,
    cfg: AsciiConfig | None = None,
    progress_bar: bool = False,
) -> None:
    """
    Convert a single image to ASCII art and save it.

    Parameters
    ----------
    in_path  : path to the input image
    out_path : path to save the output image
    cfg      : AsciiConfig (uses defaults if None)
    progress_bar : print status messages
    """
    if not os.path.exists(in_path):
        raise FileNotFoundError(f"Input file not found: {in_path}")

    cfg = cfg or AsciiConfig()
    frame = cv2.imread(in_path)
    if frame is None:
        raise ValueError(f"Could not read image: {in_path}")

    result = asciify_frame(frame, cfg)
    Image.fromarray(result, "RGB").save(out_path)

    if progress_bar:
        print(f"Saved ASCII photo to {out_path}")


def ascii_video(
    in_path: str,
    out_path: str,
    cfg: AsciiConfig | None = None,
    progress_bar: bool = True,
    chunk_size: int = 64,
) -> None:
    """
    Convert a video to ASCII art and save it, including audio.

    Parameters
    ----------
    in_path      : path to the input video
    out_path     : path to save the output video
    cfg          : AsciiConfig (uses defaults if None)
    progress_bar : show tqdm progress bars
    chunk_size   : number of frames to hold in memory at once
    """
    if not os.path.exists(in_path):
        raise FileNotFoundError(f"Input file not found: {in_path}")

    cfg = cfg or AsciiConfig()

    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {in_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Read a single frame to determine output dimensions
    ret, probe = cap.read()
    if not ret:
        raise RuntimeError("Could not read any frames from video.")
    probe_out = asciify_frame(probe, cfg)
    out_h, out_w = probe_out.shape[:2]
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Audio
    original_clip = VideoFileClip(in_path)
    audio_tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    audio_tmp.close()

    if cfg.low_res_audio:
        low = original_clip.audio.with_fps(8000)
        low.write_audiofile(audio_tmp.name, logger=None)
        audio = AudioFileClip(audio_tmp.name).with_fps(16000)
        audio_final_tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        audio_final_tmp.close()
        audio.write_audiofile(audio_final_tmp.name, logger=None)
        audio_path = audio_final_tmp.name
        os.remove(audio_tmp.name)
    else:
        original_clip.audio.write_audiofile(audio_tmp.name, logger=None)
        audio_path = audio_tmp.name
        audio_final_tmp = None

    # Video write loop
    video_tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    video_tmp.close()

    writer = FFMPEG_VideoWriter(
        video_tmp.name,
        fps=fps,
        size=(out_w, out_h),
        codec="libx264",
        logfile=None,
        threads=cfg.num_workers,
        audiofile=audio_path,
        ffmpeg_params=["-strict", "-2"],
    )

    pbar = tqdm(total=frame_count, desc="Converting", disable=not progress_bar)

    with Pool(cfg.num_workers) as pool:
        # Stream chunks through the pool to keep memory bounded
        chunk: list[tuple[np.ndarray, AsciiConfig]] = []

        def flush_chunk():
            for ascii_frame in pool.imap(_convert_frame_worker, chunk):
                writer.write_frame(ascii_frame)
                pbar.update(1)
            chunk.clear()

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            chunk.append((frame, cfg))
            if len(chunk) >= chunk_size:
                flush_chunk()

        if chunk:
            flush_chunk()

    pbar.close()
    cap.release()
    writer.close()

    # Cleanup
    original_clip.close()
    shutil.move(video_tmp.name, out_path)
    os.remove(audio_path)
    if audio_final_tmp and os.path.exists(audio_final_tmp.name):
        os.remove(audio_final_tmp.name)