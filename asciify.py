import cv2
import math
from tqdm import tqdm
from moviepy.editor import VideoFileClip, AudioFileClip
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter
from multiprocessing import Pool, cpu_count
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np
import os

def _asciify_frame(
        frame, scale_factor=0.15, return_to_original_size=False, one_char_width=8, one_char_height=8, color_brightness=1, pixel_brightness=2.15, char_set = "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\\|()1{}[]?-_+~<>i!lI;:,\"^`'. ",
        monochrome=False, filters=None, overlay_contours=True, contour_depth_minimum_threshold = 0, contour_depth_maximum_threshold = 255, progress_bar=False
    ):
    
    chars = char_set[::-1]
    charArray = list(chars)
    charLength = len(charArray)
    interval = charLength / 256
    contour_chars = "|/-\\_ "

    def apply_crt_effect(image):
        crt_image = image.copy()
        width, height = crt_image.size
        pixels = crt_image.load()
        
        for y in range(height):
            if y % 2 == 0:
                for x in range(width):
                    r, g, b = pixels[x, y]
                    pixels[x, y] = (r // 2, g // 2, b // 2)
        
        crt_image = crt_image.filter(ImageFilter.GaussianBlur(1))
        return crt_image

    def apply_sepia_filter(image):
        sepia_image = np.array(image)
        sepia_filter = np.array([[0.272, 0.534, 0.131],
                                [0.349, 0.686, 0.168],
                                [0.393, 0.769, 0.189]])
                                
        sepia_image = cv2.transform(sepia_image, sepia_filter)
        sepia_image = np.clip(sepia_image, 0, 255)
        return Image.fromarray(sepia_image.astype(np.uint8))

    def apply_color_tint(image, tint_color):
        tinted_image = Image.new("RGB", image.size)
        image = image.convert("RGB") 
        tint_color_image = Image.new("RGB", tinted_image.size, tint_color)
        blended = Image.blend(image, tint_color_image, alpha=0.3)
        return blended
    
    def getChar(inputInt):
        return charArray[math.floor(inputInt * interval)]
    
    if frame.shape[2] == 3:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    elif frame.shape[2] == 4:
        frame_rgb = cv2.cvtColor(frame[:, :, :3], cv2.COLOR_BGR2RGB)
    
    im = Image.fromarray(frame_rgb)
    fnt = ImageFont.truetype('lucon.ttf', 10)

    width, height = im.size
    original_size = (width, height)
    im = im.resize((int(scale_factor * width), int(scale_factor * height * (one_char_width / one_char_height))), Image.NEAREST)
    width, height = im.size
    pix = im.load()

    outputImage = Image.new('RGB', (int(one_char_width * width), int(one_char_height * height)), color=(0, 0, 0))
    d = ImageDraw.Draw(outputImage)

    gray = cv2.cvtColor(np.array(im), cv2.COLOR_BGR2GRAY)

    depth = cv2.Laplacian(gray, cv2.CV_64F)
    depth = cv2.convertScaleAbs(depth)
    
    _, depth_mask = cv2.threshold(depth, max(0, contour_depth_minimum_threshold), min(255, contour_depth_maximum_threshold), cv2.THRESH_BINARY)
    
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edges = cv2.Canny(gray, 50, 70)

    for i in tqdm(range(height), disable=not progress_bar, leave=False):
        for j in tqdm(range(width), disable=not progress_bar, leave=False):
            r, g, b = pix[j, i]
            
            mult_r = 0.299 * color_brightness
            mult_g = 0.587 * color_brightness
            mult_b = 0.114 * color_brightness
            
            h = int(mult_r * r + mult_g * g + mult_b * b)
            h = min(255, int(h * pixel_brightness))
            
            char = getChar(h)
            
            if overlay_contours and edges[i, j] > 0 and depth_mask[i, j] > 0:
                angle = cv2.phase(grad_x[i, j], grad_y[i, j], angleInDegrees=True)[0] % 180
                if angle < 45:
                    char = contour_chars[0]  # |
                elif angle < 90:
                    char = contour_chars[1]  # /
                elif angle < 135:
                    char = contour_chars[2]  # -
                else:
                    char = contour_chars[3]  # \
                        
            rgb = (r,g,b)
            
            if monochrome:
                rgb = (h,h,h)

            d.text((j * one_char_width, i * one_char_height), char, font=fnt, fill=rgb, align="center")
    
    outputImage = outputImage.convert('RGB')
    
    if filters:
        if 'crt' in filters:
            outputImage = apply_crt_effect(outputImage)
        if 'sepia' in filters:
            outputImage = apply_sepia_filter(outputImage)
        if 'tint' in filters and 'tint_color' in filters:
            outputImage = apply_color_tint(outputImage, filters['tint_color'])
    
    if return_to_original_size:
        outputImage = outputImage.resize(original_size, Image.NEAREST)
    
    output_frame = np.array(outputImage)
    return output_frame

def _asciify_frame_wrapper(args):
    return _asciify_frame(*args)

def ascii_photo(
        in_path, out_path, scale_factor=0.15, return_to_original_size=False, one_char_width=8, one_char_height=8, color_brightness=1, pixel_brightness=2.15, char_set = "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\\|()1{}[]?-_+~<>i!lI;:,\"^`'. ",
        monochrome=False, filters=None, overlay_contours=True, contour_depth_minimum_threshold = 0, contour_depth_maximum_threshold = 255, progress_bar=False
    ):
    """
    Asciifies a photo from a path, and saves it to a file.

    Parameters:
    - in_path: String.
        Relative path to the input image.
    - out_path: String.
        Relative path to the output image. Doesn't need to exist.
    - scale_factor: Float.
        Controls the image quality in the ASCII image.
        Default is 0.15.
    - return_to_original_size: Bool.
        If True, the frame is resized back to original size after conversion (loses quality).
        Default is False.
    - one_char_width: Int.
        Width of one character in the ASCII representation.
        Default is 8.
    - one_char_height: Int.
        Height of one character in the ASCII representation.
        Default is 8.
    - color_brightness: Float.
        Specify brightness multiplier of colors in calculations.
        Default is 1.
    - pixel_brightness: Float.
        Specify brightness multiplier of the drawn pixels in the frame.
        Default is 2.15.
    - char_set: String.
        A string containing all the ASCII/Unicode characters to represent pixels (going from lightest to darkest.)
        Default is a predertimened string for an ASCII set.
    - monochrome: Bool.
        If True, a frame are rendered using only grayscale colors.
        Default is False.
    - filters: Dict.
        A dictionary containing filters to use. crt, sepia, and tint are boolean keys, and tint requires a tint_color key with a color tuple (0-255).
        Default is None.
    - overlay_contours: Bool.
        If True, overlays contours on the image.
        Default is True.
    - contour_depth_minimum_threshold: Float.
        Specify the minimum threshold of the depth map for when point contours are drawn. Must be between (0-255).
        Default is 0.
    - contour_depth_maximum_threshold: Float.
        Specify the maximum threshold of the depth map for when point contours are drawn. Must be between (0-255).
        Default is 255.
    - progress_bar: Bool.
        If True, displays a progress bar in the console.
        Default is False.
        
    Returns: Void.
    """
    
    if not os.path.exists(in_path):
        print("Invalid input path.")
        return

    cap = cv2.imread(in_path)
    frame_args = (
        cap, scale_factor, return_to_original_size, one_char_width, one_char_height, color_brightness, pixel_brightness, char_set, monochrome, 
        filters, overlay_contours, contour_depth_minimum_threshold, contour_depth_maximum_threshold, progress_bar
    )
    ascii_frame = Image.fromarray(_asciify_frame(*frame_args), 'RGB')
    ascii_frame.save(out_path)
    if progress_bar: 
        print(f"Saved asciified photo to {out_path}")

def ascii_video(
        in_path, out_path, scale_factor = 0.15, return_to_original_size=False, one_char_width = 8, one_char_height = 8, color_brightness=1, pixel_brightness=2.15, char_set = "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\\|()1{}[]?-_+~<>i!lI;:,\"^`'. ",
        monochrome=False, filters=None, overlay_contours = True, contour_depth_minimum_threshold = 0, contour_depth_maximum_threshold = 255, low_res_audio = True, progress_bar = True, num_workers=None
    ):
    """
    Asciifies a video from a path, and saves it to a file.

    Parameters:
    - in_path: String.
        Relative path to the input video.
    - out_path: String.
        Relative path to the output video. Doesn't need to exist.
    - scale_factor: Float.
        Controls the video quality in the ASCII video.
        Default is 0.15.
    - return_to_original_size: Bool.
        If True, frames is resized back to original size after conversion (loses quality).
        Default is False.
    - one_char_width: Int.
        Width of one character in the ASCII representation.
        Default is 8.
    - one_char_height: Int.
        Height of one character in the ASCII representation.
        Default is 8.
    - color_brightness: Float.
        Specify brightness multiplier of colors in calculations.
        Default is 1.
    - pixel_brightness: Float.
        Specify brightness multiplier of the drawn pixels in the frame.
        Default is 2.15.
    - char_set: String.
        A string containing all the ASCII/Unicode characters to represent pixels (going from lightest to darkest.)
        Default is a predertimened string for an ASCII set.
    - monochrome: Bool.
        If True, frames are rendered using only grayscale colors.
        Default is False.
    - filters: Dict.
        A dictionary containing filters to use. crt, sepia, and tint are boolean keys, and tint requires a tint_color key with a color tuple (0-255).
        Default is None.
    - overlay_contours: Bool.
        If True, overlays contours on the video.
        Default is True.
    - contour_depth_minimum_threshold: Float.
        Specify the minimum threshold of the depth map for when point contours are drawn. Must be between (0-255).
        Default is 0.
    - contour_depth_maximum_threshold: Float.
        Specify the maximum threshold of the depth map for when point contours are drawn. Must be between (0-255).
        Default is 255.
    - low_res_audio: Bool.
        If True, audio quality will be lowered to 8Khz to match the video.
        Default is true.
    - progress_bar: Bool.
        If True, displays a progress bar in the console.
        Default is True.
    - num_workers: Int or None.
        Number of concurrent workers for multiprocessing.
        If None, uses the number of system's CPU cores.
        Default is None.
        
    Returns: Void
    """

    if not os.path.exists(in_path):
        print("Invalid input path.")
        return

    cap = cv2.VideoCapture(in_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_rate = cap.get(cv2.CAP_PROP_FPS)

    if not cap.isOpened():
        print('Cap is not open')
        return

    if num_workers is None:
        num_workers = cpu_count()

    frame_args = []

    for current_frame in tqdm(range(1, frame_count + 1), desc="Processing frames", disable = not progress_bar, leave=False):
        ret, frame = cap.read()
        if ret:
            frame_args.append((
                frame, scale_factor, return_to_original_size, one_char_width, one_char_height, color_brightness, pixel_brightness, char_set, monochrome, 
                filters, overlay_contours, contour_depth_minimum_threshold, contour_depth_maximum_threshold, False
            ))
        else:
            break
    with Pool(num_workers) as pool:
        ascii_frames = list(tqdm(pool.imap(_asciify_frame_wrapper, frame_args), total=len(frame_args), desc="Converting frames", disable=not progress_bar, leave=False))
        pool.close()
        pool.join()

    temp_video_path = "temp.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_video = cv2.VideoWriter(temp_video_path, fourcc, frame_rate, (ascii_frames[0].shape[1], ascii_frames[0].shape[0]))

    for frame in tqdm(ascii_frames, desc="Saving video", disable = not progress_bar, leave=False):
        output_video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    
    output_video.release()

    original_clip = VideoFileClip(in_path)
    original_audio = original_clip.audio
    
    clip = VideoFileClip(temp_video_path)
    clip = clip.to_RGB()
    fps = clip.fps

    temp_audio_path = "temp_audio.mp3"
    temp_audio_low_path = "temp_audio_low.mp3"

    if original_audio != None:
        if low_res_audio:
            low_audio = original_audio.set_fps(8000)
            low_audio.write_audiofile(temp_audio_low_path, verbose=False, logger=None)
            
            audio = AudioFileClip(temp_audio_low_path)
            audio = audio.set_fps(16000)
            audio.write_audiofile(temp_audio_path, verbose=False, logger=None)
        else:
            original_audio.write_audiofile(temp_audio_path, verbose=False, logger=None)
    else:
        temp_audio_path = None

    with FFMPEG_VideoWriter(out_path, fps=fps, size=clip.size, codec='libx264', logfile=None, threads=num_workers, audiofile=temp_audio_path, ffmpeg_params=['-strict', '-2']) as writer:
        frame_iterator = tqdm(clip.iter_frames(fps), total=int(clip.duration*fps), desc = "Adding audio", disable = not progress_bar, leave=False)

        for frame in frame_iterator:
            writer.write_frame(frame)

        writer.close()

    clip.close()
    original_clip.close()
    os.remove(temp_video_path)
    if temp_audio_path:
        if low_res_audio: os.remove(temp_audio_low_path)
        os.remove(temp_audio_path)