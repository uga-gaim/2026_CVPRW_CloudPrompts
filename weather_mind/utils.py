from __future__ import annotations

from pathlib import Path
import random
from typing import List
import numpy as np
from PIL import Image

import math

def _pick_sample_images(folder: str | Path, k: int, exts: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")) -> List[Path]:
    """
    Return up to k random image Paths from the given folder (non-recursive).
    """
    folder = Path(folder)
    candidates = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts]
    if not candidates:
        raise FileNotFoundError(f"No images with extensions {exts} found in: {folder}")

    k = min(k, len(candidates))
    return random.sample(candidates, k)


def _load_grayscale(image_path: str | Path) -> np.ndarray:
    """
    Load an image and return a 2D uint8 array (shape: H x W), with values in [0, 255].
    """
    image_path = Path(image_path)
    with Image.open(image_path) as im:
        gray = im.convert("L")
        arr = np.array(gray, dtype=np.uint8)
    return arr

def _smooth1d(x: np.ndarray, window: int) -> np.ndarray:
    """
    Simple moving average smoothing (same length as input).
    """
    if window <= 1:
        return x
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(x, kernel, mode="same")


def _estimate_bottom_bar_height_from_array(
    arr_gray: np.ndarray,
    *,
    dark_quantile: float = 0.15,
    min_dark_frac: float = 0.60,
    smooth_window: int = 5,
    max_search_ratio: float = 0.40,
) -> int:
    """
    Estimate the height (in pixels) of a black/dark bar at the bottom of a grayscale image.

    Arguments:
        arr_gray : np.ndarray
            2D uint8 array (H x W) with values in [0, 255].
        dark_quantile : float
            Quantile of pixel intensities used to define the "dark" threshold. Lower = stricter (darker).
            Example: 0.15 means the bottom 15% darkest pixels define the cutoff.
        min_dark_frac : float
            Row is considered "bar" if at least this fraction of pixels are darker than the threshold.
            Keep this < 1.0 because the bar usually contains lighter text.
        smooth_window : int
            Moving average window (in rows) to smooth the row-wise darkness fraction.
        max_search_ratio : float
            Only analyze the bottom portion of the image (e.g., 0.40 = bottom 40% of rows)
            to avoid mis-detecting dark regions elsewhere.

    Returns:
        int
            Estimated bar height in pixels (0 if not found).
    """
    if arr_gray.ndim != 2:
        raise ValueError("Expected a 2D grayscale array (H x W).")
    H, W = arr_gray.shape
    if H == 0 or W == 0:
        return 0

    search_h = max(1, int(H * max_search_ratio))
    start_row = H - search_h
    roi = arr_gray[start_row:, :]

    q_val = float(np.quantile(roi, dark_quantile))
    threshold = int(np.clip(q_val, 5, 200))
    dark_frac = (roi < threshold).mean(axis=1)

    dark_frac_s = _smooth1d(dark_frac, smooth_window)

    is_bar_row = dark_frac_s >= min_dark_frac

    height = 0
    for val in is_bar_row[::-1]:
        if val:
            height += 1
        else:
            break

    return int(height)


def _estimate_bar_height_single(
    image_path: str | Path,
    **kwargs
) -> int:
    """
    Convenience wrapper: load an image and estimate its bottom bar height.
    Any kwargs are passed to _estimate_bottom_bar_height_from_array().
    """
    arr = _load_grayscale(image_path)
    return _estimate_bottom_bar_height_from_array(arr, **kwargs)



def estimate_average_bottom_bar_height(
    folder: str | Path,
    sample_count: int,
    *,
    seed: int | None = None,
    ignore_zero: bool = True,
    verbose: bool = False,
    **kwargs,
) -> int:
    """
    Sample `sample_count` images from `folder`, estimate each one's bottom-bar height,
    and return the rounded average height (in pixels).

    Arguments:
        folder : str | Path
            Directory containing images (non-recursive).
        sample_count : int
            Number of images to sample randomly.
        seed : int | None
            Optional RNG seed for reproducible sampling.
        ignore_zero : bool
            If True, drop zero-height estimates (i.e., detections that found no bar).
        verbose : bool
            If True, prints per-image heights for quick debugging.
        **kwargs :
            Extra keyword arguments passed through to `_estimate_bar_height_single`
            (e.g., dark_quantile, min_dark_frac, smooth_window, max_search_ratio).

    Returns:
        int
            Rounded mean of the (filtered) heights, or 0 if none are usable.
    """
    if seed is not None:
        random.seed(seed)

    paths = _pick_sample_images(folder, sample_count)
    heights: list[int] = []

    for p in paths:
        try:
            h = _estimate_bar_height_single(p, **kwargs)
            if verbose:
                print(f"[bar-height] {p.name}: {h}px")
            if (h > 0) or (not ignore_zero):
                heights.append(int(h))
        except Exception as e:
            if verbose:
                print(f"[warn] failed on {p}: {e}")

    if not heights:
        if verbose:
            print("[bar-height] No valid height estimates; returning 0.")
        return 0

    avg = int(round(float(np.mean(heights))))
    if verbose:
        print(f"[bar-height] Average over {len(heights)} images: {avg}px")
    return avg




def _crop_bottom_pixels(im: Image.Image, bar_height: int) -> Image.Image:
    """
    Return a copy of `im` with `bar_height` pixels removed from the bottom.
    If `bar_height` <= 0, returns the original image.
    If `bar_height` >= image height, returns a 1-pixel-high image (safety clamp).
    """
    if not isinstance(im, Image.Image):
        raise TypeError("Expected a PIL.Image.Image")

    w, h = im.size
    if bar_height <= 0:
        return im.copy()

    cut = min(bar_height, h - 1)
    box = (0, 0, w, h - cut)
    return im.crop(box)


def _crop_file_and_return_image(image_path: str | Path, bar_height: int) -> Image.Image:
    """
    Open an image from disk, crop `bar_height` pixels from the bottom,
    and return the cropped PIL image (not saved).
    """
    image_path = Path(image_path)
    with Image.open(image_path) as im:
        im.load()
        cropped = _crop_bottom_pixels(im, bar_height)
    return cropped



def remove_bottom_bar_from_folder(
    folder: str | Path,
    bar_height: int,
    *,
    output_suffix: str = "_clean",
    exts: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"),
    overwrite: bool = False,
    verbose: bool = True,
) -> Path:
    """
    Removes `bar_height` pixels from the bottom of all images in a folder and saves
    the cropped versions into a new directory named "<folder><output_suffix>".

    Arguments:
        folder : str | Path
            Input directory containing images.
        bar_height : int
            Number of pixels to remove from the bottom (from estimate_average_bottom_bar_height()).
        output_suffix : str
            Suffix appended to the input folder name for output.
        exts : tuple[str, ...]
            Allowed image extensions.
        overwrite : bool
            If True, overwrite files in the output directory if they already exist.
        verbose : bool
            If True, prints progress.

    Returns:
        Path
            Path to the output directory.
    """
    folder = Path(folder)
    output_dir = folder.parent / f"{folder.name}{output_suffix}"
    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"[crop] Saving cropped images to: {output_dir}")
        print(f"[crop] Removing {bar_height}px from bottom of each image...")

    count = 0
    for p in folder.iterdir():
        if not p.is_file() or p.suffix.lower() not in exts:
            continue

        out_path = output_dir / p.name
        if out_path.exists() and not overwrite:
            if verbose:
                print(f"[skip] {out_path.name} (already exists)")
            continue

        try:
            cropped = _crop_file_and_return_image(p, bar_height)
            cropped.save(out_path)
            count += 1
            if verbose and count % 20 == 0:
                print(f"[crop] {count} images processed...")
        except Exception as e:
            if verbose:
                print(f"[warn] failed on {p.name}: {e}")

    if verbose:
        print(f"[done] Cropped {count} images to {output_dir}")

    return output_dir