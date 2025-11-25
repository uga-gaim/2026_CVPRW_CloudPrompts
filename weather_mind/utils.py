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
    verbose: bool = False,
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


def extract_bottom_bar_to_csv(
    folder: str | Path,
    *,
    sample_count: int = 100,
    output_csv: str | Path | None = None,
    exts: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"),
    tesseract_cmd: str | None = None,
    ocr_lang: str = "eng",
    seed: int | None = 42,
    verbose: bool = True,
) -> Path:
    """
    1) Estimate the average bottom-bar height using estimate_average_bottom_bar_height().
    2) For every image in `folder`, crop just the bottom bar, OCR it, parse fields.
    3) Write one CSV row per image with: image_path, bar_height_px, and parsed values.

    Returns
    -------
    Path : path to the written CSV.
    """

    from pathlib import Path as _Path
    import csv, re
    from typing import Dict, Any, Iterable
    from PIL import Image, ImageOps

    try:
        import pytesseract as _pt
    except Exception as e:
        raise RuntimeError(
        ) from e
    if tesseract_cmd:
        _pt.pytesseract.tesseract_cmd = str(tesseract_cmd)

    folder = _Path(folder)
    if output_csv is None:
        output_csv = folder.parent / f"{folder.name}_ocr.csv"
    output_csv = _Path(output_csv)

    def _crop_bottom_strip(image_path: _Path, bar_h: int) -> Image.Image:
        with Image.open(image_path) as im:
            w, h = im.size
            cut = max(1, min(bar_h, h - 1))
            strip = im.crop((0, h - cut, w, h))
            strip.load()
            return strip

    def _preprocess_for_ocr(im: Image.Image) -> Image.Image:
        g = im.convert("L")
        g = ImageOps.autocontrast(g)
        w, h = g.size
        return g.resize((w * 2, h * 2))
    
    def _parse_overlay(text: str) -> Dict[str, Any]:
        """Return a dict with raw values + detected unit codes."""
        t = " ".join(text.replace("\n", " ").split()).lower()
        out: Dict[str, Any] = {}
        def _to_float(s):
            try: return float(s)
            except Exception: return None

        m = re.search(r"(\d{4}-\d{2}-\d{2})\s+(\d{1,2}:\d{2})", t)
        if m:
            out["date"] = m.group(1)
            out["time"] = m.group(2)

        m = re.search(r"(?:temp[^0-9\-]*|temperature[^0-9\-]*)(-?\d+(?:\.\d+)?)\s*°?\s*([cf])", t)
        if m:
            out["temperature_value"] = _to_float(m.group(1))
            out["temperature_unit"]  = {"c": "c", "f": "f"}[m.group(2)]

        m = re.search(r"(?:feels\s*like[^0-9\-]*)(-?\d+(?:\.\d+)?)\s*°?\s*([cf])", t)
        if m:
            out["feels_like_value"] = _to_float(m.group(1))
            out["feels_like_unit"]  = {"c": "c", "f": "f"}[m.group(2)]

        m = re.search(r"(?:humidity|rh)[^0-9]{0,5}(\d{1,3})\s*%", t)
        if m:
            out["humidity_pct"] = int(m.group(1))

        m = re.search(r"(?:pressure)[^0-9\-]{0,6}(\d+(?:\.\d+)?)\s*(hpa|mb|inhg)", t)
        if m:
            out["pressure_value"] = _to_float(m.group(1))
            out["pressure_unit"]  = m.group(2)

        m = re.search(r"(?:wind)[^0-9\-]{0,6}(\d+(?:\.\d+)?)\s*(mph|mi/?h|kph|km/?h)?", t)
        if m:
            out["wind_speed_value"] = _to_float(m.group(1))
            unit = (m.group(2) or "").replace("mi/h","mph").replace("km/h","kph").replace("kmh","kph")
            out["wind_speed_unit"]  = unit if unit in {"mph","kph"} else None

        m = re.search(r"(?:gust)[^0-9\-]{0,6}(\d+(?:\.\d+)?)\s*(mph|mi/?h|kph|km/?h)?", t)
        if m:
            out["gust_speed_value"] = _to_float(m.group(1))
            unit = (m.group(2) or "").replace("mi/h","mph").replace("km/h","kph").replace("kmh","kph")
            out["gust_speed_unit"]  = unit if unit in {"mph","kph"} else None

        m = re.search(r"(?:rain\s*today|rain\s*tdy|rain)\s*[:\-]?\s*(\d+(?:\.\d+)?)\s*(in|mm|cm)", t)
        if m:
            out["rain_today_value"] = _to_float(m.group(1))
            out["rain_today_unit"]  = m.group(2)
        return out

    avg_h = estimate_average_bottom_bar_height(folder, sample_count=sample_count, seed=seed, verbose=False)
    if verbose:
        print(f"[bar-ocr] Using average bar height: {avg_h}px")

    images: Iterable[_Path] = sorted(
        p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts
    )

    rows = []
    for p in images:
        try:
            strip = _crop_bottom_strip(p, avg_h)
            strip = _preprocess_for_ocr(strip)
            raw = _pt.image_to_string(strip, lang=ocr_lang, config="--psm 6")
            parsed = _parse_overlay(raw)

            row = {"image_path": str(p), "bar_height_px": int(avg_h)}
            if "date" in parsed: row["date"] = parsed["date"]
            if "time" in parsed: row["time"] = parsed["time"]
            if "humidity_pct" in parsed: row["humidity_pct"] = parsed["humidity_pct"]

            if "temperature_value" in parsed and "temperature_unit" in parsed:
                row[f"temperature_{parsed['temperature_unit']}"] = parsed["temperature_value"]

            if "feels_like_value" in parsed and "feels_like_unit" in parsed:
                row[f"feels_like_{parsed['feels_like_unit']}"] = parsed["feels_like_value"]

            if "pressure_value" in parsed and "pressure_unit" in parsed:
                row[f"pressure_{parsed['pressure_unit']}"] = parsed["pressure_value"]

            if "wind_speed_value" in parsed and parsed.get("wind_speed_unit"):
                row[f"wind_speed_{parsed['wind_speed_unit']}"] = parsed["wind_speed_value"]

            if "gust_speed_value" in parsed and parsed.get("gust_speed_unit"):
                row[f"gust_speed_{parsed['gust_speed_unit']}"] = parsed["gust_speed_value"]

            if "rain_today_value" in parsed and "rain_today_unit" in parsed:
                row[f"rain_today_{parsed['rain_today_unit']}"] = parsed["rain_today_value"]

            rows.append(row)
        except Exception as e:
            if verbose:
                print(f"[warn] OCR failed on {p.name}: {e}")
            rows.append({"image_path": str(p), "bar_height_px": int(avg_h), "error": str(e)})

    if not rows:
        if verbose:
            print("[bar-ocr] No images found; wrote empty CSV.")
        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["image_path","bar_height_px"])
            writer.writeheader()
        return output_csv

    all_keys = set().union(*(r.keys() for r in rows))
    preferred = [
        "image_path","bar_height_px","date","time","humidity_pct",
        "temperature_c","temperature_f","feels_like_c","feels_like_f",
        "pressure_hpa","pressure_mb","pressure_inhg",
        "wind_speed_mph","wind_speed_kph","gust_speed_mph","gust_speed_kph",
        "rain_today_in","rain_today_mm","rain_today_cm",
        "error",
    ]
    header = [k for k in preferred if k in all_keys] + [k for k in sorted(all_keys) if k not in preferred]

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in header})

    if verbose:
        print(f"[bar-ocr] Wrote {len(rows)} rows to {output_csv}")
    return output_csv