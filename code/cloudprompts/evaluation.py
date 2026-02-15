from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

import warnings

try:
    import torch

    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False

PathLike = Union[str, Path]
ArrayLike = Union[np.ndarray, "torch.Tensor"]
IgnoreIndex = Optional[int]


def _to_numpy(x: ArrayLike) -> np.ndarray:
    if _HAS_TORCH and hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _normalize_2d_mask(arr: np.ndarray, *, source: str) -> np.ndarray:
    """
    Normalize common mask layouts to [H, W] int64.

    Accepts:
      - [H, W]
      - [H, W, 1]
      - [1, H, W]

    Rejects multi-channel masks like [H, W, 3] because those are usually
    color-coded labels and need explicit conversion.
    """
    arr = np.asarray(arr)

    if arr.ndim == 2:
        return arr.astype(np.int64, copy=False)

    if arr.ndim == 3:
        if arr.shape[-1] == 1:
            return arr[..., 0].astype(np.int64, copy=False)
        if arr.shape[0] == 1:
            return arr[0].astype(np.int64, copy=False)

    raise ValueError(
        f"{source}: expected a single-channel class-index mask [H,W], "
        f"got shape={arr.shape}. If this is RGB/color-coded, convert it to "
        f"class indices before evaluation."
    )


def _flatten_matching_shapes(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")

    if y_true.ndim == 2:
        return y_true.reshape(-1), y_pred.reshape(-1)

    if y_true.ndim == 3:
        return y_true.reshape(-1), y_pred.reshape(-1)

    raise ValueError(f"Expected [H,W] or [N,H,W], got ndim={y_true.ndim}")


def load_mask(path: PathLike, *, key: str = "mask") -> np.ndarray:
    """
    Load a mask from .npz/.npy/.png/.tif/.tiff as int64 [H,W].

    Notes:
      - .npz expects `key` (default: "mask")
      - multi-channel RGB masks are intentionally rejected
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)

    suffix = p.suffix.lower()

    if suffix == ".npz":
        with np.load(p) as z:
            if key not in z:
                raise KeyError(f"{p}: key '{key}' not found. Available keys: {list(z.keys())}")
            arr = z[key]

    elif suffix == ".npy":
        arr = np.load(p)

    elif suffix in {".png", ".tif", ".tiff"}:
        try:
            from PIL import Image
        except Exception as exc:
            raise ImportError(
                "Pillow is required to read .png/.tif/.tiff masks. Install with `pip install pillow`."
            ) from exc
        arr = np.array(Image.open(p))

    else:
        raise ValueError(f"Unsupported mask type: {p} (supported: .npz, .npy, .png, .tif, .tiff)")

    return _normalize_2d_mask(np.asarray(arr), source=str(p))


def _match_pairs_by_stem(
    gt_dir: Path,
    pred_dir: Path,
    *,
    gt_ext: str,
    pred_ext: str,
) -> List[Tuple[str, Path, Path]]:
    gt_files = sorted(gt_dir.glob(f"*{gt_ext}"))
    pairs: List[Tuple[str, Path, Path]] = []

    for g in gt_files:
        sid = g.stem
        p = pred_dir / f"{sid}{pred_ext}"
        if p.exists():
            pairs.append((sid, g, p))

    return pairs


def confusion_matrix_from_masks(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    num_classes: int,
    ignore_index: Optional[int] = None,
    strict: bool = True,
) -> np.ndarray:
    """
    Rows = y_true, Cols = y_pred.

    Behavior:
      - If label == num_classes: raise error (likely 1..n indexing issue)
      - If label < 0: ignore with warning (in strict=False), error in strict=True
      - If label > num_classes: ignore with warning (in strict=False), error in strict=True
    """
    yt = _to_numpy(y_true).astype(np.int64, copy=False)
    yp = _to_numpy(y_pred).astype(np.int64, copy=False)

    yt, yp = _flatten_matching_shapes(yt, yp)

    if ignore_index is not None:
        keep = yt != int(ignore_index)
        yt = yt[keep]
        yp = yp[keep]

    if yt.size == 0:
        return np.zeros((num_classes, num_classes), dtype=np.int64)

    if num_classes <= 0:
        raise ValueError(f"num_classes must be > 0, got {num_classes}")

    eq_n = (yt == num_classes) | (yp == num_classes)
    if eq_n.any():
        vals_t = np.unique(yt[yt == num_classes])[:10]
        vals_p = np.unique(yp[yp == num_classes])[:10]
        raise ValueError(
            f"Found label value == num_classes ({num_classes}). "
            f"Possible indexing mismatch (labels may be 1..n instead of 0..n-1). "
            f"Examples y_true={vals_t}, y_pred={vals_p}."
        )

    neg = (yt < 0) | (yp < 0)
    gt_n = (yt > num_classes) | (yp > num_classes)

    if strict:
        if neg.any():
            vals_t = np.unique(yt[yt < 0])[:20]
            vals_p = np.unique(yp[yp < 0])[:20]
            raise ValueError(
                f"Negative labels found. y_true examples={vals_t}, y_pred examples={vals_p}."
            )
        if gt_n.any():
            vals_t = np.unique(yt[yt > num_classes])[:20]
            vals_p = np.unique(yp[yp > num_classes])[:20]
            raise ValueError(
                f"Labels > num_classes ({num_classes}) found. "
                f"y_true examples={vals_t}, y_pred examples={vals_p}."
            )
    else:
        if neg.any():
            warnings.warn(
                f"Ignoring {int(neg.sum())} pixels with negative labels.",
                RuntimeWarning,
            )
        if gt_n.any():
            warnings.warn(
                f"Ignoring {int(gt_n.sum())} pixels with labels > num_classes ({num_classes}).",
                RuntimeWarning,
            )
        keep = ~(neg | gt_n)
        yt = yt[keep]
        yp = yp[keep]

    if yt.size == 0:
        return np.zeros((num_classes, num_classes), dtype=np.int64)

    k = yt * num_classes + yp
    cm = np.bincount(k, minlength=num_classes * num_classes).reshape(num_classes, num_classes)
    return cm.astype(np.int64)



def metrics_from_confusion(cm: np.ndarray) -> Dict[str, Any]:
    cm = np.asarray(cm, dtype=np.float64)
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp

    denom = tp + fp + fn
    per_class_iou = np.where(denom > 0, tp / denom, np.nan)

    miou = float(np.nanmean(per_class_iou)) if np.any(~np.isnan(per_class_iou)) else np.nan
    pixel_acc = float(tp.sum() / cm.sum()) if cm.sum() > 0 else np.nan

    return {
        "miou": miou,
        "pixel_acc": pixel_acc,
        "per_class_iou": per_class_iou,
    }


def evaluate_segmentation(
    *,
    gt_dir: PathLike,
    pred_dir: PathLike,
    num_classes: int,
    model: str,
    dataset: str,
    stage: str,
    run_root: PathLike = "runs",
    class_names: Optional[Sequence[str]] = None,
    ignore_index: IgnoreIndex = None,
    gt_key: str = "mask",
    pred_key: str = "mask",
    gt_ext: str = ".npz",
    pred_ext: str = ".npz",
    strict: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate segmentation by matching files by stem and writing:
      - miou_<model>_<dataset>_<stage>.csv
      - cm_<model>_<dataset>_<stage>.csv

    Expected file matching:
      gt_dir/<id><gt_ext>
      pred_dir/<id><pred_ext>

    Mask requirements per file:
      - single-channel class-index mask [H,W]
      - labels in [0, num_classes-1] unless strict=False
    """
    gt_dir = Path(gt_dir)
    pred_dir = Path(pred_dir)

    if not gt_dir.exists():
        raise FileNotFoundError(gt_dir)
    if not pred_dir.exists():
        raise FileNotFoundError(pred_dir)

    gt_files = sorted(gt_dir.glob(f"*{gt_ext}"))
    if not gt_files:
        raise RuntimeError(f"No GT files found in {gt_dir} matching *{gt_ext}")

    pairs = _match_pairs_by_stem(gt_dir, pred_dir, gt_ext=gt_ext, pred_ext=pred_ext)
    if not pairs:
        msg = f"No matching pairs between gt_dir={gt_dir} and pred_dir={pred_dir}"
        if strict:
            raise RuntimeError(msg)

        run_root = Path(run_root)
        run_root.mkdir(parents=True, exist_ok=True)
        miou_path = run_root / f"miou_{model}_{dataset}_{stage}.csv"
        cm_path = run_root / f"cm_{model}_{dataset}_{stage}.csv"
        pd.DataFrame([{"id": "__aggregate__", "miou": np.nan, "pixel_acc": np.nan}]).to_csv(miou_path, index=False)
        pd.DataFrame([{"id": "__aggregate__"}]).to_csv(cm_path, index=False)
        return {
            "miou_csv": str(miou_path),
            "cm_csv": str(cm_path),
            "num_samples": 0,
            "aggregate_miou": np.nan,
            "aggregate_pixel_acc": np.nan,
        }

    metric_rows: List[Dict[str, Any]] = []
    cm_rows: List[Dict[str, Any]] = []
    cm_total = np.zeros((num_classes, num_classes), dtype=np.int64)

    cm_cols = [f"cm_t{t}_p{p}" for t in range(num_classes) for p in range(num_classes)]

    for sid, gt_path, pr_path in pairs:
        y_true = load_mask(gt_path, key=gt_key)
        y_pred = load_mask(pr_path, key=pred_key)

        if y_true.shape != y_pred.shape:
            raise ValueError(f"{sid}: shape mismatch gt {y_true.shape} vs pred {y_pred.shape}")

        cm = confusion_matrix_from_masks(
            y_true,
            y_pred,
            num_classes=num_classes,
            ignore_index=ignore_index,
            strict=strict,
        )
        cm_total += cm

        stats = metrics_from_confusion(cm)

        row: Dict[str, Any] = {
            "id": sid,
            "miou": stats["miou"],
            "pixel_acc": stats["pixel_acc"],
        }
        for k in range(num_classes):
            cname = class_names[k] if class_names and k < len(class_names) else f"class_{k}"
            v = stats["per_class_iou"][k]
            row[f"iou_{cname}"] = float(v) if not np.isnan(v) else np.nan
        metric_rows.append(row)

        flat = cm.reshape(-1)
        cm_row = {"id": sid}
        cm_row.update({c: int(v) for c, v in zip(cm_cols, flat)})
        cm_rows.append(cm_row)

    agg = metrics_from_confusion(cm_total)

    agg_row: Dict[str, Any] = {
        "id": "__aggregate__",
        "miou": agg["miou"],
        "pixel_acc": agg["pixel_acc"],
    }
    for k in range(num_classes):
        cname = class_names[k] if class_names and k < len(class_names) else f"class_{k}"
        v = agg["per_class_iou"][k]
        agg_row[f"iou_{cname}"] = float(v) if not np.isnan(v) else np.nan
    metric_rows.append(agg_row)

    agg_flat = cm_total.reshape(-1)
    agg_cm_row = {"id": "__aggregate__"}
    agg_cm_row.update({c: int(v) for c, v in zip(cm_cols, agg_flat)})
    cm_rows.append(agg_cm_row)

    run_root = Path(run_root)
    run_root.mkdir(parents=True, exist_ok=True)

    miou_path = run_root / f"miou_{model}_{dataset}_{stage}.csv"
    cm_path = run_root / f"cm_{model}_{dataset}_{stage}.csv"

    pd.DataFrame(metric_rows).to_csv(miou_path, index=False)
    pd.DataFrame(cm_rows).to_csv(cm_path, index=False)

    return {
        "miou_csv": str(miou_path),
        "cm_csv": str(cm_path),
        "num_samples": len(pairs),
        "aggregate_miou": agg["miou"],
        "aggregate_pixel_acc": agg["pixel_acc"],
    }


__all__ = [
    "load_mask",
    "confusion_matrix_from_masks",
    "metrics_from_confusion",
    "evaluate_segmentation",
    "evaluate_metrics",
]