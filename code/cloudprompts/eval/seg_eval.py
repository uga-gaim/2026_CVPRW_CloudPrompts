from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Union, Dict, Any, Tuple, List
import numpy as np
import pandas as pd

try:
    import torch
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False

ArrayLike = Union[np.ndarray, "torch.Tensor"]
PathLike = Union[str, Path]

def _to_numpy(x: ArrayLike) -> np.ndarray:
    if _HAS_TORCH and hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def _validate_shapes(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")

    if y_true.ndim == 2:
        return y_true.reshape(-1), y_pred.reshape(-1)
    if y_true.ndim == 3:
        return y_true.reshape(-1), y_pred.reshape(-1)
    raise ValueError(f"Expected [H,W] or [B,H,W], got ndim={y_true.ndim}")

def load_mask(path: PathLike, *, key: str = "mask") -> np.ndarray:
    """
    Load a mask from:
      - .npz  (expects `key`, default "mask")
      - .npy
      - .png (uint8)
    Returns:
      np.ndarray [H,W] or [B,H,W] (int64)
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)

    suf = p.suffix.lower()
    if suf == ".npz":
        arr = np.load(p)[key]
    elif suf == ".npy":
        arr = np.load(p)
    elif suf == ".png":
        from PIL import Image
        arr = np.array(Image.open(p))
    else:
        raise ValueError(f"Unsupported mask file type: {p} (use .npz/.npy/.png)")

    return np.asarray(arr, dtype=np.int64)

def load_mask_pairs_from_dirs(
    gt_dir: PathLike,
    pred_dir: PathLike,
    *,
    gt_ext: str = ".npz",
    pred_ext: str = ".npz",
    gt_key: str = "mask",
    pred_key: str = "mask",
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load masks from two directories and align by sample id (basename).

    Expects files like:
      gt_dir/<id>.npz
      pred_dir/<id>.npz

    Returns:
      y_true: [N,H,W] int64
      y_pred: [N,H,W] int64
      ids: list of ids
    """
    gt_dir = Path(gt_dir)
    pred_dir = Path(pred_dir)
    if not gt_dir.exists(): raise FileNotFoundError(gt_dir)
    if not pred_dir.exists(): raise FileNotFoundError(pred_dir)

    gt_files = sorted(gt_dir.glob(f"*{gt_ext}"))
    if not gt_files:
        raise RuntimeError(f"No GT files found in {gt_dir} matching *{gt_ext}")

    ids = []
    y_true_list = []
    y_pred_list = []

    for gtp in gt_files:
        sid = gtp.stem
        predp = pred_dir / f"{sid}{pred_ext}"
        if not predp.exists():
            continue

        gt = load_mask(gtp, key=gt_key)
        pr = load_mask(predp, key=pred_key)

        if gt.shape != pr.shape:
            raise ValueError(f"{sid}: shape mismatch gt {gt.shape} vs pred {pr.shape}")

        if gt.ndim != 2:
            raise ValueError(f"{sid}: expected per-file mask to be [H,W], got {gt.shape}")

        ids.append(sid)
        y_true_list.append(gt)
        y_pred_list.append(pr)

    if not ids:
        raise RuntimeError(f"No matching pairs found between {gt_dir} and {pred_dir}")

    y_true = np.stack(y_true_list, axis=0)
    y_pred = np.stack(y_pred_list, axis=0)
    return y_true, y_pred, ids

def confusion_matrix_from_masks(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    num_classes: int,
    ignore_index: Optional[int] = None,
    strict: bool = True,
) -> np.ndarray:
    yt = _to_numpy(y_true).astype(np.int64, copy=False)
    yp = _to_numpy(y_pred).astype(np.int64, copy=False)

    yt, yp = _validate_shapes(yt, yp)

    if ignore_index is not None:
        keep = yt != ignore_index
        yt = yt[keep]
        yp = yp[keep]

    bad_true = (yt < 0) | (yt >= num_classes)
    bad_pred = (yp < 0) | (yp >= num_classes)

    if strict:
        if bad_true.any():
            vals = np.unique(yt[bad_true])[:20]
            raise ValueError(f"y_true has out-of-range labels (expected 0..{num_classes-1}). "
                             f"Examples: {vals} (count={bad_true.sum()})")
        if bad_pred.any():
            vals = np.unique(yp[bad_pred])[:20]
            raise ValueError(f"y_pred has out-of-range labels (expected 0..{num_classes-1}). "
                             f"Examples: {vals} (count={bad_pred.sum()})")
    else:
        keep = ~(bad_true | bad_pred)
        yt = yt[keep]
        yp = yp[keep]

    k = yt * num_classes + yp
    cm = np.bincount(k, minlength=num_classes * num_classes).reshape(num_classes, num_classes)
    return cm.astype(np.int64)

def miou_from_confusion(cm: np.ndarray) -> Dict[str, Any]:
    cm = cm.astype(np.float64)
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    denom = tp + fp + fn
    per_class_iou = np.where(denom > 0, tp / denom, np.nan)
    miou = np.nanmean(per_class_iou)
    pixel_acc = (tp.sum() / cm.sum()) if cm.sum() > 0 else np.nan
    return {"miou": float(miou), "pixel_acc": float(pixel_acc), "per_class_iou": per_class_iou}

def _load_npz_key(p: Path, key: str) -> np.ndarray:
    arr = np.load(p)[key]
    return np.asarray(arr, dtype=np.int64)

def _match_pairs_by_stem(
    gt_dir: Path,
    pred_dir: Path,
    gt_ext: str,
    pred_ext: str,
) -> List[Tuple[str, Path, Path]]:
    gt_files = sorted(gt_dir.glob(f"*{gt_ext}"))
    pairs = []
    for g in gt_files:
        sid = g.stem
        pr = pred_dir / f"{sid}{pred_ext}"
        if pr.exists():
            pairs.append((sid, g, pr))
    return pairs

def evaluate_metrics(
    *,
    gt_dir: PathLike,
    pred_dir: PathLike,
    model: str,
    dataset: str,
    stage: str,
    num_classes: int,
    run_root: PathLike = "runs",
    class_names: Optional[Sequence[str]] = None,
    ignore_index: Optional[int] = None,
    gt_key: str = "mask",
    pred_key: str = "mask",
    gt_ext: str = ".npz",
    pred_ext: str = ".npz",
    strict: bool = True,
) -> Dict[str, Any]:
    """
    Matches files by stem (basename) and computes:
      - per-sample mIoU + per-class IoU + pixel_acc
      - per-sample confusion matrix (flattened)
      - one final aggregate row ("__aggregate__") for both tables

    Writes exactly:
      <run_root>/miou_<model>_<dataset>_<stage>.csv   (many rows + aggregate)
      <run_root>/cm_<model>_<dataset>_<stage>.csv     (many rows + aggregate)

    Expects:
      gt_dir/<id>.npz contains gt_key (default "mask") -> [H,W]
      pred_dir/<id>.npz contains pred_key (default "mask") -> [H,W]
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

    pairs: List[Tuple[str, Path, Path]] = []
    for g in gt_files:
        sid = g.stem
        p = pred_dir / f"{sid}{pred_ext}"
        if p.exists():
            pairs.append((sid, g, p))

    if not pairs:
        msg = f"No matching pairs between gt_dir={gt_dir} and pred_dir={pred_dir}"
        if strict:
            raise RuntimeError(msg)
        else:
            print("WARNING:", msg)

    def _load_npz(p: Path, key: str) -> np.ndarray:
        return np.asarray(np.load(p)[key], dtype=np.int64)

    metric_rows = []
    cm_rows = []
    cm_total = np.zeros((num_classes, num_classes), dtype=np.int64)

    cm_cols = [f"cm_t{t}_p{p}" for t in range(num_classes) for p in range(num_classes)]

    for sid, gt_path, pr_path in pairs:
        y_true = _load_npz(gt_path, gt_key)
        y_pred = _load_npz(pr_path, pred_key)

        if y_true.shape != y_pred.shape:
            raise ValueError(f"{sid}: shape mismatch gt {y_true.shape} vs pred {y_pred.shape}")
        if y_true.ndim != 2:
            raise ValueError(f"{sid}: expected mask [H,W], got {y_true.shape}")

        cm = confusion_matrix_from_masks(y_true, y_pred, num_classes=num_classes, ignore_index=ignore_index)
        cm_total += cm

        stats = miou_from_confusion(cm)

        row = {
            "id": sid,
            "miou": stats["miou"],
            "pixel_acc": stats["pixel_acc"],
        }
        for k in range(num_classes):
            name = class_names[k] if class_names and k < len(class_names) else f"class_{k}"
            v = stats["per_class_iou"][k]
            row[f"iou_{name}"] = float(v) if not np.isnan(v) else np.nan

        metric_rows.append(row)

        flat = cm.reshape(-1).tolist()
        cm_row = {"id": sid}
        cm_row.update({c: int(v) for c, v in zip(cm_cols, flat)})
        cm_rows.append(cm_row)

    agg_stats = miou_from_confusion(cm_total)

    agg_row = {
        "id": "__aggregate__",
        "miou": agg_stats["miou"],
        "pixel_acc": agg_stats["pixel_acc"],
    }
    for k in range(num_classes):
        name = class_names[k] if class_names and k < len(class_names) else f"class_{k}"
        v = agg_stats["per_class_iou"][k]
        agg_row[f"iou_{name}"] = float(v) if not np.isnan(v) else np.nan
    metric_rows.append(agg_row)

    agg_cm_flat = cm_total.reshape(-1).tolist()
    agg_cm_row = {"id": "__aggregate__"}
    agg_cm_row.update({c: int(v) for c, v in zip(cm_cols, agg_cm_flat)})
    cm_rows.append(agg_cm_row)
    
    run_root = Path(run_root)
    run_root.mkdir(parents=True, exist_ok=True)

    miou_path = run_root / f"miou_{model}_{dataset}_{stage}.csv"
    cm_path   = run_root / f"cm_{model}_{dataset}_{stage}.csv"

    pd.DataFrame(metric_rows).to_csv(miou_path, index=False)
    pd.DataFrame(cm_rows).to_csv(cm_path, index=False)

    return {
        "miou_csv": str(miou_path),
        "cm_csv": str(cm_path),
        "num_samples": len(pairs),
        "aggregate_miou": agg_stats["miou"],
        "aggregate_pixel_acc": agg_stats["pixel_acc"],
    }