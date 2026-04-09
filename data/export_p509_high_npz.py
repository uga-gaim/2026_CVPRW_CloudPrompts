import argparse
from pathlib import Path
import numpy as np
import mlstac
from tqdm import tqdm

import os
import contextlib

@contextlib.contextmanager
def suppress_stdout():
    """Silence print() output from libraries while keeping exceptions/stderr visible."""
    with open(os.devnull, "w") as devnull:
        with contextlib.redirect_stdout(devnull):
            yield

RGB_IDXS = (3, 2, 1)
CM1_IDX = 13

def load_datapoint(df, idx: int):
    subset = df.iloc[idx:idx+1]
    with suppress_stdout():
        out = mlstac.core.load_data(subset)
    if isinstance(out, np.ndarray):
        return out[0] if out.ndim == 4 else out
    if isinstance(out, list):
        first = out[0]
        if isinstance(first, tuple):
            data, _meta = first
            return data
        return first
    raise TypeError(f"Unexpected load_data output type: {type(out)}")

def export_split(mlstac_path: Path, out_root: Path, split: str, crop_509: bool):
    df = mlstac.core.load_metadata(str(mlstac_path))
    df = df.sort_values("datapoint_id").reset_index(drop=True)

    img_dir = out_root / split / "images"
    msk_dir = out_root / split / "masks"
    img_dir.mkdir(parents=True, exist_ok=True)
    msk_dir.mkdir(parents=True, exist_ok=True)

    H_target = 509 if crop_509 else 512
    W_target = 509 if crop_509 else 512

    for i in tqdm(range(len(df)), desc=f"export {split}"):
        dp_id = df.loc[i, "datapoint_id"]

        dp = load_datapoint(df, i)
        rgb = np.moveaxis(dp[list(RGB_IDXS)], 0, -1).astype(np.float32) / 5000.0
        rgb = np.clip(rgb, 0.0, 1.0)

        mask = dp[CM1_IDX].astype(np.uint8)

        if crop_509:
            rgb = rgb[0:509, 3:512, :]
            mask = mask[0:509, 3:512]

        if rgb.shape[:2] != (H_target, W_target) or mask.shape != (H_target, W_target):
            raise ValueError(f"{dp_id}: shape mismatch rgb {rgb.shape}, mask {mask.shape}")

        rgb_chw = np.transpose(rgb, (2, 0, 1)).astype(np.float32)

        np.savez_compressed(img_dir / f"{dp_id}.npz", image=rgb_chw)
        np.savez_compressed(msk_dir / f"{dp_id}.npz", mask=mask)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        default="./cloudsen12plus_p509",
        help="Folder containing train/validation/test .mlstac files (default: %(default)s)"
    )
    ap.add_argument(
        "--out",
        default="./export_p509_high_npz",
        help="Output dataset folder (default: %(default)s)"
    )
    ap.add_argument("--crop509", action="store_true", help="Crop padded 512->509 (removes left+bottom padding)")
    args = ap.parse_args()

    root = Path(args.root)
    out = Path(args.out)

    train_p = root / "train" / "train_509_high.mlstac"
    val_p   = root / "validation" / "validation_509_high.mlstac"
    test_p  = root / "test" / "test_509_high.mlstac"

    for p in (train_p, val_p, test_p):
        if not p.exists():
            raise FileNotFoundError(f"Missing: {p}")

    export_split(train_p, out, "train", crop_509=args.crop509)
    export_split(val_p,   out, "val",   crop_509=args.crop509)
    export_split(test_p,  out, "test",  crop_509=args.crop509)

    print("Done. Exported to:", out)

if __name__ == "__main__":
    main()