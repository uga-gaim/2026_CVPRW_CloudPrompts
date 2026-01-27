from pathlib import Path
import random
import argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True, help="SPARCS folder containing *_photo.png and *_labels.tif")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train", type=int, default=60)
    ap.add_argument("--val", type=int, default=10)
    ap.add_argument("--test", type=int, default=10)
    args = ap.parse_args()

    root = Path(args.root)
    photos = sorted(root.glob("*_photo.png"))
    ids = [p.name.replace("_photo.png", "") for p in photos]

    ids = [sid for sid in ids if (root / f"{sid}_labels.tif").exists()]

    if len(ids) < (args.train + args.val + args.test):
        raise SystemExit(
            f"Not enough scenes found ({len(ids)}). "
            f"Need at least {args.train+args.val+args.test} for requested split."
        )

    random.seed(args.seed)
    random.shuffle(ids)

    train_ids = ids[:args.train]
    val_ids   = ids[args.train:args.train + args.val]
    test_ids  = ids[args.train + args.val:args.train + args.val + args.test]

    out = root / "splits"
    out.mkdir(exist_ok=True)

    (out / "train.txt").write_text("\n".join(train_ids) + "\n")
    (out / "val.txt").write_text("\n".join(val_ids) + "\n")
    (out / "test.txt").write_text("\n".join(test_ids) + "\n")

    print(f"Total scenes: {len(ids)}")
    print(f"train: {len(train_ids)}  val: {len(val_ids)}  test: {len(test_ids)}")
    print("Wrote:", out)

if __name__ == "__main__":
    main()
