import argparse
import importlib
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import pytorch_lightning as pl

from data.sparcs import SparcsDataset

SPARCS_LABELS = [
    "cloud shadow",
    "cloud shadow over water",
    "water",
    "snow or ice",
    "land",
    "cloud",
    "flooded area",
]

def try_import_lsegnet():
    """
    Try a few common import paths. Adjust if your repo differs.
    """
    candidates = [
        "modules.models.lseg_net",
        "models.lseg_net",
        "lseg.models.lseg_net",
        "lseg_net",
    ]
    last_err = None
    for mod in candidates:
        try:
            m = importlib.import_module(mod)
            if hasattr(m, "LSegNet"):
                return getattr(m, "LSegNet"), mod
        except Exception as e:
            last_err = e
    raise ImportError(
        "Could not import LSegNet. Edit try_import_lsegnet() candidates to match your repo.\n"
        f"Last error: {last_err}"
    )

def strip_prefix_if_present(state_dict: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    if not any(k.startswith(prefix) for k in state_dict.keys()):
        return state_dict
    return {k[len(prefix):]: v for k, v in state_dict.items()}

def load_checkpoint_into_model(model: nn.Module, ckpt_path: str) -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

    for pref in ["net.", "model.", "module."]:
        state = strip_prefix_if_present(state, pref)

    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[ckpt] Loaded with strict=False")
    print(f"[ckpt] Missing keys: {len(missing)}")
    print(f"[ckpt] Unexpected keys: {len(unexpected)}")

def confusion_matrix(pred: torch.Tensor, target: torch.Tensor, num_classes: int, ignore_index: int) -> torch.Tensor:
    """
    pred: (B,H,W) int64
    target: (B,H,W) int64
    returns: (C,C) where rows=gt, cols=pred
    """
    with torch.no_grad():
        mask = (target != ignore_index)
        pred = pred[mask]
        target = target[mask]
        cm = torch.bincount(
            (target * num_classes + pred),
            minlength=num_classes * num_classes
        ).reshape(num_classes, num_classes)
    return cm

def miou_from_cm(cm: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    cm: (C,C)
    returns: (miou, per_class_iou)
    """
    tp = torch.diag(cm).float()
    fp = cm.sum(dim=0).float() - tp
    fn = cm.sum(dim=1).float() - tp
    denom = tp + fp + fn
    iou = torch.where(denom > 0, tp / denom, torch.zeros_like(tp))
    miou = iou.mean()
    return miou, iou

class LSegSparcsFinetune(pl.LightningModule):
    def __init__(
        self,
        lsegnet_ctor,
        ckpt_path: str,
        lr: float,
        weight_decay: float,
        num_classes: int = 7,
        crop_size: int = 480,
        ignore_index: int = 255,
        backbone: str = "clip_vitl16_384",
        features: int = 256,
        arch_option: int = 0,
        block_depth: int = 0,
        activation: str = "lrelu",
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["lsegnet_ctor"])

        self.num_classes = num_classes
        self.ignore_index = ignore_index

        self.net = lsegnet_ctor(
            labels=SPARCS_LABELS,
            backbone=backbone,
            features=features,
            crop_size=crop_size,
            arch_option=arch_option,
            block_depth=block_depth,
            activation=activation,
        )

        load_checkpoint_into_model(self.net, ckpt_path)

        self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)

        self.register_buffer("val_cm", torch.zeros(num_classes, num_classes, dtype=torch.int64), persistent=False)
        self.register_buffer("test_cm", torch.zeros(num_classes, num_classes, dtype=torch.int64), persistent=False)

    def forward(self, x):
        out = self.net(x)
        if isinstance(out, (tuple, list)):
            out = out[0]
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        if logits.ndim != 4:
            raise RuntimeError(f"Expected logits (B,C,H,W), got shape {tuple(logits.shape)}")

        if logits.shape[1] != self.num_classes:
            raise RuntimeError(
                f"Expected {self.num_classes} classes, got C={logits.shape[1]}. "
                "Check your labels/head config."
            )

        loss = self.criterion(logits, y)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log("val/loss", loss, prog_bar=True)

        pred = torch.argmax(logits, dim=1).to(torch.int64)
        cm = confusion_matrix(pred, y.to(torch.int64), self.num_classes, self.ignore_index)
        self.val_cm += cm
        return loss

    def on_validation_epoch_end(self):
        miou, per_class = miou_from_cm(self.val_cm)
        self.log("val/mIoU", miou, prog_bar=True)
        # Reset
        self.val_cm.zero_()

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        pred = torch.argmax(logits, dim=1).to(torch.int64)
        cm = confusion_matrix(pred, y.to(torch.int64), self.num_classes, self.ignore_index)
        self.test_cm += cm

    def on_test_epoch_end(self):
        miou, per_class = miou_from_cm(self.test_cm)
        self.log("test/mIoU", miou, prog_bar=True)
        names = ["shad", "shad_w", "water", "snow", "land", "cloud", "flood"]
        msg = " | ".join([f"{n}:{per_class[i].item():.3f}" for i, n in enumerate(names)])
        print(f"[test] mIoU={miou.item():.4f}  {msg}")
        self.test_cm.zero_()

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.trainer.max_epochs)
        return {"optimizer": opt, "lr_scheduler": sch}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True, help="LSeg checkpoint (e.g., fss_l16.ckpt)")
    ap.add_argument("--out_dir", type=str, default="runs/sparcs_baseline")
    ap.add_argument("--crop", type=int, default=480)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "gpu"])
    args = ap.parse_args()

    pl.seed_everything(args.seed, workers=True)

    data_root = Path(args.data_root)
    for sp in ["train", "val", "test"]:
        p = data_root / "splits" / f"{sp}.txt"
        if not p.exists():
            raise FileNotFoundError(f"Missing split file: {p}. Run tools/make_sparcs_splits.py first.")

    train_ds = SparcsDataset(args.data_root, "train", crop_size=args.crop, augment=True)
    val_ds   = SparcsDataset(args.data_root, "val",   crop_size=args.crop, augment=False)
    test_ds  = SparcsDataset(args.data_root, "test",  crop_size=args.crop, augment=False)

    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.workers, pin_memory=True)
    val_dl   = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=args.workers, pin_memory=True)
    test_dl  = DataLoader(test_ds, batch_size=args.batch, shuffle=False, num_workers=args.workers, pin_memory=True)

    LSegNet, modpath = try_import_lsegnet()
    print(f"[import] Using LSegNet from: {modpath}")

    model = LSegSparcsFinetune(
        lsegnet_ctor=LSegNet,
        ckpt_path=args.ckpt,
        lr=args.lr,
        weight_decay=args.wd,
        num_classes=7,
        crop_size=args.crop,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    accelerator = "auto" if args.device == "auto" else args.device
    devices = 1 if accelerator != "cpu" else "auto"

    ckpt_cb = pl.callbacks.ModelCheckpoint(
        dirpath=str(out_dir),
        filename="best-{epoch:02d}-{val_mIoU:.4f}",
        monitor="val/mIoU",
        mode="max",
        save_top_k=1,
    )

    trainer = pl.Trainer(
        default_root_dir=str(out_dir),
        max_epochs=args.epochs,
        accelerator=accelerator,
        devices=devices,
        callbacks=[ckpt_cb],
        log_every_n_steps=10,
    )

    trainer.fit(model, train_dl, val_dl)

    best = ckpt_cb.best_model_path
    print("[done] Best checkpoint:", best)

    if best:
        model = LSegSparcsFinetune.load_from_checkpoint(
            best,
            lsegnet_ctor=LSegNet,
            ckpt_path=args.ckpt,
            lr=args.lr,
            weight_decay=args.wd,
        )
    trainer.test(model, test_dl)

if __name__ == "__main__":
    main()