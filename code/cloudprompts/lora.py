from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def _require(pkg: str, hint: str) -> None:
    try:
        __import__(pkg)
    except Exception as e:
        raise RuntimeError(
            f"Missing dependency '{pkg}'. {hint}\nOriginal error: {repr(e)}"
        ) from e


_require("transformers", "Install with: pip install transformers")
_require("peft", "Install with: pip install peft")
from transformers import TrainingArguments, Trainer, set_seed
from peft import LoraConfig, get_peft_model

try:
    from .models import get_model_adapter
except Exception:
    from models import get_model_adapter


def _csv_to_list(s: str | Sequence[str]) -> List[str]:
    if isinstance(s, str):
        return [x.strip() for x in s.split(",") if x.strip()]
    return [str(x).strip() for x in s if str(x).strip()]


def _csv_to_ints(s: str | Sequence[int]) -> List[int]:
    if isinstance(s, str):
        return [int(x.strip()) for x in s.split(",") if x.strip()]
    return [int(x) for x in s]


def dice_loss_from_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    num = 2.0 * (probs * targets).sum(dim=(1, 2)) + eps
    den = (probs + targets).sum(dim=(1, 2)) + eps
    return 1.0 - (num / den).mean()


def iou_from_logits(logits: torch.Tensor, targets: torch.Tensor, thresh: float = 0.5, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    preds = (probs > thresh).float()
    inter = (preds * targets).sum(dim=(1, 2))
    union = (preds + targets - preds * targets).sum(dim=(1, 2))
    return ((inter + eps) / (union + eps)).mean()

class ImprovedSegLoss(torch.nn.Module):
    """
    Combined loss for binary segmentation (one-vs-rest):
      total = w_focal * focal + w_tversky * tversky + w_boundary * boundary
    """
    def __init__(
        self,
        focal_alpha: float = 0.75,
        focal_gamma: float = 2.0,
        tversky_alpha: float = 0.3,   # weight on FP
        tversky_beta: float = 0.7,    # weight on FN (higher => recall-friendly)
        boundary_scale: float = 2.0,
        boundary_kernel_size: int = 3,
        min_boundary_pixels: int = 10,
        weights: tuple = (0.8, 1.0, 0.1),
        only_non_empty_tversky: bool = True,
    ):
        super().__init__()
        if boundary_kernel_size % 2 == 0:
            raise ValueError("boundary_kernel_size must be odd (e.g., 3, 5).")

        self.focal_alpha = float(focal_alpha)
        self.focal_gamma = float(focal_gamma)
        self.tversky_alpha = float(tversky_alpha)
        self.tversky_beta = float(tversky_beta)
        self.boundary_scale = float(boundary_scale)
        self.boundary_kernel_size = int(boundary_kernel_size)
        self.min_boundary_pixels = int(min_boundary_pixels)
        self.w_focal, self.w_tversky, self.w_boundary = [float(x) for x in weights]
        self.only_non_empty_tversky = bool(only_non_empty_tversky)

    def focal_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
        focal_weight = (1.0 - p_t).pow(self.focal_gamma)
        alpha_t = self.focal_alpha * targets + (1.0 - self.focal_alpha) * (1.0 - targets)
        return (alpha_t * focal_weight * ce).mean()

    def tversky_loss(self, logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        probs = torch.sigmoid(logits).flatten(1)
        t = targets.flatten(1)

        if self.only_non_empty_tversky:
            keep = t.sum(dim=1) > 0
            if keep.any():
                probs = probs[keep]
                t = t[keep]
            else:
                return logits.new_tensor(0.0)

        tp = (probs * t).sum(dim=1)
        fp = (probs * (1.0 - t)).sum(dim=1)
        fn = ((1.0 - probs) * t).sum(dim=1)

        tversky = (tp + eps) / (tp + self.tversky_alpha * fp + self.tversky_beta * fn + eps)
        return 1.0 - tversky.mean()

    def boundary_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if targets.sum() < float(self.min_boundary_pixels):
            return logits.new_tensor(0.0)

        k = self.boundary_kernel_size
        pad = k // 2

        t = targets.unsqueeze(1)
        max_pool = F.max_pool2d(t, kernel_size=k, stride=1, padding=pad)
        min_pool = -F.max_pool2d(-t, kernel_size=k, stride=1, padding=pad)
        boundary = ((max_pool - min_pool) > 0).to(targets.dtype).squeeze(1)

        if boundary.sum() < 1:
            return logits.new_tensor(0.0)

        weight = 1.0 + self.boundary_scale * boundary
        bce_px = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        return (bce_px * weight).sum() / weight.sum().clamp_min(1.0)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        focal = self.focal_loss(logits, targets)
        tversky = self.tversky_loss(logits, targets)
        boundary = self.boundary_loss(logits, targets)

        total = (
            self.w_focal * focal
            + self.w_tversky * tversky
            + self.w_boundary * boundary
        )
        return total


def print_trainable_params(model: torch.nn.Module) -> None:
    trainable = 0
    total = 0
    for p in model.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    pct = 100.0 * trainable / max(total, 1)
    print(f"Trainable params: {trainable:,} / {total:,} ({pct:.2f}%)")



class CloudSENPromptDataset(Dataset):
    """
    CloudSEN12+ NPZ dataset flattened into (image, prompt, binary_mask) samples.

    Directory contract:
      data_root/<split>/images/*.npz with key 'image' (CHW float in [0,1])
      data_root/<split>/masks/*.npz  with key 'mask'  (HW class IDs)
    """

    def __init__(
        self,
        *,
        data_root: str,
        split: str,
        processor,
        model_adapter,
        prompts: List[str],
        class_ids: List[int],
        prompt_template: str = "{label}",
        max_images: Optional[int] = None,
        dataset_name: str = "cloudsen12plus",
    ):
        self.data_root = data_root
        self.split = split
        self.processor = processor
        self.model_adapter = model_adapter
        self.prompts = prompts
        self.class_ids = class_ids
        self.prompt_template = prompt_template

        if dataset_name.strip().lower() != "cloudsen12plus":
            raise NotImplementedError(
                f"Dataset '{dataset_name}' is not implemented in LoRA pipeline yet. "
                "Current support: cloudsen12plus"
            )

        img_dir = os.path.join(data_root, split, "images")
        msk_dir = os.path.join(data_root, split, "masks")
        if not os.path.isdir(img_dir):
            raise FileNotFoundError(f"Missing images dir: {img_dir}")
        if not os.path.isdir(msk_dir):
            raise FileNotFoundError(f"Missing masks dir: {msk_dir}")

        ids = []
        for fn in os.listdir(img_dir):
            if fn.endswith(".npz"):
                stem = os.path.splitext(fn)[0]
                if os.path.exists(os.path.join(msk_dir, f"{stem}.npz")):
                    ids.append(stem)

        ids.sort()
        if max_images is not None:
            ids = ids[:max_images]

        if len(ids) == 0:
            raise RuntimeError(f"No matching image/mask pairs found in {img_dir} and {msk_dir}")

        self.ids = ids
        self.img_dir = img_dir
        self.msk_dir = msk_dir
        self.num_classes = len(prompts)
        self.total = len(self.ids) * self.num_classes

    def __len__(self) -> int:
        return self.total

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_i = idx // self.num_classes
        cls_i = idx % self.num_classes

        dp_id = self.ids[img_i]
        class_id = int(self.class_ids[cls_i])
        label_text = self.prompts[cls_i]
        prompt = self.prompt_template.format(label=label_text)

        img_npz = np.load(os.path.join(self.img_dir, f"{dp_id}.npz"))
        msk_npz = np.load(os.path.join(self.msk_dir, f"{dp_id}.npz"))

        if "image" not in img_npz:
            raise KeyError(f"Missing key 'image' in {self.img_dir}/{dp_id}.npz")
        if "mask" not in msk_npz:
            raise KeyError(f"Missing key 'mask' in {self.msk_dir}/{dp_id}.npz")

        img_chw = img_npz["image"].astype(np.float32)
        mask_hw = msk_npz["mask"].astype(np.int64)

        pil_img = self.model_adapter.prepare_image(img_chw)
        y = self.model_adapter.prepare_binary_mask(mask_hw, class_id)
        enc = self.model_adapter.encode(self.processor, prompt=prompt, image_pil=pil_img)

        return {
            "pixel_values": enc["pixel_values"],
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": y,
        }


@dataclass
class DataCollator:
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = {}
        for k in ["pixel_values", "input_ids", "attention_mask", "labels"]:
            batch[k] = torch.stack([f[k] for f in features], dim=0)
        return batch

class SegTrainer(Trainer):
    def __init__(
        self,
        loss_name: str = "improved",
        dice_weight: float = 1.0,
        pos_weight: Optional[float] = None,
        focal_alpha: float = 0.75,
        focal_gamma: float = 2.0,
        tversky_alpha: float = 0.3,
        tversky_beta: float = 0.7,
        w_focal: float = 0.8,
        w_tversky: float = 1.0,
        w_boundary: float = 0.1,
        boundary_scale: float = 2.0,
        boundary_kernel_size: int = 3,
        min_boundary_pixels: int = 10,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.loss_name = loss_name.strip().lower()

        self.dice_weight = float(dice_weight)
        self.pos_weight = pos_weight

        if self.loss_name == "improved":
            self.criterion = ImprovedSegLoss(
                focal_alpha=focal_alpha,
                focal_gamma=focal_gamma,
                tversky_alpha=tversky_alpha,
                tversky_beta=tversky_beta,
                boundary_scale=boundary_scale,
                boundary_kernel_size=boundary_kernel_size,
                min_boundary_pixels=min_boundary_pixels,
                weights=(w_focal, w_tversky, w_boundary),
                only_non_empty_tversky=True,
            )
        elif self.loss_name == "bce_dice":
            self.criterion = None
        else:
            raise ValueError(f"Unsupported loss_name='{loss_name}'. Use 'improved' or 'bce_dice'.")

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")

        outputs = model(**inputs)
        logits = outputs.logits
        if logits.ndim == 4 and logits.shape[1] == 1:
            logits = logits[:, 0, :, :]
        elif logits.ndim != 3:
            raise RuntimeError(f"Unexpected logits shape: {tuple(logits.shape)}")

        labels = labels.to(logits.dtype)

        if self.loss_name == "improved":
            loss = self.criterion(logits, labels)
        else:
            if self.pos_weight is not None:
                pw = torch.tensor([self.pos_weight], device=logits.device, dtype=logits.dtype)
                bce = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pw)
            else:
                bce = F.binary_cross_entropy_with_logits(logits, labels)

            dloss = dice_loss_from_logits(logits, labels)
            loss = bce + self.dice_weight * dloss

        return (loss, outputs) if return_outputs else loss



@dataclass
class LoRAConfig:
    technique: str = "lora"
    model_name: str = "clipseg"
    dataset_name: str = "cloudsen12plus"

    data_root: str = ""
    output_dir: str = ""

    model_id: Optional[str] = None
    seed: int = 42
    labels: str = "clear,thick cloud,thin cloud,cloud shadow"
    class_ids: str = "0,1,2,3"
    prompt_template: str = "{label}"
    image_size: Optional[int] = None

    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: Optional[str] = None

    epochs: int = 5
    lr: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    train_bs: int = 16
    eval_bs: int = 2
    grad_accum: int = 1
    num_workers: int = 4

    eval_steps: int = 1000
    save_steps: int = 1000
    logging_steps: int = 50
    save_total_limit: int = 3

    fp16: bool = False
    bf16: bool = False
    gradient_checkpointing: bool = False

    loss_name: str = "improved"

    focal_alpha: float = 0.75
    focal_gamma: float = 2.0
    tversky_alpha: float = 0.3
    tversky_beta: float = 0.7

    w_focal: float = 0.8
    w_tversky: float = 1.0
    w_boundary: float = 0.1

    boundary_scale: float = 2.0
    boundary_kernel_size: int = 3
    min_boundary_pixels: int = 10

    dice_weight: float = 1.0
    pos_weight: Optional[float] = None

    max_train_images: Optional[int] = None
    max_val_images: Optional[int] = None


def run_lora(cfg: LoRAConfig) -> None:
    if cfg.technique.strip().lower() != "lora":
        raise ValueError(f"LoRA runner received technique='{cfg.technique}'. Expected 'lora'.")

    if not cfg.data_root:
        raise ValueError("data_root is required")
    if not cfg.output_dir:
        raise ValueError("output_dir is required")

    set_seed(cfg.seed)

    prompts = _csv_to_list(cfg.labels)
    class_ids = _csv_to_ints(cfg.class_ids)
    if len(prompts) != len(class_ids):
        raise ValueError(f"labels count ({len(prompts)}) must match class_ids count ({len(class_ids)})")

    adapter = get_model_adapter(
        cfg.model_name,
        model_id=cfg.model_id,
        image_size=cfg.image_size,
    )

    print(f"[model] {adapter.spec.key} | id={adapter.model_id} | train_size={adapter.image_size}")
    print(f"[dataset] {cfg.dataset_name}")
    print(f"[technique] LoRA")
    print("[prompts]", list(zip(class_ids, prompts)))
    print("[prompt_template]", cfg.prompt_template)

    processor = adapter.build_processor()
    model = adapter.build_model()

    if cfg.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        print("Enabled gradient checkpointing.")

    target_modules = (
        _csv_to_list(cfg.target_modules)
        if cfg.target_modules
        else list(adapter.spec.default_target_modules)
    )
    lora_cfg = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=target_modules,
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)

    bad = [n for n, p in model.named_parameters() if p.requires_grad and "lora_" not in n]
    if bad:
        raise RuntimeError(
            "Pure LoRA violated. Non-LoRA trainable params found, e.g.: "
            + ", ".join(bad[:10])
        )

    print_trainable_params(model)

    train_ds = CloudSENPromptDataset(
        data_root=cfg.data_root,
        split="train",
        processor=processor,
        model_adapter=adapter,
        prompts=prompts,
        class_ids=class_ids,
        prompt_template=cfg.prompt_template,
        max_images=cfg.max_train_images,
        dataset_name=cfg.dataset_name,
    )
    val_ds = CloudSENPromptDataset(
        data_root=cfg.data_root,
        split="val",
        processor=processor,
        model_adapter=adapter,
        prompts=prompts,
        class_ids=class_ids,
        prompt_template=cfg.prompt_template,
        max_images=cfg.max_val_images,
        dataset_name=cfg.dataset_name,
    )

    targs = TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.lr,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        per_device_train_batch_size=cfg.train_bs,
        per_device_eval_batch_size=cfg.eval_bs,
        gradient_accumulation_steps=cfg.grad_accum,
        evaluation_strategy="steps",
        eval_steps=cfg.eval_steps,
        save_steps=cfg.save_steps,
        logging_steps=cfg.logging_steps,
        save_total_limit=cfg.save_total_limit,
        dataloader_num_workers=cfg.num_workers,
        remove_unused_columns=False,
        label_names=["labels"],
        fp16=cfg.fp16,
        bf16=cfg.bf16,
        load_best_model_at_end=True,
        metric_for_best_model="iou",
        greater_is_better=True,
    )

    def compute_metrics(eval_pred):
        preds = eval_pred.predictions
        if isinstance(preds, (tuple, list)):
            preds = preds[0]
        if preds.ndim == 4 and preds.shape[1] == 1:
            preds = preds[:, 0, :, :]
        labels_np = eval_pred.label_ids
        logits = torch.from_numpy(preds)
        y = torch.from_numpy(labels_np).float()
        with torch.no_grad():
            di = 1.0 - dice_loss_from_logits(logits, y).item()
            iu = iou_from_logits(logits, y).item()
        return {"dice": di, "iou": iu}

    trainer = SegTrainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollator(),
        tokenizer=processor.tokenizer,
        compute_metrics=compute_metrics,
        loss_name=cfg.loss_name,
        focal_alpha=cfg.focal_alpha,
        focal_gamma=cfg.focal_gamma,
        tversky_alpha=cfg.tversky_alpha,
        tversky_beta=cfg.tversky_beta,
        w_focal=cfg.w_focal,
        w_tversky=cfg.w_tversky,
        w_boundary=cfg.w_boundary,
        boundary_scale=cfg.boundary_scale,
        boundary_kernel_size=cfg.boundary_kernel_size,
        min_boundary_pixels=cfg.min_boundary_pixels,
        dice_weight=cfg.dice_weight,
        pos_weight=cfg.pos_weight,
    )

    print("Starting training...")
    trainer.train()

    print("Saving final adapter + trainer state...")
    trainer.save_model(cfg.output_dir)
    processor.save_pretrained(cfg.output_dir)
    print("Done. Saved to:", cfg.output_dir)


def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="LoRA finetuning runner")

    ap.add_argument("--model_name", type=str, default="clipseg")
    ap.add_argument("--dataset_name", type=str, default="cloudsen12plus")

    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--output_dir", type=str, required=True)

    ap.add_argument("--model_id", type=str, default=None)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--labels", type=str, default="clear,thick cloud,thin cloud,cloud shadow")
    ap.add_argument("--class_ids", type=str, default="0,1,2,3")
    ap.add_argument("--prompt_template", type=str, default="{label}")
    ap.add_argument("--image_size", type=int, default=None)

    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--target_modules", type=str, default=None)

    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--train_bs", type=int, default=16)
    ap.add_argument("--eval_bs", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--num_workers", type=int, default=4)

    ap.add_argument("--eval_steps", type=int, default=1000)
    ap.add_argument("--save_steps", type=int, default=1000)
    ap.add_argument("--logging_steps", type=int, default=50)
    ap.add_argument("--save_total_limit", type=int, default=3)

    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--gradient_checkpointing", action="store_true")

    ap.add_argument("--loss_name", type=str, default="improved", choices=["improved", "bce_dice"])

    ap.add_argument("--focal_alpha", type=float, default=0.75)
    ap.add_argument("--focal_gamma", type=float, default=2.0)
    ap.add_argument("--tversky_alpha", type=float, default=0.3)
    ap.add_argument("--tversky_beta", type=float, default=0.7)

    ap.add_argument("--w_focal", type=float, default=0.8)
    ap.add_argument("--w_tversky", type=float, default=1.0)
    ap.add_argument("--w_boundary", type=float, default=0.1)

    ap.add_argument("--boundary_scale", type=float, default=2.0)
    ap.add_argument("--boundary_kernel_size", type=int, default=3)
    ap.add_argument("--min_boundary_pixels", type=int, default=10)

    ap.add_argument("--dice_weight", type=float, default=1.0)
    ap.add_argument("--pos_weight", type=float, default=None)

    ap.add_argument("--max_train_images", type=int, default=None)
    ap.add_argument("--max_val_images", type=int, default=None)

    return ap


def main() -> None:
    ap = _build_arg_parser()
    ns = ap.parse_args()
    cfg = LoRAConfig(**vars(ns))
    run_lora(cfg)


if __name__ == "__main__":
    main()


__all__ = [
    "LoRAConfig",
    "run_lora",
    "CloudSENPromptDataset",
    "SegTrainer",
]
