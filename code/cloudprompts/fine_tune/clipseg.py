import os
import math
import argparse
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

def _require(pkg: str, hint: str):
    try:
        __import__(pkg)
    except Exception as e:
        raise RuntimeError(
            f"Missing dependency '{pkg}'. {hint}\nOriginal error: {repr(e)}"
        )

_require("transformers", "Install with: pip install transformers")
_require("peft", "Install with: pip install peft")
from transformers import (
    CLIPSegProcessor,
    CLIPSegForImageSegmentation,
    TrainingArguments,
    Trainer,
    set_seed,
)
from peft import LoraConfig, get_peft_model


def chw_float01_to_pil(img_chw: np.ndarray) -> Image.Image:
    """Convert float32 CHW in [0,1] to uint8 HWC PIL."""
    if img_chw.ndim != 3 or img_chw.shape[0] not in (1, 3, 4):
        raise ValueError(f"Expected CHW with C in (1,3,4). Got {img_chw.shape}")
    img_hwc = np.transpose(img_chw, (1, 2, 0))
    img_u8 = np.clip(img_hwc * 255.0, 0, 255).astype(np.uint8)
    if img_u8.shape[2] == 1:
        img_u8 = img_u8[:, :, 0]
        return Image.fromarray(img_u8, mode="L")
    if img_u8.shape[2] == 4:
        return Image.fromarray(img_u8, mode="RGBA")
    return Image.fromarray(img_u8, mode="RGB")


def mask_to_pil(mask_hw: np.ndarray) -> Image.Image:
    """Convert HW uint8 mask to PIL (mode L)."""
    if mask_hw.ndim != 2:
        raise ValueError(f"Expected HW mask, got {mask_hw.shape}")
    return Image.fromarray(mask_hw.astype(np.uint8), mode="L")


def resize_binary_mask(mask_hw: np.ndarray, size: int) -> torch.Tensor:
    """
    mask_hw is {0,1} uint8 or bool.
    Returns float tensor [H,W] in {0,1}, resized with NEAREST.
    """
    m = (mask_hw.astype(np.uint8) * 255)
    pil = Image.fromarray(m, mode="L")
    pil = pil.resize((size, size), resample=Image.NEAREST)
    arr = (np.array(pil) > 127).astype(np.float32)
    return torch.from_numpy(arr)


def dice_loss_from_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    logits: [B,H,W] (raw)
    targets: [B,H,W] float in {0,1}
    """
    probs = torch.sigmoid(logits)
    num = 2.0 * (probs * targets).sum(dim=(1, 2)) + eps
    den = (probs + targets).sum(dim=(1, 2)) + eps
    return 1.0 - (num / den).mean()


def iou_from_logits(logits: torch.Tensor, targets: torch.Tensor, thresh: float = 0.5, eps: float = 1e-6) -> torch.Tensor:
    """
    logits: [B,H,W]
    targets: [B,H,W] float in {0,1}
    """
    probs = torch.sigmoid(logits)
    preds = (probs > thresh).float()
    inter = (preds * targets).sum(dim=(1, 2))
    union = (preds + targets - preds * targets).sum(dim=(1, 2))
    return ((inter + eps) / (union + eps)).mean()


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


def set_requires_grad_by_keyword(model: torch.nn.Module, keywords: List[str], requires_grad: bool = True) -> int:
    """
    Set requires_grad for parameters whose name contains any keyword.
    Returns how many params were affected.
    """
    affected = 0
    for name, p in model.named_parameters():
        if any(k in name for k in keywords):
            p.requires_grad = requires_grad
            affected += 1
    return affected


class CloudSENClipSegNPZ(Dataset):
    """
    Flattens multi-class masks into (image, prompt, binary_mask) samples.
    """

    def __init__(
        self,
        data_root: str,
        split: str,
        processor: "CLIPSegProcessor",
        prompts: List[str],
        class_ids: List[int],
        image_size: int = 352,
        prompt_template: str = "{label}",
        max_images: Optional[int] = None,
    ):
        self.data_root = data_root
        self.split = split
        self.processor = processor
        self.prompts = prompts
        self.class_ids = class_ids
        self.image_size = image_size
        self.prompt_template = prompt_template

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

        img_chw = img_npz["image"].astype(np.float32)
        mask_hw = msk_npz["mask"].astype(np.uint8)

        pil_img = chw_float01_to_pil(img_chw)

        bin_hw = (mask_hw == class_id).astype(np.uint8)
        y = resize_binary_mask(bin_hw, self.image_size)

        enc = self.processor(
            text=prompt,
            images=pil_img,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        )

        item = {
            "pixel_values": enc["pixel_values"].squeeze(0),
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": y,
        }
        return item


@dataclass
class DataCollator:
    """
    Simple collator that stacks tensors.
    """
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = {}
        for k in ["pixel_values", "input_ids", "attention_mask", "labels"]:
            batch[k] = torch.stack([f[k] for f in features], dim=0)
        return batch


class SegTrainer(Trainer):
    def __init__(self, dice_weight: float = 1.0, pos_weight: Optional[float] = None, **kwargs):
        super().__init__(**kwargs)
        self.dice_weight = float(dice_weight)
        self.pos_weight = pos_weight

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")

        outputs = model(**inputs)
        logits = outputs.logits
        if logits.ndim == 4 and logits.shape[1] == 1:
            logits = logits[:, 0, :, :]
        elif logits.ndim != 3:
            raise RuntimeError(f"Unexpected logits shape: {tuple(logits.shape)}")

        labels = labels.to(logits.dtype)

        if self.pos_weight is not None:
            pw = torch.tensor([self.pos_weight], device=logits.device, dtype=logits.dtype)
            bce = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pw)
        else:
            bce = F.binary_cross_entropy_with_logits(logits, labels)

        dloss = dice_loss_from_logits(logits, labels)

        loss = bce + self.dice_weight * dloss
        return (loss, outputs) if return_outputs else loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True, help="Exported NPZ dataset root")
    ap.add_argument("--output_dir", type=str, required=True, help="Where to save checkpoints/adapters")

    ap.add_argument("--model_name", type=str, default="CIDAS/clipseg-rd64-refined",
                    help="HF model name for CLIPSeg")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--labels", type=str, default="clear,thick cloud,thin cloud,cloud shadow",
                    help="Comma-separated label texts used as prompts")
    ap.add_argument("--class_ids", type=str, default="0,1,2,3",
                    help="Comma-separated integer class IDs corresponding to labels")

    ap.add_argument("--prompt_template", type=str, default="{label}",
                    help="Prompt template, e.g. 'a satellite image of {label}'")

    ap.add_argument("--image_size", type=int, default=352, help="Resize size used for labels (and expected logits size)")

    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--target_modules", type=str, default="q_proj,k_proj,v_proj,out_proj",
                    help="Comma-separated module name suffixes to apply LoRA to")
    ap.add_argument("--train_head_keywords", type=str, default="decoder,segmentation,classifier,output",
                    help="Unfreeze these (comma-separated) keywords in parameter names, in addition to LoRA adapters")

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

    ap.add_argument("--dice_weight", type=float, default=1.0, help="Loss = BCE + dice_weight * DiceLoss")
    ap.add_argument("--pos_weight", type=float, default=None,
                    help="Optional BCE pos_weight (>1 helps rare positives). Example: 3.0")

    ap.add_argument("--max_train_images", type=int, default=None)
    ap.add_argument("--max_val_images", type=int, default=None)

    args = ap.parse_args()

    set_seed(args.seed)

    labels = [s.strip() for s in args.labels.split(",") if s.strip()]
    class_ids = [int(s.strip()) for s in args.class_ids.split(",") if s.strip()]
    if len(labels) != len(class_ids):
        raise ValueError(f"--labels count ({len(labels)}) must match --class_ids count ({len(class_ids)})")

    print("Labels/prompts:", list(zip(class_ids, labels)))
    print("Prompt template:", args.prompt_template)

    processor = CLIPSegProcessor.from_pretrained(args.model_name)
    model = CLIPSegForImageSegmentation.from_pretrained(args.model_name)

    if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        print("Enabled gradient checkpointing.")

    target_modules = [s.strip() for s in args.target_modules.split(",") if s.strip()]
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)

    head_keywords = [s.strip() for s in args.train_head_keywords.split(",") if s.strip()]
    affected = set_requires_grad_by_keyword(model, head_keywords, requires_grad=True)
    print(f"Unfroze params matching keywords {head_keywords}: affected {affected} tensors")

    print_trainable_params(model)

    train_ds = CloudSENClipSegNPZ(
        data_root=args.data_root,
        split="train",
        processor=processor,
        prompts=labels,
        class_ids=class_ids,
        image_size=args.image_size,
        prompt_template=args.prompt_template,
        max_images=args.max_train_images,
    )
    val_ds = CloudSENClipSegNPZ(
        data_root=args.data_root,
        split="val",
        processor=processor,
        prompts=labels,
        class_ids=class_ids,
        image_size=args.image_size,
        prompt_template=args.prompt_template,
        max_images=args.max_val_images,
    )

    targs = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,

        per_device_train_batch_size=args.train_bs,
        per_device_eval_batch_size=args.eval_bs,
        gradient_accumulation_steps=args.grad_accum,

        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        save_total_limit=args.save_total_limit,

        dataloader_num_workers=args.num_workers,
        remove_unused_columns=False,

        label_names=["labels"],

        fp16=args.fp16,
        bf16=args.bf16,

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
        dice_weight=args.dice_weight,
        pos_weight=args.pos_weight,
    )

    print("Starting training...")
    trainer.train()

    print("Saving final adapter + trainer state...")
    trainer.save_model(args.output_dir)

    processor.save_pretrained(args.output_dir)

    print("Done. Saved to:", args.output_dir)


if __name__ == "__main__":
    main()
