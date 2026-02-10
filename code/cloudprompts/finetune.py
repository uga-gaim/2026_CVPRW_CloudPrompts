from __future__ import annotations

import argparse
from dataclasses import asdict
from typing import Any, Dict, Optional

try:
    from .lora import LoRAConfig, run_lora
except Exception:
    from lora import LoRAConfig, run_lora


SUPPORTED_TECHNIQUES = ("lora",)


def run_finetune(config: Optional[LoRAConfig] = None, **kwargs: Any) -> None:
    """
    Public finetuning API.

    Current support:
      - technique: lora
      - model: clipseg (from models registry)
      - dataset: cloudsen12plus

    Usage:
      run_finetune(config=LoRAConfig(...))
      run_finetune(technique="lora", model_name="clipseg", ...)
    """
    if config is None:
        if "technique" not in kwargs:
            kwargs["technique"] = "lora"
        config = LoRAConfig(**kwargs)

    technique = config.technique.strip().lower()
    if technique not in SUPPORTED_TECHNIQUES:
        raise NotImplementedError(
            f"Unsupported technique '{config.technique}'. Supported techniques: {', '.join(SUPPORTED_TECHNIQUES)}"
        )

    if technique == "lora":
        run_lora(config)
        return

    raise NotImplementedError(f"Technique '{technique}' is not implemented")


def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Public finetuning entrypoint (router). Currently supports technique=lora."
    )

    ap.add_argument("--technique", type=str, default="lora", choices=list(SUPPORTED_TECHNIQUES))
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

    ap.add_argument("--eval_steps", type=int, default=50)
    ap.add_argument("--save_steps", type=int, default=50)
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
    run_finetune(config=cfg)


if __name__ == "__main__":
    main()


__all__ = ["run_finetune", "LoRAConfig", "SUPPORTED_TECHNIQUES"]