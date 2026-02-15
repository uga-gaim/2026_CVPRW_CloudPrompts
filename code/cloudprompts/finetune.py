from __future__ import annotations

import argparse
from dataclasses import fields
from typing import Any, Dict, Optional, Type, TypeVar, Union

try:
    from .lora import LoRAConfig, run_lora
except Exception:
    from lora import LoRAConfig, run_lora

try:
    from .fullfinetune import FullFineTuneConfig, run_full_finetune
except Exception:
    from fullfinetune import FullFineTuneConfig, run_full_finetune


SUPPORTED_TECHNIQUES = (
    "lora",
    "fullfinetune",
    "full",
    "full_finetune",
    "full-ft",
)


def _normalize_technique(value: str) -> str:
    v = value.strip().lower()
    compact = v.replace("_", "").replace("-", "")
    if compact in {"lora"}:
        return "lora"
    if compact in {"fullfinetune", "full", "fullft"}:
        return "fullfinetune"
    raise NotImplementedError(
        f"Unsupported technique '{value}'. Supported techniques: {', '.join(SUPPORTED_TECHNIQUES)}"
    )


T = TypeVar("T")


def _dataclass_from_kwargs(cls: Type[T], kwargs: Dict[str, Any]) -> T:
    allowed = {f.name for f in fields(cls)}
    filtered = {k: v for k, v in kwargs.items() if k in allowed}
    return cls(**filtered)  # type: ignore[arg-type]


def run_finetune(
    config: Optional[Union[LoRAConfig, FullFineTuneConfig]] = None,
    **kwargs: Any,
) -> None:
    """
    Public finetuning API router.

    Supported techniques:
      - lora
      - fullfinetune (aliases: full, full_finetune, full-ft)

    Usage:
      run_finetune(config=LoRAConfig(...))
      run_finetune(config=FullFineTuneConfig(...))
      run_finetune(technique="lora", model_name="clipseg", ...)
      run_finetune(technique="fullfinetune", model_name="clipseg", ...)
    """
    if config is None:
        technique = _normalize_technique(str(kwargs.get("technique", "lora")))
        kwargs["technique"] = technique
        if technique == "lora":
            config = _dataclass_from_kwargs(LoRAConfig, kwargs)
        else:
            config = _dataclass_from_kwargs(FullFineTuneConfig, kwargs)

    technique = _normalize_technique(config.technique)

    if technique == "lora":
        if not isinstance(config, LoRAConfig):
            cfg_dict = vars(config).copy()
            cfg_dict["technique"] = "lora"
            config = _dataclass_from_kwargs(LoRAConfig, cfg_dict)
        run_lora(config)
        return

    if technique == "fullfinetune":
        if not isinstance(config, FullFineTuneConfig):
            cfg_dict = vars(config).copy()
            cfg_dict["technique"] = "fullfinetune"
            config = _dataclass_from_kwargs(FullFineTuneConfig, cfg_dict)
        run_full_finetune(config)
        return

    raise NotImplementedError(f"Technique '{config.technique}' is not implemented")


def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Public finetuning entrypoint (router). Supports LoRA and full fine-tuning."
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
    ap.add_argument("--eval_bs", type=int, default=128)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--num_workers", type=int, default=4)

    ap.add_argument("--eval_strategy", type=str, default="epoch")
    ap.add_argument("--save_strategy", type=str, default="epoch")
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

    ap.add_argument("--train_data_pct", type=float, default=None)
    ap.add_argument("--val_data_pct", type=float, default=None)
    ap.add_argument("--subset_seed", type=int, default=None)


    ap.add_argument("--max_train_images", type=int, default=None)
    ap.add_argument("--max_val_images", type=int, default=None)

    return ap


def main() -> None:
    ap = _build_arg_parser()
    ns = ap.parse_args()
    kwargs = vars(ns)
    technique = _normalize_technique(kwargs.get("technique", "lora"))
    kwargs["technique"] = technique

    if technique == "lora":
        cfg = _dataclass_from_kwargs(LoRAConfig, kwargs)
    else:
        cfg = _dataclass_from_kwargs(FullFineTuneConfig, kwargs)

    run_finetune(config=cfg)


if __name__ == "__main__":
    main()


__all__ = [
    "run_finetune",
    "LoRAConfig",
    "FullFineTuneConfig",
    "SUPPORTED_TECHNIQUES",
]