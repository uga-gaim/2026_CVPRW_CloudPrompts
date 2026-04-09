import argparse
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

DEFAULT_PROMPTS = ["clear", "thick cloud", "thin cloud", "cloud shadow"]
DEFAULT_LABELS = [0, 1, 2, 3]


def _load_export_npz_as_pil(npz_path: Path) -> tuple[Image.Image, int, int]:
    with np.load(npz_path) as z:
        if "image" not in z.files:
            raise KeyError(
                f"{npz_path} does not contain key 'image'. "
                f"Keys found: {z.files}."
            )
        arr = z["image"]

    if arr.ndim != 3 or arr.shape[0] != 3:
        raise ValueError(f"Expected image shape (3,H,W). Got {arr.shape} in {npz_path}")

    arr = np.clip(arr, 0.0, 1.0)
    hwc = (np.transpose(arr, (1, 2, 0)) * 255.0).round().astype(np.uint8)
    h, w = hwc.shape[:2]
    return Image.fromarray(hwc, mode="RGB"), h, w


@torch.no_grad()
def _predict_mask_clipseg(
    model,
    processor: CLIPSegProcessor,
    image_pil: Image.Image,
    prompts: list[str],
    label_ids: list[int],
    out_hw: tuple[int, int],
    device: str,
    use_amp: bool,
) -> np.ndarray:
    """Returns uint8 mask (H,W) using argmax over prompt logits."""
    if len(prompts) != len(label_ids):
        raise ValueError("prompts and label_ids must have the same length")

    inputs = processor(
        text=prompts,
        images=[image_pil] * len(prompts),
        padding=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    if use_amp and device.startswith("cuda"):
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            outputs = model(**inputs)
    else:
        outputs = model(**inputs)

    logits = outputs.logits
    h, w = out_hw

    logits = F.interpolate(
        logits.unsqueeze(1),
        size=(h, w),
        mode="bilinear",
        align_corners=False,
    ).squeeze(1)

    pred_idx = torch.argmax(logits, dim=0).to("cpu").numpy().astype(np.uint8)
    label_lut = np.array(label_ids, dtype=np.uint8)
    pred_mask = label_lut[pred_idx]
    return pred_mask


def pick_device(user_device: str | None = None) -> str:
    """Priority: user override > CUDA > MPS > CPU"""
    if user_device:
        return user_device
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _has_any_file(d: Path, names: tuple[str, ...]) -> bool:
    return any((d / name).exists() for name in names)


def _normalize_ckpt_type(value: str | None) -> str:
    if value is None:
        return "auto"
    v = value.strip().lower()
    aliases = {
        "fullfinetune": "full",
        "full_finetune": "full",
        "full-ft": "full",
        "lora": "adapter",
    }
    return aliases.get(v, v)


def detect_checkpoint_type(checkpoint_dir: str | Path) -> Literal["adapter", "full"]:
    """
    Detect whether checkpoint is LoRA adapter or full fine-tuned model.

    - Adapter: adapter_config.json exists.
    - Full: model weight files (or index files) exist.
    """
    ckpt = Path(checkpoint_dir)
    if not ckpt.exists() or not ckpt.is_dir():
        raise FileNotFoundError(f"Checkpoint dir not found: {ckpt}")

    if (ckpt / "adapter_config.json").exists():
        return "adapter"

    full_weight_names = (
        "model.safetensors",
        "pytorch_model.bin",
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
    )
    if _has_any_file(ckpt, full_weight_names):
        return "full"

    raise ValueError(
        f"Could not detect checkpoint type in: {ckpt}.\n"
        "Expected either adapter_config.json (LoRA adapter) OR one of "
        f"{full_weight_names} (full fine-tuned)."
    )


def _load_processor_with_fallback(primary: str | Path, fallback_model_id: str) -> CLIPSegProcessor:
    try:
        return CLIPSegProcessor.from_pretrained(str(primary))
    except Exception:
        return CLIPSegProcessor.from_pretrained(fallback_model_id)


def load_model_and_processor(
    model_id: str,
    device: str,
    checkpoint_type: str = "auto",
    checkpoint_dir: str | None = None,
    use_adapter: bool = False,
    adapter_dir: str | None = None,
):
    """
    Supports three loading modes:
      - base:    model_id only (zero-shot)
      - adapter: base model_id + LoRA adapter checkpoint
      - full:    full fine-tuned checkpoint directory

    Backward compatible with --use_adapter / --adapter_dir.
    """
    ckpt_type = _normalize_ckpt_type(checkpoint_type)
    ckpt_dir = checkpoint_dir or adapter_dir

    if use_adapter and ckpt_type == "base":
        ckpt_type = "adapter"
    if use_adapter and ckpt_type == "auto" and ckpt_dir is None:
        raise ValueError("--use_adapter was set but no checkpoint directory was provided.")

    if ckpt_type == "auto":
        if ckpt_dir:
            ckpt_type = detect_checkpoint_type(ckpt_dir)
        elif use_adapter:
            ckpt_type = "adapter"
        else:
            ckpt_type = "base"

    if ckpt_type not in {"base", "adapter", "full"}:
        raise ValueError(
            f"Invalid checkpoint_type='{checkpoint_type}'. "
            "Use one of: base, adapter, full, auto"
        )

    if ckpt_type in {"adapter", "full"} and not ckpt_dir:
        raise ValueError(
            f"checkpoint_type='{ckpt_type}' requires --checkpoint_dir "
            "(or legacy --adapter_dir)."
        )

    if ckpt_type == "base":
        processor = CLIPSegProcessor.from_pretrained(model_id)
        model = CLIPSegForImageSegmentation.from_pretrained(model_id)

    elif ckpt_type == "adapter":
        try:
            from peft import PeftModel
        except Exception as e:
            raise RuntimeError(
                "peft is required for adapter checkpoints. Install with: pip install peft\n"
                f"Original error: {repr(e)}"
            )

        processor = _load_processor_with_fallback(ckpt_dir, model_id)
        base = CLIPSegForImageSegmentation.from_pretrained(model_id)
        model = PeftModel.from_pretrained(base, ckpt_dir)

    else:
        processor = _load_processor_with_fallback(ckpt_dir, model_id)
        model = CLIPSegForImageSegmentation.from_pretrained(ckpt_dir)

    model = model.to(device)
    model.eval()
    return processor, model, ckpt_type


def run(
    data_root: str,
    out_root: str,
    model_id: str = "CIDAS/clipseg-rd64-refined",
    device: str | None = None,
    amp: bool = False,
    prompts: list[str] | None = None,
    label_ids: list[int] | None = None,
    skip_existing: bool = True,
    checkpoint_type: str = "auto",
    checkpoint_dir: str | None = None,
    use_adapter: bool = False,
    adapter_dir: str | None = None,
) -> None:
    device = pick_device(device)
    prompts = prompts or DEFAULT_PROMPTS
    label_ids = label_ids or DEFAULT_LABELS

    in_images = Path(data_root)
    if not in_images.exists():
        raise FileNotFoundError(f"Input images dir not found: {in_images}")

    out_masks = Path(out_root) / "masks"
    out_masks.mkdir(parents=True, exist_ok=True)

    processor, model, resolved_ckpt_type = load_model_and_processor(
        model_id=model_id,
        device=device,
        checkpoint_type=checkpoint_type,
        checkpoint_dir=checkpoint_dir,
        use_adapter=use_adapter,
        adapter_dir=adapter_dir,
    )

    img_paths = sorted(in_images.glob("*.npz"))
    if not img_paths:
        raise FileNotFoundError(f"No .npz files found in: {in_images}")

    tag_map = {"base": "zs", "adapter": "lora", "full": "fullft"}
    run_tag = tag_map.get(resolved_ckpt_type, resolved_ckpt_type)

    for img_npz in tqdm(img_paths, desc=f"clipseg {run_tag}: {in_images.name}"):
        stem = img_npz.stem
        out_npz = out_masks / f"{stem}.npz"

        if skip_existing and out_npz.exists():
            continue

        image_pil, h, w = _load_export_npz_as_pil(img_npz)
        pred = _predict_mask_clipseg(
            model=model,
            processor=processor,
            image_pil=image_pil,
            prompts=prompts,
            label_ids=label_ids,
            out_hw=(h, w),
            device=device,
            use_amp=amp,
        )

        np.savez_compressed(out_npz, mask=pred)


def _parse_args():
    ap = argparse.ArgumentParser(
        description=(
            "Run CLIPSeg on CloudSEN export npz and write predicted masks as npz. "
            "Supports base, adapter (LoRA), and full fine-tuned checkpoints."
        )
    )
    ap.add_argument("--data_root", required=True,
                    help="Directory containing *.npz images (e.g., .../test/images)")
    ap.add_argument("--out_root", required=True,
                    help="Output root; writes out_root/masks/*.npz")
    ap.add_argument("--model_id", default="CIDAS/clipseg-rd64-refined",
                    help="Base HF model id used for base/adapter modes.")
    ap.add_argument("--device", default=None, help="cuda/cpu/mps (default auto)")
    ap.add_argument("--amp", action="store_true", help="Use autocast fp16 on CUDA")

    ap.add_argument(
        "--checkpoint_type",
        type=str,
        default="auto",
        choices=["base", "adapter", "full", "auto", "lora", "fullfinetune", "full_finetune", "full-ft"],
        help="base | adapter | full | auto (auto inspects --checkpoint_dir).",
    )
    ap.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help=(
            "Path to checkpoint folder. "
            "For adapter: folder with adapter_config.json. "
            "For full: folder with model.safetensors/pytorch_model.bin."
        ),
    )

    ap.add_argument(
        "--use_adapter",
        action="store_true",
        help="[Legacy] If set, load LoRA adapter from --adapter_dir.",
    )
    ap.add_argument(
        "--adapter_dir",
        type=str,
        default=None,
        help="[Legacy] Path to adapter checkpoint dir.",
    )

    ap.add_argument("--prompts", nargs="+", default=None,
                    help="Override prompts, e.g. --prompts clear 'thick cloud' 'thin cloud' 'cloud shadow'")
    ap.add_argument("--label_ids", nargs="+", type=int, default=None,
                    help="Override label ids aligned to prompts, e.g. --label_ids 0 1 2 3")
    ap.add_argument("--no_skip_existing", action="store_true",
                    help="If set, do NOT skip existing outputs (recompute everything).")
    return ap.parse_args()


def main():
    args = _parse_args()
    run(
        data_root=args.data_root,
        out_root=args.out_root,
        model_id=args.model_id,
        device=args.device,
        amp=args.amp,
        prompts=args.prompts,
        label_ids=args.label_ids,
        skip_existing=(not args.no_skip_existing),
        checkpoint_type=args.checkpoint_type,
        checkpoint_dir=args.checkpoint_dir,
        use_adapter=args.use_adapter,
        adapter_dir=args.adapter_dir,
    )


if __name__ == "__main__":
    main()
