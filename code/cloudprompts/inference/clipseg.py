import argparse
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

DEFAULT_PROMPTS = ["clear", "thick cloud", "thin cloud", "cloud shadow"]
DEFAULT_LABELS  = [0, 1, 2, 3]


def _load_export_npz_as_pil(npz_path: Path) -> tuple[Image.Image, int, int]:
    with np.load(npz_path) as z:
        if "image" not in z.files:
            raise KeyError(
                f"{npz_path} does not contain key 'image'. "
                f"Keys found: {z.files}. "
            )
        arr = z["image"]

    if arr.ndim != 3 or arr.shape[0] != 3:
        raise ValueError(f"Expected image shape (3,H,W). Got {arr.shape} in {npz_path}")

    arr = np.clip(arr, 0.0, 1.0)
    hwc = (np.transpose(arr, (1, 2, 0)) * 255.0).round().astype(np.uint8)
    H, W = hwc.shape[:2]
    return Image.fromarray(hwc, mode="RGB"), H, W


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
    """
    Returns uint8 mask (H,W) with values from label_ids based on argmax over prompt logits.
    """
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

    logits = outputs.logits  # [N, Hm, Wm]
    H, W = out_hw

    logits = F.interpolate(
        logits.unsqueeze(1),   # [N,1,Hm,Wm]
        size=(H, W),
        mode="bilinear",
        align_corners=False,
    ).squeeze(1)              # [N,H,W]

    pred_idx = torch.argmax(logits, dim=0).to("cpu").numpy().astype(np.uint8)
    label_lut = np.array(label_ids, dtype=np.uint8)
    pred_mask = label_lut[pred_idx]
    return pred_mask


def pick_device(user_device: str | None = None) -> str:
    """
    Priority: user override > CUDA > MPS > CPU
    """
    if user_device:
        return user_device

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model_and_processor(
    model_id: str,
    device: str,
    use_adapter: bool,
    adapter_dir: str | None,
):
    """
    If use_adapter=True, loads base CLIPSeg + LoRA adapter from adapter_dir (checkpoint folder).
    adapter_dir must contain adapter_model.safetensors + adapter_config.json.
    """
    processor = CLIPSegProcessor.from_pretrained(model_id)

    base = CLIPSegForImageSegmentation.from_pretrained(model_id)

    if use_adapter:
        if not adapter_dir:
            raise ValueError("--use_adapter was set but --adapter_dir was not provided.")
        try:
            from peft import PeftModel
        except Exception as e:
            raise RuntimeError(
                "peft is required for --use_adapter. Install with: pip install peft\n"
                f"Original error: {repr(e)}"
            )
        model = PeftModel.from_pretrained(base, adapter_dir)
    else:
        model = base

    model = model.to(device)
    model.eval()
    return processor, model


def run(
    data_root: str,
    out_root: str,
    model_id: str = "CIDAS/clipseg-rd64-refined",
    device: str | None = None,
    amp: bool = False,
    prompts: list[str] | None = None,
    label_ids: list[int] | None = None,
    skip_existing: bool = True,
    use_adapter: bool = False,
    adapter_dir: str | None = None,
) -> None:
    data_root = str(data_root)
    out_root = str(out_root)

    device = pick_device(device)
    prompts = prompts or DEFAULT_PROMPTS
    label_ids = label_ids or DEFAULT_LABELS

    in_images = Path(data_root)
    if not in_images.exists():
        raise FileNotFoundError(f"Input images dir not found: {in_images}")

    out_masks = Path(out_root) / "masks"
    out_masks.mkdir(parents=True, exist_ok=True)

    processor, model = load_model_and_processor(
        model_id=model_id,
        device=device,
        use_adapter=use_adapter,
        adapter_dir=adapter_dir,
    )

    img_paths = sorted(in_images.glob("*.npz"))
    if not img_paths:
        raise FileNotFoundError(f"No .npz files found in: {in_images}")

    tag = "lora" if use_adapter else "zs"
    for img_npz in tqdm(img_paths, desc=f"clipseg {tag}: {Path(data_root).name}"):
        stem = img_npz.stem
        out_npz = out_masks / f"{stem}.npz"

        if skip_existing and out_npz.exists():
            continue

        image_pil, H, W = _load_export_npz_as_pil(img_npz)
        pred = _predict_mask_clipseg(
            model=model,
            processor=processor,
            image_pil=image_pil,
            prompts=prompts,
            label_ids=label_ids,
            out_hw=(H, W),
            device=device,
            use_amp=amp,
        )

        np.savez_compressed(out_npz, mask=pred)


def _parse_args():
    ap = argparse.ArgumentParser(
        description="Run CLIPSeg on CloudSEN export npz and write predicted masks as npz. "
                    "Optionally load a LoRA adapter checkpoint."
    )
    ap.add_argument("--data_root", required=True,
                    help="Directory containing *.npz images (e.g., .../test/images)")
    ap.add_argument("--out_root", required=True,
                    help="Output root; writes out_root/masks/*.npz")
    ap.add_argument("--model_id", default="CIDAS/clipseg-rd64-refined")
    ap.add_argument("--device", default=None, help="cuda/cpu/mps (default auto)")
    ap.add_argument("--amp", action="store_true", help="Use autocast fp16 on CUDA")

    # ✅ new: adapter inference toggle + path
    ap.add_argument("--use_adapter", action="store_true",
                    help="If set, load a LoRA adapter from --adapter_dir (fine-tuned checkpoint).")
    ap.add_argument("--adapter_dir", type=str, default=None,
                    help="Path to checkpoint folder containing adapter_model.safetensors. "
                         "Example: .../checkpoint-1000")

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
        use_adapter=args.use_adapter,
        adapter_dir=args.adapter_dir,
    )


if __name__ == "__main__":
    main()