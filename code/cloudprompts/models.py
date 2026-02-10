from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Type

import numpy as np
from PIL import Image
import torch

from torchvision.transforms.functional import resize as tv_resize
from torchvision.transforms import InterpolationMode


def _require(pkg: str, hint: str) -> None:
    try:
        __import__(pkg)
    except Exception as e:
        raise RuntimeError(
            f"Missing dependency '{pkg}'. {hint}\nOriginal error: {repr(e)}"
        ) from e


def resize_binary_mask(mask_hw: np.ndarray, size: int) -> torch.Tensor:
    """
    Resize binary mask [H,W] -> [size,size] using torchvision.
    Keeps labels discrete and returns float tensor in {0,1}.
    """
    mask_tensor = torch.from_numpy(mask_hw.astype(np.float32)).unsqueeze(0)

    interp = getattr(InterpolationMode, "NEAREST_EXACT", InterpolationMode.NEAREST)

    resized = tv_resize(
        mask_tensor,
        [size, size],
        interpolation=interp,
        antialias=False,
    )
    return (resized.squeeze(0) > 0.5).to(torch.float32)


_require("transformers", "Install with: pip install transformers")
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation


@dataclass(frozen=True)
class ModelSpec:
    """Metadata and defaults for a trainable model family."""

    key: str
    hf_default_id: str
    default_image_size: int
    default_target_modules: Tuple[str, ...]
    supports_lora: bool = True


class BaseModelAdapter:
    """Shared contract used by finetuning and (later) inference."""

    spec: ModelSpec

    def __init__(self, model_id: Optional[str] = None, image_size: Optional[int] = None):
        self.model_id = model_id or self.spec.hf_default_id
        self.image_size = int(image_size) if image_size is not None else int(self.spec.default_image_size)

    def build_processor(self):
        raise NotImplementedError

    def build_model(self):
        raise NotImplementedError

    def prepare_image(self, img_chw: np.ndarray) -> Image.Image:
        raise NotImplementedError

    def prepare_binary_mask(self, mask_hw: np.ndarray, class_id: int) -> torch.Tensor:
        raise NotImplementedError

    def encode(self, processor, prompt: str, image_pil: Image.Image) -> Dict[str, torch.Tensor]:
        raise NotImplementedError


@dataclass(frozen=True)
class ClipSegSpec(ModelSpec):
    pass


class ClipSegAdapter(BaseModelAdapter):
    """
    Adapter for CLIPSeg preprocessing + model/proc loading.

    Notes on spatial sizing:
    - CloudSEN+ masks can originate at 509x509.
    - For CLIPSeg training, binary masks are resized to `image_size` (default 352).
    - Image tensor resizing/normalization is handled by CLIPSegProcessor during encode().
    """

    spec = ClipSegSpec(
        key="clipseg",
        hf_default_id="CIDAS/clipseg-rd64-refined",
        default_image_size=352,
        default_target_modules=("q_proj", "k_proj", "v_proj", "out_proj"),
        supports_lora=True,
    )

    def build_processor(self) -> CLIPSegProcessor:
        return CLIPSegProcessor.from_pretrained(self.model_id)

    def build_model(self) -> CLIPSegForImageSegmentation:
        return CLIPSegForImageSegmentation.from_pretrained(self.model_id)

    def prepare_image(self, img_chw: np.ndarray) -> Image.Image:
        if img_chw.ndim != 3 or img_chw.shape[0] not in (1, 3, 4):
            raise ValueError(f"Expected CHW with C in (1,3,4). Got {img_chw.shape}")

        img_hwc = np.transpose(img_chw.astype(np.float32), (1, 2, 0))
        img_u8 = np.clip(img_hwc * 255.0, 0, 255).astype(np.uint8)

        if img_u8.shape[2] == 1:
            return Image.fromarray(img_u8[:, :, 0], mode="L")
        if img_u8.shape[2] == 4:
            return Image.fromarray(img_u8, mode="RGBA")
        return Image.fromarray(img_u8, mode="RGB")

    def prepare_binary_mask(self, mask_hw: np.ndarray, class_id: int) -> torch.Tensor:
        if mask_hw.ndim != 2:
            raise ValueError(f"Expected HW mask, got {mask_hw.shape}")

        bin_hw = (mask_hw.astype(np.int64) == int(class_id)).astype(np.uint8)
        return resize_binary_mask(bin_hw, self.image_size)


    def encode(self, processor: CLIPSegProcessor, prompt: str, image_pil: Image.Image) -> Dict[str, torch.Tensor]:
        enc = processor(
            text=prompt,
            images=image_pil,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        )
        return {
            "pixel_values": enc["pixel_values"].squeeze(0),
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }


_MODEL_REGISTRY: Dict[str, Type[BaseModelAdapter]] = {
    "clipseg": ClipSegAdapter,
}


def normalize_model_name(model_name: str) -> str:
    key = model_name.strip().lower()
    aliases = {
        "clipseg-rd64": "clipseg",
        "cidas/clipseg-rd64-refined": "clipseg",
    }
    return aliases.get(key, key)


def supported_models() -> Tuple[str, ...]:
    return tuple(sorted(_MODEL_REGISTRY.keys()))


def get_model_adapter(
    model_name: str,
    *,
    model_id: Optional[str] = None,
    image_size: Optional[int] = None,
) -> BaseModelAdapter:
    key = normalize_model_name(model_name)
    if key not in _MODEL_REGISTRY:
        raise NotImplementedError(
            f"Unsupported model '{model_name}'. Supported models: {', '.join(supported_models())}"
        )
    return _MODEL_REGISTRY[key](model_id=model_id, image_size=image_size)


def get_model_spec(model_name: str) -> ModelSpec:
    return get_model_adapter(model_name).spec


__all__ = [
    "ModelSpec",
    "BaseModelAdapter",
    "ClipSegAdapter",
    "get_model_adapter",
    "get_model_spec",
    "supported_models",
]
