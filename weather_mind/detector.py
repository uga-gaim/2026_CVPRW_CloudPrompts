from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from pathlib import Path
import inspect

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor


################################################################################
# Data Structures
################################################################################


@dataclass
class DetectionResult:
    """
    Canonical representation of a single detection.

    All coordinates are in absolute pixel space using the (x_min, y_min, x_max, y_max)
    convention ("xyxy") as returned by most HF ZSOD post-processing utilities.
    """

    tag: str
    score: float
    box_xyxy: Tuple[float, float, float, float]
    model_name: str
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def box_xywh(self) -> Tuple[float, float, float, float]:
        """Return the box as (x, y, width, height) in pixels."""
        x0, y0, x1, y1 = self.box_xyxy
        return (x0, y0, x1 - x0, y1 - y0)


@dataclass
class CloudBaseGeometry:
    """
    Helper for mapping a cloud bounding box to an apparent elevation angle.
    """

    bbox_bottom_y_px: float
    elevation_angle_deg: float


################################################################################
# Default tag structure for weather detection
################################################################################


DEFAULT_TAG_GROUPS: Dict[str, List[str]] = {
    "visibility": [
        "clear",
        "fog",
        "haze",
        "smoke",
        "wildfire smoke",
        "fire",
        "wildfire sequence",
    ],
    "clouds": [
        "cloud",
        "clouds",
        "low clouds",
        "high clouds",
        "thick clouds",
        "thin clouds",
        "cloud base",
        "cloud deck",
    ],
    "precipitation": [
        "rain",
        "drizzle",
        "snow",
        "sleet",
        "mixed precipitation",
        "hail",
        "rain shaft",
        "heavy rain",
        "light rain",
    ],
    "artifacts": [
        "raindrops on lens",
        "water drops on lens",
        "snow on lens",
    ],
    "surface": [
        "wet surface",
        "wet road",
        "snow covered road",
        "snow covered surface",
        "icy road",
    ],
}


def flatten_default_tags() -> List[str]:
    """Return a deduplicated, flattened list of all tags from DEFAULT_TAG_GROUPS."""
    seen = set()
    flat: List[str] = []
    for group_tags in DEFAULT_TAG_GROUPS.values():
        for t in group_tags:
            if t not in seen:
                seen.add(t)
                flat.append(t)
    return flat


################################################################################
# Model abstraction
################################################################################


class BaseZSODModel:
    """Abstract base class for zero-shot object detectors."""

    def __call__(
        self,
        image: Image.Image,
        tags: Sequence[str],
        score_threshold: float = 0.1,
    ) -> List[DetectionResult]:
        return self.detect(image, tags, score_threshold=score_threshold)

    def detect(
        self,
        image: Image.Image,
        tags: Sequence[str],
        score_threshold: float = 0.1,
    ) -> List[DetectionResult]:
        raise NotImplementedError("Subclasses must implement detect()")


class HuggingFaceZSODModel(BaseZSODModel):
    """
    Thin wrapper around `AutoModelForZeroShotObjectDetection` + `AutoProcessor`.
    """

    def __init__(
        self,
        model_id: str,
        name: str,
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
    ) -> None:
        self.model_id = model_id
        self.name = name.upper()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self._processor = None
        self._model = None
        self._torch_dtype = torch_dtype

    @property
    def processor(self):
        if self._processor is None:
            self._processor = AutoProcessor.from_pretrained(self.model_id)
        return self._processor

    @property
    def model(self):
        if self._model is None:
            kwargs = {}
            if self._torch_dtype is not None:
                kwargs["torch_dtype"] = self._torch_dtype
            self._model = AutoModelForZeroShotObjectDetection.from_pretrained(
                self.model_id, **kwargs
            ).to(self.device)
        return self._model

    def detect(
        self,
        image: Image.Image,
        tags: Sequence[str],
        score_threshold: float = 0.1,
    ) -> List[DetectionResult]:
        """
        Run zero-shot detection on `image` given `tags`.
        """
        if len(tags) == 0:
            return []

        norm_tags: List[str] = []
        for t in tags:
            txt = t.strip().lower()
            if not txt.endswith("."):
                txt += "."
            norm_tags.append(txt)

        if self.name in {"GROUNDING_DINO", "MM_GROUNDING_DINO_TINY", "LLMDET_LARGE"}:
            text_input: Any = " ".join(norm_tags)
        else:
            text_input = norm_tags

        inputs = self.processor(
            images=image,
            text=text_input,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        h, w = image.height, image.width
        processor = self.processor
        processed = None

        if hasattr(processor, "post_process_object_detection"):
            target_sizes = torch.tensor([[h, w]], device=self.device)
            processed = processor.post_process_object_detection(
                outputs=outputs,
                threshold=score_threshold,
                target_sizes=target_sizes,
            )[0]

        elif hasattr(processor, "post_process_grounded_object_detection"):
            post_proc = processor.post_process_grounded_object_detection
            sig = inspect.signature(post_proc)
            param_names = list(sig.parameters.keys())

            call_kwargs = {
                "outputs": outputs,
                "target_sizes": [(h, w)],
            }
            if "input_ids" in param_names:
                call_kwargs["input_ids"] = inputs["input_ids"]

            if "box_threshold" in param_names:
                call_kwargs["box_threshold"] = score_threshold
            if "threshold" in param_names:
                call_kwargs["threshold"] = score_threshold
            if (
                "text_threshold" in param_names
                and "box_threshold" not in param_names
                and "threshold" not in param_names
            ):
                call_kwargs["text_threshold"] = score_threshold

            processed_raw = post_proc(**call_kwargs)
            if isinstance(processed_raw, list):
                processed = processed_raw[0]
            else:
                processed = processed_raw

        elif hasattr(processor, "image_processor") and hasattr(
            processor.image_processor, "post_process_object_detection"
        ):
            target_sizes = torch.tensor([[h, w]], device=self.device)
            processed = processor.image_processor.post_process_object_detection(
                outputs=outputs,
                threshold=score_threshold,
                target_sizes=target_sizes,
            )[0]

        if processed is None:
            raise AttributeError(
                "This processor does not expose a known object-detection "
                "post-processing method (tried post_process_object_detection, "
                "post_process_grounded_object_detection, and "
                "image_processor.post_process_object_detection)."
            )

        boxes = processed["boxes"]
        scores = processed["scores"]
        raw_labels = processed["labels"]

        if isinstance(boxes, torch.Tensor):
            boxes = boxes.detach().cpu().tolist()
        if isinstance(scores, torch.Tensor):
            scores = scores.detach().cpu().tolist()

        norm_labels: List[Any] = []
        for lab in raw_labels:
            if isinstance(lab, torch.Tensor):
                norm_labels.append(int(lab.item()))
            elif isinstance(lab, (int, np.integer)):
                norm_labels.append(int(lab))
            else:
                norm_labels.append(lab)

        detections: List[DetectionResult] = []
        for box, score, lab in zip(boxes, scores, norm_labels):
            score = float(score)
            if score < score_threshold:
                continue

            if isinstance(lab, int):
                if 0 <= lab < len(tags):
                    tag = tags[lab]
                else:
                    tag = f"label_{lab}"
                extra = {"label_idx": int(lab)}
            else:
                tag = str(lab)
                extra = {"raw_label": lab}

            detections.append(
                DetectionResult(
                    tag=tag,
                    score=score,
                    box_xyxy=tuple(float(x) for x in box),
                    model_name=self.name,
                    extra=extra,
                )
            )

        return detections



################################################################################
# Registry of ZSOD models to try
################################################################################


ZSOD_MODEL_CONFIGS: Dict[str, str] = {
    "GROUNDING_DINO": "IDEA-Research/grounding-dino-base",
    "OWL_VIT": "google/owlvit-base-patch32",
    "MM_GROUNDING_DINO_TINY": "rziga/mm_grounding_dino_tiny_o365v1_goldg_grit_v3det",
    "LLMDET_LARGE": "iSEE-Laboratory/llmdet_large",
}

SUPPORTED_ZSOD_MODEL_NAMES: List[str] = sorted(ZSOD_MODEL_CONFIGS.keys())
DEFAULT_ZSOD_MODEL_NAME: str = "GROUNDING_DINO"


def _make_hf_model(name: str) -> HuggingFaceZSODModel:
    name = name.upper()
    if name not in ZSOD_MODEL_CONFIGS:
        raise KeyError(
            f"Unknown ZSOD model name '{name}'. "
            f"Known models: {sorted(ZSOD_MODEL_CONFIGS.keys())}"
        )
    model_id = ZSOD_MODEL_CONFIGS[name]
    return HuggingFaceZSODModel(model_id=model_id, name=name)


ZSOD_MODEL_REGISTRY: Dict[str, Callable[[], BaseZSODModel]] = {
    name: (lambda n=name: _make_hf_model(n)) for name in ZSOD_MODEL_CONFIGS.keys()
}


################################################################################
# High-level manager
################################################################################


def _load_image(input_data: Any) -> Image.Image:
    """
    Normalize input into an RGB PIL.Image.

    Accepts:
      - PIL.Image.Image directly
      - str or Path -> treated as a filesystem path
    """
    if isinstance(input_data, Image.Image):
        return input_data.convert("RGB")

    if isinstance(input_data, (str, Path)):
        path = Path(input_data)
        if not path.exists():
            raise FileNotFoundError(f"Image path does not exist: {path}")
        return Image.open(path).convert("RGB")

    raise TypeError(
        f"Unsupported input type {type(input_data)!r}; expected PIL.Image or path string."
    )


class ZSODDetector:
    """
    High-level interface for running zero-shot object detection.
    """

    def __init__(
        self,
        default_model_name: str = DEFAULT_ZSOD_MODEL_NAME,
        device: Optional[str] = None,
    ) -> None:
        self.default_model_name = default_model_name.upper()
        self.device = device
        self.current_model_name: Optional[str] = None
        self.current_model: Optional[BaseZSODModel] = None

    def load_model(self, model_name: Optional[str] = None) -> None:
        if model_name is None:
            model_name = self.default_model_name

        model_name = model_name.upper()
        if model_name not in ZSOD_MODEL_REGISTRY:
            raise KeyError(
                f"Unknown ZSOD model '{model_name}'. "
                f"Available: {SUPPORTED_ZSOD_MODEL_NAMES}"
            )

        factory = ZSOD_MODEL_REGISTRY[model_name]
        model = factory()

        if isinstance(model, HuggingFaceZSODModel) and self.device is not None:
            model.device = self.device

        self.current_model_name = model_name
        self.current_model = model

    def detect(
        self,
        input_data: Any,
        tags: Sequence[str],
        score_threshold: float = 0.1,
    ) -> List[DetectionResult]:
        if self.current_model is None:
            self.load_model(self.default_model_name)

        image = _load_image(input_data)
        assert self.current_model is not None

        return self.current_model.detect(image, tags, score_threshold=score_threshold)

    def compare_models(
        self,
        input_data: Any,
        tags: Sequence[str],
        model_names: Optional[Sequence[str]] = None,
        score_threshold: float = 0.1,
    ) -> Dict[str, List[DetectionResult]]:
        image = _load_image(input_data)

        if model_names is None:
            model_names = SUPPORTED_ZSOD_MODEL_NAMES

        results: Dict[str, List[DetectionResult]] = {}
        for name in model_names:
            name = name.upper()
            if name not in ZSOD_MODEL_REGISTRY:
                continue

            model = ZSOD_MODEL_REGISTRY[name]()
            if isinstance(model, HuggingFaceZSODModel) and self.device is not None:
                model.device = self.device

            detections = model.detect(image, tags, score_threshold=score_threshold)
            results[name] = detections

        return results

    def detect_default_weather_tags(
        self,
        input_data: Any,
        score_threshold: float = 0.1,
        tag_groups: Optional[Dict[str, List[str]]] = None,
    ) -> Dict[str, List[DetectionResult]]:
        if tag_groups is None:
            tag_groups = DEFAULT_TAG_GROUPS

        image = _load_image(input_data)

        if self.current_model is None:
            self.load_model(self.default_model_name)

        assert self.current_model is not None

        output: Dict[str, List[DetectionResult]] = {}
        for group_name, tags in tag_groups.items():
            dets = self.current_model.detect(
                image, tags, score_threshold=score_threshold
            )
            output[group_name] = dets

        return output

    @staticmethod
    def estimate_cloud_base_geometry(
        bbox: Tuple[float, float, float, float],
        image_height_px: int,
        camera_vertical_fov_deg: float,
        camera_tilt_deg: float = 0.0,
    ) -> CloudBaseGeometry:
        """
        Approximate the elevation angle of the ray through the bottom of a cloud bbox.
        """
        _x_min, _y_min, _x_max, y_max = bbox
        bbox_bottom_y_px = y_max

        y_norm = bbox_bottom_y_px / float(image_height_px)
        angle_from_center_deg = (0.5 - y_norm) * camera_vertical_fov_deg
        elevation_angle_deg = camera_tilt_deg + angle_from_center_deg

        return CloudBaseGeometry(
            bbox_bottom_y_px=float(bbox_bottom_y_px),
            elevation_angle_deg=float(elevation_angle_deg),
        )

    def attach_lora_adapter(self, *args, **kwargs) -> None:
        """Placeholder stub for future LoRA integration."""
        return
