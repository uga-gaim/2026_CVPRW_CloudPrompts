from pathlib import Path
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import tifffile as tiff
import torchvision.transforms.functional as TF

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)

def _read_mask_tif(path: Path) -> np.ndarray:
    m = tiff.imread(str(path))
    if m.ndim == 3:
        m = m[0]
    return m.astype(np.int64)

class SparcsDataset(Dataset):
    """
    Expects:
      <root>/<scene_id>_photo.png
      <root>/<scene_id>_labels.tif
      <root>/splits/{train,val,test}.txt  (one scene_id per line)
    """
    def __init__(
        self,
        root: str,
        split: str,
        crop_size: int = 480,
        augment: bool = True,
        ignore_index: int = 255,
    ):
        self.root = Path(root)
        self.split = split
        self.crop_size = crop_size
        self.augment = augment and (split == "train")
        self.ignore_index = ignore_index

        ids_path = self.root / "splits" / f"{split}.txt"
        if not ids_path.exists():
            raise FileNotFoundError(f"Missing split file: {ids_path}")
        self.ids = [x.strip() for x in ids_path.read_text().splitlines() if x.strip()]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        sid = self.ids[idx]
        img_path = self.root / f"{sid}_photo.png"
        lab_path = self.root / f"{sid}_labels.tif"

        img = Image.open(img_path).convert("RGB")
        mask = _read_mask_tif(lab_path)
        mask = np.where((mask >= 0) & (mask <= 6), mask, self.ignore_index).astype(np.int64)
        mask = Image.fromarray(mask.astype(np.uint8), mode="L")

        if self.augment:
            top = random.randint(0, max(0, img.height - self.crop_size))
            left = random.randint(0, max(0, img.width - self.crop_size))
            img = TF.crop(img, top, left, self.crop_size, self.crop_size)
            mask = TF.crop(mask, top, left, self.crop_size, self.crop_size)

            if random.random() < 0.5:
                img = TF.hflip(img)
                mask = TF.hflip(mask)
        else:
            img = TF.center_crop(img, [self.crop_size, self.crop_size])
            mask = TF.center_crop(mask, [self.crop_size, self.crop_size])

        img_t = TF.to_tensor(img)
        img_t = TF.normalize(img_t, CLIP_MEAN, CLIP_STD)

        mask_t = torch.from_numpy(np.array(mask, dtype=np.int64))

        return img_t, mask_t
