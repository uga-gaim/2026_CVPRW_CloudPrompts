# Low-Data Supervised Adaptation Outperforms Prompting for Cloud Segmentation Under Domain Shift

#### [Harshith Kethavath](https://www.kethavath.com), [Weiming Hu](https://weiming.uga.edu)

**Lab for Geoinformatics and AI Modeling (GAIM), Department of Geography, The University of Georgia**

📄 [Paper](https://arxiv.org/abs/2604.08956) &nbsp;|&nbsp; 🤗 [Models](https://huggingface.co/collections/uga-gaim/2026-cvprw-cloudprompts)

---

## Overview

We evaluate [CLIPSeg](https://huggingface.co/CIDAS/clipseg-rd64-refined) on the [CloudSEN12+](https://huggingface.co/datasets/isp-uv-es/CloudSEN12Plus) dataset for four-class cloud segmentation (clear, thick cloud, thin cloud, cloud shadow). We systematically evaluate 60 prompt variants and compare against LoRA and full fine-tuning under varying amounts of labeled data.

**Key findings:**
- Prompt engineering fails comprehensively under domain shift; CLIPSeg's frozen encoder is largely insensitive to prompt variation for satellite imagery.
- Full fine-tuning (FFT) and Low Rank Adaptation (LoRA) with 0.1% labeled data (~8 images) outperforms all prompting approaches.
- 5–10% labeled data recovers ~85% of full fine-tuning performance.
- FFT consistently outperforms LoRA, with the gap most pronounced for spectrally ambiguous classes (thin cloud, cloud shadow).

---

## Installation

> **Note:** Install PyTorch for your CUDA version first from [pytorch.org](https://pytorch.org/get-started/locally/), then install this package.

```bash
pip install cloudprompts
```

Or from source:

```bash
git clone https://github.com/uga-gaim/2026_CVPRW_CloudPrompts.git
cd 2026_CVPRW_CloudPrompts
pip install -e .
```

---

## Dataset

Download and preprocess CloudSEN12+:

```bash
# Step 1: Download (defaults to ./cloudsen12plus_p509)
bash data/cloudsen12plus/download_cloudsen12plus_p509.sh

# Step 2: Export to NPZ format
python data/cloudsen12plus/export_p509_high_npz.py \
  --root ./cloudsen12plus_p509 \
  --out /path/to/data
```

The expected directory structure after preprocessing:

```
/path/to/data/
├── train/
│   ├── images/   # *.npz, key: 'image' (3, H, W) float32 in [0, 1]
│   └── masks/    # *.npz, key: 'mask'  (H, W) int64 class IDs
├── val/
│   ├── images/
│   └── masks/
└── test/
    ├── images/
    └── masks/
```

Class IDs: `0` = clear, `1` = thick cloud, `2` = thin cloud, `3` = cloud shadow.

---

## Usage

### Inference (Zero-Shot)

```bash
cloudprompts-infer \
  --data_root /path/to/data/test/images \
  --out_root /path/to/output \
  --checkpoint_type base
```

### Inference (Fine-Tuned Models)

Load a fine-tuned model from HuggingFace:

```bash
# Full fine-tuned
huggingface-cli download uga-gaim/clipseg-cloudsen12plus-fft --local-dir ./clipseg-fft

cloudprompts-infer \
  --data_root /path/to/data/test/images \
  --out_root /path/to/output \
  --checkpoint_type full \
  --checkpoint_dir ./clipseg-fft

# LoRA adapter
huggingface-cli download uga-gaim/clipseg-cloudsen12plus-lora --local-dir ./clipseg-lora

cloudprompts-infer \
  --data_root /path/to/data/test/images \
  --out_root /path/to/output \
  --checkpoint_type adapter \
  --checkpoint_dir ./clipseg-lora
```

---

## Pretrained Models

Fine-tuned models are available on HuggingFace under the [`uga-gaim`](https://huggingface.co/uga-gaim) organization.

| Model | Type | mIoU |
|---|---|---|
| [clipseg-cloudsen12plus-fft](https://huggingface.co/uga-gaim/clipseg-cloudsen12plus-fft) | Full fine-tune | 0.6572 |
| [clipseg-cloudsen12plus-lora](https://huggingface.co/uga-gaim/clipseg-cloudsen12plus-lora) | LoRA adapter | 0.5991 |


---

## Citation

If you use this code or models in your research, please cite:

```bibtex
@misc{kethavath2026lowdatasupervisedadaptationoutperforms,
      title={Low-Data Supervised Adaptation Outperforms Prompting for Cloud Segmentation Under Domain Shift}, 
      author={Harshith Kethavath and Weiming Hu},
      year={2026},
      eprint={2604.08956},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2604.08956}, 
}
```

---

## License

This code is licensed under the [Creative Commons Attribution 4.0 International License (CC BY 4.0)](LICENSE).
