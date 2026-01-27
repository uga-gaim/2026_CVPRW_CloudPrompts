#!/usr/bin/env bash
set -euo pipefail

# -----------------------------
# CloudSEN12+ downloader
# -----------------------------
# Usage:
#   bash ./code/data/cloudsen12plus/download_cloudsen12plus_p509.sh
#
# Optional env vars:
#   OUT_DIR=/workspace/CloudSEN12Plus_p509        # where the dataset files will be downloaded
#   MODE=all|high                                 # "all" downloads high+scribble+nolabel chunks; "high" downloads only high labels
#   REPO=isp-uv-es/CloudSEN12Plus                 # huggingface dataset repo
#
# Notes:
# - Uses huggingface-cli download (from huggingface_hub) + hf_transfer for faster pulls.

REPO="${REPO:-isp-uv-es/CloudSEN12Plus}"
MODE="${MODE:-high}"
OUT_DIR="./code/data/cloudsen12plus/cloudsen12plus_p509"


echo "==> Repo:    ${REPO}"
echo "==> Mode:    ${MODE}"
echo "==> OUT_DIR: ${OUT_DIR}"
echo

mkdir -p "${OUT_DIR}"

if command -v python3 >/dev/null 2>&1; then
  PY=python3
elif command -v python >/dev/null 2>&1; then
  PY=python
else
  echo "ERROR: python not found. Install Python 3.10+ and retry."
  exit 1
fi

echo "==> Using Python executable: $(command -v "$PY")"
echo "==> Python version: $("$PY" --version 2>&1)"

echo "==> Installing/Updating downloader tools (huggingface_hub + hf_transfer)..."
"${PY}" -m pip install -U pip >/dev/null
"${PY}" -m pip install -U huggingface_hub hf_transfer

export HF_HUB_ENABLE_HF_TRANSFER=1

FILES=()
if [[ "${MODE}" == "high" ]]; then
  FILES+=(
    "train/train_509_high.mlstac"
    "validation/validation_509_high.mlstac"
    "test/test_509_high.mlstac"
  )
elif [[ "${MODE}" == "all" ]]; then
  FILES+=(
    "train/train_509_high.mlstac"
    "train/train_509_scribble.mlstac"
    "train/train_509_nolabel_chunk1.mlstac"
    "train/train_509_nolabel_chunk2.mlstac"
    "validation/validation_509_high.mlstac"
    "validation/validation_509_scribble.mlstac"
    "test/test_509_high.mlstac"
    "test/test_509_scribble.mlstac"
  )
else
  echo "ERROR: MODE must be 'all' or 'high' (got '${MODE}')."
  exit 1
fi

if command -v hf >/dev/null 2>&1; then
  HF=hf
elif command -v huggingface-cli >/dev/null 2>&1; then
  HF=huggingface-cli
else
  echo "ERROR: Hugging Face CLI not found (neither 'hf' nor 'huggingface-cli')."
  echo "Try: ${PY} -m pip install -U huggingface_hub hf_transfer"
  exit 1
fi

echo "==> Using HF CLI: $(command -v "$HF")"
"$HF" --help >/dev/null 2>&1 || true

echo
echo "==> Downloading ${#FILES[@]} file(s)..."
for f in "${FILES[@]}"; do
  echo "----> ${f}"
  "$HF" download "${REPO}" "${f}" \
    --repo-type dataset \
    --local-dir "${OUT_DIR}"
done

echo
echo "==> Download complete. Quick inventory:"
find "${OUT_DIR}" -maxdepth 2 -type f -name "*.mlstac" -print | sed 's|^| - |'

echo
echo "==> Disk usage:"
du -sh "${OUT_DIR}" || true

echo
echo "==> Done."
echo "   Files are under: ${OUT_DIR}"
