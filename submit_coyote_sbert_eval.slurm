#!/bin/bash
#SBATCH --job-name=coyote-sbert
#SBATCH --account=bbqj-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --mail-user=alucic2@illinois.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=/projects/bbqj/alucic2/llamafactory/logs/coyote-sbert-%j.out
#SBATCH --error=/projects/bbqj/alucic2/llamafactory/logs/coyote-sbert-%j.err

set -euo pipefail

module reset
module load python/miniforge3_pytorch/2.11.0

# Avoid home quota (~/.cache) — use project or Taiga (override with SBERT_CACHE)
export SBERT_CACHE="${SBERT_CACHE:-/projects/bbqj/alucic2/hf_cache}"
export HF_HOME="${HF_HOME:-$SBERT_CACHE}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TMPDIR="${TMPDIR:-/projects/bbqj/alucic2/tmp}"
mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$TMPDIR"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate base
source /projects/bbqj/alucic2/envs/lfactory/bin/activate

REPO="${REPO:-/projects/bbqj/alucic2/llamafactory/classroom_demo}"
TAIGA_OUT="${TAIGA_OUT:-/taiga/ncsa/radiant/bbgp/rgpu02/owodd/coco_datasets/classroom_demo}"
CSV="${CSV:-${REPO}/coyote_metadata_comparison.csv}"
OUT="${OUT:-${TAIGA_OUT}/coyote_sbert_report.json}"
METRICS="${METRICS:-${TAIGA_OUT}/coyote_sbert_metrics}"

mkdir -p "$TAIGA_OUT"

echo "Job started on $(hostname) at $(date)"
echo "HF cache: $HF_HOME"
echo "Output:   $TAIGA_OUT"

python3 - <<'PY'
import os, torch
print("torch:", torch.__version__)
print("HF_HOME:", os.environ.get("HF_HOME"))
import sentence_transformers
print("sentence-transformers:", sentence_transformers.__version__)
PY

cd "${REPO}/.."
python3 "${REPO}/compare_model_agreement.py" \
  --csv "${CSV}" \
  --species-hint coyote \
  --output "${OUT}" \
  --metrics-dir "${METRICS}"

echo "Finished at $(date)"
echo "Report: ${OUT}"
echo "Metrics: ${METRICS}/metrics_summary.json"
echo "Image trust: ${METRICS}/image_trust_summary.json"
