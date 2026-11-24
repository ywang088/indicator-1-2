#!/bin/bash
#SBATCH --job-name=parking18-fresh
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=parking-fresh-%j.log
set -euo pipefail
cd "${PARKING_ROOT:-$SLURM_SUBMIT_DIR}"
: "${PARKING_PYTHON:?Set to the Python with SAM3 and CUDA installed}"
export EASYOCR_MODEL_DIR="${EASYOCR_MODEL_DIR:-$PWD/data/models/ocr}"
export OMP_NUM_THREADS=4
bash scripts/01_build_18.sh
bash scripts/02_compare_full_map.sh
