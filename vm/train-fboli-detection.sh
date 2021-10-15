#!/bin/bash
#
#SBATCH --job-name=train-fboli-detection
#
#SBATCH --time=5-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --nice
#SBATCH --qos=training

# Example:
# sbatch --export=NNETTRAIN_CFG="/abs/path/to/cfg.yaml" train-fboli-detection.sh


export PATH="/opt/singularity/bin:${PATH}"

echo "BEGIN TRAINING: ${NNETTRAIN_CFG}"
module load singularity
singularity exec --nv vm/multi-mouse-pose-2019-11-04.sif python3 tools/trainfecalboli.py \
      --cfg "${NNETTRAIN_CFG}" \
      --cvat-files data/fecal-boli/*.xml \
      --image-dir data/fecal-boli/images
