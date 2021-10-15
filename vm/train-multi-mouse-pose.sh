#!/bin/bash
#
#SBATCH --job-name=train-multi-mouse-pose
#
#SBATCH --time=5-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --nice
#SBATCH --partition=gpu

# Example:
# sbatch --export=MMTRAIN_CFG="/abs/path/to/cfg.yaml" train-multi-mouse-pose.sh


export PATH="/opt/singularity/bin:${PATH}"

echo "BEGIN MULTI-MOUSE POSE TRAINING: ${MMTRAIN_CFG}"
module load singularity
singularity exec --nv vm/multi-mouse-pose-2019-11-04.sif python3 tools/trainmultimouse.py --cfg "${MMTRAIN_CFG}" --cvat-files data/multi-mouse/Annotations/*.xml data/multi-mouse/Annotations_NoMarkings/*.xml --image-dir data/multi-mouse/Dataset
