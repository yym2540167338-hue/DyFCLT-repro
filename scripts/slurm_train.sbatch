#!/bin/bash
# Run one of the scripts/run_*.sh on a GPU node of a Slurm cluster.
#   sbatch -p <GPU分区名> scripts/slurm_train.sbatch                     # default: scripts/run_droneperson.sh
#   sbatch -p <GPU分区名> scripts/slurm_train.sbatch scripts/run_flir.sh
# Partition names and GPU types: `sinfo -o "%P %G %l"`. Some clusters want --gpus=1 instead of --gres.
#SBATCH -J dyfclt
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 48:00:00
#SBATCH -o slurm-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"

# module load anaconda3        # uncomment / adapt if conda is provided through environment modules
if [ -f "$HOME/venvs/dyfclt/bin/activate" ]; then        # created by setup_server_cuda.sh without conda
  source "$HOME/venvs/dyfclt/bin/activate"
elif command -v conda >/dev/null; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate dyfclt
else
  echo "no dyfclt environment found: run bash environment/setup_server_cuda.sh first"; exit 1
fi
export PYTHONNOUSERSITE=1      # ignore packages installed with `pip install --user` (~/.local)

echo "host=$(hostname) job=${SLURM_JOB_ID:-} gpus=${SLURM_JOB_GPUS:-${CUDA_VISIBLE_DEVICES:-}}"
nvidia-smi
NGPU=${NGPU:-1} bash "${1:-scripts/run_droneperson.sh}"
