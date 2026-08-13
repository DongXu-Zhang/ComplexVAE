#!/usr/bin/env bash
#SBATCH --job-name=microvae_env
#SBATCH --partition=comp
#SBATCH --qos=normal
#SBATCH --time=00:40:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/home/zhaoruij/mz32_scratch2/wuyujun/VAE/microscopy-vae/build_venv.%j.log
set -euo pipefail
cd /home/zhaoruij/mz32_scratch2/wuyujun/VAE/microscopy-vae
bash setup_env.sh
