#!/bin/bash

##extracted from https://ray.readthedocs.io/en/latest/deploying-on-slurm.html

#SBATCH --partition=normal
#SBATCH --job-name=covers10k
##SBATCH --cpus-per-task=24
#SBATCH --mem=80GB
#SBATCH --nodes=16
#SBATCH --tasks-per-node=1
#SBATCH --mail-user=dirceu.silva@co.it.pt
#SBATCH --mail-type=ALL
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/covers10k/dirceusilva/data/Covers10k/output.txt
#SBATCH --error=/home/covers10k/dirceusilva/data/Covers10k/error.txt


worker_num=15 # Must be one less that the total number of nodes

# module load Langs/Python/3.6.4 # This will vary depending on your environment
# source venv/bin/activate

source $CONDA_PREFIX/etc/profile.d/conda.sh
conda activate acoss

python coverid_ray.py -i '/home/covers10k/dirceusilva/scripts/acoss-1/acoss/data/Covers10k_p1.csv' \
-d '/home/covers10k/dirceusilva/data/Covers10k/features/' \
-r '/home/covers10k/dirceusilva/data/Covers10k/results/' \
-v '/home/covers10k/dirceusilva/data/Covers10k/csv/' \
-b '/home/covers10k/dirceusilva/data/Covers10k/batches/' \
-a 'EarlyFusionTraile' -c 'hpcp' -p 1 -n 384 -t 0 -s "covers20k"