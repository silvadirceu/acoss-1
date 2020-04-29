#!/bin/bash

##extracted from https://ray.readthedocs.io/en/latest/deploying-on-slurm.html

#SBATCH --partition=normal
#SBATCH --job-name=covers10k
##SBATCH --cpus-per-task=10
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

nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST) # Getting the node names
nodes_array=( $nodes )

node1=${nodes_array[0]}

ip_prefix=$(srun --nodes=1 --ntasks=1 -w $node1 hostname --ip-address) # Making address
suffix=':6379'
ip_head=$ip_prefix$suffix
redis_password=$(uuidgen)

export ip_head # Exporting for latter access by trainer.py

echo "STARTING HEAD at $node1"
# Starting the head
srun --nodes=1 --ntasks=1 -w $node1 ray start --block --head --redis-port=6379 --redis-password=$redis_password \
--object-store-memory=$((1024 * 1024 * 1024))  --memory=$((7 * 1024 * 1024 * 1024)) &
sleep 5

for ((  i=1; i<=$worker_num; i++ ))
do
  # Starting the workers
  node2=${nodes_array[$i]}
  srun --nodes=1 --ntasks=1 -w $node2 ray start --block --address=$ip_head --redis-password=$redis_password \
  --object-store-memory=$((1024 * 1024 * 1024))  --memory=$((7 * 1024 * 1024 * 1024)) &
  sleep 5
done

#python -u extracton.py $redis_password 15 # Pass the total number of allocated CPUs

python -u extractors_ray.py -d '/home/covers10k/dirceusilva/scripts/acoss-1/acoss/data/Covers10k_1.csv' \
-a '/home/covers10k/dirceusilva/data/Covers10k/Audios/' \
-p '/home/covers10k/dirceusilva/data/Covers10k/features/' \
-b '/home/covers10k/dirceusilva/scripts/acoss-1/acoss/batches_1/' \
-f 'hpcp' 'key_extractor' 'madmom_features' 'mfcc_htk' 'chroma_cens' 'crema' \
-m 'parallel' -c 1 -n 160 -r 0 -w $redis_password