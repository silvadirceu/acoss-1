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
--object-store-memory=$((1024 * 1024 * 1024))  --memory=$((2 * 1024 * 1024 * 1024)) &
sleep 5

for ((  i=1; i<=$worker_num; i++ ))
do
  # Starting the workers
  node2=${nodes_array[$i]}
  srun --nodes=1 --ntasks=1 -w $node2 ray start --block --address=$ip_head --redis-password=$redis_password \
  --object-store-memory=$((1024 * 1024 * 1024))  --memory=$((2 * 1024 * 1024 * 1024)) &
  sleep 5
done


python coverid_ray.py -i '/home/covers10k/dirceusilva/scripts/acoss-1/acoss/data/Covers10k_p1.csv' \
-d '/home/covers10k/dirceusilva/data/Covers10k/features/' \
-r '/home/covers10k/dirceusilva/data/Covers10k/cache/' \
-v '/home/covers10k/dirceusilva/data/Covers10k/csv/' \
-b '/home/covers10k/dirceusilva/data/Covers10k/batches/' \
-a 'EarlyFusionTraile' -c 'hpcp' -p 1 -n 384 -t 1 -s "coversBR" -w $redis_password