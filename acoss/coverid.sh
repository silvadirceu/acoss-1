#!/bin/bash

##python extractors_ray.py -d '/home/ecad/acoss-1/acoss/data/Covers10k.csv' \
##-a '/home/ecad/AUDIOBASE/Covers10k/Audios/' \
##-p '/home/ecad/AUDIOBASE/Covers10k/Results/features/Covers10k_11khz/' \
##-f 'hpcp' 'key_extractor' 'madmom_features' 'mfcc_htk' 'chroma_cens' 'crema' \
##-m 'parallel' -n 18 -r 0

#python extractors_ray.py -d '/mnt/Data/dirceusilva/development/libs/acoss-1/acoss/data/Covers10k_p.csv' \
#-a '/mnt/Data/dirceusilva/dados/ECAD/Covers10k/Audios/' \
#-p '/mnt/Data/dirceusilva/Results/features/Covers10k_11khz/features/' \
#-f 'hpcp' 'key_extractor' 'madmom_features' 'mfcc_htk' 'chroma_cens' 'crema' \
#-m 'single' -n 1 -r 1

python coverid_ray.py -i 'data/Covers10k_p2.csv' \
-d '/mnt/Data/dirceusilva/Results/features/Covers10k_11khz/features/' \
-r '/mnt/Data/dirceusilva/Results/features/Covers10k_11khz/results/' \
-v '/mnt/Data/dirceusilva/Results/features/Covers10k_11khz/csv/' \
-b '/mnt/Data/dirceusilva/Results/features/Covers10k_11khz/batches/' \
-a 'FTM2D' -c 'hpcp' -p 1 -n 8 -t 0 -s "coversBR"


#python coverid_ray.py -i 'data/Covers10k_p.csv' \
#-d '/media/dirceu/Backup2T/Testes/Cover/Covers10k_11khz/features/' \
#-r '/media/dirceu/Backup2T/Testes/Cover/Covers10k_11khz/results/' \
#-v '/media/dirceu/Backup2T/Testes/Cover/Covers10k_11khz/csv/' \
#-b '/media/dirceu/Backup2T/Testes/Cover/Covers10k_11khz/batches/' \
#-a 'EarlyFusion' -c 'hpcp' -p 1 -n 4 -t 0


