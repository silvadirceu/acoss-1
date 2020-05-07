#!/bin/bash

##python extractors_ray.py -d '/home/ecad/acoss-1/acoss/data/Covers10k.csv' \
##-a '/home/ecad/AUDIOBASE/Covers10k/Audios/' \
##-p '/home/ecad/AUDIOBASE/Covers10k/Results/features/Covers10k_11khz/' \
##-f 'hpcp' 'key_extractor' 'madmom_features' 'mfcc_htk' 'chroma_cens' 'crema' \
##-m 'parallel' -n 18 -r 0

python extractors_ray.py -d '/mnt/Data/dirceusilva/development/libs/acoss-1/acoss/data/Covers10k_p3.csv' \
-a '/mnt/Data/dirceusilva/dados/ECAD/CoversBR/Audios/' \
-p '/mnt/Data/dirceusilva/Results/features/' \
-f 'hpcp' 'key_extractor' 'madmom_features' 'mfcc_htk' 'chroma_cens' 'crema' \
-m 'single' -n 1 -r 1


#python extractors.py -d 'data/Covers10k_p.csv' \
#-a '/media/dirceu/Backup2T/BaseDados/Covers10k/Audios/' \
#-p '/media/dirceu/Backup2T/Testes/Cover/Covers10k_11khz/features/' \
#-f 'hpcp' 'key_extractor' 'madmom_features' 'mfcc_htk' 'chroma_cens' 'crema' \
#-m 'parallel' -n 8 -r 0
