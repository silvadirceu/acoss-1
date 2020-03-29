#!/bin/bash

python extractors_ray.py -d '/home/ecad/acoss-1/acoss/data/Covers10k.csv' \
-a '/home/ecad/AUDIOBASE/Covers10k/Audios/' \
-p '/home/ecad/Results/features/Covers10k_11khz/' \
-f 'hpcp' 'key_extractor' 'madmom_features' 'mfcc_htk' 'chroma_cens' 'crema' \
-m 'parallel' -n 18

##python extractors_ray.py -d '/mnt/Data/dirceusilva/dados/ECAD/Covers10k/Covers10k.csv' \
##-a '/mnt/Data/dirceusilva/dados/ECAD/Covers10k/Audios/' \
##-p '/mnt/Data/dirceusilva/Results/features/Covers10k_11khz/' \
##-f 'hpcp' 'key_extractor' 'madmom_features' 'mfcc_htk' 'chroma_cens' 'crema' \
##-m 'parallel' -n 8


##python extractors.py -d '/mnt/Data/dirceusilva/dados/Cover/Covers80/lists/covers80_complete.csv' \
##-a '/mnt/Data/dirceusilva/dados/Cover/Covers80/covers32k/' \
##-p '/mnt/Data/dirceusilva/Results/features/covers80_11khz/' \
##-f 'hpcp' 'key_extractor' 'madmom_features' 'mfcc_htk' 'chroma_cens' 'crema' \
##-m 'parallel'
