#!/bin/bash

conda create -n acoss python=3.5
conda activate acoss

conda install -c anaconda cython psutil
conda install -c bioconda ray
conda install -c ska tables
conda install -c conda-forge deepdish progress
pip install numpy==1.16.5 tensorflow==1.13.1 keras==2.2.4
pip install numba==0.43.0 pandas==0.25.3 scipy==1.2.1 scikit-learn==0.19.2

#git clone https://github.com/silvadirceu/acoss-1.git
#cd acoss-1
python setup.py install
#cd ..

pip install crema librosa==0.6.2 pumpp==0.4 essentia madmom