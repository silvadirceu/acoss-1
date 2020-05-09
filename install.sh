#!/bin/zsh
# pyenv install for CentOS 6.5 x86_64

#yum install -y  gcc gcc-c++ make git patch openssl-devel zlib-devel readline-devel sqlite-devel bzip2-devel

#git clone git://github.com/yyuu/pyenv.git ~/.pyenv

#export PATH="$HOME/.pyenv/bin:$PATH"
#eval "$(pyenv init -)"
#
#cat << _PYENVCONF_ >> ~/.zshrc
#export PATH="$HOME/.pyenv/bin:$PATH"
#eval "$(pyenv init -)"
#_PYENVCONF_

# hdf5
#sudo yum -y install hdf5-devel

#After pyenv instalation

#pyenv install 3.5.9

# in git project folder

git pull
git checkout develop

# scripts folder
cd acoss

pyenv uninstall -f acoss
pyenv virtualenv 3.6.9 acoss
pyenv activate acoss

#pip install numpy==1.16.5 cython  tensorflow==1.13.1 keras==2.2.4 tables ray psutil scikit-image h5py
#pip install numba==0.43.0 pandas==0.25.3 scipy==1.2.1 scikit-learn==0.19.2 deepdish progress
#pip install crema librosa==0.6.2 pumpp essentia madmom

pip install numpy cython  tensorflow keras tables ray psutil scikit-image h5py
pip install numba pandas scipy scikit-learn deepdish progress
pip install crema librosa==0.6.2 pumpp essentia madmom

#git clone https://github.com/silvadirceu/acoss-1.git
#cd acoss-1
python setup.py install
#cd ..





