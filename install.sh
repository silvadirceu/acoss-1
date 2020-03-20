#!/bin/zsh
# pyenv install for CentOS 6.5 x86_64

yum install -y  gcc gcc-c++ make git patch openssl-devel zlib-devel readline-devel sqlite-devel bzip2-devel

git clone git://github.com/yyuu/pyenv.git ~/.pyenv

export PATH="$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init -)"

cat << _PYENVCONF_ >> ~/.zshrc
export PATH="$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init -)"
_PYENVCONF_


#After pyenv instalation

pyenv install 3.5.9
pyenv virtualenv 3.5.9 acoss
pyenv activate acoss

pip install numpy==1.16.2 cython  tensorflow==1.13.1 keras==2.2.4

git clone https://github.com/silvadirceu/acoss-1.git
cd acoss-1
python setup.py install
cd ..

pip install crema librosa==0.6.2 pumpp==0.4 essentia

