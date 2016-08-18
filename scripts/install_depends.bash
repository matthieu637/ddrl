#!/bin/bash

###########################################
# INSTRUCTION FOR INSTALLATION ON UBUNTU  #
###########################################
# In case you use Arch, just install the provided PKGBUILD

LIB=$(dirname "${BASH_SOURCE[0]}")
cd $LIB

. locate.bash
goto_root

#####################
# GENERAL PACKAGES  #
#####################
sudo apt-get install python cmake libode-dev astyle cppcheck libtbb-dev libglew-dev libgtest-dev unzip libboost-all-dev doxygen valgrind

echo "check your gcc version isn't too old ( <= 4.6 )"

########################
# gtest lib (required) #
########################
cd /usr/src/gtest
sudo cmake .
sudo make
sudo mv libg* /usr/local/lib/

######################
# cpplint (optional) #
######################
wget https://google-styleguide.googlecode.com/svn/trunk/cpplint/cpplint.py
sudo mv cpplint.py /usr/local/bin/cpplint
sudo chmod ugo+rx /usr/local/bin/cpplint

#######################################################
# fann (required by agents : qlearning, cacla, cmaes) #
#######################################################
mkdir extern
cd extern
wget http://downloads.sourceforge.net/sourceforge/fann/FANN-2.2.0-Source.zip
unzip FANN-2.2.0-Source.zip
cd FANN-2.2.0-Source
cmake .
make
sudo make install

########################################################
# opt++ (required by agents : qlearning, cacla, cmaes) #
########################################################
sudo apt-get install gfortran libblas-dev

goto_root
cd scripts/extern
#wget https://software.sandia.gov/opt++/downloads/optpp-2.4.tar.gz
tar -xvf optpp-2.4.tar.gz
wget https://matthieu-zimmer.net/~matthieu/patches/optpp-2.4.patch
cd optpp-2.4/
patch -Np1 -i ../optpp-2.4.patch
./configure --prefix=/usr/local --includedir=/usr/include/opt++ --enable-static=no --enable-shared=yes
make
sudo make install

#################################
# eigen (required by dmp-power) #
#################################
sudo apt-get install libeigen3-dev

##############################
# caffe (required by deepqn) #
##############################
sudo apt-get install nvidia-cuda-dev nvidia-cuda-toolkit libprotobuf-dev libleveldb-dev libsnappy-dev protobuf-compiler libopenblas-dev libgflags-dev libgoogle-glog-dev liblmdb-dev libhdf5-serial-dev

goto_root
cd scripts/extern
git clone https://github.com/BVLC/caffe.git
wget https://matthieu-zimmer.net/~matthieu/patches/caffe-git.patch
patch -Np0 -i caffe-git.patch
mkdir caffe/build
cd caffe/build
cmake ../ -DBLAS=Open -DBUILD_python=OFF -DUSE_OPENCV=OFF -DCMAKE_C_COMPILER=gcc-5 \
                -DCMAKE_INSTALL_PREFIX=/usr/local -DCUDA_ARCH_NAME=All -DCMAKE_CXX_FLAGS="-D_FORCE_INLINES"
make all -j4
sudo make install

