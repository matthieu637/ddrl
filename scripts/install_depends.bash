#!/bin/bash

LIB=$(dirname "${BASH_SOURCE[0]}")
cd $LIB

. locate.bash
goto_root

sudo apt-get install python cmake libode-dev astyle cppcheck libtbb-dev libglew-dev libgtest-dev unzip libboost-all-dev doxygen valgrind

#cpplint
wget https://google-styleguide.googlecode.com/svn/trunk/cpplint/cpplint.py
sudo mv cpplint.py /usr/local/bin/cpplint
sudo chmod ugo+rx /usr/local/bin/cpplint

#fann
mkdir extern
cd extern
wget http://downloads.sourceforge.net/sourceforge/fann/FANN-2.2.0-Source.zip
unzip FANN-2.2.0-Source.zip
cd FANN-2.2.0-Source
cmake .
make
sudo make install

#gtest lib
cd /usr/src/gtest
sudo cmake .
sudo make
sudo mv libg* /usr/local/lib/

echo "check your gcc version isn't too old ( <= 4.6 )"

#opt++
sudo apt-get install gfortran libblas-dev

goto_root
cd scripts/extern
#wget https://software.sandia.gov/opt++/downloads/optpp-2.4.tar.gz
tar -xvf optpp-2.4.tar.gz
cd optpp-2.4/
./configure --prefix=/usr/local --includedir=/usr/include/opt++ --enable-static=no --enable-shared=yes
make
sudo make install
