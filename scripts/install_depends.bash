#!/bin/bash

LIB=$(dirname "${BASH_SOURCE[0]}")
cd $LIB

. locate.bash
goto_root

sudo apt-get install python
sudo apt-get install cmake
sudo apt-get install libode-dev
sudo apt-get install astyle
sudo apt-get install cppcheck
sudo apt-get install libtbb-dev
sudo apt-get install libglew-dev
sudo apt-get install libgtest-dev
sudo apt-get install unzip

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
