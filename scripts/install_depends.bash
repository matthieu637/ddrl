#!/bin/bash

LIB=$(dirname "${BASH_SOURCE[0]}")
cd $LIB

. locate.bash
goto_root

sudo apt-get install python cmake libode-dev astyle cppcheck libtbb-dev libglew-dev libgtest-dev unzip libboost-all-dev

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

