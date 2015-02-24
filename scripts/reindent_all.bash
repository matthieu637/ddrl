#!/bin/bash

LIB=$(dirname "${BASH_SOURCE[0]}")
cd $LIB

. locate.bash
goto_root

find . | egrep "(cpp|hpp)$" | grep -v build | xargs astyle -L -A14 -N -H -c -p --indent=spaces=4
find . | grep ".orig$" |xargs rm
