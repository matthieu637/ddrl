#!/bin/bash

LIB=$(dirname "${BASH_SOURCE[0]}")
cd $LIB

. locate.bash
goto_root

find . | egrep "(cpp|hpp)$" | grep -v build | xargs  astyle -A14 --indent=spaces=2 

if [ `find . | grep ".orig$" | wc -l` -ne 0 ] ; then
	find . | grep ".orig$" |xargs rm
fi
