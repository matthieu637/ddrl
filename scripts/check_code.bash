#!/bin/bash

LIB=$(dirname "${BASH_SOURCE[0]}")
cd $LIB

. locate.bash
. rm_build.bash

rm_build

goto_root

all_sources=`ls -d */src */*/src */include */*/include`

cppcheck --enable=all --inconclusive --suppress=missingIncludeSystem $all_sources

