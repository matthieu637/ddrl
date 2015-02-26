#!/bin/bash

LIB=$(dirname "${BASH_SOURCE[0]}")
cd $LIB

. locate.bash
. rm_build.bash

rm_build

function check_code_cppcheck(){
	goto_root
	all_sources=`ls -d */src */*/src */include */*/include`
	cppcheck --enable=all --inconclusive --suppress=missingIncludeSystem $all_sources
}

function check_code_cpplint(){
	goto_root
	all_files=`find . -type f -name '*cpp' -o -name '*.hpp' | grep -v extern`
	echo $all_files | xargs cpplint --filter=-legal/copyright --extensions=hpp,cpp --linelength=120
}

check_code_cppcheck
check_code_cpplint
