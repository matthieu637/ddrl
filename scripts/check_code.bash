#!/bin/bash

LIB=$(dirname "${BASH_SOURCE[0]}")
cd $LIB

. locate.bash
. rm_build.bash

rm_build

function check_code_cppcheck(){
	goto_root
	all_sources=`ls -d */src */*/src`
	all_includes=`ls -d */include */*/include | xargs -I% echo -n "-I% "`
	cppcheck --enable=all --inconclusive --suppress=missingIncludeSystem --std=c++11 $all_includes $all_sources
}

function check_code_cpplint(){
	goto_root
	all_files=`find . -type f -name '*cpp' -o -name '*.hpp' | grep -v extern`
	echo $all_files | xargs cpplint --filter=-legal/copyright,-build/c++11 --extensions=hpp,cpp --linelength=120 |& grep -v 'Include the directory when naming' |& grep -v 'All parameters should be named' |& grep -v 'Archive &ar' |& grep -v 'Is this a non-const reference.*ostream'

	#EXCEPTIONS RULES : coryright, enable c++11, linelength 120
	#include "dir/fann.h" hasn't been done like this by the fann library
	#Archive* instead of Archive& : hasn't been done like this by the boost library
	#ostream* instead of ostream& : ugly
	#All parameters should be named : conflit with gcc warning unused parameters
}

check_code_cppcheck
check_code_cpplint
