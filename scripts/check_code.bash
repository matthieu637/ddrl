#!/bin/bash

LIB=$(dirname "${BASH_SOURCE[0]}")
cd $LIB

. locate.bash
. rm_build.bash
. reindent_all.bash --norun

rm_build

function cppcheck_run(){
	goto_root
	all_sources=`ls -d */src */*/src`
	all_includes=`ls -d */include */*/include | xargs -I% echo -n "-I% "`
	cppcheck --enable=all --inconclusive --suppress=missingIncludeSystem --std=c++11 $all_includes $all_sources $@
}

function check_code_cppcheck(){
	tmp=`mktemp`
	cppcheck_run >& $tmp
	cat $tmp | grep -v 'Checking' | grep -v 'files checked' | grep -v "Cppcheck cannot find all"
	error=`cat $tmp | grep -v 'Checking' | grep -v 'files checked' | grep -v "Cppcheck cannot find all" | grep -v 'is never used' | wc -l`
	rm $tmp
	if [ $error -ne 0 ] ; then
		echo "ERROR : $error"
		exit 1
	fi
}

function check_code_cpplint(){
	goto_root
	all_files=`find . -type f -name '*cpp' -o -name '*.hpp' | grep -v extern`
	tmp=`mktemp`
	echo $all_files | xargs cpplint --filter=-legal/copyright,-build/c++11 --extensions=hpp,cpp --linelength=120 |& grep -v 'Include the directory when naming' |& grep -v 'All parameters should be named' |& grep -v 'Archive &ar' |& grep -v 'Is this a non-const reference.*ostream' | grep -v 'Done processing' | grep -v 'Total errors' >& $tmp
	error=`cat $tmp | wc -l`
	cat $tmp
	rm $tmp
	if [ $error -ne 0 ] ; then
		echo "ERROR : $error"
		exit 1
	fi

	#EXCEPTIONS RULES : coryright, enable c++11, linelength 120
	#include "dir/fann.h" hasn't been done like this by the fann library
	#Archive* instead of Archive& : hasn't been done like this by the boost library
	#ostream* instead of ostream& : ugly
	#All parameters should be named : conflit with gcc warning unused parameters
}

function check_code_astyle(){
	tmp=`mktemp`
	reindentation --dry-run >& $tmp
	error=`cat $tmp | grep -e 'Forma' | wc -l`
	cat $tmp | grep -e 'Forma'
	if [ $error -ne 0 ] ; then
		echo "ERROR : $error"
		exit 1
	fi
}

if [ $# -eq 0 ] ; then
	echo "################### CPPCHECK ###################"
	check_code_cppcheck
	echo "################### CPPLINT ###################"
	check_code_cpplint
	echo "################### ASTYLE ###################"
	check_code_astyle
else
	cppcheck_run --xml --xml-version=2
fi

