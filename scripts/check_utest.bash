#!/bin/bash

LIB=$(dirname "${BASH_SOURCE[0]}")
cd $LIB

. locate.bash

set -e

function run_all_test(){
	goto_root
	find {common,agent,environment} -type f -name 'unit-test' | grep -v old | while read atest ; do
		cd $(dirname $atest)
		#1h timeout
		timeout 3600 ./unit-test $@
		if [ $? -ne 0 ] ; then
			exit 1
		fi
		cd -
	done
}


if [ $# -eq 0 ] ; then
	run_all_test
else
	run_all_test --gtest_output="xml:./unit-test.xml"
fi

exit 0

