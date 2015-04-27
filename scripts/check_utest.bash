#!/bin/bash

LIB=$(dirname "${BASH_SOURCE[0]}")
cd $LIB

. locate.bash

function run_all_test(){
	goto_root
	find . -type f -name 'unit-test' | while read atest ; do
		cd $(dirname $atest)
		#15 min timeout
		timeout 900 ./unit-test $@
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

