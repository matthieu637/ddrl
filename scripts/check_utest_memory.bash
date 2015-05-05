#!/bin/bash

LIB=$(dirname "${BASH_SOURCE[0]}")
cd $LIB

. locate.bash

set -e

function run_all_test(){
	goto_root
	find . -type f -name 'unit-test' | while read atest ; do
		cd $(dirname $atest)
		tmp=`mktemp`
		#30 min timeout
		timeout 1800 valgrind --tool=memcheck --leak-check=full --show-reachable=yes --track-origins=yes --leak-resolution=high ./unit-test >& $tmp
		if [ $? -ne 0 ] ; then
			cat $tmp
			rm $tmp
			exit 1
		fi
		if [ `cat $tmp |& grep ERROR |& grep '0 errors' | wc -l` -ne 1 ] ; then
			cat $tmp
			rm $tmp
			exit 1
		fi
		rm $tmp
		cd -
	done
}


run_all_test

exit 0

