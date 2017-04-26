#!/bin/bash

LIB=$(dirname "${BASH_SOURCE[0]}")
cd $LIB

. locate.bash

set -e

function run_all_test(){
	goto_root
	find {common,agent,environment} -type f -name 'unit-test' | grep -v old | while read atest ; do
		cd $(dirname $atest)
		tmp=`mktemp`
		#30 min timeout
		echo "testing $atest"
		valgrind --tool=memcheck --leak-check=full --show-reachable=yes --track-origins=yes --leak-resolution=high ./unit-test -valgrind >& $tmp
		#if [ $? -ne 0 ] ; then
		#	cat $tmp
		#	rm $tmp
		#	exit 1
		#fi
		if [ `cat $tmp |& grep ERROR |& grep '0 errors' | wc -l` -ne 1 ] ; then
			cat $tmp
			echo "more than 0 errors"
			rm $tmp
			exit 1
		fi

		if [ `cat $tmp |& grep -A 6 'LEAK SUMMARY' |& grep '0 bytes in 0 blocks' | wc -l` -ne 4 ] ; then
			cat $tmp
			rm $tmp
			echo "more than just still reacheable memory"
			exit 1
		fi

		rm $tmp
		cd -
	done
}


run_all_test

exit 0

