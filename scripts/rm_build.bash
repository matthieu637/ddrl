#!/bin/bash

LIB=$(dirname "${BASH_SOURCE[0]}")
cd $LIB

. locate.bash

function rm_build(){
	goto_root
	if [ `find . -name 'build' -type d | wc -l` -ne 0 ] ; then
		find . -name 'build' -type d | xargs rm -r 
	fi
}
