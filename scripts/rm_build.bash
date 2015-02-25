#!/bin/bash

LIB=$(dirname "${BASH_SOURCE[0]}")
cd $LIB

. locate.bash

function rm_build_lib(){
        goto_root
        if [ `find . -name 'lib' -type d | wc -l` -ne 0 ] ; then
                find . -name 'lib' -type d | xargs rm -r  
        fi 
}

function rm_build(){
	rm_build_lib

	goto_root
	if [ `find . -name 'build' -type d | wc -l` -ne 0 ] ; then
		find . -name 'build' -type d | xargs rm -r 
	fi
}
