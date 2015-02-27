#!/bin/bash

LIB=$(dirname "${BASH_SOURCE[0]}")
cd $LIB

. locate.bash
goto_root

#cpplint to inject missing namespace comment [readability/namespace]
./scripts/check_code.bash |& grep 'readability/namespace' | while read prob ; do
	file=`echo "$prob" | cut -d ':' -f1`
	line=`echo "$prob" | cut -d ':' -f2`
	injection=`echo "$prob" | cut -d ':' -f3 | sed -e 's/^.*"\(\/\/ namespace [a-zA-Z]*\)".*$/\1/'`
	injection=`echo $injection | sed -e 's/\//\\\%/g' | sed -e 's/%/\//g'`

	sed -si "${line},${line}s/}/} $injection/" $file
	#echo "${file} changed! ($line)"
done

./scripts/reindent_all.bash

