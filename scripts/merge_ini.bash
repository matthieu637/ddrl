#!/bin/bash

if [ $# -ne 3 ] ; then
	echo "Usage $0 : <base> <injector> <destination>"
	exit 1
fi

base=$1
injector=$2
destination=$3

if [ ! -e $base ] ; then
	echo "$base doesn't exist !"
fi

if [ ! -e $injector ] ; then
	echo "$injector doesn't exist !"
fi

tmp=`mktemp`
diff -DVERSION1 $base $injector > $tmp
grep -v '^#if' $tmp | grep -v '^#endif' | grep -v '#else' > $destination
mv $destination $tmp

duplicates="`cat $tmp | grep '=' | grep -v -e '^#' | cut -d'=' -f1 | sort | uniq -d`"

echo "$duplicates" | while read dupl ; do
	sed -i "0,/\($dupl\)/s//#\1/" $tmp
	#dupl=`echo $dupl | sed -e 's/[_]/[_]/g'`
	#sed -i "s/$dupl//" $tmp
	#echo "$dupl"
done

mv $tmp $destination
