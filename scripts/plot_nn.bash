#!/bin/bash

CAFFE_PATH=~/git/caffe/

if [ $# -eq 0 ] ; then
	echo "usage $0 <dir with a config.ini>"
fi

if [ ! -e $1 ] ; then
	echo "$1 doesn't exists"	
fi 

if [ ! -e $1/config.ini ] ; then
	echo "$1/config.ini doesn't exists"
fi

#prepare a run with a verry low number of ep
tmp=`mktemp -d`
cd $1
cp *.ini $tmp
cd $tmp
sed -i 's/^max_episode=.*$/max_episode=3/g' *.ini

#looking for an executable
cd $1
if [ -e ../../rules.xml ] ; then
	COMMAND=$(xml sel -t -m "/xml/command" -v @value ../../rules.xml)
	ARGS=$(xml sel -t -m "/xml/args" -v @value ../../rules.xml)
	cp $COMMAND $tmp
else
	echo "code me! cannot find an executable"
	exit 1
fi
cd ../..

#run
cd $tmp
#$COMMAND $ARGS
$COMMAND $ARGS >& /dev/null

for file in *.struct.data ; do
	python2 $CAFFE_PATH/python/draw_net.py $file $file.pdf
	python2 $CAFFE_PATH/python/draw_net.py $file $file.dot
#	python2 $CAFFE_PATH/python/draw_net_old.py $file $file.old.pdf
done

#xdot Actor.struct.data.dot
#xdot Critic.struct.data.dot
#xdot Actor.1.struct.data.dot
#xdot Critic.1.struct.data.dot
#xdot Critic.1.struct.data.old.dot

echo $tmp
okular *.pdf

#rm -rf $tmp

