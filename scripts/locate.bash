#!/bin/bash

PROJECT_NAME='drl'

function root_path(){
	here=`pwd`
	if [ `echo $here | grep $PROJECT_NAME | wc -l` -eq 1 ] ; then
		#go up
		reduction=`echo $here | sed "s/^.*\/$PROJECT_NAME/$PROJECT_NAME/"`
		nb_dir=`echo $reduction | grep -o '/' | wc -l`
		path='.'
		for i in $(seq 1 1 $nb_dir) ; do
			path="$path/.."
		done
		echo "$path/"
	elif [ -e $PROJECT_NAME ] ; then
		#go down
		echo $PROJECT_NAME
	else
		#scripts dir
		cd "$(dirname "${BASH_SOURCE[0]}")"
		pwd | sed "s/scripts//"
	fi
}

function goto_root(){
	path=$(root_path)
	cd $path
}

