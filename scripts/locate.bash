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
	else
		#go down
		echo $PROJECT_NAME
	fi

}

function goto_root(){
	path=$(root_path)
	cd $path
}

