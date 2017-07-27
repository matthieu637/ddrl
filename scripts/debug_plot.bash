#!/bin/bash

STAT_FILE=".learning.data"
LHPO_PATH="$(pwd)/scripts/extern/lhpo"

echoerr() { echo "$@" 1>&2; }

if [ $# -lt 1 ] ; then
	echo "Usage : $0 <dir>"
	exit 1
fi

function read_input_until {
	init=""
	declare -a list=("${!1}")
	while [ 1 ] ; do
		read -s -n 1 init
		
		if [[ ${list[@]} =~ $init || $init == "" ]] ; then
			break
		fi
	done

	if [[ $init == ""  ]] ; then
		init=${list[0]}
	fi
	echo $init
}

function ask_learning_testing(){
	echoerr "Learning or Testing ? (L/T)"
	arg=(L T)
	read_input_until arg[@]
}

function ask_save_best(){
	echoerr "Save best ? (0/1/2)"
	arg=(0 1 2)
	read_input_until arg[@]
}

function ask_dimension(){
	echoerr "Which dimension ? (0-9)"
	arg=$(seq 0 9)
	read_input_until arg[@]
}

function ask_higher_better(){
	echoerr "Is higher value better on this dimension ? (0/1)"
	arg=(0 1)
	read_input_until arg[@]
}

lot=`ask_learning_testing`
if [[ $lot == "T" ]] ; then
	STAT_FILE=".testing.data"
fi

dimension=`ask_dimension`
save_best=0
save_best=`ask_save_best`
higher_better=0
if [[ $save_best != "0" ]] ; then
	higher_better=`ask_higher_better`
fi

COMMAND="one_by_one.m $STAT_FILE $dimension $save_best $higher_better more"

while [ 1 ] ; do
	cd $1
	echo "OCTAVE_PATH=$LHPO_PATH/utils octave $LHPO_PATH/utils/$COMMAND"
	OCTAVE_PATH=$LHPO_PATH/utils octave $LHPO_PATH/utils/$COMMAND
	sleep 1s
done

