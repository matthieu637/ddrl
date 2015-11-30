#!/bin/bash

CALLED_PATH=$(pwd)

LIB=$(dirname "${BASH_SOURCE[0]}")
cd $LIB
LIB=$(pwd)

. ./check_program.bash
check_all

if [ ! -e extern/lhpo ] ; then
	cd extern
	git clone https://github.com/matthieu637/lhpo.git
else
	cd extern/lhpo
	timeout 2 git pull > /dev/null

	if [ $? -ne 0 ] ; then
		echo "timeout cannot pull"
	fi
fi

if [ $# -ne 3 ] ; then
	echo "$0 <experiment_name> <agent_executable> <number_of_run>"
	exit 1
fi

cd $CALLED_PATH

if [ ! -e $2 ] ; then
	echo "cannot find $2"
	exit 1
fi

if [ ! -e $LIB/../gen-data/ ] ; then
	mkdir $LIB/../gen-data/
fi

if [ ! -e $LIB/../gen-data/$1 ] ; then
	mkdir $LIB/../gen-data/$1
	echo "Create gen-data/$1"
else
	echo "$1 already exists"
	exit 1
fi

if [ ! -e $(dirname $2)/config.ini ] ; then
	echo "cannot found a config.ini near $2"
	echo "you must do it yourself"
else
	cp $(dirname $2)/config.ini $LIB/../gen-data/$1/
fi

RULES="$LIB/../gen-data/$1/rules.xml"
echo "<xml>" >> $RULES
echo "	<command value='$CALLED_PATH/$2' />" >> $RULES
echo "	<args value='' />" >> $RULES
echo "	<ini_file value='config.ini' />" >> $RULES
echo "	<end_file value='time_elapsed' />" >> $RULES
echo "	<default_stat_file value='learning.data' />" >> $RULES
echo "" >> $RULES
echo "	<fold name='main'>" >> $RULES
echo "		<!-- <param name='some_parameter' values='a,b,c' /> --> " >> $RULES
echo "		<param name='run' values='1:$3' />" >> $RULES
echo "	</fold>" >> $RULES
echo "</xml>" >> $RULES

cd $LIB/extern/lhpo/
$LIB/extern/lhpo/parsing_rules.bash $LIB/../gen-data/$1

echo "Pre-run ok."
echo "####### If you want to change parameters : ######"
echo "edit $LIB/../gen-data/$1/rules.xml"
echo "cd $LIB/extern/lhpo/ && ./parsing_rules.bash $LIB/../gen-data/$1"

echo "####### To finaly run experiment : ######"
echo "cd $LIB/extern/lhpo/ && ./optimizer.bash $LIB/../gen-data/$1"

echo "####### To monitor : ######"
echo "cd $LIB/extern/lhpo/ && ./count.bash $LIB/../gen-data/$1"

echo "####### Statistics : ######"
echo "cd $LIB/extern/lhpo/ && ./stats.bash $LIB/../gen-data/$1"

