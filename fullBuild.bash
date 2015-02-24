#!/bin/bash

LIB=$(dirname "${BASH_SOURCE[0]}")
cd $LIB

. scripts/cpu.bash
#check cmake if installed
. scripts/check_program.bash

function hr(){
	echo "---------------------------------------------------------------------------------"
}

#if the last command failed display the log and stop the program
function stopOnError(){
	error=$1
	log=$2
	if [ $error -ne 0 ] ; then
		cat $log
        	exit
        fi  
}

#launch cmake and make in a multicore maner
#if the build fails, it displays the logs
function cmakeBuild(){
	directory=$1
	target=$2

	tmplog=`mktemp`

	mkdir build/$1
	cd build/$1

	cmake ../.. -DCMAKE_BUILD_TYPE=$2 >& $tmplog
	stopOnError $? $tmplog
	echo '' > $tmplog

	make -j $(nbcpu) >& $tmplog
	stopOnError $? $tmplog

	rm $tmplog
	cd ../..
}

#found all the CMakeLists.txt and create 3 targets to build (release, debug, release with debug info)
function buildDir(){
	dir=$1
	if [ ! -e $dir ] ; then
		echo "Please cd into the root directory of DRL"
		exit
	fi
	
	here=`pwd`
	find $dir -name 'CMakeLists.txt' -printf '%h\n' | while read subdir ; do
		cd $here/$subdir
		
		if [ -e build ]; then
			if [ $FORCE_REMOVE -eq 0 ]; then
				echo "INFO : $subdir already contains a build directory. Passing..."
				continue
			else
				echo "INFO : $subdir already contains a build directory. Removing it..."
				rm -rf build/
			fi
		fi

		if [[ -e lib && $FORCE_REMOVE -eq 1 ]]; then
			rm -rf lib/
		fi
		
		#building
		mkdir build
		cmakeBuild release Release
		cmakeBuild debug Debug
		cmakeBuild relwithdeb RelWithDebInfo

		echo "INFO : $subdir well builed. Congratz."
		hr
	done
}

echo "INFO : cmake well founded. Look what following to know if you need other software."
echo "QUESTION : if a subdirectory already contains a build, should I remove it ? (y/n) [y]:"

force_remove="a"
while [[ $force_remove != "" && $force_remove != "y" &&  $force_remove != "n"  ]] ; do
	read force_remove
done
hr

if [[ $force_remove == "n" ]] ; then
	export FORCE_REMOVE=0
else
	export FORCE_REMOVE=1
fi

buildDir common
buildDir environment
buildDir agent
