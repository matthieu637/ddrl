#!/bin/bash

LIB=$(dirname "${BASH_SOURCE[0]}")
cd $LIB

. scripts/cpu.bash
#check cmake if installed
. scripts/check_program.bash
check_all

function hr(){
	echo "---------------------------------------------------------------------------------"
}

#if the last command failed display the log and stop the program
function stopOnError(){
	error=$1
	log=$2
	if [ $error -ne 0 ] ; then
		cat $log
        	exit 1
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
	
	if [[ "${CMAKE_ARGS[@]}" == '' ]] ; then
		cmake ../.. -DCMAKE_BUILD_TYPE=$2 >& $tmplog
	else
		cmake ../.. -DCMAKE_BUILD_TYPE=$2 -G "${CMAKE_ARGS[@]}" >& $tmplog
	fi

	stopOnError $? $tmplog
	echo '' > $tmplog

	make -j $(nbcpu) >& $tmplog
	stopOnError $? $tmplog

	mv $tmplog build.log
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
        dirs=`find $dir -name 'CMakeLists.txt' -printf '%h\n'`
        for subdir in "$dirs" ; do
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

function merge_report(){
	find . -name 'build.log' -type f | xargs cat > build_report.log
}

function echo_usage(){
	echo "usage : $0 [options...] "
	echo "Following options are possible : "
	echo "codeblocks | CB : generates Codeblocks projects"
	echo "--force : always remove old build without asking"
	echo "--help | -h : displays this message"
}

export CMAKE_ARGS=''
export FORCE_REMOVE=''
REPORT=0

for ARG in $*
do
	case $ARG in
		"--help" | "-h")
			echo_usage
			exit 0
			;;
		"codeblocks" | "CB")
			echo "Will generate Codeblocks projects"
			export CMAKE_ARGS='CodeBlocks - Unix Makefiles'
			;;
		"--force")
			export FORCE_REMOVE='1'
			;;
		"--report")
			REPORT=1
			;;
	esac
done



echo "INFO : cmake well founded. Look what following to know if you need other software."

if [[ "$FORCE_REMOVE" == '' ]] ; then
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
fi

buildDir common
buildDir environment
buildDir agent

if [ $REPORT -eq 1 ] ; then
	merge_report
fi

