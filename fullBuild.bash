#!/bin/bash

OUTSIDE_PATH="eclipse"

LIB=$(dirname "${BASH_SOURCE[0]}")
cd $LIB

. scripts/os.bash
. scripts/cpu.bash
#check cmake if installed
. scripts/check_program.bash
check_all

export CPU=$(nbcpu)

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
	cmakelist='../../'
	if [ $# -eq 3 ] ; then
		cmakelist=$3
	else
		mkdir build/$1
		cd build/$1
	fi

	tmplog=`$MKTEMP`


	if [[ "${CMAKE_ARGS[@]}" == '' ]] ; then
		cmake $cmakelist -DCMAKE_BUILD_TYPE=$2 >& $tmplog
	else
		eval "cmake $cmakelist -DCMAKE_BUILD_TYPE=$2 ${CMAKE_ARGS[@]} >& $tmplog"
	fi

	stopOnError $? $tmplog
	cp $tmplog cmake.log
	echo '' > $tmplog

	if [ $XCODE -eq 0 ] ; then
		make -j $CPU >& $tmplog
		stopOnError $? $tmplog
	else
		xcodebuild >& $tmplog
		stopOnError $? $tmplog
	fi

	mv $tmplog build.log
	cd ../..
}

#recall cmake & make in a multicore maner
function cmakeBuildRecall(){
	directory=$1

	cmakelist='../../'
	if [ $# -eq 2 ] ; then
		cmakelist=$2
	else
		cd build/$directory
	fi

	tmplog=`$MKTEMP`
	cmake $cmakelist >& $tmplog
	stopOnError $? $tmplog
	cp $tmplog cmake.log
	echo '' > $tmplog

	if [ $XCODE -eq 0 ] ; then
		make -j $CPU >& $tmplog
		stopOnError $? $tmplog
	else
		xcodebuild >& $tmplog
		stopOnError $? $tmplog
	fi

	mv $tmplog build.log

	cd ../..
}

#found all the CMakeLists.txt and create 3 targets to build (release, debug, release with debug info)
function buildDir(){
        dir=$1
        if [ ! -e $dir ] ; then
                echo "Please cd into the root directory of DRL"
                exit 1
        fi

        here=`pwd`
        for subdir in $($FIND $dir/ -maxdepth 2 -name 'CMakeLists.txt' -printf '%h\n' | sort -rn | grep -v old) ; do
		
                cd $here/$subdir

		if [ $CLEAR -eq 1 ] ; then
			rm -rf build
			rm -rf lib
			echo "INFO : $subdir cleared."
			continue
		fi

                if [ -e build ]; then
                        if [ $FORCE_REMOVE -eq 0 ]; then
                                echo "INFO : $subdir already contains a build directory. Just recall..."
				if [ $BUILD_DEBUG -eq  1 ] ; then
					if [ -e build/debug ] ; then
						cmakeBuildRecall debug
					else
						cmakeBuild debug Debug
					fi
				else
					if [ -e build/release ] ; then
						cmakeBuildRecall release
					else
						cmakeBuild release Release
					fi

					if [ -e build/debug ] ; then
						cmakeBuildRecall debug
					else
						cmakeBuild debug Debug
					fi
	
					if [ $BUILD_RELWITHDEB -eq 1 ] ; then
						if [ -e build/relwithdeb ] ; then
							cmakeBuildRecall relwithdeb
						else
		                			cmakeBuild relwithdeb RelWithDebInfo
						fi
					fi
				fi

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
		if [ $BUILD_DEBUG -eq  1 ] ; then
			cmakeBuild debug Debug
		else
	                cmakeBuild release Release
	                cmakeBuild debug Debug
			if [ $BUILD_RELWITHDEB -eq 1 ] ; then
		                cmakeBuild relwithdeb RelWithDebInfo
			fi
		fi

                echo "INFO : $subdir well builed. Congratz."
                hr
        done
	cd $here
}

#found all the CMakeLists.txt and create 3 targets to build (release, debug, release with debug info)
function buildDirOutside(){
        dir="../$1"
        if [ ! -e $dir ] ; then
                echo "Please cd into the root directory of DRL"
                exit 1
        fi

        here=`pwd`
        for subdir in $($FIND $dir/ -name 'CMakeLists.txt' -printf '%h\n') ; do
		#echo "init subdir $subdir"
		linkdepth=`echo "$subdir" | sed -e 's/^\(\([.][.]\/\)*\).*$/\1/'`
		rsubdir="$subdir"
		subdir=`echo "$subdir" | sed 's/[.][.]\///g'`
		cd $here

		if [ $CLEAR -eq 1 ] ; then
			rm -rf project.$subdir/pfiles/lib
			rm -rf project.$subdir
			echo "INFO : $subdir cleared."
			continue
		fi

		if [ $FORCE_REMOVE -eq 0 ]; then
			if [ -e project.$subdir ] ; then
				echo "INFO : $subdir already contains a build directory. Just recall..."
				cd project.$subdir
				cmakeBuildRecall debug pfiles/
				continue
			fi
		else
			echo "INFO : $subdir already contains a build directory. Removing it..."
			rm -rf project.$subdir
		fi

		mkdir -p project.$subdir
                cd project.$subdir
		if [ ! -e pfiles ] ; then
	                nb_dir=`echo $rsubdir | grep -o '/' | wc -l`
	                path='.'
	                for i in $(seq 1 1 $nb_dir) ; do
		               path="$path/.."
		        done
			
			#ln -s $path/${linkdepth}${subdir} pfiles
			fp=$(dirname $here)
			ln -s $fp/$subdir pfiles
		fi

                if [[ -e pfiles/lib && $FORCE_REMOVE -eq 1 ]]; then
                        rm -rf pfiles/lib/
                fi

                #building

		export CMAKE_ARGS="-DROOT_DRL_PATH=$(dirname $here)  $CMAKE_ARGS"
                cmakeBuild debug Debug pfiles/

                echo "INFO : $subdir well builed. Congratz."
                hr
        done
	cd $here
}



function merge_report(){
	$FIND . -name 'build.log' -type f | xargs cat > build_report.log
}

function outside_prepare(){
	mkdir project.$1
	cd project.$1
	ln -s ../../$1 $1
}

function echo_usage(){
	echo "usage : $0 [options...] "
	echo "Following options are possible : "
	echo "codeblocks | CB : generates Codeblocks projects"
	echo "eclipse | EC : generates Eclipse projects"
	echo "xcode | XC : generates Xcode projects"
	echo "--force : always remove old build without asking"
	echo "--clear : remove old build (without build)"
	echo "--debug : build only debug"
	echo "--report : merge all report into one (build_report.log) usefull for continuous integration"
	echo "-j n : limit the build with n cpu (use all by default)"
	echo "-with-relwithdeb : build also relwithdeb"
	echo "--help | -h : displays this message"
}

export CMAKE_ARGS=''
export FORCE_REMOVE=''
export CLEAR=0
export XCODE=0
export BUILD_DEBUG=0
export BUILD_RELWITHDEB=0
REPORT=0
BUILD_OUTSIDE=0

for ARG in $*
do
	case $ARG in
		"--help" | "-h")
			echo_usage
			exit 0
			;;
		"codeblocks" | "CB")
			echo "Will generate Codeblocks projects"
			export CMAKE_ARGS='-G "CodeBlocks - Unix Makefiles"'
			;;
		"eclipse" | "EC")
			echo "Will generate Eclipse projects"
			export CMAKE_ARGS='-G "Eclipse CDT4 - Unix Makefiles"'
			BUILD_OUTSIDE=1
			;;
		"xcode" | "XC")
			echo "Will generate Xcode projects"
			export CMAKE_ARGS='-G "Xcode"'
			XCODE=1
			;;
		"--force")
			export FORCE_REMOVE='1'
			;;
		"--report")
			REPORT=1
			;;
		"--clear")
			CLEAR=1
			;;
		"--debug")
			export BUILD_DEBUG=1
			;;
		"-j")
			shift
			export CPU=$1
			;;
		"-with-relwithdeb")
			export BUILD_RELWITHDEB=1
			;;
	esac
done


echo "INFO : $CPU CPU used"
echo "INFO : cmake well founded. Look what following to know if you need other software."

if [[ "$FORCE_REMOVE" == '' && $CLEAR -eq 0 ]]  ; then
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

if [ $BUILD_OUTSIDE -eq 1 ] ; then
	if [ ! -e $OUTSIDE_PATH ] ; then
		mkdir $OUTSIDE_PATH
	fi

	here_r=$(pwd)
	cd $OUTSIDE_PATH
	buildDirOutside common
	buildDirOutside environment
	buildDirOutside agent

	cd $here_r
	if [ $CLEAR -eq 1 ]; then
		rm -rf $OUTSIDE_PATH
	fi
else
	buildDir common
	buildDir environment
	buildDir agent
fi

if [ $REPORT -eq 1 ] ; then
	merge_report
fi

