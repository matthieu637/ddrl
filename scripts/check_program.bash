#!/bin/bash

function checker(){
	program=$1
	which $program >& /dev/null
	error=$?
	if [[ $error -ne 0 && $# -eq 1 ]] ; then
	        echo "ERROR : Please install '$program' to run this project."
		exit 1
	elif [[ $error -ne 0 ]] ; then
		echo "optional install : $program"
	fi
}

function check_all_optional(){
	checker astyle 0
	checker cppcheck 0
	checker cpplint 0
	checker valgrind 0
	checker xml 0
	checker wget 0
	checker tar 0
	checker unzip 0
	checker python 0
	checker gdb 0
}

function check_all(){
	checker cmake
	checker make
	checker g++
	checker git
#	checker ode-config #if ode is not installed, we will compile it
#	if it's here, check the version
	which ode-config >& /dev/null
	if [ $? -eq 0 ] ; then # installed
		vercomp $(ode-config --version) 0.14
		if [ $? -eq 2 ] ; then
			echo "ode version is too low (0.14 required)"
			exit 1
		fi	
	else #not install need libtoolize for ode building
		checker libtoolize
	fi
	check_all_optional
}

vercomp () {
    if [[ $1 == $2 ]]
    then
        return 0
    fi
    local IFS=.
    local i ver1=($1) ver2=($2)
    # fill empty fields in ver1 with zeros
    for ((i=${#ver1[@]}; i<${#ver2[@]}; i++))
    do
        ver1[i]=0
    done
    for ((i=0; i<${#ver1[@]}; i++))
    do
        if [[ -z ${ver2[i]} ]]
        then
            # fill empty fields in ver2 with zeros
            ver2[i]=0
        fi
        if ((10#${ver1[i]} > 10#${ver2[i]}))
        then
            return 1
        fi
        if ((10#${ver1[i]} < 10#${ver2[i]}))
        then
            return 2
        fi
    done
    return 0
}

