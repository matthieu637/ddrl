#!/bin/bash

function checker(){
	program=$1
	which $program >& /dev/null
	if [[ $? -ne 0 && $# -eq 1 ]] ; then
	        echo "ERROR : Please install '$program' to run this project."
		exit 1
	elif [[ $? -ne 0 ]] ; then
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
	checker ode-config
	check_all_optional
}
