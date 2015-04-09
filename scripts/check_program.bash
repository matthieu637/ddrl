#!/bin/bash

function checker(){
	program=$1
	which $program >& /dev/null
	if [ $? -ne 0 ]; then
	        echo "ERROR : Please install '$program' to run this project."
		exit 1
	fi
}

function check_all(){
	checker wget
	checker tar
	checker unzip
	checker python
	checker cmake
	checker make
	checker g++
	checker ode-config
	checker astyle
	checker cppcheck
	checker cpplint
	checker valgrind
}

