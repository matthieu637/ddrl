#!/bin/bash

function checker(){
	program=$1
	which $program >& /dev/null
	if [ $? -ne 0 ]; then
	        echo "ERROR : Please install '$program' to run this project."
		exit
	fi
}

checker cmake
checker make
checker c++
checker ode-config
checker astyle
checker cppcheck

