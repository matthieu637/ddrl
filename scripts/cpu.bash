#!/bin/bash

function nbcpu(){
	if [ $MAC ]; then
		sysctl machdep.cpu.thread_count | sed 's/^.*: //'
	else
		cat /proc/cpuinfo | grep processor | wc -l
	fi
}

