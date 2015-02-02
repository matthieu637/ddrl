#!/bin/bash

function nbcpu(){
	cat /proc/cpuinfo | grep processor | wc -l
}

