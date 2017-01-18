#!/bin/bash

if [ "$(uname)" == "Darwin" ]; then
	export MAC=1
	export LINUX=0
	export FIND=gfind
else
	export MAC=0
	export LINUX=1
	export FIND=find
fi
