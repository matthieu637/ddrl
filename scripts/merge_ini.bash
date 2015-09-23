#!/bin/bash

if [ $# -ne 3 ] ; then
	echo "Usage $0 : <base> <injector> <destination>"
	exit 1
fi

base=$1
injector=$2
destination=$3

if [ ! -e $base ] ; then
	echo "$base doesn't exist !"
	exit 1
fi

if [ ! -e $injector ] ; then
	echo "$injector doesn't exist !"
	exit 1
fi

tmp=`mktemp`

#
# Read the base file and copy it to $tmp by remplacing the duplicate
#
while read line
do
	isComment=`echo "$line" | grep -e '^#' | wc -l`
	isSection=`echo "$line" | grep -e '^[[]' | wc -l`
	isBlank=`echo "$line" | grep -v -e '^#' | grep -v -e '^[[]' | grep -v '[=]' | wc -l`

	if [[ $isComment -eq 1 || $isSection -eq 1 || $isBlank -eq 1 ]] ; then
		echo "$line" >> $tmp
	else
		key=`echo "$line" | sed -e 's/[_]/\\\_/g' | cut -d'=' -f1`
		isDuplicate=`grep -i "^$key" $injector | wc -l`

		if [ $isDuplicate -eq 1 ]; then
			newL=`grep -i "$key" $injector`
			echo "$newL" >> $tmp
		else
			echo "$line" >> $tmp
		fi
	fi

done < $base

#
# Inject a line before a specific [section]
#
# <file> <section> <line>
function inject_line_before_section(){
	file=$1
	section=$2
	line=$3

	str_sections_base=`cat $base | grep -e '^[[]'`
	read -a sections_base <<< $str_sections_base
	
	i=0;
	secfinal=""
	for sec in "${sections_base[@]}"
	do	
		if [[ "$sec" == "$section"  ]] ;
		then
			i=`expr $i + 1`
			secfinal="${sections_base[$i]}"
			break
		fi
		i=`expr $i + 1`
	done

	section="$secfinal"

	isBlankSec=`echo "$section" | grep -v -e '^[[]' | wc -l`
	if [[ $isBlankSec -eq 1 ]]; then
		echo "$line" >> $file
	else
		#escape [section] to \[section\]
		section=`echo $section | sed -e 's/[[]/\\\[/' | sed -e 's/[]]/\\\]/' `
		sed -i "s/\($section\)/$line\n\1/" $file
	fi
}

#
# Put the rest (new key=value + comment) from injector to $tmp
#

currentSection=-1
str_sections=`cat $injector | grep -e '^[[]'`
read -a sections <<< $str_sections

while read line
do
	isComment=`echo "$line" | grep -e '^#' | wc -l`
	isSection=`echo "$line" | grep -e '^[[]' | wc -l`
	isKey=`echo "$line" | grep -v -e '^#' | grep -v -e '^[[]' | grep -e '[=]' | wc -l`

	if [ $isSection -eq 1 ] ; 
	then
		currentSection=`expr $currentSection + 1`
	elif [[ $isComment -eq 1 || $isKey -eq 0 ]] ; 
	then
		inject_line_before_section $tmp "${sections[$currentSection]}" "$line"
	elif [ $isKey -eq 1 ] ; 
	then
		key=`echo "$line" | sed -e 's/[_]/\\\_/g' | cut -d'=' -f1`
		isDuplicate=`grep -i "^$key" $base | wc -l`

		if [ $isDuplicate -eq 0 ] ; then
			inject_line_before_section $tmp "${sections[$currentSection]}" "$line"
		fi
	fi

done < $injector

mv $tmp $destination
