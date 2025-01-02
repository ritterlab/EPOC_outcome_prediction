#!/bin/bash

PROCESSNUM=$1

ARCHDIR=/ritter/share/data/EPOC_BIDS/EPOC_rawdata
DERIVDIR=/ritter/share/data/EPOC_BIDS/EPOC_derivatives
FILES=(${ARCHDIR}/sub-epoc*)

cmds_per_file=23
initial_index=$(( PROCESSNUM * cmds_per_file))
last_index=$(( initial_index + cmds_per_file))


for ((y=$initial_index; y<$last_index; y++));
do
    SUBJID_DIR=${FILES[y]/"${ARCHDIR}/"/}
    echo $SUBJID_DIR
    ./1_subj_level_reconall_EPOC.sh $SUBJID_DIR $ARCHDIR $DERIVDIR
done

