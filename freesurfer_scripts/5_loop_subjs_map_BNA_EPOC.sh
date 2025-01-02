#!/bin/bash

ARCHDIR=/ritter/share/data/EPOC/EPOC_BIDS/EPOC_derivatives

# Loop over directories
for DIR in ${ARCHDIR}/sub-epoc*
do
	SUBJID=${DIR/"${ARCHDIR}/"/}
	echo $SUBJID
	if [ ! -e ${ARCHDIR}/${SUBJID}/freesurfer/touch/bna.done ]
	then
        ./4_subj_level_map_BNA_EPOC.sh $SUBJID $ARCHDIR
	fi
done;

