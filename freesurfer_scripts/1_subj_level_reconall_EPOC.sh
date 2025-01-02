#!/bin/bash

export FREESURFER_HOME=/opt/freesurfer
export FS_LICENSE=$/opt/freesurfer/license.txt 
source $FREESURFER_HOME/SetUpFreeSurfer.sh

# Define paths to data
SUBJID=$1
ARCHDIR=$2
DERIVDIR=$3

T1DIR=`ls ${ARCHDIR}/${SUBJID}/anat/*_T1w.nii.gz`

#if [ ! -e ${DERIVDIR}/${SUBJID}/freesurfer ] && [ -e $T1DIR ]
#then

        mkdir ${DERIVDIR}/${SUBJID}
        #mkdir ${DERIVDIR}/${SUBJID}/freesurfer
        start=$(date)
        echo "Start running recon-all $SUBJID"
        recon-all -subjid freesurfer -i ${T1DIR} -all -sd ${DERIVDIR}/${SUBJID} # /freesurfer
        echo ${start} > $SDERIVDIR/${SUBJID}/date_script.txt
        date >> $DERIVDIR/${SUBJID}/date_script.txt
#fi





#ARCHDIR=/ritter/share/data/EPOC_BIDS/EPOC_rawdata
#DERIVDIR=/ritter/share/data/EPOC_BIDS/EPOC_derivatives
#SUBJID= the sub- folder inside raw

