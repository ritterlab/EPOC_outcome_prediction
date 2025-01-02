#!/bin/bash

export FREESURFER_HOME=/opt/freesurfer
export FS_LICENSE=$/opt/freesurfer/license.txt 
source $FREESURFER_HOME/SetUpFreeSurfer.sh


SUBJECTS_DIR=/ritter/share/data/EPOC/EPOC_BIDS/EPOC_derivatives
PREFIX=sub-epoc*/freesurfer
OUTPUT=/ritter/marija/output/EPOC
curdir=`pwd` 

cd $SUBJECTS_DIR

aparcstats2table --subjects ${PREFIX} --hemi lh --parc BN_Atlas --meas thickness --tablefile \
$OUTPUT/lh.thickness.txt --skip
 

aparcstats2table --subjects ${PREFIX} --hemi rh --parc BN_Atlas --meas thickness --tablefile \
$OUTPUT/rh.thickness.txt --skip
 

aparcstats2table --subjects ${PREFIX} --hemi lh --parc BN_Atlas --meas volume --tablefile \
$OUTPUT/lh.volume.txt --skip
 

aparcstats2table --subjects ${PREFIX} --hemi rh --parc BN_Atlas --meas volume --tablefile \
$OUTPUT/rh.volume.txt --skip
 

asegstats2table --subjects ${PREFIX} --tablefile $OUTPUT/aseg_stats.txt --skip

 
cd $curdir

