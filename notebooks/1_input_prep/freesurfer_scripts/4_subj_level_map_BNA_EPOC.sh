#!/bin/bash

export FREESURFER_HOME=/opt/freesurfer
export FS_LICENSE=$/opt/freesurfer/license.txt 
source $FREESURFER_HOME/SetUpFreeSurfer.sh


SUBJID=$1
ARCHDIR=$2
ATLASDIR="/ritter/marija/atlas/BN_Atlas_freesurfer"
SUBJECTS_DIR=${ARCHDIR}/${SUBJID}
DIR="freesurfer"
touch ${SUBJECTS_DIR}/${DIR}/touch/bna.done

# Map BNA to subjectslabel

if [ ! -e ${SUBJECTS_DIR}/${DIR}/label/rh.BN_Atlas.annot ]
then
        echo ${SUBJECTS_DIR}/${DIR}/surf/lh.sphere.reg

        mris_ca_label -t ${ATLASDIR}/BN_Atlas_210_LUT.txt \
        -l ${SUBJECTS_DIR}/${DIR}/label/lh.cortex.label \
        ${DIR} lh ${SUBJECTS_DIR}/${DIR}/surf/lh.sphere.reg \
        ${ATLASDIR}/lh.BN_Atlas.gcs \
        ${SUBJECTS_DIR}/${DIR}/label/lh.BN_Atlas.annot

 
        mris_ca_label -t ${ATLASDIR}/BN_Atlas_210_LUT.txt \
        -l ${SUBJECTS_DIR}/${DIR}/label/rh.cortex.label \
        ${DIR} rh ${SUBJECTS_DIR}/${DIR}/surf/rh.sphere.reg \
        ${ATLASDIR}/rh.BN_Atlas.gcs \
        ${SUBJECTS_DIR}/${DIR}/label/rh.BN_Atlas.annot
        
                
fi

if [ ! -e ${SUBJECTS_DIR}/${DIR}/stats/rh.BN_Atlas.stats ]
then
        mris_anatomical_stats -mgz -cortex ${SUBJECTS_DIR}/${DIR}/label/lh.cortex.label -f \
        ${SUBJECTS_DIR}/${DIR}/stats/lh.BN_Atlas.stats -b -a ${SUBJECTS_DIR}/${DIR}/label/lh.BN_Atlas.annot -c \
        ${ATLASDIR}/BN_Atlas_210_LUT.txt ${DIR} lh white

 
        mris_anatomical_stats -mgz -cortex ${SUBJECTS_DIR}/${DIR}/label/rh.cortex.label -f \
        ${SUBJECTS_DIR}/${DIR}/stats/rh.BN_Atlas.stats -b -a ${SUBJECTS_DIR}/${DIR}/label/rh.BN_Atlas.annot -c \
        ${ATLASDIR}/BN_Atlas_210_LUT.txt ${DIR} rh white
fi
