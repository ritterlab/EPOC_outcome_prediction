#!/bin/bash

for i in {0..10}
do
   echo "Start process $i"
   ./2_loop_subjs_reconall_EPOC.sh $i &
   sleep 1
done