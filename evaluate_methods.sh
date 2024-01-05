#!/bin/bash

#names=("mask3d-partnetsim-20-10queries-7threshold" "mask3d-partnetsim-20-10queries-8threshold" "partnetsim_pointgroup_elastic_off_normals" "pointnext-pointgroup-bsize8" "pointnext-clustering-20-best")
names=("swin3d-pointgroup-full-val-bsize8")
iou_modes=("iou_mode" "giou_bb_mode" "giou_ch_mode")
for name in "${names[@]}"
do
   for iou_mode in "${iou_modes[@]}"
   do
      python demo.py --exp ${name} --${iou_mode} --save_metric
   done
done