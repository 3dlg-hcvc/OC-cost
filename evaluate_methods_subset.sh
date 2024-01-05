#!/bin/bash

names=("subset_partnetsim_20k_vertices/mask3d-10queries-7threshold" "subset_partnetsim_20k_vertices/mask3d-10queries-8threshold" "subset_partnetsim_20k_vertices/pointgroup_best" "subset_partnetsim_20k_vertices/pointnext_clustering_best" "subset_partnetsim_20k_vertices/pointnext_pointgroup_best")

iou_modes=("iou_mode" "giou_bb_mode" "giou_ch_mode")
for name in "${names[@]}"
do
   for iou_mode in "${iou_modes[@]}"
   do
      python demo.py --exp ${name} --${iou_mode} --save_metric -gt "../minsu3d/data/partnetsim-20k-vertices-subset/val"
   done
done