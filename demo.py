import numpy as np
from oc_cost3d.oc_cost import OC_Cost3D
from tqdm import tqdm
import json
import argparse
import math
import glob
import os
import torch

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='calculate OC cost')

    parser.add_argument(
        '-exp', '--experiment_name', help='experiment name', type=str)
    parser.add_argument(
        '-gt', '--truth', help='minsu3d gt directory', type=str, default="../minsu3d/data/partnetsim/val")
    parser.add_argument(
        '-lm', '--lam', help='Lambda parameter', default=1
    )
    parser.add_argument(
        '-b', '--beta', help='beta parameter', default=0.5
    )
    parser.add_argument(
        '--iou_mode', help="turn on iou mode", action='store_true'
    )
    parser.add_argument(
        '--giou_bb_mode', help="turn on giou bounding box mode", action='store_true'
    )
    parser.add_argument(
        '--giou_ch_mode', help="turn on giou convex hull mode", action='store_true'
    )

    args = parser.parse_args()

    experiment_name = args.experiment_name
    pred_path = f"../minsu3d/output/PartNetSim/PointGroup/{experiment_name}/inference/val/predictions/instance"

    gt_path = args.truth
    model_ids = [path.split('/')[-1].split('.')[0] for path in glob.glob(f"{pred_path}/*.txt")]
    total_occost = 0

    for model_id in tqdm(model_ids):
        with open(f"{pred_path}/{model_id}.txt") as file:
            lines = file.readlines()
            lines = [line.rstrip() for line in lines]
        instanceFileNames = []
        labelIndexes = []
        confidenceScores = []
        predicted_mask_list = []
        for i in lines:
            splitedLine = i.split()
            instanceFileNames.append(os.path.join(pred_path, splitedLine[0]))
            labelIndexes.append(splitedLine[1])
            confidenceScores.append(float(splitedLine[2]))

        for instanceFileName in instanceFileNames:
            predicted_mask_list.append(np.loadtxt(instanceFileName, dtype=bool))
        
        preds = {"pred_labels": labelIndexes, "conf": confidenceScores, "masks": predicted_mask_list}

        gt_data = torch.load(f"{gt_path}/{model_id}.pth")
        gt_instance_ids =  gt_data["instance_ids"]
        gt_semantic_instance_labels = []
        gt_instance_masks = []
        for instance_id in np.unique(gt_instance_ids):
            instance_mask = gt_data["instance_ids"] == instance_id
            gt_instance_masks.append(instance_mask)
            gt_semantic_instance_labels.append(np.where(gt_data["instance_ids"] == instance_id)[0])
        
        gt = {"gt_labels": gt_semantic_instance_labels, "masks": gt_instance_masks, "xyz": gt_data["xyz"]}

        occost = OC_Cost3D(float(args.lam), args.iou_mode, args.giou_bb_mode, args.giou_ch_mode)
    
        c_matrix = occost.build_C_matrix(gt, preds)
        pi_tilde_matrix = occost.optim(float(args.beta))
        cost = np.sum(np.multiply(pi_tilde_matrix, occost.opt.cost))
        if math.isnan(cost):
            cost = 0
        total_occost += cost

    oc_cost = total_occost / len(model_ids)
    print(oc_cost)
