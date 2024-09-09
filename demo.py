import argparse
import glob
import json
import math
import os

import numpy as np
import torch
from oc_cost3d import OC_Cost3D
from tqdm import tqdm

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='calculate OC cost')

    parser.add_argument(
        '-exp', '--experiment_name', help='experiment name', type=str)
    parser.add_argument(
        '--exp_dir', help='experiment directory', type=str, default=None
    )
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
    parser.add_argument(
        '--sm', help="single model", type=str, default=None
    )
    parser.add_argument(
        '--save_metric', help="save metric to the folder", action='store_true'
    )

    args = parser.parse_args()


    iou_type = None
    if args.iou_mode:
        iou_type = "iou"
    elif args.giou_bb_mode:
        iou_type = "giou_bb"
    elif args.giou_ch_mode:
        iou_type = "giou_ch"
    else:
        raise Exception("IoU type not specified, choose one among: --iou_mode, --giou_bb_mode, --giou_ch_mode")

    save_folder = None
    if args.save_metric:
        save_folder = f"./metric_results/{args.experiment_name}/{iou_type}"
        os.makedirs(f"{save_folder}", exist_ok=True)

    if not args.exp_dir:
        experiment_name = args.experiment_name
        pred_path = f"../minsu3d/output/PartNetSim/PointGroup/{experiment_name}/inference/val/predictions/instance"
    else:
        pred_path = f"{args.exp_dir}/inference/val/predictions/instance"
        experiment_name = args.exp_dir.split("/")[-5]

    gt_path = args.truth
    model_ids = [path.split('/')[-1].split('.')[0] for path in glob.glob(f"{pred_path}/*.txt")]
    total_occost = 0

    if args.sm:
        model_ids = [args.sm]

    per_part_cost = {"drawer": 0.0, "door": 0.0, "lid": 0.0, "base": 0.0}
    per_part_count = {"drawer": 0.0, "door": 0.0, "lid": 0.0, "base": 0.0}

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

        if args.save_metric:
            with open(f"{save_folder}/{model_id}.txt", "w+") as f:
                f.write(f"{cost}")


    oc_cost = total_occost / len(model_ids)
    if args.save_metric and not args.sm:
        with open(f"{save_folder}/average.txt", "w+") as f:
            f.write(f"{oc_cost}")
    print("OC-cost: ", oc_cost)
