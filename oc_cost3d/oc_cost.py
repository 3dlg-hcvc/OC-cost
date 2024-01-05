import numpy as np
from .optimization import OCOpt
from .Annotations import BBox, predBBox, Annotations
import pulp
from scipy.spatial import ConvexHull, HalfspaceIntersection, Delaunay
from scipy.optimize import linprog


class OC_Cost3D:
    def __init__(self, lm=1, iou_mode=False, giou_bb_mode=False, giou_ch_mode=False):
        self.lm = lm
        if iou_mode:
            self.mode = "iou"
        elif giou_bb_mode:
            self.mode = "giou_bb"
        elif giou_ch_mode:
            self.mode = "giou_ch"
        self.mask_labels = []

    def getIntersectUnion(self, gt_mask, pred_mask):
        if self.mode == "giou_bb" or self.mode == "iou":
            # https://pbr-book.org/3ed-2018/Geometry_and_Transformations/Bounding_Boxes#:~:text=The%20intersection%20of%20two%20bounding,Intersection%20of%20Two%20Bounding%20Boxes.
            max_min = np.maximum(np.min(gt_mask["xyz"][gt_mask["mask"]], axis=0), np.min(pred_mask["xyz"][pred_mask["mask"]], axis=0))
            min_max = np.minimum(np.max(gt_mask["xyz"][gt_mask["mask"]], axis=0), np.max(pred_mask["xyz"][pred_mask["mask"]], axis=0))

            intersection_dims = np.maximum(0, min_max - max_min)
            intersection_volume = np.prod(intersection_dims)

            gt_volume = np.prod(np.max(gt_mask["xyz"][gt_mask["mask"]], axis=0) - np.min(gt_mask["xyz"][gt_mask["mask"]], axis=0))
            pred_volume = np.prod(np.max(pred_mask["xyz"][pred_mask["mask"]], axis=0) - np.min(pred_mask["xyz"][pred_mask["mask"]], axis=0))
            union_volume = gt_volume + pred_volume - intersection_volume
        if self.mode == "giou_ch":
            gt_hull = ConvexHull(gt_mask["xyz"][gt_mask["mask"]])
            pred_hull = ConvexHull(pred_mask["xyz"][pred_mask["mask"]])
            
            # adapted from https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.HalfspaceIntersection.html#scipy.spatial.HalfspaceIntersection
            halfspaces = np.vstack([gt_hull.equations, pred_hull.equations])
            norm_vector = np.reshape(np.linalg.norm(halfspaces[:, :-1], axis=1), (halfspaces.shape[0], 1))
            c = np.zeros((halfspaces.shape[1],))
            c[-1] = -1
            A = np.hstack((halfspaces[:, :-1], norm_vector))
            b = - halfspaces[:, -1:]
            res = linprog(c, A_ub=A, b_ub=b, bounds=(None, None))
            feasible_point = res.x[:-1]
            intersection = HalfspaceIntersection(halfspaces, feasible_point)
            intersection_volume = ConvexHull(intersection.intersections).volume

            union_volume = gt_hull.volume + pred_hull.volume - intersection_volume
        
        return intersection_volume, union_volume

    def getIOU(self, gt_mask, pred_mask):

        intersect, union = self.getIntersectUnion(gt_mask, pred_mask)

        iou = intersect / (union)
        return iou

    def getGIOU(self, gt_mask, pred_mask):
        intersect, union = self.getIntersectUnion(gt_mask, pred_mask)
        iou = intersect / union

        if self.mode == "giou_bb":
            min = np.minimum(np.min(gt_mask["xyz"][gt_mask["mask"]], axis=0), np.min(pred_mask["xyz"][pred_mask["mask"]], axis=0))
            max = np.maximum(np.max(gt_mask["xyz"][gt_mask["mask"]], axis=0), np.max(pred_mask["xyz"][pred_mask["mask"]], axis=0))
            
            min_enclosing_bbox_dims = max - min
            c_volume = np.prod(min_enclosing_bbox_dims)
        else:
            all_points = np.concatenate((gt_mask["xyz"][gt_mask["mask"]], pred_mask["xyz"][pred_mask["mask"]]))
            min_eclosing_hull = ConvexHull(all_points)
            c_volume = min_eclosing_hull.volume
        
        giou = iou - (c_volume - union) / c_volume
        return giou

    def getCloc(self, gt_mask, pred_mask):
        cost: float = 0
        if self.mode == "iou":
            cost = (1 - self.getIOU(gt_mask, pred_mask)) / 2
        else:
            cost = (1 - self.getGIOU(gt_mask, pred_mask)) / 2
        
        return cost

    def getCcls(self, gt_mask, pred_mask):
        clt = gt_mask["label"]
        clp = pred_mask["label"]

        preci = pred_mask["conf"]
        ccls = 0.5
        if clt == clp:
            ccls = (1 - preci) / 2
        else:
            ccls = (1 + preci) / 2
        return ccls

    def getoneCost(self, gt_mask, pred_mask):
        Cloc = self.getCloc(gt_mask, pred_mask)
        CCls = self.getCcls(gt_mask, pred_mask)

        return (self.lm * Cloc) + ((1 - self.lm) * CCls)

    def build_C_matrix(self, gt, preds):
        n = len(gt["masks"])
        m = len(preds["masks"])

        self.cost = np.zeros((m, n))

        for i in range(m):
            for j in range(n):
                gt_mask = {"mask": gt["masks"][j], "label": gt["gt_labels"][j], "xyz": gt["xyz"]}
                pred_mask = {"mask": preds["masks"][i], "label": preds["pred_labels"][i], "conf": preds["conf"][i], "xyz": gt["xyz"]}
                self.cost[i][j] = self.getoneCost(gt_mask, pred_mask)
        return self.cost

    def optim(self, beta):
        m = self.cost.shape[0] + 1
        n = self.cost.shape[1] + 1
        opt = OCOpt(m, n, beta)
        opt.set_cost_matrix(self.cost)
        opt.setVariable()
        opt.setObjective()
        opt.setConstrain()

        result = opt.prob.solve(pulp.PULP_CBC_CMD(
            msg=0, timeLimit=100))
        p_matrix = np.zeros((m, n))

        # print('objective value: {}'.format(pulp.value(opt.prob.objective)))
        # print('solution')
        for i in range(opt.m):
            for j in range(opt.n):
                #print(f'{opt.variable[j][i]} = {pulp.value(opt.variable[j][i])}')
                p_matrix[j][i] = pulp.value(opt.variable[j][i])
        p_matrix[-1][-1] = 0
        p_tilde_matrix = p_matrix / np.sum(p_matrix)
        self.p_matrix = p_matrix
        self.p_tilde_matrix = p_tilde_matrix
        self.opt = opt
        return p_tilde_matrix
