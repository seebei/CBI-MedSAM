# -*- coding: utf-8 -*-
import numpy as np
from medpy import metric
from scipy.spatial.distance import directed_hausdorff

def compute_dice_coefficient(mask_gt, mask_pred):
  volume_sum = mask_gt.sum() + mask_pred.sum()
  if volume_sum == 0:
    return np.NaN
  volume_intersect = (mask_gt & mask_pred).sum()
  return 2*volume_intersect / volume_sum
 

def compute_HD_distances(mask_gt, mask_pred):
  if mask_gt.sum() <= 0 or mask_pred.sum() <= 0:
    return 100.0
  hd = metric.binary.hd(mask_pred, mask_gt)
  hd = np.fabs(hd)
  return hd


def compute_HD95(gt_mask, pred_mask):
    """
    Compute 95th percentile Hausdorff Distance between predicted and ground truth masks.
    Args:
        gt_mask (ndarray): binary ground truth mask [H, W]
        pred_mask (ndarray): binary predicted mask [H, W]
    Returns:
        float: HD95 distance
    """
    # 提取边缘点坐标
    gt_coords = np.argwhere(gt_mask > 0)
    pred_coords = np.argwhere(pred_mask > 0)

    if len(gt_coords) == 0 or len(pred_coords) == 0:
        return 999.0  # 代表预测或 GT 全空，最大惩罚

    # 计算双向最近距离
    from scipy.spatial import cKDTree
    tree1 = cKDTree(gt_coords)
    tree2 = cKDTree(pred_coords)

    dists1, _ = tree1.query(pred_coords)
    dists2, _ = tree2.query(gt_coords)

    hd95 = max(np.percentile(dists1, 95), np.percentile(dists2, 95))
    return hd95

def compute_metrics(y_true, y_pred):
    """
    输入:
    - y_true: [H, W] 或 [B, H, W] 的 ground truth mask，类型 uint8，0/1
    - y_pred: 与 y_true 相同形状的预测 mask，类型 uint8，0/1

    输出:
    - acc: accuracy
    - se: sensitivity (recall)
    - sp: specificity
    """

    y_true = y_true.astype(np.bool_)
    y_pred = y_pred.astype(np.bool_)

    tp = np.logical_and(y_true, y_pred).sum()
    tn = np.logical_and(~y_true, ~y_pred).sum()
    fp = np.logical_and(~y_true, y_pred).sum()
    fn = np.logical_and(y_true, ~y_pred).sum()

    acc = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    se = tp / (tp + fn + 1e-8)
    sp = tn / (tn + fp + 1e-8)

    return acc, se, sp