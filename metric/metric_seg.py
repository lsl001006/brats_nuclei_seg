import torch
import numpy as np
from scipy.spatial.distance import directed_hausdorff
from numpy.core.umath_tests import inner1d
from skimage import morphology


def accuracy(output, target):
    with torch.no_grad():
        pred = torch.sigmoid(output) > 0.5
        pred = pred.view([pred.size(0), -1]).float()
        target = target.view([target.size(0), -1]).float()
        assert pred.size() == target.size()

        TP = torch.sum((pred == 1) * (target == 1), dim=1).float()
        FP = torch.sum((pred == 1) * (target == 0), dim=1).float()
        TN = torch.sum((pred == 0) * (target == 0), dim=1).float()
        FN = torch.sum((pred == 0) * (target == 1), dim=1).float()
        acc = torch.mean((TP + TN) / (TP + TN + FP + FN + 1e-8)).item()
    return acc


def dice(output, target):
    with torch.no_grad():
        pred = torch.sigmoid(output) > 0.5
        pred = pred.view([pred.size(0), -1]).float()
        target = target.view([target.size(0), -1]).float()
        assert pred.size() == target.size()

        TP = torch.sum((pred == 1) * (target == 1), dim=1).float()
        FP = torch.sum((pred == 1) * (target == 0), dim=1).float()
        FN = torch.sum((pred == 0) * (target == 1), dim=1).float()
        dice = torch.mean((2*TP) / (2*TP + FP + FN + 1e-8)).item()
    return dice


def sensitivity(output, target):
    with torch.no_grad():
        pred = torch.sigmoid(output) > 0.5
        pred = pred.view([pred.size(0), -1]).float()
        target = target.view([target.size(0), -1]).float()
        assert pred.size() == target.size()

        TP = torch.sum((pred == 1) * (target == 1), dim=1).float()
        FN = torch.sum((pred == 0) * (target == 1), dim=1).float()
        sensi_value = torch.mean(TP / (TP + FN + 1e-8)).item()
    return sensi_value


def specificity(output, target):
    with torch.no_grad():
        pred = torch.sigmoid(output) > 0.5
        pred = pred.view([pred.size(0), -1]).float()
        target = target.view([target.size(0), -1]).float()
        assert pred.size() == target.size()

        FP = torch.sum((pred == 1) * (target == 0), dim=1).float()
        TN = torch.sum((pred == 0) * (target == 0), dim=1).float()
        speci_value = torch.mean(TN / (TN + FP + 1e-8)).item()
    return speci_value


def hausdorff95(output, target):
    with torch.no_grad():
        # pred = torch.sigmoid(output) > 0.5
        
        pred = output > 0.5
        target = target > 0.5
        assert pred.size() == target.size()
        
        pred = pred.detach().cpu().numpy()
        target = target.detach().cpu().numpy()
        
        
        
        hd95 = 0.0
        n = 0
        for k in range(pred.shape[0]):
            if np.count_nonzero(pred[k]) > 0:  # need to handle blank prediction
                n += 1
                pred_contours = pred[k] & (~morphology.binary_erosion(pred[k]))
                # print(type(target[k]))
                # import pdb;pdb.set_trace()
                target_contours = target[k] & (~morphology.binary_erosion(target[k]))
                pred_ind = np.argwhere(pred_contours)
                target_ind = np.argwhere(target_contours)
                hd95 += _haus_dist_95(pred_ind, target_ind)
    return hd95, n


def _haus_dist_95(A, B):
    """ compute the 95 percentile hausdorff distance """
    # Find pairwise distance
    D_mat = np.sqrt(inner1d(A, A)[np.newaxis].T + inner1d(B, B) - 2 * (np.dot(A, B.T)))
    dist1 = np.min(D_mat, axis=0)
    dist2 = np.min(D_mat, axis=1)
    hd95 = np.percentile(np.hstack((dist1, dist2)), 95)

    # hd = np.max(np.array([np.max(np.min(D_mat, axis=0)), np.max(np.min(D_mat, axis=1))]))

    return hd95


