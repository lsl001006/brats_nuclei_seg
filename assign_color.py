

import os, random, json
import numpy as np
from scipy import misc
import utils


def get_random_color():
    ''' generate rgb using a list comprehension '''
    r, g, b = [random.random() for i in range(3)]
    return r, g, b



# list_postfix = ['isbi19', 'point', 'rand', 'un']
# data_dir = './experiments/LC/final'
# name = 'MV-5'

list_postfix = ['FullNet', 'point', 'rand', 'un']
data_dir = './experiments/MO/final'
name = 'Breast_TCGA-E2-A14V-01Z-00-DX1'

gt_path = '{:s}/{:s}_label.png'.format(data_dir, name)
gt = misc.imread(gt_path)
unique_vals = np.unique(gt)
Ng = len(unique_vals) - 1

# colorize gt
gt_colored = np.zeros((gt.shape[0], gt.shape[1], 3))
color_list = {}
for k in range(1, Ng+1):
    idx = unique_vals[k]
    color_list[idx] = np.array(get_random_color())
    gt_colored[gt == idx, :] = color_list[idx]

save_path = '{:s}/{:s}-gt.png'.format(data_dir, name)
misc.imsave(save_path, gt_colored)

# assign same color for TPs
for postfix in list_postfix:
    pred_path = '{:s}/{:s}_seg_{:s}.tiff'.format(data_dir, name, postfix)
    pred_img = misc.imread(pred_path)

    unique_vals_pred = np.unique(pred_img)
    Ns = len(unique_vals_pred) - 1

    pred_colored = np.zeros((gt.shape[0], gt.shape[1], 3))
    gt_copy = np.copy(gt)
    for i in range(1, Ns + 1):
        idx = unique_vals_pred[i]
        pred_i = np.where(pred_img == idx, 1, 0)
        overlap_parts = pred_i * gt_copy

        # get intersection objects numbers in image
        obj_no = np.unique(overlap_parts)
        obj_no = obj_no[obj_no != 0]

        if obj_no.size == 0:
            pred_colored[pred_img == idx, :] = np.array(get_random_color())
        elif obj_no.size == 1:
            pred_colored[pred_img == idx, :] = color_list[obj_no[0]]
            gt_copy[gt_copy == obj_no[0]] = 0
        else:  # more than one overlapped preds
            #  find max overlap object
            obj_areas = [np.sum(overlap_parts == i) for i in obj_no]
            seg_obj = obj_no[np.argmax(obj_areas)]  # segmented object number
            pred_colored[pred_img == idx, :] = color_list[seg_obj]
            gt_copy[gt_copy == obj_no[0]] = 0

        # utils.show_figures((gt_colored, pred_img, pred_colored, overlap_parts))

    # save colored images
    save_path = '{:s}/{:s}-seg-{:s}.png'.format(data_dir, name, postfix)
    misc.imsave(save_path, pred_colored)






