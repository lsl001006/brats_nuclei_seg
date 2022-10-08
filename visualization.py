from skimage import io
import numpy as np

root = "./experiments/lifelong_eccv/exp/models"
lifelong = "{:s}/lifelong_nuclei_exp3.4_feb25/0304_172626/test_results/".format(root) #lifelong
jl = "{:s}/lifelong_nuclei_nodropout_exp1.4_mar3/0304_171131/test_results".format(root) #joint learning
ft = "{:s}/lifelong_nuclei_nodropout_exp4.3_mar2/0304_172916/test_results".format(root) #finetune
local = "{:s}/lifelong_nuclei_nodropout_exp2.3_mar2/0304_221720/test_results".format(root) #localgan

nuclei_list = ["Breast_TCGA-E2-A1B5-01Z-00-DX1",
               "Breast_TCGA-E2-A14V-01Z-00-DX1",
               "Kidney_TCGA-B0-5698-01Z-00-DX1",
               "Kidney_TCGA-B0-5710-01Z-00-DX1",
               "Liver_TCGA-21-5784-01Z-00-DX1",
               "Liver_TCGA-21-5786-01Z-00-DX1",
               # "Prostate_TCGA-CH-5767-01Z-00-DX1",
               "Prostate_TCGA-G9-6362-01Z-00-DX1"]


def collect_results_brain():

    w = 1000
    space = 10
    # for i in range(7):
    montage_img = np.ones((2*w+2*space, 6*w + 10*space, 3), dtype=np.uint8) * 255

    imgname = nuclei_list[1]
    real = io.imread('{:s}/{:s}_orig.png'.format(lifelong, imgname))
    gt = io.imread('{:s}/{:s}_label.png'.format(lifelong, imgname))

    seg_lf = io.imread('{:s}/{:s}_pred.png'.format(lifelong, imgname))
    seg_ft = io.imread('{:s}/{:s}_pred.png'.format(ft, imgname))
    seg_jl = io.imread('{:s}/{:s}_pred.png'.format(jl, imgname))
    seg_local = io.imread('{:s}/{:s}_pred.png'.format(local, imgname))

    montage_img[:w, :w, :] = real
    montage_img[:w, w+2*space:2*w+2*space, :] = gt
    montage_img[:w, 2*w+4*space:3*w+4*space, :] = seg_lf
    montage_img[:w, 3*w+6*space:4*w+6*space, :] = seg_ft
    montage_img[:w, 4*w+8*space:5*w+8*space, :] = seg_jl
    montage_img[:w, 5*w+10*space:6*w+10*space, :] = seg_local

    imgname = nuclei_list[6]
    real = io.imread('{:s}/{:s}_orig.png'.format(lifelong, imgname))
    gt = io.imread('{:s}/{:s}_label.png'.format(lifelong, imgname))

    seg_lf = io.imread('{:s}/{:s}_pred.png'.format(lifelong, imgname))
    seg_ft = io.imread('{:s}/{:s}_pred.png'.format(ft, imgname))
    seg_jl = io.imread('{:s}/{:s}_pred.png'.format(jl, imgname))
    seg_local = io.imread('{:s}/{:s}_pred.png'.format(local, imgname))

    montage_img[w+2*space:2*w+2*space, :w, :] = real
    montage_img[w+2*space:2*w+2*space, w + 2 * space:2 * w + 2 * space, :] = gt
    montage_img[w+2*space:2*w+2*space, 2 * w + 4 * space:3 * w + 4 * space, :] = seg_lf
    montage_img[w+2*space:2*w+2*space, 3 * w + 6 * space:4 * w + 6 * space, :] = seg_ft
    montage_img[w+2*space:2*w+2*space, 4 * w + 8 * space:5 * w + 8 * space, :] = seg_jl
    montage_img[w+2*space:2*w+2*space, 5 * w + 10 * space:6 * w + 10 * space, :] = seg_local

    io.imsave('./experiments/eccv/Nuclei_vis/cmp.png'.format(imgname), montage_img)

collect_results_brain()