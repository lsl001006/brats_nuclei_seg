import os
import h5py
import SimpleITK as sitk
import nibabel as nib
from PIL import Image, ImageEnhance
import numpy as np

path = "/a2il/data/xuangong/BRATS/2018_seg/HGG_tumor_size_split_10/BraTS18_tumor_size_0/train.h5"
data = h5py.File(path, 'r')
print(data.keys())
# print(np.unique(data['labels']))
import pdb;pdb.set_trace()
# 

# path = "/a2il/data/xuangong/BRATS/2018/HGG/Brats18_TCIA08_469_1"
# t2 = "/a2il/data/xuangong/BRATS/2018/HGG/Brats18_TCIA08_469_1/Brats18_TCIA08_469_1_t2.nii.gz"
# seg = "/a2il/data/xuangong/BRATS/2018/HGG/Brats18_TCIA08_469_1/Brats18_TCIA08_469_1_seg.nii.gz"
# # s = sitk.ReadImage(seg)
# # s = sitk.GetArrayFromImage(s)
# s = nib.load(t2).get_fdata()
# s = np.array(s)
# print(s.shape)
# print(type(s))

# for i in range(s.shape[2]):
#     if np.count_nonzero(s[:,:,i]) > 0:
#         print(f'{i} not zero')
#         img = Image.fromarray(s[:,:,i]).convert('RGB')
#         # img = ImageEnhance.Contrast(img).enhance()
#         img.save('./1.png')
#         print(img.size)
#         # print(np.unique(img))
        
#         import pdb;pdb.set_trace()



