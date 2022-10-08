import numpy as np
import h5py
import os
import parse
# h5_file = h5py.File('/share_hd1/db/BRATS/brats18_for_seg/BraTS18_whole_syn/brats_multilabel_whole_syn_db_resnet_9blocks_new_x1_epoch200.h5','r')
#
# labels = h5_file['labels']
# keys = list(labels.keys())
#
# for key in keys:
#     label = labels[key][()]
#     print(np.unique(label))


def parse_results(data_dir):
    folders = os.listdir(data_dir)
    results = {}
    for folder in sorted(folders):
        filename = '{:s}/{:s}/info.log'.format(data_dir, folder)
        with open(filename, 'r') as file:
            line1 = file.readline()
            line2 = file.readline()
            epoch = parse.search('epoch{:d}', line1)[0]
            dice = parse.search("'dice': {:.f}", line2)[0]
            sensitivity = parse.search("'sensitivity': {:.f}", line2)[0]
            specificity = parse.search("'specificity': {:.f}", line2)[0]
            if epoch not in results:
                results[epoch] = {'dice': dice, 'sensitivity': sensitivity, 'specificity': specificity}
    return results

results = parse_results('./experiments/seg/real_imgs_unet/log/HGG_LGG_All_seg_test')
for k, data in results.items():
    print('{:d}\tdice: {:.4f}\tsensitivity: {:.4f}\tspecificity: {:.4f}'.format(k, data['dice'], data['sensitivity'],
                                                                                  data['specificity']))