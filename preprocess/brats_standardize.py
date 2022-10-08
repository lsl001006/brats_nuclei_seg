import os
# path = '/a2il/data/xuangong/BraTS2018/BraTS2018_Train'
path = '/a2il/data/xuangong/BraTS2018/BraTS2018_Validation'
dst = '/a2il/data/xuangong/BRATS/2018'
# if not os.path.exists(dst):
#     os.mkdir(dst)
hgg_dst = os.path.join(dst, "HGG")
lgg_dst = os.path.join(dst, "LGG")

if not os.path.exists(hgg_dst):
    os.makedirs(hgg_dst, exist_ok=True)
if not os.path.exists(lgg_dst):
    os.makedirs(lgg_dst, exist_ok=True)

dirs = os.listdir(path)

for d in dirs:
    if d[:3] == 'HGG':
        ori_p = os.path.join(path, d)
        save_p = os.path.join(hgg_dst, d[4:])
        os.system(f'cp -r {ori_p} {save_p}')
        print(f'copying {ori_p} -> {save_p}')
    if d[:3] == 'LGG':
        ori_p = os.path.join(path, d)
        save_p = os.path.join(lgg_dst, d[4:])
        os.system(f'cp -r {ori_p} {save_p}')
        print(f'copying {ori_p} -> {save_p}')
print('complete')