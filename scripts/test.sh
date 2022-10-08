for n in `seq 1 80`
do
  python test_seg.py -c config_seg_all_HGG_LGG.json -d 2 -r /share_hd1/projects/code_seg_v2/experiments/seg/real_imgs_unet/models/LGG_All_seg_train/1103_215750/checkpoint-epoch${n}.pth
done
