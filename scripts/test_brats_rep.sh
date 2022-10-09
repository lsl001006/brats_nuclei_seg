for n in `seq 1 10`
do
    python test_seg.py \
    -c /home/csgrad/xuangong/dffed/brats_nuclei_seg/config_files/config_brats_test.json \
    -d 2 -r /home/csgrad/xuangong/dffed/brats_nuclei_seg/experiments/seg/real_imgs_unet_test/models/brats_seg_test/1008_234154/model_best.pth
done