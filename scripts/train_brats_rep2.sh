for n in `seq 5 9`
do
    CUDA_VISIBLE_DEVICES=1 python train_seg.py \
    -c /home/csgrad/xuangong/dffed/brats_nuclei_seg/config_files/config_seg_tumor_size_${n}.json \
    -d 1
done