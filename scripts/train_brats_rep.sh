for n in `seq 0 4`
do
    CUDA_VISBLE_DEVICES=1 python train_seg.py \
    -c /home/csgrad/xuangong/dffed/brats_nuclei_seg/config_files/config_seg_tumor_size_${n}.json \
    -d 1
done