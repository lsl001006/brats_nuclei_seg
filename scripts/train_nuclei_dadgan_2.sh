for n in 1 2 
do
  python train_nuclei_seg.py -c config_nuclei_seg_all_dadgan_epoch300_x${n}.json -d 0
done
