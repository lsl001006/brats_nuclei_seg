for n in 1 2 
do
  python train_nuclei_seg.py -c config_nuclei_seg_all_fake_x${n}.json -d 2
done
