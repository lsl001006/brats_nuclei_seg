for n in `seq 0 9`
do
  python train_seg.py -c config_files/config_seg_random_${n}.json -d 2
done
