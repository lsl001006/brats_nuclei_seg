for n in 'liver' 'kidney' 'prostate' 
do
  python train_nuclei_seg.py -c config_nuclei_seg_${n}.json -d 3
done
