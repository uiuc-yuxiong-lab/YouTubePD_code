#!/bin/sh
test_dir=$1
src_dir=$2
gt_dir=$3

echo "Calculate FID Score..."
python -m pytorch_fid $src_dir $test_dir

echo "Calculate IS Score..."
python inception_score.py --path $test_dir

echo "Calculate CLS score..."
python cls_score.py --path $test_dir 

echo "Calculate Direction Score..."
python vgg_sim.py --path_src $src_dir --path_gt $gt_dir --path_gen $test_dir