#!/bin/bash

for perc_pullation in 0 5 25 50 100
do
    echo "Manual Seed: 0"
    echo "Running MNIST, Abnormal Digit: 2"
    echo "Percentage level of pullation: $perc_pullation/100"
    python train.py --dataset mnist --isize 32 --nc 1 --niter 15 --abnormal_class 2 --manualseed 0 --perc_pullation $perc_pullation
done
exit 0