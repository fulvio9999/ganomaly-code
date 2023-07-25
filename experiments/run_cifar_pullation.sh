#!/bin/bash

for perc_pullation in 5 25 50 100
do
    echo "Manual Seed: 0"
    echo "Running CIFAR. Anomaly Class: frog "
    echo "Percentage level of pullation: $perc_pullation/100"
    python train.py --dataset cifar10 --isize 32 --niter 15 --abnormal_class "frog" --manualseed 0 --perc_pullation $perc_pullation
done
exit 0

