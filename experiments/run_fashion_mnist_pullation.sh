#!/bin/bash

for perc_pullation in 5 25 50 100
do
    echo "Manual Seed: 0"
    echo "Running Fashion-MNIST. Anomaly Class: Bag "
    echo "Percentage level of pullation: $perc_pullation/100"
    python train.py --dataset fashion_mnist --isize 32 --nc 1 --niter 15 --abnormal_class "Bag" --manualseed 0 --perc_pullation $perc_pullation
done
exit 0