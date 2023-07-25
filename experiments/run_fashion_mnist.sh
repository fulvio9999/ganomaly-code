#!/bin/bash


# Run FASHION-MNIST experiment on ganomaly

# declare -a 
# arr="T-shirt/top" "Trouser" "Pullover" "Dress" "Coat" "Sandal" "Shirt" "Sneaker" "Bag" "Ankle boot"
for m in $(seq 0 2)
do
    echo "Manual Seed: $m"
    for i in "T-shirt/top" "Trouser" "Pullover" "Dress" "Coat" "Sandal" "Shirt" "Sneaker" "Bag" "Ankle boot": #"${arr[@]}";
    do
        echo "Running FASHION-MNIST. Anomaly Class: $i "
        python train.py --dataset fashion_mnist --isize 32 --nc 1 --niter 15 --abnormal_class $i --manualseed $m
    done
done
exit 0



# for m in $(seq 0 2)
# do
#     echo "Manual Seed: $m"
#     echo "Running MNIST, Abnormal Digit: 2"
#     python train.py --dataset mnist --isize 32 --nc 1 --niter 15 --abnormal_class 2 --manualseed $m --contamination_perc 0.1
# done
# exit 0


# for perc_pullation in 0 5 25 50 100
# do
#     echo "Manual Seed: 0"
#     echo "Running MNIST, Abnormal Digit: 2"
#     echo "Percentage level of pullation: $perc_pullation/100"
#     python train.py --dataset mnist --isize 32 --nc 1 --niter 15 --abnormal_class 2 --manualseed 0 --perc_pullation $perc_pullation
# done
# exit 0