#!/bin/bash

for p in $(seq 0 5 25 50 100)
do
    echo "Percentuale contaminazione: $p"
    for c in $(seq 0 1 2 3 4 5)
    do
        echo "Count: $c "
        python test.py --dataset fashion_mnist --isize 32 --nc 1 --niter 15 --abnormal_class "Bag" --manualseed 0 --class_test "Bag" --perc_pullation $p --count_test $c
    done
done
exit 0