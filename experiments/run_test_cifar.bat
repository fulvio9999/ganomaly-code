@echo off

for %%P in (0 5 25 50 100) do (
    for %%C in (0 1 2 3 4 5) do (
        python test.py --dataset cifar10 --isize 32 --nc 3 --niter 15 --abnormal_class "frog" --manualseed 0 --class_test "frog" --perc_pullation %%P --count_test %%C
    )
)

exit /b 0
